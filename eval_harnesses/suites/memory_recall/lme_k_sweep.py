"""K-sweep on a single qid: vary (max_tokens, top_k) to find the
budget/recall frontier. Prints per-config: facts_retrieved, history_in_topk,
predicted, judge_verdict. Single qid only — cheap, exploratory."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Tuple

from config.global_config import HINDSIGHT_URL
from src.memory.backend.hindsight import HindsightRESTClient

from .lme_judge import (
    ANSWER_PROMPT, JUDGE_PROMPT, _fact_text, _gemini_call, _hit_session_id,
    _parse_verdict,
)
from .lme_smoke import TIER_FILES


SWEEP: List[Tuple[int, int]] = [
    (128, 5),
    (256, 10),
    (512, 10),
    (1024, 15),
    (2048, 20),
]


async def main(tier: str, qid: str, bank: str, model: str) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    data = json.loads(Path(TIER_FILES[tier]).read_text(encoding="utf-8"))
    q = next(d for d in data if d["question_id"] == qid)
    gold_sessions = set(q["answer_session_ids"])
    print(f"QID: {qid}  bucket: {q['question_type']}")
    print(f"Q: {q['question']}")
    print(f"GOLD: {q['answer'][:240]}...")
    print()

    client = HindsightRESTClient(HINDSIGHT_URL, timeout=120.0)
    rows = []
    for max_toks, top_k in SWEEP:
        hits = await client.arecall(
            bank, q["question"], tags=[qid], max_tokens=max_toks,
        ) or []
        top = hits[:top_k]
        sids = list(dict.fromkeys(
            s for s in (_hit_session_id(h) for h in top) if s
        ))
        sess_hit = bool(set(sids) & gold_sessions)
        ctx = "\n".join(f"- {_fact_text(h)}" for h in top if _fact_text(h))
        # Detect whether genre-specific facts surfaced
        history_in_topk = any(
            "history" in (_fact_text(h) or "").lower() for h in top
        )

        ans = _gemini_call(ANSWER_PROMPT.format(context=ctx, question=q["question"]), model=model)
        jr = _gemini_call(
            JUDGE_PROMPT.format(question=q["question"], gold=q["answer"], predicted=ans or "(empty)"),
            model=model,
        )
        v = _parse_verdict(jr)

        rows.append({
            "max_tokens": max_toks, "top_k": top_k,
            "n_hits_returned": len(hits), "n_facts_in_context": len(top),
            "session_hit_any": sess_hit, "history_in_topk": history_in_topk,
            "predicted": ans, "judge_raw": jr, "judge_label": v,
        })
        print(f"[max_tok={max_toks:>4} top_k={top_k:>2}] hits={len(hits):>2} "
              f"history={'Y' if history_in_topk else 'n'} judge={v}  "
              f"pred={(ans or '')[:120]!r}")

    out = Path(".eval_cache/lme_results") / f"k_sweep_{qid}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="s", choices=list(TIER_FILES.keys()))
    ap.add_argument("--qid", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--model", default="gemini-2.5-flash")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.tier, args.qid, args.bank, args.model)))
