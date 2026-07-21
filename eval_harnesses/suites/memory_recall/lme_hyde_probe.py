"""One-shot HyDE probe on a single LongMemEval qid.

For each variant bank requested:
  1. Generate a hypothetical answer to the question (gemini-CLI).
  2. Use the hypothetical answer (not the question) as the recall query.
  3. Answer the question from the retrieved facts.
  4. Judge predicted vs gold.

Compare against baseline lme_judge results.

Usage:
    python -m eval_harnesses.suites.memory_recall.lme_hyde_probe \
        --tier s --qid 1c0ddc50 \
        --banks lme_s_1c0ddc50,lme_s_1c0ddc50_v2a,lme_s_1c0ddc50_v3a \
        --out .eval_cache/lme_results/hyde_1c0ddc50.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from config.global_config import HINDSIGHT_URL
from src.memory.backend.hindsight import HindsightRESTClient

from .lme_judge import (
    _stream_load_qids, _gemini_call, _parse_verdict,
    _fact_text, _hit_session_id, ANSWER_PROMPT, JUDGE_PROMPT,
)
from .lme_smoke import TIER_FILES


HYDE_PROMPT = """You are generating a hypothetical answer to a question for retrieval purposes.

The hypothetical answer will be embedded and used to retrieve relevant facts from a memory store. It does not need to be true or correct — the goal is to produce text whose embedding overlaps the embedding of true facts that answer the question.

Be specific. Use likely concrete terms (named entities, categories, activities) a real answer would contain. Aim for ~80-150 words. Do not hedge or note that the answer is hypothetical.

Question: {question}

Hypothetical answer:"""


async def probe_bank(
    client: HindsightRESTClient, bank: str, q: Dict[str, Any],
    hypothetical: str, top_k: int, max_tokens: int,
    model_answer: str, model_judge: str,
) -> Dict[str, Any]:
    qid = q["question_id"]
    gold_sessions = set(q["answer_session_ids"])

    t0 = time.monotonic()
    hits = await client.arecall(bank, hypothetical, tags=[qid], max_tokens=max_tokens)
    t_recall = time.monotonic() - t0
    hits = hits or []
    top = hits[:top_k]

    retrieved_sids = [s for s in (_hit_session_id(h) for h in top) if s]
    unique_sids = list(dict.fromkeys(retrieved_sids))
    sess_hit_any = bool(set(unique_sids) & gold_sessions)

    context = "\n".join(f"- {_fact_text(h)}" for h in top if _fact_text(h))
    ans_prompt = ANSWER_PROMPT.format(context=context, question=q["question"])

    t0 = time.monotonic()
    predicted = _gemini_call(ans_prompt, model=model_answer)
    t_answer = time.monotonic() - t0

    judge_prompt = JUDGE_PROMPT.format(
        question=q["question"], gold=q["answer"], predicted=predicted or "(empty)",
    )
    t0 = time.monotonic()
    judge_raw = _gemini_call(judge_prompt, model=model_judge)
    t_judge = time.monotonic() - t0

    verdict = _parse_verdict(judge_raw)
    return {
        "bank": bank,
        "hypothetical": hypothetical,
        "n_facts_retrieved": len(hits),
        "top_k": top_k,
        "retrieved_session_ids": unique_sids,
        "gold_session_ids": sorted(gold_sessions),
        "session_hit_any": sess_hit_any,
        "predicted_answer": predicted,
        "judge_raw": judge_raw,
        "judge_verdict": "yes" if verdict is True else "no" if verdict is False else "unparseable",
        "judge_label": verdict,
        "timings_s": {
            "recall": round(t_recall, 2),
            "answer": round(t_answer, 2),
            "judge": round(t_judge, 2),
        },
    }


async def main(tier: str, qid: str, banks: List[str], top_k: int, max_tokens: int,
               model_answer: str, model_judge: str, out: Path) -> None:
    tier_file = TIER_FILES[tier]
    qs = _stream_load_qids(tier_file, {qid})
    if qid not in qs:
        raise SystemExit(f"qid {qid} not found in {tier_file}")
    q = qs[qid]

    print(f"Question: {q['question']}")
    print(f"Gold:     {q['answer'][:200]}...\n")

    # Single hypothetical, reused across variants for apples-to-apples.
    print("Generating hypothetical answer via gemini-cli...")
    hypothetical = _gemini_call(HYDE_PROMPT.format(question=q["question"]),
                                model=model_answer)
    print(f"\nHypothetical:\n{hypothetical}\n")
    print("=" * 70)

    results = []
    client = HindsightRESTClient(HINDSIGHT_URL, timeout=300.0)
    for bank in banks:
        print(f"\nProbe bank: {bank}")
        r = await probe_bank(client, bank, q, hypothetical, top_k, max_tokens,
                             model_answer, model_judge)
        print(f"  sess_hit_any={r['session_hit_any']} judge={r['judge_verdict']}")
        print(f"  predicted: {r['predicted_answer'][:200]}...")
        results.append(r)

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "qid": qid,
        "tier": tier,
        "question": q["question"],
        "gold_answer": q["answer"],
        "hypothetical": hypothetical,
        "model_answer": model_answer,
        "model_judge": model_judge,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["s", "m"])
    ap.add_argument("--qid", required=True)
    ap.add_argument("--banks", required=True,
                    help="comma-separated bank ids to probe with same hypothetical")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--model-answer", default="gemini-2.5-flash")
    ap.add_argument("--model-judge", default="gemini-2.5-flash")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    banks = [b.strip() for b in args.banks.split(",") if b.strip()]
    asyncio.run(main(args.tier, args.qid, banks, args.top_k, args.max_tokens,
                     args.model_answer, args.model_judge, args.out))
