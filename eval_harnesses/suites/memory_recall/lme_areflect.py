"""A-2: re-test `areflect` as an alternate recall path against the LME banks.

`areflect` is Hindsight's synthesis endpoint: instead of returning a flat
top-K fact list (`arecall`), it runs a query against the bank and returns a
reasoned/structured result. It timed out at 300s during M-tier ingest in the
original sprint; the banks are idle now, so this re-runs it cleanly.

Per qid:
    1. areflect(bank, question, tags=[qid]) — time it, capture the full raw result
    2. extract a text answer from the result (best-effort across likely fields)
    3. judge that answer vs gold with the same gemini judge as lme_judge

Output rows are shaped to be diffable against an arecall baseline via
`lme_compare.py` (same qid / judge_label / judge_verdict / question_type keys).
Because areflect synthesizes its own answer, there is no separate answer-LLM
call and no top-K context window; session_hit / n_facts fields are left null.

The result schema is NOT assumed — `areflect` was never observed returning
successfully here, so the full payload is stored under `reflect_raw` and the
answer is pulled from the first present of a few likely text fields, falling
back to a JSON dump of the whole payload. Inspect `reflect_raw` after the
first run to pin the real field, then tighten `_extract_answer` if useful.

Usage:
    python -m eval_harnesses.suites.memory_recall.lme_areflect \\
        --tier s --qids 1c0ddc50,1c549ce4 --bank-prefix lme_s \\
        --model-judge gemini-2.5-flash \\
        --out .eval_cache/lme_results/areflect_s.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.global_config import HINDSIGHT_URL
from src.memory.backend.hindsight import HindsightAPIError, HindsightRESTClient

from .lme_judge import (
    JUDGE_PROMPT,
    _gemini_call,
    _parse_verdict,
    _stream_load_qids,
)
from .lme_smoke import TIER_FILES

# Fields areflect might carry the synthesized answer in. Ordered by likelihood;
# first present, non-empty string wins. Unknown until first successful run.
_ANSWER_FIELDS = ("answer", "response", "summary", "result", "text", "content")


def _extract_answer(reflect: Any) -> str:
    """Best-effort pull of a text answer from an areflect payload."""
    if isinstance(reflect, str):
        return reflect
    if isinstance(reflect, dict):
        for f in _ANSWER_FIELDS:
            v = reflect.get(f)
            if isinstance(v, str) and v.strip():
                return v
        # nested {"result": {...}} or similar — dig one level
        for v in reflect.values():
            if isinstance(v, dict):
                for f in _ANSWER_FIELDS:
                    inner = v.get(f)
                    if isinstance(inner, str) and inner.strip():
                        return inner
    return json.dumps(reflect, ensure_ascii=False)


async def _reflect_one(
    client: HindsightRESTClient, q: Dict[str, Any], bank: str,
    model_judge: str, timeout: float,
) -> Dict[str, Any]:
    qid = q["question_id"]
    reflect_raw: Any = None
    reflect_error: Optional[str] = None
    t0 = time.monotonic()
    try:
        reflect_raw = await asyncio.wait_for(
            client.areflect(bank, q["question"], tags=[qid]), timeout=timeout
        )
    except asyncio.TimeoutError:
        reflect_error = f"timeout>{timeout:.0f}s"
    except HindsightAPIError as e:
        reflect_error = f"{e.status_code} {e}"
    except Exception as e:  # noqa: BLE001 — capture, don't crash the sweep
        reflect_error = repr(e)
    t_reflect = round(time.monotonic() - t0, 2)

    predicted = "" if reflect_error else _extract_answer(reflect_raw)

    judge_raw = ""
    judge_label: Optional[bool] = None
    judge_error: Optional[str] = None
    if predicted.strip():
        try:
            judge_raw = _gemini_call(
                JUDGE_PROMPT.format(
                    question=q["question"], gold=q["answer"], predicted=predicted,
                ),
                model_judge,
            )
            judge_label = _parse_verdict(judge_raw)
        except Exception as e:  # noqa: BLE001
            judge_error = repr(e)

    verdict = ("yes" if judge_label is True
               else "no" if judge_label is False
               else "unparseable")

    return {
        "qid": qid,
        "question_type": q["question_type"],
        "bank": bank,
        "question": q["question"],
        "gold_answer": q["answer"],
        # null fields: areflect has no top-K context window or fact list
        "n_facts_retrieved": None,
        "retrieved_session_ids": None,
        "session_hit_any": None,
        "session_hit_rate": None,
        "predicted_answer": predicted,
        "reflect_raw": reflect_raw,
        "reflect_error": reflect_error,
        "judge_raw": judge_raw,
        "judge_error": judge_error,
        "judge_verdict": verdict,
        "judge_label": judge_label,
        "judge_model": model_judge,
        "answer_model": "areflect-native",
        "timings_s": {"reflect": t_reflect},
    }


async def main(
    tier: str, qids: List[str], bank_prefix: str, model_judge: str,
    out: Path, bank_suffix: str = "", timeout: float = 300.0,
) -> int:
    src = TIER_FILES[tier]
    if not src.exists():
        raise SystemExit(f"missing {tier} JSON at {src}")
    print(f"Streaming {tier} for {len(qids)} qids...", file=sys.stderr)
    by_id = _stream_load_qids(src, set(qids))
    missing = [q for q in qids if q not in by_id]
    if missing:
        raise SystemExit(f"qids not in {tier}: {missing}")

    client = HindsightRESTClient(HINDSIGHT_URL, timeout=timeout + 30.0)
    results: List[Dict[str, Any]] = []
    for qid in qids:
        q = by_id[qid]
        bank = f"{bank_prefix}_{qid}{bank_suffix}"
        print(f"\n--- {qid} [{q['question_type']}] bank={bank} ---", file=sys.stderr)
        r = await _reflect_one(client, q, bank, model_judge, timeout)
        results.append(r)
        status = r["reflect_error"] or f"judge={r['judge_verdict']}"
        print(
            f"  reflect={r['timings_s']['reflect']}s {status}  "
            f"pred={r['predicted_answer'][:100].replace(chr(10), ' ')!r}",
            file=sys.stderr,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}", file=sys.stderr)

    n = len(results)
    yes = sum(1 for r in results if r["judge_label"] is True)
    errs = sum(1 for r in results if r["reflect_error"])
    print(f"OVERALL n={n}  judge_yes={yes}  reflect_errors={errs}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIER_FILES.keys()), default="s")
    ap.add_argument("--qids", required=True,
                    help="comma-separated question_ids")
    ap.add_argument("--bank-prefix", required=True,
                    help="bank name is f'{prefix}_{qid}{suffix}'")
    ap.add_argument("--bank-suffix", default="")
    ap.add_argument("--model-judge", default="gemini-2.5-flash")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-qid areflect timeout seconds (default 300)")
    ap.add_argument("--out", type=Path,
                    default=Path(".eval_cache/lme_results/areflect.json"))
    args = ap.parse_args()
    qids_list = [q.strip() for q in args.qids.split(",") if q.strip()]
    raise SystemExit(asyncio.run(main(
        args.tier, qids_list, args.bank_prefix, args.model_judge,
        args.out, args.bank_suffix, args.timeout,
    )))
