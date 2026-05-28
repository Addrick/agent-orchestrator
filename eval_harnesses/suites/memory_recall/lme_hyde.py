"""HyDE (Hypothetical Document Embeddings) probe for LongMemEval recall.

Targets the 1c0ddc50 retrieval-ranking failure: the natural question vector
("activities during my commute") sits far from the gold fact vectors (specific
history-podcast titles). HyDE generates a hypothetical *answer* to the question
and recalls on that instead — the hypothetical answer is lexically/semantically
closer to the stored facts than the bare question is.

Per qid this runs BOTH:
    baseline:  arecall(bank, question)        -> answer -> judge
    hyde:      arecall(bank, hypothetical)    -> answer -> judge

so the delta is directly comparable in one output file. Everything downstream
of recall (answer model, judge model, context budget, K) is identical to
lme_judge, which this imports from.

Usage:
    python -m eval_harnesses.suites.memory_recall.lme_hyde \
        --tier s --qids 1c0ddc50 --bank-prefix lme_s \
        --out .eval_cache/lme_results/hyde_1c0ddc50.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from config.global_config import HINDSIGHT_URL
from src.memory.backend.hindsight import HindsightRESTClient

from .lme_smoke import TIER_FILES
from .lme_judge import (
    _stream_load_qids,
    _score_at_k,
    _gemini_call,
)

# Classic HyDE: ask the model to write the passage it *would* expect to find,
# not to answer for real. No bank context — the point is a topical decoy vector.
HYDE_PROMPT = """A user asked the following question of a personal assistant \
that has access to the user's own conversation history.

QUESTION: {question}

Write a short, plausible passage (2-4 sentences) of the kind that might appear \
in that user's history and would directly answer this question. Invent concrete \
specifics (names, titles, preferences) as needed — this is a retrieval decoy, \
not a real answer. Write only the passage, no preamble."""


async def _recall_score(
    client: HindsightRESTClient, q: Dict[str, Any], bank: str,
    query_text: str, top_k: int, max_tokens: int,
    model_answer: str, model_judge: str,
) -> Dict[str, Any]:
    gold_sessions = set(q["answer_session_ids"])
    t0 = time.monotonic()
    hits = await client.arecall(bank, query_text, tags=[q["question_id"]],
                                max_tokens=max_tokens)
    t_recall = time.monotonic() - t0
    scored = _score_at_k(q, hits or [], gold_sessions, top_k,
                         model_answer, model_judge)
    scored["n_facts_retrieved"] = len(hits or [])
    scored["recall_query"] = query_text
    scored["timings_s"]["recall"] = round(t_recall, 2)
    return scored


async def main(
    tier: str, qids: List[str], bank_prefix: str, bank_suffix: str,
    model_answer: str, model_judge: str, top_k: int, max_tokens: int,
    out: Path,
) -> int:
    src = TIER_FILES[tier]
    if not src.exists():
        raise SystemExit(f"missing {tier} JSON at {src}")
    by_id = _stream_load_qids(src, set(qids))
    missing = [q for q in qids if q not in by_id]
    if missing:
        raise SystemExit(f"qids not in {tier}: {missing}")

    client = HindsightRESTClient(HINDSIGHT_URL, timeout=300.0)
    results: List[Dict[str, Any]] = []
    for qid in qids:
        q = by_id[qid]
        bank = f"{bank_prefix}_{qid}{bank_suffix}"
        print(f"\n--- {qid} [{q['question_type']}] bank={bank} ---", file=sys.stderr)

        # Baseline: recall on the raw question.
        base = await _recall_score(client, q, bank, q["question"],
                                   top_k, max_tokens, model_answer, model_judge)
        print(f"  baseline: sess={'OK' if base['session_hit_any'] else 'MISS'} "
              f"judge={base['judge_verdict']} "
              f"pred={base['predicted_answer'][:80]!r}", file=sys.stderr)

        # HyDE: generate hypothetical passage, recall on that.
        hyde_doc = _gemini_call(HYDE_PROMPT.format(question=q["question"]),
                                model=model_answer)
        print(f"  hyde_doc={hyde_doc[:120]!r}", file=sys.stderr)
        hyde = await _recall_score(client, q, bank, hyde_doc,
                                   top_k, max_tokens, model_answer, model_judge)
        print(f"  hyde:     sess={'OK' if hyde['session_hit_any'] else 'MISS'} "
              f"judge={hyde['judge_verdict']} "
              f"pred={hyde['predicted_answer'][:80]!r}", file=sys.stderr)

        results.append({
            "qid": qid,
            "question_type": q["question_type"],
            "bank": bank,
            "question": q["question"],
            "gold_answer": q["answer"],
            "gold_session_ids": sorted(gold_sessions := set(q["answer_session_ids"])),
            "answer_model": model_answer,
            "judge_model": model_judge,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "hyde_doc": hyde_doc,
            "baseline": base,
            "hyde": hyde,
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}", file=sys.stderr)

    print("\n=== baseline vs HyDE ===", file=sys.stderr)
    for r in results:
        b, h = r["baseline"], r["hyde"]
        print(f"{r['qid']:<12} baseline: sess_hit={b['session_hit_any']} "
              f"judge={b['judge_verdict']:<11} | "
              f"hyde: sess_hit={h['session_hit_any']} "
              f"judge={h['judge_verdict']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIER_FILES.keys()), default="s")
    ap.add_argument("--qids", required=True)
    ap.add_argument("--bank-prefix", required=True)
    ap.add_argument("--bank-suffix", default="")
    ap.add_argument("--model-answer", default="gemini-2.5-flash")
    ap.add_argument("--model-judge", default="gemini-2.5-flash")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    qids_list = [q.strip() for q in args.qids.split(",") if q.strip()]
    raise SystemExit(asyncio.run(main(
        args.tier, qids_list, args.bank_prefix, args.bank_suffix,
        args.model_answer, args.model_judge, args.top_k, args.max_tokens,
        args.out,
    )))
