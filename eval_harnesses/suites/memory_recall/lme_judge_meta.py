"""Judge meta-evaluation: isolate judge-model quality from answer quality.

Motivation: the LongMemEval paper pins a single judge (gpt-4o-2024-08-06,
>97% human agreement) and varies the *answerer*. Our harness substitutes a
local gemini judge and to date pins answer-model == judge-model. Before fixing
a judge we want to know which gemini judge matches 4o-grade judgment — and the
only way to compare judges cleanly is to grade the *same* predicted answers
with each, so answer-generation variance doesn't confound the verdict.

Pipeline, per qid:
    1. arecall(bank, question, tags=[qid], max_tokens) -> top-K facts
    2. answer-generation ONCE with a fixed `--answer-model` -> frozen prediction
    3. for each `--judge-models` entry: grade the frozen (q, gold, pred) triple

Output:
    - JSON: per-qid frozen prediction + every judge's verdict
    - Markdown table (`<out>.md`): qid | qtype | gold | predicted | <judge cols>
      | agree? | human (blank) — sorted so judge-disagreement rows float to top
    - stdout: pairwise judge agreement %, and the disagreement rows to hand-check

The markdown table's `human` column is intentionally empty: fill it in to run
the paper-style agreement meta-eval (judge vs human) per candidate.

Specs are `tier:qid` pairs so a single run can mix S/M tiers:
    --specs s:1c0ddc50,s:50635ada,m:8fb83627
Bank name is f"lme_{tier}_{qid}" (matches the baseline ingest convention).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config.global_config import HINDSIGHT_URL
from src.memory.backend.hindsight import HindsightRESTClient

from .lme_judge import (
    ANSWER_PROMPT,
    JUDGE_PROMPT,
    _fact_text,
    _gemini_call,
    _parse_verdict,
    _stream_load_qids,
)
from .lme_smoke import TIER_FILES


def _parse_specs(raw: str) -> List[Tuple[str, str]]:
    specs: List[Tuple[str, str]] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise SystemExit(f"bad spec {tok!r}; expected tier:qid")
        tier, qid = tok.split(":", 1)
        if tier not in TIER_FILES:
            raise SystemExit(f"unknown tier {tier!r} in spec {tok!r}")
        specs.append((tier, qid.strip()))
    return specs


async def _frozen_prediction(
    client: HindsightRESTClient, q: Dict[str, Any], bank: str,
    top_k: int, max_tokens: int, answer_model: str,
) -> Dict[str, Any]:
    """Recall + generate the answer ONCE. This is the fixed input every judge sees."""
    hits = await client.arecall(bank, q["question"], tags=[q["question_id"]],
                                max_tokens=max_tokens)
    hits = hits or []
    facts = [_fact_text(h) for h in hits[:top_k] if _fact_text(h)]
    context = "\n".join(f"- {t}" for t in facts)
    ans_prompt = ANSWER_PROMPT.format(context=context, question=q["question"])
    try:
        predicted = _gemini_call(ans_prompt, model=answer_model)
        err = None
    except Exception as e:
        predicted, err = "", str(e)[:200]
    return {
        "qid": q["question_id"],
        "question_type": q["question_type"],
        "question": q["question"],
        "gold_answer": q["answer"],
        "n_facts": len(facts),
        "predicted_answer": predicted,
        "answer_error": err,
        "answer_model": answer_model,
    }


def _judge(pred: Dict[str, Any], judge_model: str) -> Dict[str, Any]:
    prompt = JUDGE_PROMPT.format(
        question=pred["question"], gold=pred["gold_answer"],
        predicted=pred["predicted_answer"] or "(empty)",
    )
    try:
        raw = _gemini_call(prompt, model=judge_model)
        err = None
    except Exception as e:
        raw, err = "", str(e)[:200]
    v = _parse_verdict(raw)
    return {
        "verdict": "yes" if v is True else "no" if v is False else "unparseable",
        "label": v, "raw": raw, "error": err,
    }


def _md_table(rows: List[Dict[str, Any]], judges: List[str]) -> str:
    def cell(s: str, n: int = 60) -> str:
        return (s or "").replace("\n", " ").replace("|", "\\|")[:n]

    head = ["qid", "qtype", "gold", "predicted"] + judges + ["agree?", "human"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    # disagreement rows first
    def disagree(r):
        labs = {r["judges"][j]["label"] for j in judges}
        return len(labs) > 1
    for r in sorted(rows, key=lambda r: (not disagree(r), r["qid"])):
        verdicts = [r["judges"][j]["verdict"] for j in judges]
        agree = "" if disagree(r) else "✓"
        lines.append("| " + " | ".join([
            r["qid"], r["question_type"], cell(r["gold_answer"], 40),
            cell(r["predicted_answer"], 70), *verdicts, agree, "",
        ]) + " |")
    return "\n".join(lines) + "\n"


def _print_summary(rows: List[Dict[str, Any]], judges: List[str]) -> None:
    n = len(rows)
    print(f"\n=== Judge meta-eval ({n} qids) ===")
    print("Per-judge yes-rate:")
    for j in judges:
        yes = sum(1 for r in rows if r["judges"][j]["label"] is True)
        unp = sum(1 for r in rows if r["judges"][j]["label"] is None)
        print(f"  {j:<16} yes={yes}/{n}  unparseable={unp}")

    print("\nPairwise judge agreement (over parseable-by-both):")
    for a, b in combinations(judges, 2):
        both = [r for r in rows
                if r["judges"][a]["label"] is not None
                and r["judges"][b]["label"] is not None]
        agr = sum(1 for r in both
                  if r["judges"][a]["label"] == r["judges"][b]["label"])
        d = len(both)
        print(f"  {a} vs {b}: {agr}/{d}"
              f"  ({100*agr/d:.0f}%)" if d else f"  {a} vs {b}: n/a")

    print("\nDisagreement rows (hand-check these):")
    any_d = False
    for r in rows:
        labs = {j: r["judges"][j]["verdict"] for j in judges}
        if len({r["judges"][j]["label"] for j in judges}) > 1:
            any_d = True
            print(f"  {r['qid']} [{r['question_type']}] gold={r['gold_answer'][:40]!r}")
            print(f"      pred={r['predicted_answer'][:90]!r}")
            print(f"      {labs}")
    if not any_d:
        print("  (none — all judges agreed)")


async def main(
    specs: List[Tuple[str, str]], answer_model: str, judge_models: List[str],
    top_k: int, max_tokens: int, out: Path,
) -> int:
    # Group qids by tier so each big tier file is streamed once.
    by_tier: Dict[str, List[str]] = defaultdict(list)
    for tier, qid in specs:
        by_tier[tier].append(qid)
    loaded: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for tier, qids in by_tier.items():
        print(f"Streaming {tier} for {qids}...", file=sys.stderr)
        got = _stream_load_qids(TIER_FILES[tier], set(qids))
        for qid in qids:
            if qid not in got:
                raise SystemExit(f"qid {qid} not in tier {tier}")
            loaded[(tier, qid)] = got[qid]

    client = HindsightRESTClient(HINDSIGHT_URL, timeout=300.0)
    rows: List[Dict[str, Any]] = []
    for tier, qid in specs:
        q = loaded[(tier, qid)]
        bank = f"lme_{tier}_{qid}"
        print(f"\n--- {tier}:{qid} [{q['question_type']}] bank={bank} ---",
              file=sys.stderr)
        pred = await _frozen_prediction(
            client, q, bank, top_k, max_tokens, answer_model)
        print(f"  pred={pred['predicted_answer'][:90]!r} "
              f"(n_facts={pred['n_facts']})", file=sys.stderr)
        pred["bank"] = bank
        pred["judges"] = {}
        for jm in judge_models:
            jr = _judge(pred, jm)
            pred["judges"][jm] = jr
            print(f"    judge {jm:<16} -> {jr['verdict']}", file=sys.stderr)
        rows.append(pred)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"answer_model": answer_model, "judge_models": judge_models,
         "top_k": top_k, "max_tokens": max_tokens, "rows": rows},
        indent=2, ensure_ascii=False), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(_md_table(rows, judge_models), encoding="utf-8")
    print(f"\nWrote {out}\nWrote {md} (fill the `human` column to score agreement)",
          file=sys.stderr)
    _print_summary(rows, judge_models)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True,
                    help="comma list of tier:qid, e.g. s:1c0ddc50,m:8fb83627")
    ap.add_argument("--answer-model", default="lme-t0",
                    help="FIXED across the run; frozen prediction every judge sees")
    ap.add_argument("--judge-models", default="lme-t0,lme-25pro-t0,lme-g3-t0",
                    help="comma list of judge models/aliases to compare")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", type=Path,
                    default=Path(".eval_cache/lme_results/judge_meta.json"))
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(
        _parse_specs(args.specs), args.answer_model,
        [j.strip() for j in args.judge_models.split(",") if j.strip()],
        args.top_k, args.max_tokens, args.out,
    )))
