"""CC-3: diff two LME judge result JSONs.

Both inputs are flat lists of per-qid rows as emitted by lme_judge.py.
Matching is by `qid`. Prints per-qid verdict diff, slice deltas
(session_hit / judge_yes / n_facts_retrieved), and a qtype breakdown.

Usage:
    python -m eval_harnesses.suites.memory_recall.lme_compare \\
        --baseline .eval_cache/lme_results/sprint_v2_tight.json \\
        --variant  .eval_cache/lme_results/v3a_verbose.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load(p: Path) -> Dict[str, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = json.loads(p.read_text(encoding="utf-8"))
    return {r["qid"]: r for r in rows}


def _verdict(r: Dict[str, Any]) -> str:
    # judge_label is the bool; verdict can be "yes"/"no"/"error" etc.
    if r.get("judge_label") is True:
        return "yes"
    if r.get("judge_label") is False:
        return "no"
    return r.get("judge_verdict") or "err"


def _icon(v: str) -> str:
    return {"yes": "PASS", "no": "FAIL"}.get(v, v.upper())


def _agg(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(rows) or 1
    yes = sum(1 for r in rows if _verdict(r) == "yes")
    hit = sum(1 for r in rows if r.get("session_hit_any"))
    facts = sum(r.get("n_facts_retrieved", 0) for r in rows)
    return {"n": n, "judge_yes_pct": yes / n * 100,
            "session_hit_pct": hit / n * 100,
            "avg_n_facts": facts / n}


def main(baseline: Path, variant: Path, only: Optional[List[str]] = None) -> int:
    b = _load(baseline)
    v = _load(variant)
    if only:
        b = {k: b[k] for k in only if k in b}
        v = {k: v[k] for k in only if k in v}
    qids = sorted(set(b) | set(v))

    print(f"baseline: {baseline}  ({len(b)} rows)")
    print(f"variant:  {variant}  ({len(v)} rows)")
    print()

    flips: Dict[str, List[str]] = defaultdict(list)
    by_qtype: Dict[str, List[tuple]] = defaultdict(list)
    print(f"{'qid':<24} {'qtype':<26} {'base':>5}  {'var':>5}  diff")
    print("-" * 80)
    for qid in qids:
        br, vr = b.get(qid), v.get(qid)
        if br is None:
            print(f"{qid:<24} {'(variant-only)':<26}  ---    {_icon(_verdict(vr)):>5}")
            continue
        if vr is None:
            print(f"{qid:<24} {'(baseline-only)':<26} {_icon(_verdict(br)):>5}   ---")
            continue
        bv, vv = _verdict(br), _verdict(vr)
        qtype = br.get("question_type", "?")
        flips[f"{bv}->{vv}"].append(qid)
        by_qtype[qtype].append((bv, vv))
        marker = "" if bv == vv else (" <- regress" if (bv, vv) == ("yes", "no")
                                       else " <- recover")
        df_facts = vr.get("n_facts_retrieved", 0) - br.get("n_facts_retrieved", 0)
        print(f"{qid:<24} {qtype:<26} {_icon(bv):>5}  {_icon(vv):>5}  "
              f"facts {df_facts:+d}{marker}")

    print()
    print("Flip summary:")
    for k in ("yes->yes", "yes->no", "no->yes", "no->no"):
        if flips.get(k):
            print(f"  {k:<10} n={len(flips[k]):<3} {flips[k]}")

    print()
    print("Aggregate:")
    common = [b[q] for q in qids if q in b and q in v]
    common_v = [v[q] for q in qids if q in b and q in v]
    ab, av = _agg(common), _agg(common_v)
    print(f"  judge_yes%:    baseline {ab['judge_yes_pct']:5.1f}  ->  "
          f"variant {av['judge_yes_pct']:5.1f}  "
          f"(delta {av['judge_yes_pct']-ab['judge_yes_pct']:+5.1f})")
    print(f"  session_hit%:  baseline {ab['session_hit_pct']:5.1f}  ->  "
          f"variant {av['session_hit_pct']:5.1f}  "
          f"(delta {av['session_hit_pct']-ab['session_hit_pct']:+5.1f})")
    print(f"  avg_n_facts:   baseline {ab['avg_n_facts']:5.1f}  ->  "
          f"variant {av['avg_n_facts']:5.1f}  "
          f"(delta {av['avg_n_facts']-ab['avg_n_facts']:+5.1f})")

    print()
    print("By qtype:")
    for qtype, pairs in sorted(by_qtype.items()):
        n = len(pairs)
        b_yes = sum(1 for bv, _ in pairs if bv == "yes")
        v_yes = sum(1 for _, vv in pairs if vv == "yes")
        print(f"  {qtype:<28} n={n} "
              f"baseline {b_yes}/{n} -> variant {v_yes}/{n}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--variant", type=Path, required=True)
    ap.add_argument("--only", default=None,
                    help="comma-separated qids to restrict the diff to")
    args = ap.parse_args()
    only = [q.strip() for q in args.only.split(",")] if args.only else None
    raise SystemExit(main(args.baseline, args.variant, only))
