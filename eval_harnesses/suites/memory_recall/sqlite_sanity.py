"""SQLite-branch sanity run against Adam's real user_memory.db.

Exercises:
- variant.db_path (skip temp-DB + don't unlink)
- embedding step in recall_driver (encode user_request → bytes via EmbeddingService)
- existing segment-id scenarios + RetrievalHitsGrader

Usage:
    python -m eval_harnesses.suites.memory_recall.sqlite_sanity \\
        [--db data/user_memory.db] [--limit 5]
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from eval_harnesses.framework.fixtures import build_fixture
from eval_harnesses.framework.grading import resolve as resolve_graders
from eval_harnesses.framework.scenarios import load_scenarios
from eval_harnesses.framework.variants import MemoryVariant, PromptVariant

from .driver import recall_driver

_HERE = Path(__file__).parent


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/user_memory.db")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    scenarios = load_scenarios(_HERE / "scenarios.json")
    mem_var = MemoryVariant(
        id="sqlite_real_db",
        description="real user_memory.db, embedding-driven retrieval",
        sqlite_summaries=True,
        hindsight=False,
        retrieval_params={"limit": args.limit},
        db_path=args.db,
    )
    prompt_var = PromptVariant(id="default")
    graders = resolve_graders(["retrieval_hits"])

    for scen in scenarios:
        with build_fixture(scen, mem_var, prompt_var, live=False) as bundle:
            out = await recall_driver(bundle, scen, mem_var, prompt_var)
            print(f"\n== {scen.id}")
            print(f"  query: {scen.user_request!r}")
            print(f"  expected: {scen.expectations.get('expected_segments')}")
            if out.error:
                print(f"  ERROR: {out.error}")
            else:
                print(f"  retrieved seg_ids[:5]: {out.retrieved_summary_ids[:5]}")
                for g in graders:
                    res = g.grade(scen, out)
                    print(f"  [{g.name}] passed={res.passed} {res.notes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
