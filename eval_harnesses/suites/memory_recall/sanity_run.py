"""Manual sanity run for the semantic recall pipeline against a live bank.

Bypasses runner+fixture wiring; drives resolver, arecall, and grader
directly so we can confirm the pieces compose on real data.

Usage:
    python -m eval_harnesses.suites.memory_recall.sanity_run

Reads `fixtures/claudecode_sanity.json` and hits the live Hindsight bank
named in each scenario's meta.bank.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from config.global_config import HINDSIGHT_URL
from src.memory.backend.hindsight import HindsightRESTClient

from eval_harnesses.framework.grading import SemanticRecallGrader
from eval_harnesses.framework.results import RunOutput
from eval_harnesses.framework.scenarios import load_scenarios

from .resolver import load_seed_data, resolve_facts

_HERE = Path(__file__).parent


async def _run_one(client, seed_data, scenario) -> None:
    bank = scenario.meta.get("bank")
    facts = (
        list(scenario.expectations.get("expected_facts", []))
        + list(scenario.expectations.get("noise_facts", []))
    )
    resolution = await resolve_facts(
        bank, facts, rest_client=client, seed_data=seed_data
    )
    hits = await client.arecall(bank, scenario.user_request, max_tokens=800)

    out = RunOutput()
    out.hindsight_hits = hits
    out.raw["resolved_ids"] = resolution.resolved

    grade = SemanticRecallGrader().grade(scenario, out)
    print(f"\n== {scenario.id} (bank={bank})")
    print(f"  query: {scenario.user_request!r}")
    print(f"  resolved: {resolution.resolved}")
    if resolution.unresolved:
        print(f"  UNRESOLVED: {resolution.unresolved}")
        print(f"  diag: {resolution.diagnostics}")
    print(f"  hits[:5]: {[h.get('id') for h in hits[:5]]}")
    print(f"  passed={grade.passed} notes={grade.notes}")
    print(f"  per_k: {json.dumps(grade.detail['per_k'], indent=2)}")


async def main() -> None:
    client = HindsightRESTClient(HINDSIGHT_URL)
    seed_data = load_seed_data(_HERE / "fixtures" / "test_persona_seed.json")
    scenarios = load_scenarios(_HERE / "fixtures" / "claudecode_sanity.json")
    for s in scenarios:
        await _run_one(client, seed_data, s)


if __name__ == "__main__":
    asyncio.run(main())
