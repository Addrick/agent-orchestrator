"""Memory recall suite — measures whether the right summaries surface for
known queries against a seeded memory state. See suite README."""

from pathlib import Path

from eval_harnesses.framework.runner import SuiteSpec
from eval_harnesses.framework.scenarios import load_scenarios
from eval_harnesses.framework.variants import load_variants

from .driver import recall_driver

_HERE = Path(__file__).parent


def build_suite() -> SuiteSpec:
    scenarios = load_scenarios(_HERE / "scenarios.json")
    pairs_path = _HERE / "fixtures" / "test_persona_pairs.json"
    if pairs_path.exists():
        scenarios = scenarios + load_scenarios(pairs_path)
    return SuiteSpec(
        name="memory_recall",
        scenarios=scenarios,
        variants=load_variants(_HERE / "variants.json"),
        driver=recall_driver,
        default_graders=["retrieval_hits"],
    )
