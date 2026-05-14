"""ambient_attribution suite — does the ambient bank correctly attribute
statements to the named speaker, or does it credit 'ambient' / 'narrator'?

Targets the bug where multi-speaker transcripts ingested into the
`ambient` bank produce extracted memories like "ambient said X" or
"narrator believes Y" instead of the actual person who spoke.

Iteration axes: retain_mission text + content format.
"""

from pathlib import Path

from eval_harnesses.framework.runner import SuiteSpec
from eval_harnesses.framework.scenarios import load_scenarios
from eval_harnesses.framework.variants import load_variants

from .driver import ambient_attribution_driver
from . import graders as _graders  # noqa: F401  registers attribution_correctness

_HERE = Path(__file__).parent


def build_suite() -> SuiteSpec:
    return SuiteSpec(
        name="ambient_attribution",
        scenarios=load_scenarios(_HERE / "scenarios.json"),
        variants=load_variants(_HERE / "variants.json"),
        driver=ambient_attribution_driver,
        default_graders=["attribution_correctness"],
        skip_fixture=True,
    )
