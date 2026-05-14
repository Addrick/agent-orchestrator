"""backfill_dates suite — does bulk-imported content end up with the
correct `mentioned_at` dates, or does everything collapse to import day?

Targets the bug where `scratch/hindsight_migration/hindsight_import.py`
posts items with `timestamp: "unset"`, causing the server to stamp every
extracted memory with the import time instead of the date present in the
source content.
"""

from pathlib import Path

from eval_harnesses.framework.runner import SuiteSpec
from eval_harnesses.framework.scenarios import load_scenarios
from eval_harnesses.framework.variants import load_variants

from .driver import backfill_dates_driver
from . import graders as _graders  # noqa: F401  registers date_attribution

_HERE = Path(__file__).parent


def build_suite() -> SuiteSpec:
    return SuiteSpec(
        name="backfill_dates",
        scenarios=load_scenarios(_HERE / "scenarios.json"),
        variants=load_variants(_HERE / "variants.json"),
        driver=backfill_dates_driver,
        default_graders=["date_attribution"],
        skip_fixture=True,
    )
