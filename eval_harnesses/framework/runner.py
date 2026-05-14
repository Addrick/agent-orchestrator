from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, List, Optional

from .fixtures import FixtureBundle, build_fixture
from .grading import Grader, resolve as resolve_graders
from .results import CellResult, RunOutput, SuiteRun
from .scenarios import Scenario
from .variants import MemoryVariant, PromptVariant, VariantMatrix


# A driver is the suite-specific function that actually exercises the
# ChatSystem and produces a RunOutput. Different suites measure different
# things (a recall-quality suite calls retrieve_relevant_summaries directly;
# a behavioral suite calls generate_response). Keeping this pluggable means
# the framework stays generic.
Driver = Callable[[FixtureBundle, Scenario, MemoryVariant, PromptVariant], "asyncio.Future[RunOutput]"]


@dataclass
class SuiteSpec:
    name: str
    scenarios: List[Scenario]
    variants: VariantMatrix
    driver: Driver
    default_graders: List[str]
    # If True, runner skips build_fixture() and passes None as bundle.
    # Use for suites that talk directly to external services (e.g. Hindsight)
    # and don't need a ChatSystem.
    skip_fixture: bool = False


async def _run_cell(
    spec: SuiteSpec,
    scenario: Scenario,
    mem_var: MemoryVariant,
    prompt_var: PromptVariant,
    *,
    live: bool,
) -> CellResult:
    output: RunOutput
    try:
        started = datetime.utcnow()
        if spec.skip_fixture:
            output = await spec.driver(None, scenario, mem_var, prompt_var)
        else:
            with build_fixture(scenario, mem_var, prompt_var, live=live) as bundle:
                output = await spec.driver(bundle, scenario, mem_var, prompt_var)
        output.duration_s = output.duration_s or (datetime.utcnow() - started).total_seconds()
    except Exception as e:  # framework-level failure; record and move on
        output = RunOutput(error=f"{type(e).__name__}: {e}")

    grader_names = scenario.graders or spec.default_graders
    graders: List[Grader] = resolve_graders(grader_names)
    grade_dicts: List[dict] = []
    all_pass = True
    for g in graders:
        try:
            gr = g.grade(scenario, output)
        except Exception as e:
            from .grading import GradeResult
            gr = GradeResult(grader=g.name, passed=False, notes=f"grader error: {e}")
        grade_dicts.append({
            "grader": gr.grader,
            "passed": gr.passed,
            "score": gr.score,
            "notes": gr.notes,
            "detail": gr.detail,
        })
        all_pass = all_pass and gr.passed

    return CellResult(
        scenario_id=scenario.id,
        memory_variant_id=mem_var.id,
        prompt_variant_id=prompt_var.id,
        output=output,
        grades=grade_dicts,
        passed=all_pass and not output.error,
    )


async def run_suite(
    spec: SuiteSpec,
    *,
    live: bool = False,
    scenario_filter: Optional[List[str]] = None,
    variant_filter: Optional[List[str]] = None,
    progress: Optional[Callable[[CellResult], None]] = None,
) -> SuiteRun:
    scenarios = spec.scenarios
    if scenario_filter:
        wanted = set(scenario_filter)
        scenarios = [s for s in scenarios if s.id in wanted]

    cells = spec.variants.cells()
    if variant_filter:
        wanted = set(variant_filter)
        cells = [(m, p) for (m, p) in cells if m.id in wanted or p.id in wanted]

    run = SuiteRun(
        suite=spec.name,
        started_at=datetime.utcnow().isoformat(),
        finished_at=None,
        live=live,
    )

    for scenario in scenarios:
        for mem_var, prompt_var in cells:
            cell = await _run_cell(spec, scenario, mem_var, prompt_var, live=live)
            run.cells.append(cell)
            if progress:
                progress(cell)

    run.finished_at = datetime.utcnow().isoformat()
    return run
