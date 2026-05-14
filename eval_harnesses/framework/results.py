from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunOutput:
    """What one cell-run produced. Graders consume this."""
    response_text: str = ""
    response_type: Optional[str] = None
    duration_s: float = 0.0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_summary_ids: List[int] = field(default_factory=list)
    retrieved_summaries: List[Dict[str, Any]] = field(default_factory=list)
    hindsight_hits: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)  # suite-specific extras


@dataclass
class CellResult:
    scenario_id: str
    memory_variant_id: str
    prompt_variant_id: str
    output: RunOutput
    grades: List[Dict[str, Any]] = field(default_factory=list)
    passed: bool = False  # all graders passed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SuiteRun:
    suite: str
    started_at: str
    finished_at: Optional[str]
    live: bool
    cells: List[CellResult] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        total = len(self.cells)
        passed = sum(1 for c in self.cells if c.passed)
        per_grader: Dict[str, Dict[str, int]] = {}
        per_variant: Dict[str, Dict[str, Any]] = {}
        for cell in self.cells:
            for g in cell.grades:
                slot = per_grader.setdefault(g["grader"], {"pass": 0, "fail": 0})
                slot["pass" if g["passed"] else "fail"] += 1
            key = f"{cell.memory_variant_id}|{cell.prompt_variant_id}"
            slot = per_variant.setdefault(key, {"total": 0, "passed": 0, "scores": []})
            slot["total"] += 1
            if cell.passed:
                slot["passed"] += 1
            for g in cell.grades:
                slot["scores"].append(g.get("score", 0.0))
        for slot in per_variant.values():
            scores = slot.pop("scores")
            slot["avg_score"] = (sum(scores) / len(scores)) if scores else 0.0
        return {
            "total": total,
            "passed": passed,
            "per_grader": per_grader,
            "per_variant": per_variant,
        }


def write_run(run: SuiteRun, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = out / f"{run.suite}_{ts}.json"
    payload = {
        "suite": run.suite,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "live": run.live,
        "summary": run.summary(),
        "cells": [c.to_dict() for c in run.cells],
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def diff_runs(a_path: str | Path, b_path: str | Path) -> Dict[str, Any]:
    """Crude side-by-side comparison of two SuiteRun JSON files."""
    a = json.loads(Path(a_path).read_text(encoding="utf-8"))
    b = json.loads(Path(b_path).read_text(encoding="utf-8"))
    a_pass = {(c["scenario_id"], c["memory_variant_id"], c["prompt_variant_id"]): c["passed"] for c in a["cells"]}
    b_pass = {(c["scenario_id"], c["memory_variant_id"], c["prompt_variant_id"]): c["passed"] for c in b["cells"]}
    keys = sorted(set(a_pass) | set(b_pass))
    flips = [k for k in keys if a_pass.get(k) != b_pass.get(k)]
    return {
        "a_summary": a["summary"],
        "b_summary": b["summary"],
        "flips": [{"cell": k, "a": a_pass.get(k), "b": b_pass.get(k)} for k in flips],
    }
