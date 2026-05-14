from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional

from eval_harnesses.framework.grading import GradeResult, register

_DATE_ONLY = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def _to_date(value: Any) -> Optional[str]:
    """Normalize an ISO-ish timestamp to a date string (YYYY-MM-DD).

    Tolerates: full ISO-8601 with or without timezone, date-only strings,
    None. Returns None if unparseable.
    """
    if not value:
        return None
    if isinstance(value, str):
        m = _DATE_ONLY.match(value)
        if m:
            return m.group(1)
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            return None
    return None


@dataclass
class DateAttributionGrader:
    """For backfill_dates suite: were extracted memory items dated correctly?

    Reads scenario.expectations:
        documents: [{"doc_date": "...", "id": "..."}],
        # Optional: dates that *must* show up across the recalled items.
        expected_dates: ["2026-03-12", "2026-04-01"],
        # Optional: dates that must NOT show up (typically the import date).
        forbidden_dates: ["2026-05-13"],
        # Optional tolerance in days for fuzzy matches.
        tolerance_days: 0

    Reads run_output.raw:
        mentioned_at: [iso strings]
        recalled_items: [...]

    Score = fraction of expected_dates that show up in recalled
    mentioned_at set, minus 1.0 for any forbidden hit. passed = score >= 1.0.
    """
    name: str = "date_attribution"

    def grade(self, scenario: Any, run_output: Any) -> GradeResult:
        if run_output.error:
            return GradeResult(self.name, passed=False, notes=f"driver error: {run_output.error}")

        raw = run_output.raw or {}
        recalled_dates: List[Optional[str]] = [_to_date(v) for v in raw.get("mentioned_at", [])]
        recalled_set = {d for d in recalled_dates if d}
        if not recalled_set:
            return GradeResult(
                self.name,
                passed=False,
                notes="no mentioned_at values in recalled items",
                detail={"recalled": raw.get("recalled_items", [])},
            )

        # Default expected dates = doc_date of every input document.
        docs = scenario.expectations.get("documents", [])
        default_expected = [_to_date(d.get("doc_date")) for d in docs]
        expected = scenario.expectations.get("expected_dates") or [d for d in default_expected if d]
        forbidden = scenario.expectations.get("forbidden_dates", [])

        hits = [d for d in expected if d in recalled_set]
        misses = [d for d in expected if d not in recalled_set]
        forbidden_hits = [d for d in forbidden if d in recalled_set]

        score = (len(hits) / len(expected)) if expected else 0.0
        passed = bool(expected) and not misses and not forbidden_hits

        return GradeResult(
            grader=self.name,
            passed=passed,
            score=score,
            notes=(
                f"hits={len(hits)}/{len(expected)} "
                f"misses={misses or 'none'} "
                f"forbidden_hits={forbidden_hits or 'none'}"
            ),
            detail={
                "expected": expected,
                "recalled": sorted(recalled_set),
                "misses": misses,
                "forbidden_hits": forbidden_hits,
                "raw_mentioned_at": raw.get("mentioned_at"),
            },
        )


register("date_attribution", lambda: DateAttributionGrader())
