from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List

from eval_harnesses.framework.grading import GradeResult, register

# Tokens that indicate the LLM has fallen back to attributing the statement
# to the bank/format itself instead of an actual participant.
_BAD_SUBJECTS = ["ambient", "narrator", "system", "the bank", "the channel", "observer"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


@dataclass
class AttributionCorrectnessGrader:
    """For ambient_attribution suite.

    Per fact in scenario.expectations.facts:
        {"speaker": "Alice", "must_contain": ["wagyu", "discount"]}

    A fact is correctly attributed iff some recalled item contains all of
    `must_contain` AND the speaker's name AND none of _BAD_SUBJECTS appearing
    as the apparent subject of the same sentence.

    Bad-subject hits in any item count against the score regardless of
    whether facts pass — even if attribution is right elsewhere, "ambient
    said X" anywhere is a failure mode worth surfacing.

    Score = (facts_correct / total_facts) - 0.5 * (bad_subject_items / total_items),
    floored at 0. passed = facts_correct == total_facts AND no bad-subject hits.
    """
    name: str = "attribution_correctness"
    bad_subjects: List[str] = field(default_factory=lambda: list(_BAD_SUBJECTS))

    def grade(self, scenario: Any, run_output: Any) -> GradeResult:
        if run_output.error:
            return GradeResult(self.name, passed=False, notes=f"driver error: {run_output.error}")

        items: List[str] = [_norm(t) for t in run_output.raw.get("recalled_text", [])]
        if not items:
            return GradeResult(self.name, passed=False, notes="no recalled items")

        facts = scenario.expectations.get("facts", [])
        speakers_known = {s.lower() for s in scenario.expectations.get("speakers", [])}

        fact_results = []
        correct = 0
        for fact in facts:
            speaker = fact["speaker"].lower()
            keys = [_norm(k) for k in fact.get("must_contain", [])]
            matched_idx = None
            for i, text in enumerate(items):
                if speaker in text and all(k in text for k in keys):
                    matched_idx = i
                    break
            ok = matched_idx is not None
            if ok:
                correct += 1
            fact_results.append({
                "speaker": fact["speaker"],
                "must_contain": fact.get("must_contain", []),
                "passed": ok,
                "matched_item_idx": matched_idx,
            })

        # Detect bad-subject misattributions. Hit if a known-bad token appears
        # near a verb of speech ("said", "claims", "noted", "mentioned") OR is
        # the leading subject of an item.
        bad_hits = []
        verb_re = re.compile(r"\b(said|says|claims|noted|notes|mentioned|mentions|stated|states|told|asked|asks|reports|reported)\b")
        for i, text in enumerate(items):
            for bad in self.bad_subjects:
                if bad in text:
                    near_verb = bool(re.search(rf"\b{re.escape(bad)}\b[^.]{{0,40}}{verb_re.pattern}", text)) or text.startswith(bad)
                    if near_verb:
                        bad_hits.append({"item_idx": i, "subject": bad})
                        break

        total_facts = max(len(facts), 1)
        total_items = max(len(items), 1)
        score = max(0.0, (correct / total_facts) - 0.5 * (len(bad_hits) / total_items))
        passed = correct == len(facts) and not bad_hits

        return GradeResult(
            grader=self.name,
            passed=passed,
            score=score,
            notes=f"facts {correct}/{len(facts)} bad_attribution={len(bad_hits)}",
            detail={
                "fact_results": fact_results,
                "bad_attribution_hits": bad_hits,
                "speakers_known": sorted(speakers_known),
                "item_count": len(items),
            },
        )


register("attribution_correctness", lambda: AttributionCorrectnessGrader())
