from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Protocol


@dataclass
class GradeResult:
    grader: str
    passed: bool
    score: float = 0.0  # 0..1; binary graders use 1.0/0.0
    notes: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


class Grader(Protocol):
    name: str

    def grade(self, scenario: Any, run_output: Any) -> GradeResult: ...


# ---------- Stub graders ----------

@dataclass
class ContainsGrader:
    """Pass if response text contains all `expected_substrings` from
    scenario.expectations."""
    name: str = "contains"

    def grade(self, scenario: Any, run_output: Any) -> GradeResult:
        needles = scenario.expectations.get("contains", [])
        text = (run_output.response_text or "").lower()
        missing = [n for n in needles if n.lower() not in text]
        passed = not missing
        return GradeResult(
            grader=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            notes="all present" if passed else f"missing: {missing}",
            detail={"missing": missing, "needles": needles},
        )


@dataclass
class RetrievalHitsGrader:
    """For memory-recall scenarios: did the retrieved summaries include the
    expected segment_ids?

    Reads run_output.retrieved_summary_ids and scenario.expectations:
        {"expected_segments": [int], "anti_segments": [int]}
    Reports precision@K, recall@K, MRR. Treats hit if any expected id appears.
    """
    name: str = "retrieval_hits"
    k: int = 5

    def grade(self, scenario: Any, run_output: Any) -> GradeResult:
        expected = set(scenario.expectations.get("expected_segments", []))
        anti = set(scenario.expectations.get("anti_segments", []))
        retrieved: List[int] = list(getattr(run_output, "retrieved_summary_ids", []))[: self.k]
        if not expected:
            return GradeResult(self.name, passed=False, notes="no expected_segments declared")
        hits = [i for i, sid in enumerate(retrieved) if sid in expected]
        anti_hits = [sid for sid in retrieved if sid in anti]
        precision = len([s for s in retrieved if s in expected]) / max(len(retrieved), 1)
        recall = len([s for s in retrieved if s in expected]) / len(expected)
        mrr = (1.0 / (hits[0] + 1)) if hits else 0.0
        passed = bool(hits) and not anti_hits
        return GradeResult(
            grader=self.name,
            passed=passed,
            score=mrr,
            notes=f"P@{self.k}={precision:.2f} R@{self.k}={recall:.2f} MRR={mrr:.2f}",
            detail={
                "retrieved": retrieved,
                "expected": list(expected),
                "anti_hits": anti_hits,
                "precision": precision,
                "recall": recall,
                "mrr": mrr,
            },
        )


@dataclass
class SemanticRecallGrader:
    """Fact-anchored recall grader for semantic banks (Hindsight, etc).

    Each fact is a *concept* whose locators may resolve to multiple bank rows
    (near-dupes). Resolver returns id-sets per fact. Grader counts a fact as
    "hit" if any of its ids appears in retrieved window.

    Recall counts distinct *facts* (no double-credit for dupes).
    Precision/noise count *slots* (every retrieved item costs a K slot).

    RunOutput must carry:
        run_output.hindsight_hits       # ranked list of {"id": ..., "text": ...}
        run_output.raw["resolved_ids"]  # {fact_key: [memory_id, ...]}  (lists,
                                        #  JSON-safe; treated as sets here)

    Reports per-K metrics: precision@k, recall@k, mrr, noise_rate@k.
    Pass = recall@k_pass >= recall_threshold AND noise_rate@k_pass <= noise_threshold.
    """
    name: str = "semantic_recall"
    k_sweep: tuple = (1, 3, 5, 10)
    k_pass: int = 5
    recall_threshold: float = 0.5
    noise_threshold: float = 0.25

    def grade(self, scenario: Any, run_output: Any) -> GradeResult:
        expected = scenario.expectations.get("expected_facts", [])
        noise = scenario.expectations.get("noise_facts", [])
        raw_resolved = (run_output.raw or {}).get("resolved_ids", {}) or {}

        # Per-scenario overrides (loader shunts unknown JSON keys into meta).
        meta = getattr(scenario, "meta", {}) or {}
        k_sweep = tuple(meta.get("k_sweep") or self.k_sweep)
        thresholds = meta.get("thresholds") or {}
        k_pass = thresholds.get("k_pass", self.k_pass)
        recall_threshold = thresholds.get("recall", self.recall_threshold)
        noise_threshold = thresholds.get("noise_rate", self.noise_threshold)
        if not expected:
            return GradeResult(self.name, passed=False, notes="no expected_facts declared")

        # Normalize to sets of strings; JSON round-trip turns sets into lists.
        def _ids(key):
            v = raw_resolved.get(key)
            if v is None:
                return None
            if isinstance(v, (list, set, tuple)):
                return {str(x) for x in v}
            return {str(v)}  # tolerate legacy single-id shape

        exp_id_sets = {f["key"]: _ids(f["key"]) for f in expected}
        noise_id_sets = {f["key"]: _ids(f["key"]) for f in noise}
        unresolved = [k for k, ids in exp_id_sets.items() if not ids]
        # All expected ids (any-of, across all expected facts) for slot-precision math.
        all_expected_ids = set().union(*(s for s in exp_id_sets.values() if s)) if exp_id_sets else set()
        all_noise_ids = set().union(*(s for s in noise_id_sets.values() if s)) if noise_id_sets else set()

        hits = [str(h.get("id")) for h in (run_output.hindsight_hits or []) if h.get("id")]

        def _fact_hit(ids, window):
            return bool(ids) and bool(ids & set(window))

        per_k: Dict[int, Dict[str, float]] = {}
        for k in k_sweep:
            window = hits[:k]
            window_set = set(window)
            facts_hit = sum(1 for ids in exp_id_sets.values() if _fact_hit(ids, window_set))
            slot_exp_hits = [h for h in window if h in all_expected_ids]
            slot_noise_hits = [h for h in window if h in all_noise_ids]
            precision = len(slot_exp_hits) / max(len(window), 1)
            recall = facts_hit / max(len(expected), 1)
            mrr = next(
                (1.0 / (i + 1) for i, h in enumerate(window) if h in all_expected_ids),
                0.0,
            )
            noise_rate = len(slot_noise_hits) / max(len(window), 1)
            per_k[k] = {
                "precision": precision,
                "recall": recall,
                "mrr": mrr,
                "noise_rate": noise_rate,
            }

        gate = per_k.get(k_pass, {})
        passed = (
            not unresolved
            and gate.get("recall", 0.0) >= recall_threshold
            and gate.get("noise_rate", 1.0) <= noise_threshold
        )
        notes_bits = [
            f"k={k_pass}",
            f"R={gate.get('recall', 0):.2f}",
            f"P={gate.get('precision', 0):.2f}",
            f"noise={gate.get('noise_rate', 0):.2f}",
        ]
        if unresolved:
            notes_bits.append(f"UNRESOLVED:{unresolved}")
        return GradeResult(
            grader=self.name,
            passed=passed,
            score=gate.get("recall", 0.0),
            notes=" ".join(notes_bits),
            detail={
                "per_k": per_k,
                "expected_id_sets": {k: sorted(v or []) for k, v in exp_id_sets.items()},
                "noise_id_sets": {k: sorted(v or []) for k, v in noise_id_sets.items()},
                "unresolved": unresolved,
                "retrieved": hits,
            },
        )


@dataclass
class LLMJudgeGrader:
    """Stub. Real implementation will call a judge model with rubric.

    For now: returns passed=False, score=0, notes="not implemented" so it
    shows up in the result schema without faking signal.
    """
    name: str = "llm_judge"
    rubric: str = ""
    model: str = "gemini-1.5-flash"

    def grade(self, scenario: Any, run_output: Any) -> GradeResult:
        return GradeResult(
            grader=self.name,
            passed=False,
            score=0.0,
            notes="stub: llm judge not yet implemented",
            detail={"rubric": self.rubric, "model": self.model},
        )


# ---------- Registry ----------

GraderFactory = Callable[[], Grader]

_REGISTRY: Dict[str, GraderFactory] = {
    "contains": lambda: ContainsGrader(),
    "retrieval_hits": lambda: RetrievalHitsGrader(),
    "semantic_recall": lambda: SemanticRecallGrader(),
    "llm_judge": lambda: LLMJudgeGrader(),
}


def register(name: str, factory: GraderFactory) -> None:
    _REGISTRY[name] = factory


def resolve(names: List[str]) -> List[Grader]:
    out: List[Grader] = []
    for n in names:
        if n not in _REGISTRY:
            raise KeyError(f"unknown grader: {n} (known: {sorted(_REGISTRY)})")
        out.append(_REGISTRY[n]())
    return out
