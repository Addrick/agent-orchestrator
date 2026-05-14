from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SeededInteraction:
    """One pre-existing message to inject into memory before the scenario runs."""
    role: str  # "user" | "assistant"
    content: str
    channel: str = "eval"
    user_identifier: str = "eval_user"
    persona_name: str = "default"
    # Optional pre-summarized form: bypasses the summarizer pipeline.
    pre_summary: Optional[str] = None


@dataclass
class Scenario:
    """A single eval scenario: seeded state + a query + expectations."""
    id: str
    description: str
    user_request: str
    context: str = ""
    persona_name: str = "default"
    channel: str = "eval"
    user_identifier: str = "eval_user"
    server_id: Optional[str] = None
    # Seed memory state before the run. Each entry may be a raw turn or a
    # pre-built summary (skips summarizer for determinism).
    seed_memory: List[SeededInteraction] = field(default_factory=list)
    # Free-form expectations. Graders consume what they care about.
    expectations: Dict[str, Any] = field(default_factory=dict)
    # Which graders to apply (names resolved against the suite's grader registry).
    graders: List[str] = field(default_factory=list)
    # Optional metadata (tags, source notes).
    meta: Dict[str, Any] = field(default_factory=dict)


_SCENARIO_FIELDS = {
    "id", "description", "user_request", "context", "persona_name",
    "channel", "user_identifier", "seed_memory", "expectations",
    "graders", "meta",
}


def load_scenarios(path: str | Path) -> List[Scenario]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: List[Scenario] = []
    for item in raw:
        seed = [SeededInteraction(**s) for s in item.pop("seed_memory", [])]
        meta = dict(item.pop("meta", {}) or {})
        # Shunt any unknown top-level keys into meta so suite-specific fields
        # (bank, k_sweep, thresholds, etc.) survive loading without bloating
        # the generic Scenario dataclass.
        for k in list(item.keys()):
            if k not in _SCENARIO_FIELDS:
                meta[k] = item.pop(k)
        out.append(Scenario(seed_memory=seed, meta=meta, **item))
    return out
