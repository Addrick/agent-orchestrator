from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MemoryVariant:
    """One memory-stack configuration to evaluate.

    Axes the runner respects:
      - sqlite_summaries: enable/disable local KNN summary retrieval
      - hindsight: enable/disable hindsight backend recall
      - retrieval_params: kwargs forwarded to retrieve_relevant_summaries()
                         (e.g. {"limit": 8, "memory_mode": "user"})
      - hindsight_params: kwargs forwarded to hindsight recall
                          (e.g. {"banks": ["resident"], "limit": 5})
      - extra: free-form bag for grader-side use
    """
    id: str
    description: str = ""
    sqlite_summaries: bool = True
    hindsight: bool = False
    retrieval_params: Dict[str, Any] = field(default_factory=dict)
    hindsight_params: Dict[str, Any] = field(default_factory=dict)
    # When set, build_fixture uses this DB path directly (read-only intent;
    # not unlinked on teardown). None => fresh temp DB per cell-run.
    db_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptVariant:
    """One persona/prompt configuration to evaluate."""
    id: str
    description: str = ""
    # Replaces the persona's prompt body entirely.
    persona_prompt: Optional[str] = None
    # Appended to the system prompt at runtime (per-turn).
    system_addendum: Optional[str] = None
    # Persona name override (which persona to load); defaults to scenario's.
    persona_name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VariantMatrix:
    memory: List[MemoryVariant]
    prompt: List[PromptVariant]

    def cells(self) -> List[tuple[MemoryVariant, PromptVariant]]:
        return list(itertools.product(self.memory, self.prompt))


def load_variants(path: str | Path) -> VariantMatrix:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    mem = [MemoryVariant(**m) for m in raw.get("memory", [])]
    prm = [PromptVariant(**p) for p in raw.get("prompt", [])]
    if not mem:
        mem = [MemoryVariant(id="default")]
    if not prm:
        prm = [PromptVariant(id="default")]
    return VariantMatrix(memory=mem, prompt=prm)
