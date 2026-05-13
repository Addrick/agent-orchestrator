# src/memory/router.py
"""MemoryRouter — fan-out recall across multiple persona banks.

Sprint 4 (DP-111) groundwork for the future Meta-Agent / cross-persona reasoning.
By default, Hindsight banks are persona-isolated (`bank_id = persona_name`).
This router parallelises `MemoryBackend.recall` across N banks, merges + dedupes
the hits, and returns them sorted by relevance score.

`list_visible_personas()` returns the names of personas with `meta_visible=True`,
the canonical input for `recall_many` when invoked by the future meta-agent. No
production caller wires this in yet — Sprint 5 introduces the metabank dual-write
path; until then this is a callable seam that tests + ad-hoc tooling can use.

See plans/memory_backend_abc.md (Sprint 4) and
memory/project/tasks/DP-111.md.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Mapping, Tuple

from src.memory.backend.base import MemoryBackend, MemoryHit
from src.persona import Persona

logger = logging.getLogger(__name__)


class MemoryRouter:
    """Routes recall queries across one or more persona banks."""

    def __init__(
        self,
        backend: MemoryBackend,
        personas: Mapping[str, Persona],
    ) -> None:
        self.backend = backend
        self.personas = personas

    def list_visible_personas(self) -> List[str]:
        """Names of personas with `meta_visible=True`. Eligible for fan-out
        recall when invoked by a meta-agent."""
        return [name for name, p in self.personas.items() if p.get_meta_visible()]

    async def recall_many(
        self,
        personas: List[str],
        query: str,
        **kwargs: Any,
    ) -> List[MemoryHit]:
        """Fan-out recall across N persona banks in parallel.

        `personas` is a list of bank ids (== persona names per the bank scheme
        in plans/memory_backend_abc.md). `**kwargs` is forwarded verbatim to
        `MemoryBackend.recall` (k, types, tag_filter, max_tokens, budget).

        Per-bank failures are logged and dropped; the surviving banks still
        return results. Hits are deduped by `id` (highest-score wins) and
        sorted by score desc, then timestamp desc as a tiebreaker.
        """
        if not personas:
            return []

        tasks = [self.backend.recall(bank, query, **kwargs) for bank in personas]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged: Dict[str, MemoryHit] = {}
        for bank, result in zip(personas, results):
            if isinstance(result, BaseException):
                logger.warning("recall failed for bank %s: %s", bank, result)
                continue
            for hit in result:
                existing = merged.get(hit.id)
                if existing is None or hit.score > existing.score:
                    merged[hit.id] = hit

        def sort_key(h: MemoryHit) -> Tuple[float, float]:
            ts = h.timestamp.timestamp() if isinstance(h.timestamp, datetime) else 0.0
            return (h.score, ts)

        return sorted(merged.values(), key=sort_key, reverse=True)
