# tests/memory/test_memory_router.py
"""Tests for MemoryRouter (DP-111, Sprint 4).

Sprint-1 SQLite backend leaves the new-shape `recall` as a NotImplementedError
stub on `MemoryBackend`. To exercise the router's fan-out + dedupe logic
end-to-end against the ABC contract, these tests use a small `_FakeBackend`
that satisfies the same interface and stores hits in-memory keyed by bank.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest

from src.memory.backend.base import MemoryBackend, MemoryHit
from src.memory.router import MemoryRouter
from src.persona import Persona


class _FakeBackend(MemoryBackend):
    """ABC subclass used purely to drive the router."""

    def __init__(self) -> None:
        self.banks: Dict[str, List[MemoryHit]] = {}
        self.fail_for: set[str] = set()

    async def recall(self, bank_id, query, *, k=10, types=None, tag_filter=None,
                     max_tokens=None, budget=None):
        if bank_id in self.fail_for:
            raise RuntimeError(f"simulated failure for {bank_id}")
        return list(self.banks.get(bank_id, []))[:k]

    # The legacy abstract methods aren't exercised here. Provide minimal
    # stubs so the class instantiates.
    def log_agent_action(self, *a, **k): raise NotImplementedError
    def update_agent_action_outcome(self, *a, **k): raise NotImplementedError
    def add_action_contexts(self, *a, **k): raise NotImplementedError
    def get_relevant_agent_actions(self, *a, **k): raise NotImplementedError
    def get_action_steps(self, *a, **k): raise NotImplementedError
    def store_message_embedding(self, *a, **k): raise NotImplementedError
    def get_unembedded_messages(self, *a, **k): raise NotImplementedError
    def store_segment(self, *a, **k): raise NotImplementedError
    def store_summary(self, *a, **k): raise NotImplementedError
    def get_summaries_for_channel(self, *a, **k): raise NotImplementedError
    def get_unsegmented_embedded_messages(self, *a, **k): raise NotImplementedError
    def retrieve_relevant_summaries(self, *a, **k): raise NotImplementedError
    def record_segment_failure(self, *a, **k): raise NotImplementedError
    def get_failed_segment_ranges(self, *a, **k): raise NotImplementedError
    def clear_segment_failure(self, *a, **k): raise NotImplementedError
    def get_active_channels(self, *a, **k): raise NotImplementedError
    def get_last_segment_tail_embeddings(self, *a, **k): raise NotImplementedError


def _hit(hit_id: str, score: float, *, ts: Optional[datetime] = None,
         content: str = "") -> MemoryHit:
    return MemoryHit(id=hit_id, content=content or hit_id, score=score, timestamp=ts)


def _make_persona(name: str, *, meta_visible: bool = False) -> Persona:
    return Persona(persona_name=name, model_name="m", prompt="p", meta_visible=meta_visible)


# ---------- list_visible_personas ----------

def test_list_visible_personas_filters_on_meta_visible():
    backend = _FakeBackend()
    personas = {
        "alpha": _make_persona("alpha", meta_visible=True),
        "bravo": _make_persona("bravo", meta_visible=False),
        "charlie": _make_persona("charlie", meta_visible=True),
    }
    router = MemoryRouter(backend, personas)
    assert sorted(router.list_visible_personas()) == ["alpha", "charlie"]


def test_list_visible_personas_empty_when_none_visible():
    backend = _FakeBackend()
    personas = {"alpha": _make_persona("alpha")}
    router = MemoryRouter(backend, personas).list_visible_personas()
    assert router == []


# ---------- recall_many: merge + dedupe ----------

@pytest.mark.asyncio
async def test_recall_many_merges_and_sorts_by_score_desc():
    backend = _FakeBackend()
    backend.banks["alpha"] = [_hit("a1", 0.5), _hit("a2", 0.9)]
    backend.banks["bravo"] = [_hit("b1", 0.7)]
    router = MemoryRouter(backend, {})

    results = await router.recall_many(["alpha", "bravo"], "q")
    assert [h.id for h in results] == ["a2", "b1", "a1"]


@pytest.mark.asyncio
async def test_recall_many_dedupes_by_id_keeping_highest_score():
    backend = _FakeBackend()
    backend.banks["alpha"] = [_hit("shared", 0.4, content="from-alpha")]
    backend.banks["bravo"] = [_hit("shared", 0.9, content="from-bravo")]
    router = MemoryRouter(backend, {})

    results = await router.recall_many(["alpha", "bravo"], "q")
    assert len(results) == 1
    assert results[0].id == "shared"
    assert results[0].score == 0.9
    assert results[0].content == "from-bravo"


@pytest.mark.asyncio
async def test_recall_many_breaks_score_tie_by_timestamp_desc():
    backend = _FakeBackend()
    older = datetime(2025, 1, 1)
    newer = datetime(2026, 1, 1)
    backend.banks["alpha"] = [_hit("x", 0.5, ts=older)]
    backend.banks["bravo"] = [_hit("y", 0.5, ts=newer)]
    router = MemoryRouter(backend, {})

    results = await router.recall_many(["alpha", "bravo"], "q")
    assert [h.id for h in results] == ["y", "x"]


@pytest.mark.asyncio
async def test_recall_many_drops_failed_banks_returns_remaining():
    backend = _FakeBackend()
    backend.banks["alpha"] = [_hit("a1", 0.6)]
    backend.banks["bravo"] = [_hit("b1", 0.8)]
    backend.fail_for.add("bravo")
    router = MemoryRouter(backend, {})

    results = await router.recall_many(["alpha", "bravo"], "q")
    assert [h.id for h in results] == ["a1"]


@pytest.mark.asyncio
async def test_recall_many_empty_persona_list_returns_empty():
    router = MemoryRouter(_FakeBackend(), {})
    assert await router.recall_many([], "q") == []


@pytest.mark.asyncio
async def test_recall_many_forwards_kwargs_to_backend():
    captured: List[Dict[str, Any]] = []

    class _CaptureBackend(_FakeBackend):
        async def recall(self, bank_id, query, **kwargs):
            captured.append({"bank": bank_id, "query": query, **kwargs})
            return []

    router = MemoryRouter(_CaptureBackend(), {})
    await router.recall_many(
        ["alpha"], "find me", k=5, tag_filter=["scope:global"], max_tokens=100
    )
    assert captured == [{
        "bank": "alpha", "query": "find me",
        "k": 5, "tag_filter": ["scope:global"], "max_tokens": 100,
    }]


# ---------- "integration" fan-out across multiple banks ----------

@pytest.mark.asyncio
async def test_recall_many_fan_out_across_three_banks():
    """Simulates Sprint-4 fan-out: three populated banks, mixed scores,
    one duplicate id across two banks."""
    backend = _FakeBackend()
    base = datetime(2026, 5, 1)
    backend.banks["arbitr"] = [
        _hit("e1", 0.3, ts=base),
        _hit("shared", 0.5, ts=base, content="weak"),
    ]
    backend.banks["joy"] = [
        _hit("e2", 0.95, ts=base + timedelta(hours=1)),
        _hit("shared", 0.85, ts=base + timedelta(hours=2), content="strong"),
    ]
    backend.banks["it-help"] = [_hit("e3", 0.7, ts=base)]

    personas = {
        "arbitr": _make_persona("arbitr", meta_visible=True),
        "joy": _make_persona("joy", meta_visible=True),
        "it-help": _make_persona("it-help", meta_visible=False),
    }
    router = MemoryRouter(backend, personas)

    visible = router.list_visible_personas()
    assert sorted(visible) == ["arbitr", "joy"]

    results = await router.recall_many(visible, "q")
    # joy's e2 highest, then deduped 'shared' (joy wins, 0.85), then arbitr's e1
    assert [h.id for h in results] == ["e2", "shared", "e1"]
    shared_hit = next(h for h in results if h.id == "shared")
    assert shared_hit.content == "strong"
