# tests/memory/test_router_edge_cases.py
"""DP-199 Batch 3 — MemoryRouter fan-out edge cases.

The existing `tests/memory/test_memory_router.py` covers the happy-path
fan-out, dedupe, score-sort, tie-break-by-timestamp, single-bank-failure,
empty-personas-list, and kwarg-forwarding cases. This file fills the
remaining gaps from the DP-199 router-fan-out checklist plus a few
high-value adjacent edges discovered while reading `src/memory/router.py`.

Notes on scope:
- The current MemoryRouter routes across multiple BANKS on a SINGLE backend.
  "Multi-backend registration" and "router-level retain fan-out" are not
  implemented features and are skipped per DP-199 ground rules (no new
  features, no enshrining latent bugs).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pytest

from src.memory.backend.base import MemoryHit
from src.memory.router import MemoryRouter
from src.persona import Persona

# Reuse the existing fake backend so router tests share one ABC stub.
from tests.memory.test_memory_router import _FakeBackend  # noqa: F401


def _hit(hit_id: str, score: float, *, ts=None, content: str = "") -> MemoryHit:
    return MemoryHit(id=hit_id, content=content or hit_id, score=score, timestamp=ts)


def _persona(name: str, *, meta_visible: bool = False) -> Persona:
    return Persona(persona_name=name, model_name="m", prompt="p", meta_visible=meta_visible)


# ---------- visibility / caller contract ----------

@pytest.mark.asyncio
async def test_recall_many_does_not_filter_by_visibility():
    """Router fans out across whatever banks the caller passes; it does NOT
    consult `personas[name].meta_visible`. Visibility filtering is the
    caller's job (typically via `list_visible_personas`). DP-199 plan item
    `test_router_recall_many_non_visible_persona_undefined`.
    """
    backend = _FakeBackend()
    backend.banks["hidden"] = [_hit("h1", 0.7)]
    personas = {"hidden": _persona("hidden", meta_visible=False)}
    router = MemoryRouter(backend, personas)

    results = await router.recall_many(["hidden"], "q")
    assert [h.id for h in results] == ["h1"]


# ---------- failure modes ----------

@pytest.mark.asyncio
async def test_recall_many_all_banks_fail_returns_empty():
    backend = _FakeBackend()
    backend.banks["a"] = [_hit("a1", 0.9)]
    backend.banks["b"] = [_hit("b1", 0.9)]
    backend.fail_for.update({"a", "b"})
    router = MemoryRouter(backend, {})

    assert await router.recall_many(["a", "b"], "q") == []


@pytest.mark.asyncio
async def test_recall_many_failure_does_not_block_concurrent_bank():
    """Verify return_exceptions=True semantics — a raising coroutine in
    asyncio.gather doesn't cancel its siblings. The 'good' bank's result
    still arrives even when listed BEFORE the failing one.
    """
    backend = _FakeBackend()
    backend.banks["good"] = [_hit("g1", 0.4), _hit("g2", 0.8)]
    backend.banks["bad"] = [_hit("ignored", 0.99)]
    backend.fail_for.add("bad")
    router = MemoryRouter(backend, {})

    results = await router.recall_many(["bad", "good"], "q")
    assert [h.id for h in results] == ["g2", "g1"]


# ---------- duplicate / pathological persona lists ----------

@pytest.mark.asyncio
async def test_recall_many_duplicate_bank_ids_deduped_by_hit_id():
    """Passing the same bank twice should not duplicate its hits — dedupe
    is keyed on `MemoryHit.id`, so identical hits collapse to one.
    """
    backend = _FakeBackend()
    backend.banks["alpha"] = [_hit("a1", 0.6), _hit("a2", 0.4)]
    router = MemoryRouter(backend, {})

    results = await router.recall_many(["alpha", "alpha"], "q")
    assert [h.id for h in results] == ["a1", "a2"]


# ---------- kwarg forwarding edges ----------

@pytest.mark.asyncio
async def test_recall_many_k_kwarg_applied_per_bank():
    """`k` is forwarded verbatim to each backend.recall call — meaning the
    *per-bank* slice is `k`, and the merged total can exceed `k`. This
    documents the actual contract so callers know to post-trim if they
    want a global cap.
    """
    backend = _FakeBackend()
    backend.banks["a"] = [_hit(f"a{i}", 0.9 - i * 0.01) for i in range(5)]
    backend.banks["b"] = [_hit(f"b{i}", 0.5 - i * 0.01) for i in range(5)]
    router = MemoryRouter(backend, {})

    results = await router.recall_many(["a", "b"], "q", k=2)
    # _FakeBackend honours k by slicing [:k]; merged result has 2 per bank.
    assert len(results) == 4
    assert sorted(h.id for h in results) == ["a0", "a1", "b0", "b1"]


@pytest.mark.asyncio
async def test_recall_many_forwards_identical_kwargs_to_each_bank():
    captured: List[Dict[str, Any]] = []

    class _Capture(_FakeBackend):
        async def recall(self, bank_id, query, **kwargs):
            captured.append({"bank": bank_id, **kwargs})
            return []

    router = MemoryRouter(_Capture(), {})
    await router.recall_many(
        ["a", "b", "c"], "q", tag_filter=["scope:global"], k=3, max_tokens=50,
    )
    # Same kwargs reach every bank — no per-bank rewriting.
    assert len(captured) == 3
    for entry in captured:
        assert entry["tag_filter"] == ["scope:global"]
        assert entry["k"] == 3
        assert entry["max_tokens"] == 50
    assert [c["bank"] for c in captured] == ["a", "b", "c"]


# ---------- sort / tie-break edges ----------

@pytest.mark.asyncio
async def test_recall_many_tie_break_none_timestamp_treated_as_zero():
    """When one hit has a real timestamp and another has None at the same
    score, the real timestamp wins (None → 0.0 epoch).
    """
    backend = _FakeBackend()
    backend.banks["a"] = [_hit("with_ts", 0.5, ts=datetime(2026, 1, 1))]
    backend.banks["b"] = [_hit("no_ts", 0.5, ts=None)]
    router = MemoryRouter(backend, {})

    results = await router.recall_many(["a", "b"], "q")
    assert [h.id for h in results] == ["with_ts", "no_ts"]


@pytest.mark.asyncio
async def test_recall_many_dedupe_keeps_metadata_of_winning_hit():
    """Dedupe by id keeps the higher-score MemoryHit *whole* — not a
    field-by-field merge. The winning hit's tags/metadata/untrusted flag
    are the ones that survive.
    """
    backend = _FakeBackend()
    backend.banks["a"] = [MemoryHit(
        id="shared", content="weak", score=0.3,
        untrusted=True, tags=["loser"], metadata={"src": "a"},
    )]
    backend.banks["b"] = [MemoryHit(
        id="shared", content="strong", score=0.9,
        untrusted=False, tags=["winner"], metadata={"src": "b"},
    )]
    router = MemoryRouter(backend, {})

    results = await router.recall_many(["a", "b"], "q")
    assert len(results) == 1
    hit = results[0]
    assert hit.content == "strong"
    assert hit.untrusted is False
    assert hit.tags == ["winner"]
    assert hit.metadata == {"src": "b"}


@pytest.mark.asyncio
async def test_recall_many_dedupe_first_score_wins_on_exact_tie():
    """When two banks return the same id with identical scores, the first
    one encountered wins (router uses `>` not `>=`). Bank-iteration order
    follows the `personas` argument order.
    """
    backend = _FakeBackend()
    ts = datetime(2026, 1, 1)
    backend.banks["a"] = [_hit("shared", 0.5, ts=ts, content="from-a")]
    backend.banks["b"] = [_hit("shared", 0.5, ts=ts, content="from-b")]
    router = MemoryRouter(backend, {})

    results = await router.recall_many(["a", "b"], "q")
    assert len(results) == 1
    assert results[0].content == "from-a"


# ---------- features not yet implemented ----------

@pytest.mark.asyncio
async def test_router_retain_fan_out_not_implemented():
    """The router does not expose a `retain_many` / retain-fan-out method.
    Retain still goes per-bank via `backend.retain_turn(bank_id, ...)`.
    Implementing fan-out retain would be a new feature — out of scope for
    DP-199.
    """
    router = MemoryRouter(_FakeBackend(), {})
    pytest.skip(
        "DP-199 deferred: router-level retain fan-out is not an implemented "
        "feature (router multiplexes recall only; retain is per-bank)."
    )
    assert not hasattr(router, "retain_many")


@pytest.mark.asyncio
async def test_router_multi_backend_registration_not_implemented():
    """MemoryRouter takes exactly one backend. Multiplexing recall across
    multiple *backends* (e.g. SQLite + Hindsight concurrently) is not a
    current feature. Mentioned in the DP-199 audit prompt; skipped here
    until/unless that becomes a real surface.
    """
    pytest.skip(
        "DP-199 deferred: MemoryRouter accepts a single backend; multi-"
        "backend registration is not an implemented feature."
    )
