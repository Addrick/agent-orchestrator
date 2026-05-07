"""HindsightBackend tests.

Two layers:
  1. Unit — mock the underlying httpx client; verify tag round-trip,
     untrusted-bit conformance, retry-storm absence, fail-soft retain.
  2. Live (`@pytest.mark.hindsight_live`) — exercises a real container at
     HINDSIGHT_LIVE_URL. Auto-skipped when env var absent.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.memory.backend.hindsight import (
    HindsightAPIError,
    HindsightBackend,
    HindsightRESTClient,
    TRUSTED_TAG,
    UNTRUSTED_TAG,
    _read_untrusted,
)


# ---------- Pure-function unit tests ----------


def test_read_untrusted_explicit_true() -> None:
    assert _read_untrusted([UNTRUSTED_TAG]) is True


def test_read_untrusted_explicit_false() -> None:
    assert _read_untrusted([TRUSTED_TAG]) is False


def test_read_untrusted_absent_defaults_true() -> None:
    """Fail-closed: pre-bit data is treated as suspect, not silently trusted."""
    assert _read_untrusted(["persona:alice", "role:user"]) is True


# ---------- Mock-backed unit tests ----------


@pytest.fixture
def backend(tmp_path) -> HindsightBackend:
    return HindsightBackend(
        url="http://stub:8888",
        override_db_path=str(tmp_path / "overrides.db"),
    )


@pytest.mark.asyncio
async def test_retain_turn_threads_untrusted_tag(backend: HindsightBackend) -> None:
    captured: Dict[str, Any] = {}

    async def fake_aretain(bank_id: str, content: str, tags: List[str]) -> Dict[str, Any]:
        captured["bank_id"] = bank_id
        captured["tags"] = tags
        return {"id": "unit-123"}

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=fake_aretain):
        hit_id = await backend.retain_turn(
            "alice", "user", "hello",
            timestamp=datetime.now(timezone.utc),
            scope_tags=["channel:c1"],
            source_persona="alice",
            untrusted=True,
        )
        # Fire-and-forget: ID isn't known to the caller, drain to verify worker did POST.
        await backend.aclose()
    assert hit_id == ""
    assert captured["bank_id"] == "alice"
    assert UNTRUSTED_TAG in captured["tags"]
    assert "persona:alice" in captured["tags"]
    assert "role:user" in captured["tags"]
    assert "channel:c1" in captured["tags"]


@pytest.mark.asyncio
async def test_retain_turn_trusted_default(backend: HindsightBackend) -> None:
    captured: Dict[str, Any] = {}

    async def fake_aretain(bank_id: str, content: str, tags: List[str]) -> Dict[str, Any]:
        captured["tags"] = tags
        return {"id": "x"}

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=fake_aretain):
        await backend.retain_turn(
            "alice", "assistant", "ack",
            timestamp=datetime.now(timezone.utc),
            scope_tags=[],
            source_persona="alice",
        )
        await backend.aclose()
    assert TRUSTED_TAG in captured["tags"]
    assert UNTRUSTED_TAG not in captured["tags"]


@pytest.mark.asyncio
async def test_recall_recovers_untrusted_bit(backend: HindsightBackend) -> None:
    """Cross-backend conformance: bit set on retain MUST surface on recall."""
    fake_results = [
        {"id": "1", "content": "trusted msg", "score": 0.9, "tags": [TRUSTED_TAG, "role:user"]},
        {"id": "2", "content": "untrusted msg", "score": 0.8, "tags": [UNTRUSTED_TAG]},
        {"id": "3", "content": "untagged msg", "score": 0.7, "tags": []},
    ]

    async def fake_arecall(bank_id: str, query: str, k: int = 10, tags=None):
        return fake_results

    client = backend._get_client()
    with patch.object(client, "arecall", side_effect=fake_arecall):
        hits = await backend.recall("alice", "anything", k=10)

    assert len(hits) == 3
    assert hits[0].untrusted is False
    assert hits[1].untrusted is True
    assert hits[2].untrusted is True  # fail-closed default


@pytest.mark.asyncio
async def test_retain_turn_fails_soft_on_network_error(backend: HindsightBackend) -> None:
    """Plan §1.3 — fire-and-forget. Network error must NOT crash the user turn."""
    client = backend._get_client()
    with patch.object(
        client, "aretain", side_effect=httpx.ConnectError("kobold offline")
    ):
        hit_id = await backend.retain_turn(
            "alice", "user", "hi",
            timestamp=datetime.now(timezone.utc),
            scope_tags=[],
            source_persona="alice",
        )
        # Drain — worker must swallow ConnectError and stay alive.
        await backend.aclose()
    assert hit_id == ""


@pytest.mark.asyncio
async def test_recall_fails_soft_on_api_error(backend: HindsightBackend) -> None:
    client = backend._get_client()
    with patch.object(
        client, "arecall", side_effect=HindsightAPIError(503, "bank down")
    ):
        hits = await backend.recall("alice", "x")
    assert hits == []


@pytest.mark.asyncio
async def test_request_does_not_retry_storm() -> None:
    """Plan §1.3 — no retry storm. _request must hit the network exactly once."""
    client = HindsightRESTClient(base_url="http://stub:8888")
    inner = AsyncMock(side_effect=httpx.ConnectError("nope"))
    with patch.object(client.client, "request", inner):
        with pytest.raises(httpx.ConnectError):
            await client._request("POST", "/banks/x/recall", json={})
    assert inner.call_count == 1


@pytest.mark.asyncio
async def test_legacy_methods_raise_not_implemented(backend: HindsightBackend) -> None:
    """Flipping SEMANTIC_BACKEND with un-migrated callers must fail loud."""
    with pytest.raises(NotImplementedError):
        backend.store_segment()
    with pytest.raises(NotImplementedError):
        backend.retrieve_relevant_summaries()
    with pytest.raises(NotImplementedError):
        backend.log_agent_action()


@pytest.mark.asyncio
async def test_queue_preserves_intra_bank_order(backend: HindsightBackend) -> None:
    """Plan §1.3 — intra-bank FIFO. Cross-bank ordering not guaranteed (parallel)."""
    seen: Dict[str, List[str]] = {"alice": [], "bob": []}

    async def fake_aretain(bank_id: str, content: str, tags: List[str]) -> Dict[str, Any]:
        seen[bank_id].append(content)
        return {"id": content}

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=fake_aretain):
        for i in range(5):
            await backend.retain_turn(
                "alice", "user", f"a{i}",
                timestamp=datetime.now(timezone.utc),
                scope_tags=[], source_persona="alice",
            )
            await backend.retain_turn(
                "bob", "user", f"b{i}",
                timestamp=datetime.now(timezone.utc),
                scope_tags=[], source_persona="bob",
            )
        await backend.aclose()

    assert seen["alice"] == [f"a{i}" for i in range(5)]
    assert seen["bob"] == [f"b{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_worker_survives_connect_error(backend: HindsightBackend) -> None:
    """ConnectError on item N must not kill the worker; item N+1 still drains."""
    calls: List[str] = []

    async def flaky(bank_id: str, content: str, tags: List[str]) -> Dict[str, Any]:
        calls.append(content)
        if content == "boom":
            raise httpx.ConnectError("kobold offline")
        return {"id": content}

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=flaky):
        for c in ("ok1", "boom", "ok2"):
            await backend.retain_turn(
                "alice", "user", c,
                timestamp=datetime.now(timezone.utc),
                scope_tags=[], source_persona="alice",
            )
        await backend.aclose()

    assert calls == ["ok1", "boom", "ok2"]


@pytest.mark.asyncio
async def test_mark_trusted_audit_and_recall_override(backend: HindsightBackend) -> None:
    """Flip → recall → assert MemoryHit.untrusted reflects new value. Flip back → re-verify."""
    fake_results = [
        {"id": "u1", "content": "x", "score": 0.9, "tags": [UNTRUSTED_TAG]},
    ]

    async def fake_arecall(bank_id: str, query: str, k: int = 10, tags=None):
        return fake_results

    client = backend._get_client()
    with patch.object(client, "arecall", side_effect=fake_arecall):
        # Storage tag is untrusted → hit reads untrusted.
        hits = await backend.recall("alice", "q")
        assert hits[0].untrusted is True

        # Operator marks trusted.
        await backend.mark_trusted("alice", "u1", operator_id="adam", reason="vetted")
        hits = await backend.recall("alice", "q")
        assert hits[0].untrusted is False

        # Flip back.
        await backend.mark_untrusted("alice", "u1", operator_id="adam", reason="recanted")
        hits = await backend.recall("alice", "q")
        assert hits[0].untrusted is True

    # Audit log captured both flips with prior + new.
    rows = backend._overrides._get().execute(
        "SELECT prior, new, operator_id, reason FROM Unit_Trust_Audit "
        "WHERE bank_id=? AND hit_id=? ORDER BY audit_id", ("alice", "u1"),
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["prior"] is None and rows[0]["new"] == 0  # first flip: no prior override
    assert rows[1]["prior"] == 0 and rows[1]["new"] == 1     # second flip: prior trusted → untrusted
    assert all(r["operator_id"] == "adam" for r in rows)
    await backend.aclose()


# ---------- Live container smoke ----------


@pytest.mark.hindsight_live
@pytest.mark.asyncio
async def test_live_retain_recall_round_trip() -> None:
    """End-to-end: retain a tagged turn, recall it, assert untrusted bit preserved."""
    url = os.environ["HINDSIGHT_LIVE_URL"]
    backend = HindsightBackend(url=url)
    bank = "test-conformance"
    await backend.ensure_bank(bank, mission="conformance test bank")
    try:
        await backend.retain_turn(
            bank, "user", "the quick brown fox jumps over the lazy dog",
            timestamp=datetime.now(timezone.utc),
            scope_tags=["test:conformance"],
            source_persona="alice",
            untrusted=True,
        )
        hits = await backend.recall(bank, "fox", k=5)
        assert any(h.untrusted is True for h in hits), \
            "untrusted bit must round-trip through Hindsight tags"
    finally:
        await backend.delete_bank(bank)
