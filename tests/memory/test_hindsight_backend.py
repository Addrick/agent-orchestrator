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
def backend() -> HindsightBackend:
    return HindsightBackend(url="http://stub:8888")


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
    assert hit_id == "unit-123"
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
async def test_mark_trusted_pending_upstream(backend: HindsightBackend) -> None:
    with pytest.raises(NotImplementedError, match="DP-110"):
        await backend.mark_trusted("a", "1", operator_id="adam", reason="t")


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
