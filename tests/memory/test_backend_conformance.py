"""Cross-backend conformance for the new-shape MemoryBackend surface.

DP-113 §5: assert that record → recall round-trip on both
SqliteSemanticBackend and HindsightBackend yields MemoryHit objects with the
same shape (fields, types, untrusted-bit propagation). Catches a backend that
silently drops origin metadata or returns the wrong field shapes.

The legacy SQLite path doesn't run `retain_turn` synchronously (it's a noop —
batch MemoryAgent does the work out-of-band), so the SQLite leg seeds via
the existing `store_segment` + `store_summary` write path and asserts the
new-shape `recall` returns equivalent hits. The Hindsight leg mocks the REST
client so we can assert the same hit shape without a live server.
"""
from __future__ import annotations

import asyncio
import struct
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

from config.global_config import EMBEDDING_DIMENSION, EMBEDDING_MODEL
from src.memory.backend.base import MemoryHit
from src.memory.backend.hindsight import HindsightBackend, UNTRUSTED_TAG
from src.memory.backend.sqlite import SqliteSemanticBackend
from src.memory.memory_manager import MemoryManager


def _embed(seed: float = 0.5) -> bytes:
    return struct.pack(f"{EMBEDDING_DIMENSION}f", *[seed] * EMBEDDING_DIMENSION)


class _StubEmbeddingService:
    """Minimal EmbeddingService stand-in: returns a fixed BLOB regardless of input."""
    model_name = EMBEDDING_MODEL

    async def encode(self, texts):
        return [_embed(0.5) for _ in texts]


# ---------- SQLite leg ----------


@pytest.mark.asyncio
async def test_sqlite_recall_round_trip_returns_memory_hit() -> None:
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    try:
        backend = SqliteSemanticBackend(mm, embedding_service=_StubEmbeddingService())
        now = datetime.now(timezone.utc)
        seg_id = backend.store_segment("c1", None, "alice", 1, 3, 3, now)
        backend.store_summary(
            seg_id, "alice likes blue", _embed(0.5),
            EMBEDDING_MODEL, now, summary_level=1, untrusted=True,
        )

        hits = await backend.recall("alice", "favourite color", k=5, tag_filter=["channel:c1"])
        assert hits, "expected at least one hit"
        h = hits[0]
        _assert_hit_shape(h)
        assert h.content == "alice likes blue"
        assert h.untrusted is True  # legacy bit surfaced from Memory_Summaries.untrusted
        assert "channel:c1" in h.tags
        assert "persona:alice" in h.tags
    finally:
        mm.close()


@pytest.mark.asyncio
async def test_sqlite_recall_clean_summary_is_trusted() -> None:
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    try:
        backend = SqliteSemanticBackend(mm, embedding_service=_StubEmbeddingService())
        now = datetime.now(timezone.utc)
        seg_id = backend.store_segment("c1", None, "alice", 1, 3, 3, now)
        backend.store_summary(
            seg_id, "alice likes blue", _embed(0.5),
            EMBEDDING_MODEL, now, summary_level=1, untrusted=False,
        )
        hits = await backend.recall("alice", "x", k=5, tag_filter=["channel:c1"])
        assert hits and hits[0].untrusted is False
    finally:
        mm.close()


# ---------- Hindsight leg (mocked REST) ----------


@pytest.mark.asyncio
async def test_hindsight_recall_returns_memory_hit(tmp_path) -> None:
    backend = HindsightBackend(
        url="http://stub:8888",
        override_db_path=str(tmp_path / "overrides.db"),
        doc_scope_db_path=str(tmp_path / "doc_scope.db"),
    )
    fake_results: List[Dict[str, Any]] = [{
        "id": "h-1",
        "content": "alice likes blue",
        "score": 0.93,
        "tags": ["channel:c1", "persona:alice", UNTRUSTED_TAG],
        "metadata": {"interaction_id": 7},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }]
    client = backend._get_client()
    try:
        with patch.object(client, "arecall", AsyncMock(return_value=fake_results)):
            hits = await backend.recall("alice", "favourite color", k=5, tag_filter=["channel:c1"])
        assert hits, "expected at least one hit"
        h = hits[0]
        _assert_hit_shape(h)
        assert h.content == "alice likes blue"
        assert h.untrusted is True
        assert "channel:c1" in h.tags
    finally:
        await backend.aclose()


# ---------- shared shape assertion ----------


def _assert_hit_shape(h: MemoryHit) -> None:
    assert isinstance(h, MemoryHit)
    assert isinstance(h.id, str) and h.id
    assert isinstance(h.content, str)
    assert isinstance(h.score, float)
    assert isinstance(h.untrusted, bool)
    assert isinstance(h.tags, list)
    assert isinstance(h.metadata, dict)
    # timestamp is optional but must be datetime when present
    assert h.timestamp is None or isinstance(h.timestamp, datetime)
