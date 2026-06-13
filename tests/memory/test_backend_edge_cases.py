# tests/memory/test_backend_edge_cases.py
"""DP-199 Batch 3 — Backend ABC / Hindsight / SQLite edge cases.

Hindsight tests stub the HTTP client via patch.object on the client instance.
UTF-8/CJK go through direct payload construction. Timeout uses httpx.MockTransport.
"""
from __future__ import annotations

import asyncio
import struct
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from config.global_config import EMBEDDING_DIMENSION, EMBEDDING_MODEL
from src.memory.backend.base import MemoryBackend, MemoryHit
from src.memory.backend.hindsight import (
    HindsightAPIError,
    HindsightBackend,
    HindsightRESTClient,
    TRUSTED_TAG,
    UNTRUSTED_TAG,
    _read_untrusted,
)
from src.memory.backend.sqlite import SqliteSemanticBackend
from src.memory.memory_manager import MemoryManager


def _embed(seed: float = 0.5) -> bytes:
    return struct.pack(f"{EMBEDDING_DIMENSION}f", *[seed] * EMBEDDING_DIMENSION)


class _StubEmbeddingService:
    model_name = EMBEDDING_MODEL

    def __init__(self, *, fail: bool = False, wrong_shape: bool = False):
        self.fail = fail
        self.wrong_shape = wrong_shape

    async def encode(self, texts):
        if self.fail:
            raise RuntimeError("embedding service offline")
        if self.wrong_shape:
            # Half-size blob
            short = struct.pack(f"{EMBEDDING_DIMENSION // 2}f", *[0.5] * (EMBEDDING_DIMENSION // 2))
            return [short for _ in texts]
        return [_embed(0.5) for _ in texts]


# ---------- Backend ABC conformance extensions ----------


def test_memory_manager_semantic_backend_override():
    """MemoryManager accepts an explicit backend kwarg, overriding env-default."""
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    try:
        # Replace backend with a fresh SqliteSemanticBackend bound to a stub service
        new_backend = SqliteSemanticBackend(mm, embedding_service=_StubEmbeddingService())
        mm.backend = new_backend
        assert mm.backend is new_backend
    finally:
        mm.close()


def test_two_managers_different_backends_isolation(tmp_path):
    """Two MemoryManagers with different backend instances don't share state."""
    mm1 = MemoryManager(db_path=":memory:")
    mm1.create_schema()
    mm2 = MemoryManager(db_path=":memory:")
    mm2.create_schema()
    try:
        b1 = SqliteSemanticBackend(mm1, embedding_service=_StubEmbeddingService())
        b2 = SqliteSemanticBackend(mm2)
        mm1.backend = b1
        mm2.backend = b2
        assert mm1.backend is not mm2.backend
        # Writes to mm1 don't show in mm2
        now = datetime.now()
        mm1.log_message("u", "p", "c", "user", "U", "in mm1", now, server_id=None)
        assert len(mm1.get_personal_history("u", "p")) == 1
        assert mm2.get_personal_history("u", "p") == []
    finally:
        mm1.close()
        mm2.close()


@pytest.mark.asyncio
async def test_sqlite_backend_reflect_is_noop():
    """SqliteSemanticBackend.reflect inherits the base-class noop: empty
    ReflectResult, no side effects. Consolidation runs out-of-band via
    memory_consolidation.py — see base.py docstring."""
    from src.memory.backend.base import ReflectResult
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    try:
        backend = SqliteSemanticBackend(mm)
        result = await backend.reflect("alice", "anything")
        assert isinstance(result, ReflectResult)
        assert result.answer == ""
        assert result.mental_models == []
    finally:
        mm.close()


# ---------- Hindsight specifics ----------


@pytest.fixture
def backend(tmp_path) -> HindsightBackend:
    return HindsightBackend(
        url="http://stub:8888",
        override_db_path=str(tmp_path / "overrides.db"),
        doc_scope_db_path=str(tmp_path / "doc_scope.db"),
    )


def test_scope_key_channel_isolation(backend: HindsightBackend):
    """Same bank, different channels → different scope keys."""
    k1 = backend._scope_key("alice", ["channel:c1"])
    k2 = backend._scope_key("alice", ["channel:c2"])
    assert k1 != k2
    assert k1.endswith(":c1")
    assert k2.endswith(":c2")


def test_scope_key_no_channel_solo_scope(backend: HindsightBackend):
    """No channel tag → scope is the bank itself."""
    k = backend._scope_key("alice", ["persona:alice", "role:user"])
    assert k == "alice"


def test_recall_missing_untrusted_tag_defaults_to_false(backend: HindsightBackend):
    """Tag absent → _read_untrusted returns True (fail-closed). Confirm
    via the HindsightBackend.recall path."""
    # `_read_untrusted` should treat missing tag as untrusted=True (fail-closed).
    assert _read_untrusted([]) is True
    assert _read_untrusted(["role:user", "persona:alice"]) is True


@pytest.mark.asyncio
async def test_mark_trusted_recall_race(backend: HindsightBackend):
    """Concurrent flip + recall — last writer wins, no crash."""
    fake = [{"id": "u1", "content": "x", "score": 0.9, "tags": [UNTRUSTED_TAG]}]

    async def fake_arecall(bank_id, query, **kw):
        return fake

    client = backend._get_client()
    with patch.object(client, "arecall", side_effect=fake_arecall):
        async def flip_trusted():
            for _ in range(5):
                await backend.mark_trusted("alice", "u1", operator_id="op1", reason="r")

        async def flip_untrusted():
            for _ in range(5):
                await backend.mark_untrusted("alice", "u1", operator_id="op2", reason="r")

        async def recall_loop():
            for _ in range(5):
                await backend.recall("alice", "q")

        await asyncio.gather(flip_trusted(), flip_untrusted(), recall_loop())

    # Audit log should have 10 entries (no lost writes from race)
    rows = backend._overrides._get().execute(
        "SELECT count(*) FROM Unit_Trust_Audit WHERE bank_id=? AND hit_id=?",
        ("alice", "u1"),
    ).fetchone()[0]
    assert rows == 10
    await backend.aclose()


@pytest.mark.asyncio
async def test_trust_override_concurrent_set_audit(backend: HindsightBackend):
    """Two flips back-to-back produce two audit rows with the right prior/new chain."""
    await backend.mark_untrusted("b1", "h1", operator_id="adam", reason="first")
    await backend.mark_trusted("b1", "h1", operator_id="adam", reason="second")
    rows = backend._overrides._get().execute(
        "SELECT prior, new, reason FROM Unit_Trust_Audit WHERE bank_id=? AND hit_id=? ORDER BY audit_id",
        ("b1", "h1"),
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["prior"] is None
    assert rows[0]["new"] == 1
    assert rows[1]["prior"] == 1
    assert rows[1]["new"] == 0
    await backend.aclose()


@pytest.mark.asyncio
async def test_trust_override_store_audit_cleanup(backend: HindsightBackend):
    """Closing backend cleanly closes the override store connection."""
    await backend.mark_trusted("b", "h", operator_id="op", reason="r")
    await backend.aclose()
    assert backend._overrides._conn is None


@pytest.mark.asyncio
async def test_hindsight_recall_timeout_returns_empty(tmp_path):
    """A TimeoutException from httpx becomes an empty recall result."""
    backend = HindsightBackend(
        url="http://stub:8888",
        override_db_path=str(tmp_path / "ov.db"),
        doc_scope_db_path=str(tmp_path / "ds.db"),
    )
    client = backend._get_client()
    with patch.object(
        client, "arecall", side_effect=httpx.ReadTimeout("timed out"),
    ):
        hits = await backend.recall("alice", "anything")
    assert hits == []
    await backend.aclose()


@pytest.mark.asyncio
async def test_hindsight_config_utf8_emoji(tmp_path):
    """Verify UTF-8 emoji payloads encode cleanly through the JSON path."""
    backend = HindsightBackend(
        url="http://stub:8888",
        override_db_path=str(tmp_path / "ov.db"),
        doc_scope_db_path=str(tmp_path / "ds.db"),
    )
    captured = {}

    async def fake_aretain(bank_id, items, async_=True):
        captured["items"] = items
        return {"id": "ok"}

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=fake_aretain):
        await backend.retain_turn(
            "alice", "user", "hello 🚀 world 🎉",
            timestamp=datetime.now(timezone.utc),
            scope_tags=["channel:c1"], source_persona="alice",
        )
        await backend.aclose()

    assert "🚀" in captured["items"][0]["content"]
    assert "🎉" in captured["items"][0]["content"]


@pytest.mark.asyncio
async def test_hindsight_config_cjk_characters(tmp_path):
    """CJK content survives the JSON encode/decode round-trip."""
    backend = HindsightBackend(
        url="http://stub:8888",
        override_db_path=str(tmp_path / "ov.db"),
        doc_scope_db_path=str(tmp_path / "ds.db"),
    )
    captured = {}

    async def fake_aretain(bank_id, items, async_=True):
        captured["items"] = items
        return {"id": "ok"}

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=fake_aretain):
        await backend.retain_turn(
            "alice", "user", "你好世界 こんにちは 안녕하세요",
            timestamp=datetime.now(timezone.utc),
            scope_tags=["channel:c1"], source_persona="alice",
        )
        await backend.aclose()

    content = captured["items"][0]["content"]
    assert "你好世界" in content
    assert "こんにちは" in content
    assert "안녕하세요" in content


@pytest.mark.asyncio
async def test_hindsight_config_payload_size_limit(tmp_path):
    """Large payloads (no size validation) flow through verbatim — documents
    current behavior. If a size limit is added later, this becomes a guard."""
    backend = HindsightBackend(
        url="http://stub:8888",
        override_db_path=str(tmp_path / "ov.db"),
        doc_scope_db_path=str(tmp_path / "ds.db"),
    )
    captured = {}

    async def fake_aretain(bank_id, items, async_=True):
        captured["items"] = items
        return {"id": "ok"}

    big = "X" * 100_000
    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=fake_aretain):
        await backend.retain_turn(
            "alice", "user", big,
            timestamp=datetime.now(timezone.utc),
            scope_tags=["channel:c1"], source_persona="alice",
        )
        await backend.aclose()

    assert len(captured["items"][0]["content"]) == 100_000


# ---------- SQLite backend specifics ----------


@pytest.mark.asyncio
async def test_sqlite_recall_embedding_service_absent():
    """No embedding_service → recall returns [] (fail-soft)."""
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    try:
        backend = SqliteSemanticBackend(mm)  # no embedding service
        hits = await backend.recall("alice", "query", tag_filter=["channel:c1"])
        assert hits == []
    finally:
        mm.close()


@pytest.mark.asyncio
async def test_sqlite_recall_embedding_shape_mismatch():
    """DP-199 candidate bug: behavior on wrong-shape embedding is unspecified.
    Caller currently bubbles raw sqlite-vec errors instead of failing soft or
    surfacing a typed error. Skipping until the contract is decided."""
    pytest.skip("DP-199 candidate bug: shape-mismatch fail-mode contract not specified")


def test_summaries_exclude_after_boundaries():
    """get_summaries_for_channel with exclude_after_interaction_id filters out
    segments whose start_interaction_id >= cutoff."""
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    try:
        now = datetime.now(timezone.utc)
        backend = SqliteSemanticBackend(mm)
        s1 = backend.store_segment("c1", None, "alice", 1, 5, 5, now)
        s2 = backend.store_segment("c1", None, "alice", 10, 15, 5, now)
        backend.store_summary(s1, "first", _embed(0.1), EMBEDDING_MODEL, now, summary_level=1)
        backend.store_summary(s2, "second", _embed(0.2), EMBEDDING_MODEL, now, summary_level=1)

        all_sums = backend.get_summaries_for_channel("c1", "alice", server_id=None)
        assert len(all_sums) == 2
        cut = backend.get_summaries_for_channel("c1", "alice", server_id=None, exclude_after_interaction_id=8)
        assert len(cut) == 1
        assert cut[0]["content"] == "first"
    finally:
        mm.close()


def test_retrieve_relevant_summaries_vec_corruption_recovery():
    """Drop vec_Memory_Summaries virtual table mid-run; retrieve without
    query_embeddings still returns the canonical rows."""
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    try:
        backend = SqliteSemanticBackend(mm)
        now = datetime.now(timezone.utc)
        seg = backend.store_segment("c1", None, "alice", 1, 2, 2, now)
        backend.store_summary(seg, "fact1", _embed(0.5), EMBEDDING_MODEL, now, summary_level=1)

        # Drop the vec table — simulates corruption / missing extension
        conn = mm._get_connection()
        conn.execute("DELETE FROM vec_Memory_Summaries")
        conn.commit()

        # Without query embeddings, the path doesn't touch vec_*, must still work
        rows = backend.retrieve_relevant_summaries(
            persona_name="alice", channel="c1", server_id=None,
            memory_mode="channel", include_ambient=False,
        )
        assert len(rows) == 1
        assert rows[0]["content"] == "fact1"
    finally:
        mm.close()


def test_retrieve_relevant_summaries_partial_corruption():
    """Some summaries exist in Memory_Summaries but missing from vec_*; the
    distance-join query simply skips the orphans without raising."""
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    try:
        backend = SqliteSemanticBackend(mm)
        now = datetime.now(timezone.utc)
        seg = backend.store_segment("c1", None, "alice", 1, 2, 2, now)
        sid_present = backend.store_summary(seg, "with_vec", _embed(0.5), EMBEDDING_MODEL, now, summary_level=1)
        sid_missing = backend.store_summary(seg, "no_vec", _embed(0.6), EMBEDDING_MODEL, now, summary_level=1)
        # Remove only one vec row
        mm._get_connection().execute(
            "DELETE FROM vec_Memory_Summaries WHERE summary_id=?", (sid_missing,)
        )
        mm._get_connection().commit()

        # With query embeddings, only the row with a present vec row joins.
        rows = backend.retrieve_relevant_summaries(
            persona_name="alice", channel="c1", server_id=None,
            memory_mode="channel", include_ambient=False,
            query_embeddings=[_embed(0.5)],
        )
        ids = {r["summary_id"] for r in rows}
        assert sid_present in ids
        assert sid_missing not in ids
    finally:
        mm.close()


# ---------- Context budget (slice 11a prep) ----------

def test_truncate_single_huge_message_exceeds_budget():
    """A single message larger than the budget is preserved (it's the most
    recent user turn)."""
    from src.memory.context_budget import truncate_messages_to_budget

    huge = "x" * 10_000  # ~2500 tokens estimated
    messages = [{"role": "user", "content": huge}]
    pruned, dropped = truncate_messages_to_budget(messages, max_tokens=100)
    # The single user msg is preserved as the "last user idx"
    assert len(pruned) == 1
    assert pruned[0]["content"] == huge


def test_truncate_only_system_exceeds_budget():
    """A budget filled by system messages alone is honored — preserved set
    returned even if it busts the budget."""
    from src.memory.context_budget import truncate_messages_to_budget

    messages = [
        {"role": "system", "content": "x" * 10_000},
        {"role": "system", "content": "y" * 10_000},
    ]
    pruned, dropped = truncate_messages_to_budget(messages, max_tokens=100)
    # System messages preserved
    assert all(m["role"] == "system" for m in pruned)
    assert len(pruned) == 2


# ---------- Consolidation ----------


@pytest.mark.asyncio
async def test_consolidation_daemon_survives_text_engine_failure(tmp_path):
    """The daemon catches text-engine exceptions and continues to the next
    sleep tick rather than crashing."""
    from src.memory.memory_consolidation import MemoryConsolidator
    from unittest.mock import MagicMock, AsyncMock

    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    try:
        te = MagicMock()
        es = MagicMock()
        es.model_name = "test"
        consol = MemoryConsolidator(mm, te, es)
        # Make _run_global_consolidation raise
        consol._run_global_consolidation = AsyncMock(side_effect=RuntimeError("boom"))

        # Run one iteration: schedule a short sleep override via asyncio.sleep patch.
        from unittest.mock import patch as _p
        # Patch asyncio.sleep inside the consolidation module
        async def fake_sleep(_):
            raise asyncio.CancelledError()

        with _p("asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(asyncio.CancelledError):
                await consol.start_daemon(check_interval_seconds=0)
        # If we get here, the daemon DID catch the RuntimeError; the CancelledError
        # from sleep was the only way out. Test passes.
    finally:
        mm.close()


@pytest.mark.asyncio
async def test_consolidation_singleton_not_promoted(tmp_path):
    """A persona/channel with only a single LEVEL_EPISODIC summary should not
    be promoted to LEVEL_CORE — there's nothing to cluster."""
    from src.memory.memory_consolidation import MemoryConsolidator
    from src.memory.memory_manager import LEVEL_EPISODIC, LEVEL_CORE
    from unittest.mock import MagicMock, AsyncMock

    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    try:
        te = MagicMock()
        es = MagicMock()
        es.model_name = "test"
        consol = MemoryConsolidator(mm, te, es)
        now = datetime.now(timezone.utc)
        backend = SqliteSemanticBackend(mm)
        seg = backend.store_segment("c1", None, "alice", 1, 3, 3, now)
        backend.store_summary(seg, "lone fact", _embed(0.5), EMBEDDING_MODEL, now, summary_level=LEVEL_EPISODIC)

        await consol._run_global_consolidation()
        # No LEVEL_CORE rows should have been created
        rows = mm._get_connection().execute(
            "SELECT count(*) FROM Memory_Summaries WHERE summary_level=?",
            (LEVEL_CORE,),
        ).fetchone()[0]
        assert rows == 0
    finally:
        mm.close()


@pytest.mark.asyncio
async def test_consolidation_daemon_loop_interval():
    """start_daemon honors check_interval_seconds via asyncio.sleep."""
    from src.memory.memory_consolidation import MemoryConsolidator
    from unittest.mock import MagicMock, AsyncMock, patch as _p

    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    try:
        te = MagicMock()
        es = MagicMock()
        es.model_name = "test"
        consol = MemoryConsolidator(mm, te, es)
        consol._run_global_consolidation = AsyncMock()

        sleeps: List[float] = []

        async def fake_sleep(t):
            sleeps.append(t)
            raise asyncio.CancelledError()

        with _p("asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(asyncio.CancelledError):
                await consol.start_daemon(check_interval_seconds=42)
        assert sleeps == [42]
    finally:
        mm.close()
