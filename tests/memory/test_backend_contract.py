# tests/memory/test_backend_contract.py
"""ABC contract tests for MemoryBackend.

Sprint 1 (DP-108) ships SqliteSemanticBackend only. This suite exercises the
ABC surface the backend must satisfy. When HindsightBackend lands in Sprint 2,
parametrize the `backend` fixture across both implementations.
"""
import asyncio
import os
import struct

import pytest

from config.global_config import EMBEDDING_DIMENSION, EMBEDDING_MODEL, TEST_DATABASE_DIR
from src.memory.backend import MemoryBackend, SqliteSemanticBackend
from src.memory.memory_manager import MemoryManager


def _embed(seed: float = 0.1) -> bytes:
    """Build a deterministic embedding blob of the right dimension."""
    return struct.pack(f"{EMBEDDING_DIMENSION}f", *[seed] * EMBEDDING_DIMENSION)


@pytest.fixture
def backend() -> MemoryBackend:
    os.makedirs(TEST_DATABASE_DIR, exist_ok=True)
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    yield mm.backend
    mm.close()


# ---------- legacy-shape: episodic ----------


def test_log_agent_action_returns_id(backend: MemoryBackend) -> None:
    aid = backend.log_agent_action("test_agent", "test_action", trigger_context="ctx")
    assert isinstance(aid, int) and aid > 0


def test_action_steps_round_trip(backend: MemoryBackend) -> None:
    parent = backend.log_agent_action("agent_a", "parent_op")
    backend.log_agent_action("agent_a", "step", parent_id=parent)
    backend.log_agent_action("agent_a", "step", parent_id=parent)
    steps = backend.get_action_steps(parent)
    assert len(steps) == 2
    assert all(s["parent_id"] == parent for s in steps)


def test_get_relevant_agent_actions_filters_by_context(backend: MemoryBackend) -> None:
    a1 = backend.log_agent_action("agent_b", "remind")
    a2 = backend.log_agent_action("agent_b", "remind")
    backend.add_action_contexts(a1, [("user", "alice")])
    backend.add_action_contexts(a2, [("user", "bob")])
    hits = backend.get_relevant_agent_actions(
        "agent_b",
        match_contexts=[("user", "alice")],
        match_types=["remind"],
        limit=10,
    )
    ids = [h["id"] for h in hits]
    # Both come back (since the match scores boost a1 above a2), a1 first.
    assert ids[0] == a1


def test_update_agent_action_outcome(backend: MemoryBackend) -> None:
    aid = backend.log_agent_action("agent_c", "send")
    backend.update_agent_action_outcome(aid, "failed", outcome_payload="boom")
    # Failure outcomes get bumped above non-failures in get_relevant_agent_actions.
    hits = backend.get_relevant_agent_actions("agent_c", limit=5)
    assert hits[0]["outcome"] == "failed"


# ---------- legacy-shape: semantic ----------


def test_segment_summary_round_trip(backend: MemoryBackend) -> None:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    seg_id = backend.store_segment(
        channel="chan-1", server_id=None, persona_name="alice",
        start_id=1, end_id=5, message_count=5, created_at=now,
    )
    assert seg_id > 0
    summary_id = backend.store_summary(
        segment_id=seg_id, content="summary text", embedding=_embed(),
        model_name=EMBEDDING_MODEL, created_at=now, summary_level=1,
    )
    assert summary_id > 0

    rows = backend.get_summaries_for_channel("chan-1", "alice")
    assert len(rows) == 1
    assert rows[0]["content"] == "summary text"
    assert rows[0]["segment_id"] == seg_id


def test_retrieve_relevant_summaries_returns_seeded_row(backend: MemoryBackend) -> None:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    seg_id = backend.store_segment("chan-x", None, "p", 1, 3, 3, now)
    backend.store_summary(seg_id, "hello world", _embed(0.5), EMBEDDING_MODEL, now, summary_level=1)
    rows = backend.retrieve_relevant_summaries(
        persona_name="p", channel="chan-x", memory_mode="channel",
        query_embeddings=[_embed(0.5)], limit=5,
    )
    assert len(rows) >= 1
    assert any(r["content"] == "hello world" for r in rows)


def test_retrieve_relevant_summaries_ticket_mode_returns_empty(backend: MemoryBackend) -> None:
    rows = backend.retrieve_relevant_summaries(
        persona_name="p", channel="c", memory_mode="ticket",
    )
    assert rows == []


def test_segment_failure_round_trip(backend: MemoryBackend) -> None:
    backend.record_segment_failure("c", None, "p", 1, 5, 5, error_reason="boom")
    blocked = backend.get_failed_segment_ranges("c", "p", max_attempts=3)
    assert len(blocked) == 1
    assert blocked[0]["error_reason"] == "boom"
    # Second attempt increments
    backend.record_segment_failure("c", None, "p", 1, 5, 5, error_reason="boom2")
    blocked2 = backend.get_failed_segment_ranges("c", "p", max_attempts=3)
    assert blocked2[0]["attempts"] == 2

    backend.clear_segment_failure("c", "p", None, 1, 5)
    assert backend.get_failed_segment_ranges("c", "p", max_attempts=3, cooldown_hours=0) == []


def test_active_channels_includes_unembedded(backend: MemoryBackend, tmp_path) -> None:
    # Insert a raw User_Interactions row directly via the backend's DB.
    from datetime import datetime, timezone
    mm = backend._mm  # type: ignore[attr-defined]
    with mm.transaction() as conn:
        conn.execute(
            "INSERT INTO User_Interactions (user_identifier, persona_name, channel, author_role, content, timestamp)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("u1", "alice", "chan-y", "user", "hi", datetime.now(timezone.utc)),
        )
    rows = backend.get_active_channels()
    assert ("chan-y", "alice", None) in rows


# ---------- new-shape: defaults / NotImplementedError ----------


def test_new_shape_reflect_is_noop_on_sqlite(backend: MemoryBackend) -> None:
    result = asyncio.run(backend.reflect("bank", "anything"))
    assert result.answer == ""
    assert result.mental_models == []


def test_new_shape_list_mental_models_empty_on_sqlite(backend: MemoryBackend) -> None:
    result = asyncio.run(backend.list_mental_models("bank"))
    assert result == []


def test_new_shape_ensure_bank_noop_on_sqlite(backend: MemoryBackend) -> None:
    # Should not raise; SQLite has implicit banks.
    asyncio.run(backend.ensure_bank("bank", retain_mission="m", reflect_mission="rm"))


def test_new_shape_retain_turn_is_noop_on_sqlite(backend: MemoryBackend) -> None:
    # DP-113: legacy MemoryAgent batch loop continues to drive consolidation
    # under sqlite_legacy. retain_turn is a deliberate noop returning "".
    from datetime import datetime, timezone
    result = asyncio.run(backend.retain_turn(
        "bank", "user", "hi",
        timestamp=datetime.now(timezone.utc),
        scope_tags=["channel:c"],
        source_persona="alice",
    ))
    assert result == ""


def test_new_shape_recall_returns_empty_without_embedding_service(backend: MemoryBackend) -> None:
    # DP-113: recall translates query → embedding via injected EmbeddingService.
    # When none is set (e.g. raw MemoryManager construction), fail soft.
    hits = asyncio.run(backend.recall("bank", "query"))
    assert hits == []


# ---------- delegation sanity ----------


def test_memory_manager_uses_sqlite_backend_by_default() -> None:
    mm = MemoryManager(db_path=":memory:")
    try:
        assert isinstance(mm.backend, SqliteSemanticBackend)
    finally:
        mm.close()


def test_memory_manager_accepts_injected_backend() -> None:
    class StubBackend(SqliteSemanticBackend):
        pass

    mm = MemoryManager(db_path=":memory:")
    stub = StubBackend(mm)
    mm2 = MemoryManager(db_path=":memory:", backend=stub)
    try:
        assert mm2.backend is stub
    finally:
        mm.close()
        mm2.close()


# ---------- DP-118: retain_document on sqlite is noop+warn ----------


@pytest.mark.asyncio
async def test_sqlite_retain_document_is_noop(backend: MemoryBackend, caplog) -> None:
    from datetime import datetime, timezone
    import logging
    caplog.set_level(logging.WARNING)
    result = await backend.retain_document(
        "alice", "notes/x.md", "content",
        tags=["ingest"],
        metadata={"source_path": "notes/x.md"},
        timestamp=datetime.now(timezone.utc),
    )
    assert result is None
    assert any("sqlite backend active" in rec.message for rec in caplog.records)
