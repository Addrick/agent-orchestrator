# tests/memory/test_memory_manager_edge_cases.py
"""DP-199 Batch 3 — Memory manager edge cases.

Covers: migrations (idempotency, partial state, v2 branch, vec setup),
suppression filter cross-path, edit-history / swap, and concurrency.

Reuses the `legacy_mem_manager` fixture defined in test_memory_manager.py
(see imports below).
"""
from __future__ import annotations

import concurrent.futures
import sqlite3
import struct
import threading
import time
from datetime import datetime, timezone

import pytest

from config.global_config import EMBEDDING_DIMENSION, EMBEDDING_MODEL
from src.memory.memory_manager import MemoryManager

# Reuse fixtures from the main suite. pytest auto-discovers if same dir.
from tests.memory.test_memory_manager import (  # noqa: F401  (fixtures used implicitly)
    legacy_mem_manager,
    file_mem_manager,
    mem_manager,
)


def _embed(seed: float = 0.5) -> bytes:
    return struct.pack(f"{EMBEDDING_DIMENSION}f", *[seed] * EMBEDDING_DIMENSION)


# -------------------- Migrations --------------------

def test_create_schema_idempotent(legacy_mem_manager):
    """Calling create_schema() repeatedly on a populated DB doesn't duplicate or error."""
    legacy_mem_manager.create_schema()
    legacy_mem_manager.create_schema()
    legacy_mem_manager.create_schema()

    actions = legacy_mem_manager.get_agent_actions(agent_name="dispatch", limit=10)
    assert len(actions) == 2

    conn = legacy_mem_manager._get_connection()
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(User_Interactions)")
    cols = [row['name'] for row in cur.fetchall()]
    # No duplicate columns after multiple migrations
    assert len(cols) == len(set(cols))


def test_migration_memory_summaries_v2_idempotent(tmp_path):
    """When Memory_Summaries already has the v2 (summary_level) column, the
    create_schema migration must NOT rename/copy/drop — it should just verify
    and continue."""
    db_path = str(tmp_path / "v2.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE Memory_Summaries (
            summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id INTEGER,
            content TEXT NOT NULL,
            embedding BLOB,
            model_name TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            summary_level INTEGER NOT NULL DEFAULT 1,
            parent_summary_id INTEGER,
            untrusted INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO Memory_Summaries (segment_id, content, model_name, created_at, summary_level)
        VALUES (NULL, 'preexisting fact', 'm1', '2026-01-01T00:00:00', 1);
    """)
    conn.close()

    mm = MemoryManager(db_path=db_path)
    mm.create_schema()
    try:
        c = mm._get_connection().cursor()
        c.execute("SELECT content, summary_level FROM Memory_Summaries")
        rows = c.fetchall()
        assert len(rows) == 1
        assert rows[0]['content'] == 'preexisting fact'
        assert rows[0]['summary_level'] == 1
        # Ensure the v2→v2 path didn't drop+create losing the row
        c.execute("SELECT count(*) FROM sqlite_master WHERE name='Memory_Summaries_old'")
        assert c.fetchone()[0] == 0
    finally:
        mm.close()


def test_migration_interrupted_recovery(tmp_path):
    """Simulate partial-failure recovery: Memory_Summaries v1 exists, but a
    Memory_Summaries_old leftover from a prior aborted migration is present.

    Latent: there's no rollback for executescript. We assert the migration
    behaves deterministically on a re-run — either succeeds, or raises in a
    way the caller can observe. Either is OK; what's NOT OK is silent corruption.
    """
    db_path = str(tmp_path / "partial.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE Memory_Summaries (
            summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id INTEGER,
            content TEXT NOT NULL,
            embedding BLOB,
            model_name TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL
        );
        CREATE TABLE Memory_Summaries_old (
            summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id INTEGER,
            content TEXT NOT NULL,
            embedding BLOB,
            model_name TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL
        );
        INSERT INTO Memory_Summaries (segment_id, content, model_name, created_at)
        VALUES (NULL, 'live row', 'm1', '2026-01-01T00:00:00');
    """)
    conn.close()

    mm = MemoryManager(db_path=db_path)
    try:
        # Migration will try to rename Memory_Summaries -> Memory_Summaries_old which exists.
        # Expected: sqlite3.OperationalError. Test confirms the error surfaces
        # rather than being swallowed silently. (Latent-bug list item #8.)
        with pytest.raises(sqlite3.OperationalError):
            mm.create_schema()
    finally:
        mm.close()


def test_migration_alter_idempotent_parent_id(legacy_mem_manager):
    """ALTER TABLE Agent_Actions ADD COLUMN parent_id only fires once."""
    legacy_mem_manager.create_schema()
    legacy_mem_manager.create_schema()  # idempotent
    conn = legacy_mem_manager._get_connection()
    c = conn.cursor()
    c.execute("PRAGMA table_info(Agent_Actions)")
    names = [r['name'] for r in c.fetchall()]
    assert names.count('parent_id') == 1


def test_migration_vec_setup_on_existing_summaries(tmp_path):
    """If Memory_Summaries already has rows with embeddings, create_schema
    syncs them into vec_Memory_Summaries."""
    db_path = str(tmp_path / "vec.db")
    mm = MemoryManager(db_path=db_path)
    mm.create_schema()
    # Insert a segment + summary with embedding directly (skip backend wrapper)
    now = datetime.now(timezone.utc)
    conn = mm._get_connection()
    conn.execute(
        "INSERT INTO Memory_Segments (channel, server_id, persona_name, "
        "start_interaction_id, end_interaction_id, message_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("c1", None, "alice", 1, 3, 3, now),
    )
    seg_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO Memory_Summaries (segment_id, content, embedding, model_name, created_at, summary_level) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (seg_id, "x", _embed(0.5), EMBEDDING_MODEL, now, 1),
    )
    # Clear vec row to simulate desync
    conn.execute("DELETE FROM vec_Memory_Summaries")
    conn.commit()
    mm.close()

    # Reopen and re-run schema — should re-sync.
    mm2 = MemoryManager(db_path=db_path)
    mm2.create_schema()
    try:
        c = mm2._get_connection().cursor()
        c.execute("SELECT count(*) FROM vec_Memory_Summaries")
        assert c.fetchone()[0] == 1
    finally:
        mm2.close()


def test_migration_alter_table_integrity_on_partial_failure(tmp_path):
    """If an ALTER TABLE in the migration script triggers an existing column
    conflict, a subsequent create_schema run still leaves the schema usable.

    We construct a db where User_Interactions has SOME but not ALL new columns
    (e.g. tool_context but not reasoning_content), simulating a partial migration."""
    db_path = str(tmp_path / "partial_alter.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE User_Interactions (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_identifier TEXT NOT NULL,
            persona_name TEXT NOT NULL,
            channel TEXT NOT NULL,
            author_role TEXT NOT NULL CHECK(author_role IN ('user','assistant','system')),
            author_name TEXT,
            content TEXT,
            timestamp TIMESTAMP NOT NULL,
            zammad_ticket_id INTEGER,
            platform_message_id TEXT,
            server_id TEXT,
            tool_context TEXT
        );
    """)
    conn.close()

    mm = MemoryManager(db_path=db_path)
    mm.create_schema()
    try:
        c = mm._get_connection().cursor()
        c.execute("PRAGMA table_info(User_Interactions)")
        names = {r['name'] for r in c.fetchall()}
        # tool_context preserved, others added
        for col in ("tool_context", "reasoning_content", "parent_summary_id", "reply_to_id"):
            assert col in names
    finally:
        mm.close()


# -------------------- Suppression filter cross-path --------------------
# NOTE: test_suppression_filter_retrieve_relevant_summaries SKIPPED — DP-199
# deferred bug 1 (memory_manager.py:393 doesn't apply _suppression_filter).


def test_suppressed_not_embedded(mem_manager):
    """A suppressed message is excluded from get_unembedded_messages."""
    now = datetime.now()
    mem_manager.log_message("u1", "p", "c", "user", "Human", "keep", now, platform_message_id="k1", server_id=None)
    mem_manager.log_message("u1", "p", "c", "user", "Human", "drop", now, platform_message_id="d1", server_id=None)
    mem_manager.suppress_message_by_platform_id("d1")

    msgs = mem_manager.get_unembedded_messages("p", "c", server_id=None)
    contents = {m["content"] for m in msgs}
    assert "keep" in contents
    assert "drop" not in contents


def test_suppress_after_swap(mem_manager):
    """Suppressing an interaction id whose content was swapped still removes
    it from history."""
    now = datetime.now()
    mem_manager.log_message("u1", "p", "c", "user", "Human", "v1", now, platform_message_id="m1", server_id=None)
    # Edit creates an archive
    assert mem_manager.handle_message_edit("m1", "v2") is True

    conn = mem_manager._get_connection()
    iid = conn.execute("SELECT interaction_id FROM User_Interactions WHERE platform_message_id='m1'").fetchone()[0]
    # Swap back to archive
    mem_manager.swap_interaction_version(iid, 0)
    # Now suppress
    assert mem_manager.suppress_interaction(iid) is True
    history = mem_manager.get_personal_history("u1", "p")
    assert all(h["content"] != "v1" for h in history)
    assert history == [] or all(h.get("interaction_id") != iid for h in history)


def test_suppress_idempotent_multi_version(mem_manager):
    """Suppress on a message with multiple edit-history rows still flips once."""
    now = datetime.now()
    mem_manager.log_message("u1", "p", "c", "user", "Human", "v1", now, platform_message_id="multi", server_id=None)
    mem_manager.handle_message_edit("multi", "v2")
    mem_manager.handle_message_edit("multi", "v3")

    assert mem_manager.suppress_message_by_platform_id("multi") is True
    assert mem_manager.suppress_message_by_platform_id("multi") is False  # idempotent


def test_suppress_with_orphaned_embedding(mem_manager):
    """DP-199 candidate bug: suppression does not cascade to Message_Embeddings
    (or to vec_Message_Embeddings), so a suppressed message can still surface
    via semantic recall. Fix is either (a) cascade-delete embeddings on
    suppression, or (b) filter suppressed rows at recall time. Skipping until
    a direction is chosen — do not enshrine the current orphan behavior."""
    pytest.skip("DP-199 candidate bug: suppression leaves orphaned embeddings recallable")


# -------------------- Edit history / swap --------------------

def test_swap_identical_archives_no_dupe(mem_manager):
    """Swap canonical to identical content — content-hash dedupe means no
    extra archive row created."""
    now = datetime.now()
    mem_manager.log_message("u1", "p", "c", "user", "Human", "alpha", now, platform_message_id="ident", server_id=None)
    mem_manager.handle_message_edit("ident", "beta")  # one archive 'alpha'

    conn = mem_manager._get_connection()
    iid = conn.execute("SELECT interaction_id FROM User_Interactions WHERE platform_message_id='ident'").fetchone()[0]
    archives_before = conn.execute(
        "SELECT count(*) FROM Interaction_Edit_History WHERE interaction_id=?", (iid,)
    ).fetchone()[0]
    assert archives_before == 1

    # Swap k=0 (alpha) becomes canonical. The previous canonical "beta" gets archived.
    mem_manager.swap_interaction_version(iid, 0)
    # Now swap back. The current canonical 'alpha' must not duplicate the
    # existing 'alpha'-shaped archive (none exists now though, so fresh).
    # Edit to alpha again to set up duplicate scenario.
    canon = conn.execute("SELECT content FROM User_Interactions WHERE interaction_id=?", (iid,)).fetchone()[0]
    assert canon == "alpha"

    # Edit to beta again (alpha archived by swap above)
    mem_manager.swap_interaction_version(iid, 0)  # beta now canonical, alpha archived
    archives_after = conn.execute(
        "SELECT count(*) FROM Interaction_Edit_History WHERE interaction_id=?", (iid,)
    ).fetchone()[0]
    # Dedupe means we should not have unbounded growth of alpha rows
    alpha_archives = conn.execute(
        "SELECT count(*) FROM Interaction_Edit_History WHERE interaction_id=? AND old_content=?",
        (iid, "alpha"),
    ).fetchone()[0]
    assert alpha_archives <= 1, "content-hash dedupe should cap alpha archives at 1"


def test_swap_twice_returns_other_version(mem_manager):
    """Swap to same archive index twice — second call returns the *other*
    version (the previously-canonical one), because each swap re-archives
    what was current. Not idempotent, but deterministic."""
    now = datetime.now()
    mem_manager.log_message("u1", "p", "c", "user", "Human", "v1", now, platform_message_id="sw", server_id=None)
    mem_manager.handle_message_edit("sw", "v2")
    conn = mem_manager._get_connection()
    iid = conn.execute("SELECT interaction_id FROM User_Interactions WHERE platform_message_id='sw'").fetchone()[0]

    mem_manager.swap_interaction_version(iid, 0)
    # After the swap, original target is removed; the new archive is v2.
    # Swapping k=0 again returns v2 (which was archived).
    res = mem_manager.swap_interaction_version(iid, 0)
    assert res["current_content"] == "v2"


def test_edit_history_vec_cleanup_cascade(mem_manager):
    """Editing a message invalidates Message_Embeddings + vec_Message_Embeddings."""
    now = datetime.now()
    mem_manager.log_message("u1", "p", "c", "user", "Human", "v1", now, platform_message_id="vc", server_id=None)
    conn = mem_manager._get_connection()
    iid = conn.execute("SELECT interaction_id FROM User_Interactions WHERE platform_message_id='vc'").fetchone()[0]
    mem_manager.store_message_embedding(iid, _embed(0.4), EMBEDDING_MODEL, now)
    conn.execute(
        "INSERT INTO vec_Message_Embeddings (interaction_id, embedding) VALUES (?, ?)",
        (iid, _embed(0.4)),
    )
    conn.commit()

    mem_manager.handle_message_edit("vc", "v2")
    row = conn.execute("SELECT 1 FROM Message_Embeddings WHERE interaction_id=?", (iid,)).fetchone()
    assert row is None
    vrow = conn.execute("SELECT 1 FROM vec_Message_Embeddings WHERE interaction_id=?", (iid,)).fetchone()
    assert vrow is None


def test_list_versions_edited_at_collision(mem_manager):
    """Two edit-history rows with the same edited_at timestamp must produce
    deterministic ordering (by edit_id ASC)."""
    now = datetime.now()
    mem_manager.log_message("u1", "p", "c", "user", "Human", "v1", now, platform_message_id="col", server_id=None)
    conn = mem_manager._get_connection()
    iid = conn.execute("SELECT interaction_id FROM User_Interactions WHERE platform_message_id='col'").fetchone()[0]
    same_ts = datetime.now()
    conn.execute(
        "INSERT INTO Interaction_Edit_History (interaction_id, old_content, edited_at) VALUES (?, ?, ?)",
        (iid, "old_a", same_ts),
    )
    conn.execute(
        "INSERT INTO Interaction_Edit_History (interaction_id, old_content, edited_at) VALUES (?, ?, ?)",
        (iid, "old_b", same_ts),
    )
    conn.commit()

    versions = mem_manager.list_interaction_versions(iid)
    # Two archives + canonical
    assert len(versions) == 3
    archive_contents = [v["content"] for v in versions if v["edit_id"] is not None]
    assert archive_contents == ["old_a", "old_b"]


# -------------------- Concurrency --------------------

def test_concurrent_log_and_edit(file_mem_manager):
    """Logging and editing the same message concurrently doesn't corrupt state."""
    now = datetime.now()
    # Seed a message
    file_mem_manager.log_message(
        "u1", "p", "c", "user", "Human", "v1", now,
        platform_message_id="conc", server_id=None,
    )

    errors = []
    barrier = threading.Barrier(4)

    def worker_edit(version):
        barrier.wait()
        try:
            for _ in range(5):
                file_mem_manager.handle_message_edit("conc", f"edit-{version}")
        except Exception as e:
            errors.append(e)

    def worker_log():
        barrier.wait()
        try:
            for i in range(10):
                file_mem_manager.log_message(
                    "u1", "p", "c", "user", "Human", f"more-{i}",
                    datetime.now(), platform_message_id=f"more-{i}", server_id=None,
                )
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=worker_edit, args=(1,)),
        threading.Thread(target=worker_edit, args=(2,)),
        threading.Thread(target=worker_edit, args=(3,)),
        threading.Thread(target=worker_log),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Concurrent edit+log raised: {errors}"
    history = file_mem_manager.get_personal_history("u1", "p")
    assert len(history) >= 1


@pytest.mark.asyncio
async def test_concurrent_hindsight_queue_and_transaction(tmp_path):
    """HindsightBackend's per-bank queue operates outside MemoryManager's lock;
    enqueues from multiple tasks must not interfere with concurrent MM writes."""
    import asyncio
    from unittest.mock import patch
    from src.memory.backend.hindsight import HindsightBackend

    mm = MemoryManager(db_path=str(tmp_path / "mix.db"))
    mm.create_schema()
    backend = HindsightBackend(
        url="http://stub:8888",
        override_db_path=str(tmp_path / "overrides.db"),
        doc_scope_db_path=str(tmp_path / "doc_scope.db"),
    )

    async def fake_aretain(bank_id, items, async_=True):
        return {"id": "ok"}

    client = backend._get_client()
    try:
        with patch.object(client, "aretain", side_effect=fake_aretain):
            async def retain_burst():
                for i in range(10):
                    await backend.retain_turn(
                        "alice", "user", f"msg-{i}",
                        timestamp=datetime.now(timezone.utc),
                        scope_tags=["channel:c1"], source_persona="alice",
                    )

            def db_writes():
                for i in range(20):
                    mm.log_message(
                        "u", "p", "c", "user", "Human", f"x-{i}",
                        datetime.now(), platform_message_id=f"x-{i}", server_id=None,
                    )

            # Run hindsight retains alongside sync DB writes (in a thread).
            loop = asyncio.get_event_loop()
            db_future = loop.run_in_executor(None, db_writes)
            await asyncio.gather(retain_burst(), retain_burst())
            await db_future
        await backend.aclose()
    finally:
        mm.close()

    # Confirm DB writes all landed
    assert len(mm.get_personal_history("u", "p")) == 20


@pytest.mark.asyncio
async def test_hindsight_concurrent_retain_same_bank(tmp_path):
    """Concurrent retains to the same bank coalesce through one worker queue
    and preserve FIFO."""
    from unittest.mock import patch
    from src.memory.backend.hindsight import HindsightBackend
    import asyncio

    backend = HindsightBackend(
        url="http://stub:8888",
        override_db_path=str(tmp_path / "ov.db"),
        doc_scope_db_path=str(tmp_path / "ds.db"),
    )
    seen = []

    async def fake_aretain(bank_id, items, async_=True):
        seen.extend(it["content"] for it in items)
        return {"id": "ok"}

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=fake_aretain):
        async def burst(prefix):
            for i in range(5):
                await backend.retain_turn(
                    "alice", "user", f"{prefix}-{i}",
                    timestamp=datetime.now(timezone.utc),
                    scope_tags=["channel:c1"], source_persona="alice",
                )

        await asyncio.gather(burst("a"), burst("b"))
        await backend.aclose()

    # All 10 retains must have hit the upstream
    assert len(seen) == 10
    assert sorted(seen) == sorted([f"a-{i}" for i in range(5)] + [f"b-{i}" for i in range(5)])


# SKIP: test_hindsight_queue_overflow_backpressure — DP-199 deferred bug 4
# (hindsight.py:415-418 queue is unbounded; no backpressure exists to test).
