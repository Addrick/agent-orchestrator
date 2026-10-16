# tests/memory/test_inline_reasoning_migration.py
"""DP-434: `create_schema` moves inline `<think>` out of assistant content.

Rows written before DP-434 carry the turn's reasoning folded into `content`.
Left there they replay into every prompt and leak to Hindsight on retain, so
the startup migration splits them into `reasoning_content`.
"""

import sqlite3
from datetime import datetime, timezone

import pytest

from src.memory.memory_manager import MemoryManager

_TS = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def legacy_db(tmp_path):
    """A pre-reasoning_content DB holding pre-DP-434 rows; create_schema not run."""
    db_path = str(tmp_path / "legacy_think.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE User_Interactions (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_identifier TEXT NOT NULL,
            persona_name TEXT NOT NULL,
            channel TEXT NOT NULL,
            author_role TEXT NOT NULL CHECK(author_role IN ('user', 'assistant', 'system')),
            author_name TEXT,
            content TEXT,
            timestamp TIMESTAMP NOT NULL,
            zammad_ticket_id INTEGER,
            platform_message_id TEXT,
            server_id TEXT
        );
    """)
    rows = [
        ("user", "<think>a user typed this</think> hi"),
        ("assistant", "<think>\nthe user wants a greeting\n</think>\nHello!"),
        ("assistant", "<think>\nstopped mid-thought"),
        ("assistant", "Plain reply."),
    ]
    for role, content in rows:
        conn.execute(
            "INSERT INTO User_Interactions (user_identifier, persona_name, channel,"
            " author_role, author_name, content, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("u", "p", "c", role, "p", content, _TS),
        )
    conn.commit()
    conn.close()
    manager = MemoryManager(db_path=db_path)
    yield manager
    manager.close()


def _rows(manager):
    cur = manager._get_connection().execute(
        "SELECT content, reasoning_content FROM User_Interactions ORDER BY interaction_id")
    return [(r["content"], r["reasoning_content"]) for r in cur.fetchall()]


def test_migration_moves_inline_think_into_reasoning_content(legacy_db):
    legacy_db.create_schema()
    assert _rows(legacy_db) == [
        ("<think>a user typed this</think> hi", None),   # user text is never touched
        ("Hello!", "the user wants a greeting"),
        ("", "stopped mid-thought"),                     # unclosed block is all reasoning
        ("Plain reply.", None),
    ]


def test_migration_is_idempotent(legacy_db):
    legacy_db.create_schema()
    first = _rows(legacy_db)
    legacy_db.create_schema()
    assert _rows(legacy_db) == first


def test_migration_covers_archives_keeps_existing_reasoning_and_drops_embeddings(tmp_path):
    manager = MemoryManager(db_path=str(tmp_path / "m.db"))
    manager.create_schema()
    iid = manager.log_message(
        "u", "p", "c", "assistant", "p", "<think>\nnew\n</think>\nAnswer", _TS,
        reasoning_content="earlier",
    )
    conn = manager._get_connection()
    conn.execute(
        "INSERT INTO Interaction_Edit_History (interaction_id, old_content, edited_at)"
        " VALUES (?, ?, ?)", (iid, "<think>\nold\n</think>\nOld answer", _TS))
    conn.execute(
        "INSERT INTO Message_Embeddings (interaction_id, embedding, model_name, created_at)"
        " VALUES (?, ?, ?, ?)", (iid, b"\x00" * 8, "m", _TS))
    conn.commit()

    manager.create_schema()

    assert _rows(manager) == [("Answer", "earlier\n\nnew")]
    arch = conn.execute(
        "SELECT old_content, old_reasoning_content FROM Interaction_Edit_History").fetchone()
    assert (arch["old_content"], arch["old_reasoning_content"]) == ("Old answer", "old")
    assert conn.execute(
        "SELECT COUNT(*) FROM Message_Embeddings WHERE interaction_id = ?", (iid,)
    ).fetchone()[0] == 0
    manager.close()


def test_reasoning_only_reply_stays_in_history_and_is_the_retry_target(tmp_path):
    """A reply that was only thinking is a visible turn: dropping it from the
    history getters hid it after reload and made a retry insert a duplicate."""
    manager = MemoryManager(db_path=str(tmp_path / "r.db"))
    manager.create_schema()
    manager.log_message("u", "p", "c", "user", "u", "hi", _TS)
    aid = manager.log_message("u", "p", "c", "assistant", "p", "", _TS,
                              reasoning_content="ran out of tokens thinking")
    history = manager.get_channel_history("c", "p")
    assert [r["interaction_id"] for r in history][-1] == aid
    assert manager.handle_portal_retry("p", "u", "c") == aid
    manager.close()
