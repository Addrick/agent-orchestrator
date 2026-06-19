# src/self_edit/store.py
"""SQLite persistence for the fixr agent registry (DP-233).

Why: the registry was in-memory only, so a derpr restart vaporized every
in-flight agent record (exactly what happened to DP-ZAM-001 — the container
restarted and fixr reported the agent had "disappeared"). This persists records
so they survive a restart.

On load, any agent still marked RUNNING/WAITING is flipped to ORPHANED: its
detached `claude` process and per-agent bridge task did NOT survive the restart
(the bridge is an in-process asyncio task), so the record can no longer be
resumed — it is surfaced as orphaned rather than falsely shown as live. Bridge
re-attachment is a deliberate non-goal for v1.

Access is single-connection (check_same_thread=False); the owning AgentRegistry
serializes every call behind its asyncio.Lock, so the connection is never
touched concurrently.
"""

from __future__ import annotations

import sqlite3
from typing import List

from src.self_edit.registry import AgentRecord, ORPHANED, RUNNING, WAITING

_COLUMNS = [
    "agent_id", "bug_id", "description", "worktree", "branch", "raw_log",
    "events_log", "pid", "session_id", "status", "pr_url", "last_event",
    "discord_thread_id", "archived", "created_at", "updated_at",
]


class AgentStore:
    """SQLite-backed store of ``AgentRecord``s. One row per dispatched agent."""

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.create_schema()

    def create_schema(self) -> None:
        """Idempotent — safe to call on every startup."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fixr_agents (
                agent_id          TEXT PRIMARY KEY,
                bug_id            TEXT NOT NULL,
                description       TEXT NOT NULL,
                worktree          TEXT NOT NULL,
                branch            TEXT NOT NULL,
                raw_log           TEXT NOT NULL,
                events_log        TEXT NOT NULL,
                pid               INTEGER,
                session_id        TEXT,
                status            TEXT NOT NULL,
                pr_url            TEXT,
                last_event        TEXT,
                discord_thread_id TEXT,
                archived          INTEGER NOT NULL DEFAULT 0,
                created_at        REAL NOT NULL,
                updated_at        REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fixr_agents_bug ON fixr_agents(bug_id)"
        )
        self._migrate_add_archived()
        self._conn.commit()

    def _migrate_add_archived(self) -> None:
        """DP-237: add the ``archived`` column to a pre-DP-237 DB. The CREATE
        above only fires on a fresh DB (IF NOT EXISTS), so an existing table from
        an earlier deploy lacks the column — add it idempotently."""
        cols = {row["name"] for row in self._conn.execute(
            "PRAGMA table_info(fixr_agents)"
        ).fetchall()}
        if "archived" not in cols:
            self._conn.execute(
                "ALTER TABLE fixr_agents ADD COLUMN archived INTEGER NOT NULL DEFAULT 0"
            )

    def upsert(self, record: AgentRecord) -> None:
        d = record.to_dict()
        placeholders = ", ".join("?" for _ in _COLUMNS)
        self._conn.execute(
            f"INSERT OR REPLACE INTO fixr_agents ({', '.join(_COLUMNS)}) "
            f"VALUES ({placeholders})",
            [d[c] for c in _COLUMNS],
        )
        self._conn.commit()

    def delete(self, agent_id: str) -> None:
        self._conn.execute("DELETE FROM fixr_agents WHERE agent_id = ?", (agent_id,))
        self._conn.commit()

    def orphan_stale(self) -> int:
        """Flip RUNNING/WAITING rows to ORPHANED (their process + bridge did not
        survive the restart). Returns the number of rows orphaned."""
        cur = self._conn.execute(
            "UPDATE fixr_agents SET status = ? WHERE status IN (?, ?)",
            (ORPHANED, RUNNING, WAITING),
        )
        self._conn.commit()
        return cur.rowcount

    def load_all(self) -> List[AgentRecord]:
        rows = self._conn.execute(
            "SELECT * FROM fixr_agents ORDER BY created_at"
        ).fetchall()
        return [_row_to_record(row) for row in rows]


def _row_to_record(row: sqlite3.Row) -> AgentRecord:
    """SQLite stores ``archived`` as 0/1; coerce it back to bool so the in-memory
    record matches the dataclass type."""
    fields = {c: row[c] for c in _COLUMNS}
    fields["archived"] = bool(fields["archived"])
    return AgentRecord(**fields)
