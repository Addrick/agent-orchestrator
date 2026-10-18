# src/memory/backend/sqlite_store.py
"""Lazy-connect SQLite scaffolding shared by the backend's small local stores.

Sync sqlite3 in async land is fine at their volume: one indexed upsert per
event, under a re-entrant lock. A subclass declares `SCHEMA` and, when an
older file on disk needs it, `_migrate`.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional


class LazySqliteStore:
    SCHEMA = ""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None

    def _get(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.db_path != ":memory:":
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(self.SCHEMA)
            self._migrate(self._conn)
            self._conn.commit()
        return self._conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Bring a file written by an older schema up to date. Default: none."""

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
