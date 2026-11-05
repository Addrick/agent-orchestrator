"""The retained-turn format (DP-402) and the rolling doc-scope store.

DP-409's retry rebuild now works per block (DP-495): see
tests/memory/test_block_retain.py."""

import sqlite3
from datetime import datetime, timezone

import pytest

from config import global_config
from src.memory.backend.hindsight import _DocScopeStore
from src.turn_persistence import format_retained_turn


@pytest.fixture(autouse=True)
def eastern(monkeypatch):
    monkeypatch.setattr(global_config, "LOCAL_TZ", "America/New_York")


# ---------- format_retained_turn ----------

def test_assistant_turn_is_retained_as_its_content():
    """Reasoning lives in its own column (DP-434), so content is retained as is."""
    ts = datetime(2026, 9, 26, 18, 5, tzinfo=timezone.utc)
    out = format_retained_turn("assistant", "explr", "Hello.\n", ts)
    assert out == "explr: Hello."


def test_user_text_mentioning_think_is_untouched():
    ts = datetime(2026, 9, 26, 18, 5, tzinfo=timezone.utc)
    out = format_retained_turn("user", "Adam", "why <think>x</think>?", ts)
    assert out.endswith("Adam: why <think>x</think>?")


# ---------- _DocScopeStore (still used by agent experiences) ----------

def test_pre_dp409_store_gains_untrusted_column(tmp_path):
    path = tmp_path / "ds.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE Doc_Scope (scope_key TEXT PRIMARY KEY,"
                 " document_id TEXT NOT NULL, last_ts TEXT NOT NULL)")
    conn.execute("INSERT INTO Doc_Scope VALUES ('b:c', 'b:c:2026-09-27T00:00:00+00:00',"
                 " '2026-09-27T00:00:00+00:00')")
    conn.commit()
    conn.close()
    store = _DocScopeStore(str(path))
    doc, mode = store.resolve("b:c", datetime(2026, 9, 27, 0, 5, tzinfo=timezone.utc), untrusted=True)
    assert (doc, mode) == ("b:c:2026-09-27T00:00:00+00:00", "append")
    cols = {r["name"] for r in store._get().execute("PRAGMA table_info(Doc_Scope)")}
    assert "untrusted" in cols
    store.close()
