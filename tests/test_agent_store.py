# tests/test_agent_store.py
"""DP-233: SQLite persistence for the fixr agent registry — write-through,
restart recovery (orphan stale), and round-trip integrity."""

import sqlite3

import pytest

from src.self_edit import registry as reg
from src.self_edit.registry import AgentRecord, AgentRegistry
from src.self_edit.store import AgentStore

# Pre-DP-237 schema (no `archived` column) — used to test the migration against
# an existing on-disk DB, which a fresh :memory: store would never exercise.
_LEGACY_CREATE = """
CREATE TABLE fixr_agents (
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
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL
)
"""
_LEGACY_COLS = [
    "agent_id", "bug_id", "description", "worktree", "branch", "raw_log",
    "events_log", "pid", "session_id", "status", "pr_url", "last_event",
    "discord_thread_id", "created_at", "updated_at",
]


def _seed_legacy_db(db_path, *, agent_id="legacy1", status=reg.DONE):
    """Create a pre-DP-237 DB (no `archived` column) with one row, then close."""
    conn = sqlite3.connect(db_path)
    conn.execute(_LEGACY_CREATE)
    rec = _rec(agent_id=agent_id, status=status, pr_url="http://pr/legacy")
    d = rec.to_dict()
    conn.execute(
        f"INSERT INTO fixr_agents ({', '.join(_LEGACY_COLS)}) "
        f"VALUES ({', '.join('?' for _ in _LEGACY_COLS)})",
        [d[c] for c in _LEGACY_COLS],
    )
    conn.commit()
    conn.close()


def _rec(agent_id="DP-1-1", bug_id="DP-1", status=reg.RUNNING, **kw):
    return AgentRecord(
        agent_id=agent_id, bug_id=bug_id, description="d", worktree="/w",
        branch="b", raw_log="/r", events_log="/e", status=status, **kw,
    )


def _store(tmp_path):
    return AgentStore(str(tmp_path / "fixr_registry.db"))


# --- store-level --------------------------------------------------------------

def test_create_schema_idempotent(tmp_path):
    db = str(tmp_path / "r.db")
    AgentStore(db)
    s2 = AgentStore(db)  # second open re-runs create_schema; must not raise
    s2.create_schema()
    assert s2.load_all() == []


def test_upsert_load_roundtrip_preserves_all_fields(tmp_path):
    s = _store(tmp_path)
    rec = _rec(pid=42, session_id="sid", pr_url="http://pr/1",
               last_event="done", discord_thread_id="9001")
    s.upsert(rec)
    loaded = s.load_all()
    assert len(loaded) == 1
    assert loaded[0].to_dict() == rec.to_dict()


def test_upsert_replaces_existing_row(tmp_path):
    s = _store(tmp_path)
    s.upsert(_rec(status=reg.RUNNING))
    s.upsert(_rec(status=reg.DONE, pr_url="http://pr/9"))
    loaded = s.load_all()
    assert len(loaded) == 1
    assert loaded[0].status == reg.DONE
    assert loaded[0].pr_url == "http://pr/9"


def test_delete(tmp_path):
    s = _store(tmp_path)
    s.upsert(_rec())
    s.delete("DP-1-1")
    assert s.load_all() == []


def test_orphan_stale_only_flips_active(tmp_path):
    s = _store(tmp_path)
    s.upsert(_rec(agent_id="run", status=reg.RUNNING))
    s.upsert(_rec(agent_id="wait", status=reg.WAITING))
    s.upsert(_rec(agent_id="done", status=reg.DONE))
    s.upsert(_rec(agent_id="err", status=reg.ERROR))
    n = s.orphan_stale()
    assert n == 2
    by_id = {r.agent_id: r.status for r in s.load_all()}
    assert by_id == {
        "run": reg.ORPHANED, "wait": reg.ORPHANED,
        "done": reg.DONE, "err": reg.ERROR,
    }


# --- registry write-through ---------------------------------------------------

async def test_registry_add_update_remove_write_through(tmp_path):
    s = _store(tmp_path)
    r = AgentRegistry(store=s)
    await r.add(_rec(agent_id="a1"))
    assert {x.agent_id for x in s.load_all()} == {"a1"}

    await r.update("a1", status=reg.DONE, pr_url="http://pr/3")
    persisted = {x.agent_id: x for x in s.load_all()}["a1"]
    assert persisted.status == reg.DONE and persisted.pr_url == "http://pr/3"

    await r.remove("a1")
    assert s.load_all() == []


async def test_registry_without_store_does_not_touch_disk(tmp_path):
    """Back-compat: a store-less registry behaves exactly as before (in-memory)."""
    r = AgentRegistry()
    await r.add(_rec(agent_id="a1"))
    assert (await r.get("a1")) is not None


# --- restart recovery ---------------------------------------------------------

async def test_records_survive_restart_and_active_become_orphaned(tmp_path):
    db = str(tmp_path / "r.db")
    # First "process": dispatch two agents, finish one.
    r1 = AgentRegistry(store=AgentStore(db))
    await r1.add(_rec(agent_id="live", status=reg.RUNNING))
    await r1.add(_rec(agent_id="fin", status=reg.RUNNING))
    await r1.update("fin", status=reg.DONE)

    # Second "process": fresh registry over the same DB (simulates a restart).
    r2 = AgentRegistry(store=AgentStore(db))
    live = await r2.get("live")
    fin = await r2.get("fin")
    assert live is not None and live.status == reg.ORPHANED  # was RUNNING → gone
    assert fin is not None and fin.status == reg.DONE         # terminal preserved
    # has_active_for_bug must not count an orphaned agent as in-flight.
    assert (await r2.has_active_for_bug("DP-1")) is False


# --- DP-237 archived migration ------------------------------------------------

def test_migrate_adds_archived_column_preserving_data(tmp_path):
    """Opening a pre-DP-237 DB adds the `archived` column, defaults existing rows
    to False, and preserves all other data."""
    db = str(tmp_path / "legacy.db")
    _seed_legacy_db(db, agent_id="legacy1")

    s = AgentStore(db)  # __init__ runs create_schema → migration
    loaded = s.load_all()
    assert len(loaded) == 1
    assert loaded[0].agent_id == "legacy1"
    assert loaded[0].pr_url == "http://pr/legacy"   # data preserved
    assert loaded[0].archived is False               # defaulted


def test_migrate_is_idempotent(tmp_path):
    """A second open of an already-migrated DB must not re-add or fail."""
    db = str(tmp_path / "legacy.db")
    _seed_legacy_db(db)
    AgentStore(db)
    s2 = AgentStore(db)
    s2.create_schema()  # extra explicit call
    assert len(s2.load_all()) == 1


def test_archived_round_trips_on_migrated_db(tmp_path):
    """After migration the column is fully usable: set archived, reload it."""
    db = str(tmp_path / "legacy.db")
    _seed_legacy_db(db, agent_id="a1")
    s = AgentStore(db)
    rec = s.load_all()[0]
    rec.archived = True
    s.upsert(rec)
    reloaded = AgentStore(db).load_all()[0]
    assert reloaded.archived is True
