# tests/test_agent_store.py
"""DP-233: SQLite persistence for the fixr agent registry — write-through,
restart recovery (orphan stale), and round-trip integrity."""

import pytest

from src.self_edit import registry as reg
from src.self_edit.registry import AgentRecord, AgentRegistry
from src.self_edit.store import AgentStore


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
