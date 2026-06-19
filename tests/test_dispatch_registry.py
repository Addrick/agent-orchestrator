# tests/test_dispatch_registry.py
"""Unit tests for the DP-227 in-memory agent registry."""

import pytest

from src.self_edit import registry as reg
from src.self_edit.registry import AgentRecord, AgentRegistry

pytestmark = pytest.mark.asyncio


def _rec(agent_id="a1", bug_id="DP-9", status=reg.RUNNING) -> AgentRecord:
    return AgentRecord(
        agent_id=agent_id, bug_id=bug_id, description="d",
        worktree="/w", branch="bugfix/DP-9-fix",
        raw_log="/w/.fixr/raw.jsonl", events_log="/w/.fixr/events.jsonl",
        status=status,
    )


async def test_add_get_remove():
    r = AgentRegistry()
    await r.add(_rec())
    got = await r.get("a1")
    assert got is not None and got.bug_id == "DP-9"
    removed = await r.remove("a1")
    assert removed is not None
    assert await r.get("a1") is None


async def test_add_duplicate_raises():
    r = AgentRegistry()
    await r.add(_rec())
    with pytest.raises(KeyError):
        await r.add(_rec())


async def test_update_fields_and_unknown_is_none():
    r = AgentRegistry()
    await r.add(_rec())
    updated = await r.update("a1", status=reg.DONE, pr_url="http://pr/1")
    assert updated is not None
    assert updated.status == reg.DONE
    assert updated.pr_url == "http://pr/1"
    assert await r.update("nope", status=reg.DONE) is None


async def test_list_active_only():
    r = AgentRegistry()
    await r.add(_rec("a1", "DP-1", reg.RUNNING))
    await r.add(_rec("a2", "DP-2", reg.DONE))
    await r.add(_rec("a3", "DP-3", reg.WAITING))
    everything = await r.list()
    assert len(everything) == 3
    active = await r.list(active_only=True)
    assert {x.agent_id for x in active} == {"a1", "a3"}


async def test_archive_hides_from_default_list():
    r = AgentRegistry()
    await r.add(_rec("a1", "DP-1", reg.DONE))
    await r.add(_rec("a2", "DP-2", reg.RUNNING))
    archived = await r.archive("a1")
    assert archived is not None and archived.archived is True
    # default list hides archived; include_archived shows it.
    assert {x.agent_id for x in await r.list()} == {"a2"}
    assert {x.agent_id for x in await r.list(include_archived=True)} == {"a1", "a2"}


async def test_archive_unknown_is_none():
    r = AgentRegistry()
    assert await r.archive("ghost") is None


async def test_has_active_for_bug():
    r = AgentRegistry()
    await r.add(_rec("a1", "DP-1", reg.RUNNING))
    await r.add(_rec("a2", "DP-2", reg.DONE))
    assert await r.has_active_for_bug("DP-1") is True
    assert await r.has_active_for_bug("DP-2") is False   # done, not active
    assert await r.has_active_for_bug("DP-404") is False
