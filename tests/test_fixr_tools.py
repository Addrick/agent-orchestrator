# tests/test_fixr_tools.py
"""Unit tests for the DP-227 fixr tool handlers (dispatch/inspect/answer/kill/discord)."""

import pytest

from src.self_edit import registry as reg
from src.self_edit.dispatcher import DispatcherError
from src.self_edit.fixr_tools import FixrToolHandler
from src.self_edit.registry import AgentRecord, AgentRegistry

pytestmark = pytest.mark.asyncio


class _FakeDispatcher:
    def __init__(self):
        self.dispatched = []
        self.answered = []
        self.killed = []
        self.pruned_calls = []
        self.raise_on_dispatch = None

    async def dispatch(self, bug_id, description):
        if self.raise_on_dispatch:
            raise DispatcherError(self.raise_on_dispatch)
        self.dispatched.append((bug_id, description))
        return AgentRecord(
            agent_id=f"{bug_id}-1", bug_id=bug_id, description=description,
            worktree="/w", branch=f"bugfix/{bug_id}-fix",
            raw_log="/r", events_log="/e", status=reg.RUNNING)

    async def answer_agent(self, agent_id, message):
        self.answered.append((agent_id, message))
        return AgentRecord(
            agent_id=agent_id, bug_id="DP-1", description="x", worktree="/w",
            branch="b", raw_log="/r", events_log="/e", status=reg.RUNNING)

    async def kill(self, agent_id, remove_worktree=False):
        self.killed.append((agent_id, remove_worktree))
        return True

    async def prune(self, *, agent_id=None, max_age_hours=None):
        self.pruned_calls.append((agent_id, max_age_hours))
        return [AgentRecord(
            agent_id="DP-1-1", bug_id="DP-1", description="x", worktree="/w",
            branch="b", raw_log="/r", events_log="/e", status=reg.DONE,
            archived=True)]


async def _handler():
    registry = AgentRegistry()
    dispatcher = _FakeDispatcher()
    return FixrToolHandler(dispatcher, registry), dispatcher, registry


async def test_dispatch_fix_ok_and_error():
    h, d, _ = await _handler()
    out = await h._dispatch_fix("DP-7", "broken thing")
    assert out["status"] == "dispatched"
    assert out["agent"]["bug_id"] == "DP-7"
    assert d.dispatched == [("DP-7", "broken thing")]

    d.raise_on_dispatch = "already in flight"
    out2 = await h._dispatch_fix("DP-7", "again")
    assert out2["status"] == "error"
    assert "already in flight" in out2["message"]


async def test_inspect_agents_list_and_single():
    h, _, registry = await _handler()
    await registry.add(AgentRecord(
        agent_id="a1", bug_id="DP-1", description="x", worktree="/w",
        branch="b", raw_log="/r", events_log="/e", status=reg.RUNNING))
    listed = await h._inspect_agents()
    assert listed["count"] == 1
    assert listed["agents"][0]["agent_id"] == "a1"

    one = await h._inspect_agents(agent_id="a1")
    assert one["found"] is True
    missing = await h._inspect_agents(agent_id="nope")
    assert missing["found"] is False


async def test_answer_and_kill():
    h, d, _ = await _handler()
    out = await h._answer_agent("a1", "use option B")
    assert out["status"] == "resumed"
    assert d.answered == [("a1", "use option B")]

    out2 = await h._kill_agent("a1", remove_worktree=True)
    assert out2["status"] == "killed"
    assert d.killed == [("a1", True)]


async def test_prune_agents_forwards_args_and_reports():
    h, d, _ = await _handler()
    out = await h._prune_agents(agent_id="DP-1-1", max_age_hours=24)
    assert out["status"] == "pruned"
    assert out["count"] == 1
    assert out["agents"][0]["agent_id"] == "DP-1-1"
    assert out["agents"][0]["archived"] is True
    assert d.pruned_calls == [("DP-1-1", 24)]


async def test_prune_agents_invokes_close_callback():
    registry = AgentRegistry()
    dispatcher = _FakeDispatcher()
    closed = []

    async def on_close(records):
        closed.extend(r.agent_id for r in records)

    h = FixrToolHandler(dispatcher, registry, on_prune_close=on_close)
    await h._prune_agents()
    assert closed == ["DP-1-1"]


async def test_inspect_agents_hides_archived_by_default():
    h, _, registry = await _handler()
    await registry.add(AgentRecord(
        agent_id="live", bug_id="DP-1", description="x", worktree="/w",
        branch="b", raw_log="/r", events_log="/e", status=reg.RUNNING))
    await registry.add(AgentRecord(
        agent_id="gone", bug_id="DP-2", description="x", worktree="/w",
        branch="b", raw_log="/r", events_log="/e", status=reg.DONE, archived=True))

    default = await h._inspect_agents()
    assert {a["agent_id"] for a in default["agents"]} == {"live"}
    full = await h._inspect_agents(include_archived=True)
    assert {a["agent_id"] for a in full["agents"]} == {"live", "gone"}
