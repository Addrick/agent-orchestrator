# tests/test_fixr_tools.py
"""Unit tests for the DP-227 fixr tool handlers (dispatch/inspect/answer/kill/discord)."""

import pytest

from config import global_config
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


class _FakeNotifier:
    def __init__(self, ok=True):
        self.ok = ok
        self.sent = []

    async def send(self, channel, recipient, subject, body):
        self.sent.append((channel, recipient, subject, body))
        return self.ok


async def _handler():
    registry = AgentRegistry()
    dispatcher = _FakeDispatcher()
    notifier = _FakeNotifier()
    return FixrToolHandler(dispatcher, registry, notifier), dispatcher, registry, notifier


async def test_dispatch_fix_ok_and_error():
    h, d, _, _ = await _handler()
    out = await h._dispatch_fix("DP-7", "broken thing")
    assert out["status"] == "dispatched"
    assert out["agent"]["bug_id"] == "DP-7"
    assert d.dispatched == [("DP-7", "broken thing")]

    d.raise_on_dispatch = "already in flight"
    out2 = await h._dispatch_fix("DP-7", "again")
    assert out2["status"] == "error"
    assert "already in flight" in out2["message"]


async def test_inspect_agents_list_and_single():
    h, _, registry, _ = await _handler()
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
    h, d, _, _ = await _handler()
    out = await h._answer_agent("a1", "use option B")
    assert out["status"] == "resumed"
    assert d.answered == [("a1", "use option B")]

    out2 = await h._kill_agent("a1", remove_worktree=True)
    assert out2["status"] == "killed"
    assert d.killed == [("a1", True)]


async def test_send_discord_requires_recipient(monkeypatch):
    h, _, _, notifier = await _handler()
    monkeypatch.setattr(global_config, "CC_FIXR_DISCORD_CHANNEL", "")
    out = await h._send_discord("subj", "body")
    assert out["status"] == "error"
    assert notifier.sent == []


async def test_send_discord_uses_config_default(monkeypatch):
    h, _, _, notifier = await _handler()
    monkeypatch.setattr(global_config, "CC_FIXR_DISCORD_CHANNEL", "chan-9")
    out = await h._send_discord("subj", "body")
    assert out["status"] == "sent"
    assert notifier.sent == [("discord", "chan-9", "subj", "body")]


async def test_send_discord_explicit_recipient_overrides(monkeypatch):
    h, _, _, notifier = await _handler()
    monkeypatch.setattr(global_config, "CC_FIXR_DISCORD_CHANNEL", "default")
    out = await h._send_discord("s", "b", recipient="override")
    assert out["recipient"] == "override"
    assert notifier.sent[0][1] == "override"
