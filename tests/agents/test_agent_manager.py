"""AgentManager config-application tests.

Covers the agent schedule wiring, including DP-221's removal of the legacy
`poll_interval` config key (the old format `agents.json` used before the nested
`schedule` block). After the cleanup, `schedule` is the only accepted form and a
bare `poll_interval` is ignored.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from src.agents.agent_manager import AgentManager
from src.agents.base import Agent


class _NoopAgent(Agent):
    """Minimal agent whose deploy loop idles until stopped."""

    def __init__(self, chat_system):
        super().__init__(chat_system, inject_personas=False)

    async def deploy(self) -> None:  # pragma: no cover - never iterates in tests
        return None


def _make_manager() -> AgentManager:
    chat_system = MagicMock()
    # Point the manager at a non-existent config path so no file config leaks in.
    mgr = AgentManager(
        chat_system=chat_system,
        memory_manager=chat_system.memory_manager,
        config_path=__import__("pathlib").Path("/nonexistent/agents.json"),
    )
    mgr.register("noop", _NoopAgent)
    return mgr


@pytest.mark.asyncio
async def test_schedule_block_is_applied():
    mgr = _make_manager()
    await mgr.start_agent("noop", config_overrides={"schedule": {"interval": 5}})
    try:
        instance = mgr._running["noop"].instance
        assert instance.schedule == {"interval": 5}
    finally:
        await mgr.stop_agent("noop")
        await asyncio.gather(mgr._running["noop"].task, return_exceptions=True)


@pytest.mark.asyncio
async def test_legacy_poll_interval_is_ignored():
    """DP-221: the retired `poll_interval` key no longer maps to a schedule."""
    mgr = _make_manager()
    await mgr.start_agent("noop", config_overrides={"poll_interval": 30})
    try:
        instance = mgr._running["noop"].instance
        # schedule is untouched (class default), not derived from poll_interval.
        assert instance.schedule == {}
        assert "interval" not in instance.schedule
    finally:
        await mgr.stop_agent("noop")
        await asyncio.gather(mgr._running["noop"].task, return_exceptions=True)


# ---------- single-shot inference agents (DP-292) ----------

class _InferenceAgent:
    """Minimal single-shot agent: takes chat_system, optionally a router."""

    def __init__(self, chat_system, notification_router=None):
        self.chat_system = chat_system
        self.notification_router = notification_router

    async def tag(self, body):
        return "ok"


class _RouterlessAgent:
    def __init__(self, chat_system):
        self.chat_system = chat_system


def test_register_and_get_inference_agent_builds_and_caches():
    mgr = _make_manager()
    mgr.register_inference_agent("infer", _RouterlessAgent)
    a = mgr.get_inference_agent("infer")
    assert isinstance(a, _RouterlessAgent)
    assert a.chat_system is mgr._chat_system
    # Cached: same instance on second lookup.
    assert mgr.get_inference_agent("infer") is a


def test_get_unregistered_inference_agent_returns_none():
    mgr = _make_manager()
    assert mgr.get_inference_agent("nope") is None


def test_inference_agent_receives_convention_di():
    """A single-shot agent whose __init__ wants notification_router gets it —
    the same DI path scheduled agents use (this is what unblocks the DM feature)."""
    mgr = _make_manager()
    router = MagicMock()
    mgr.notification_router = router
    mgr.register_inference_agent("infer", _InferenceAgent)
    a = mgr.get_inference_agent("infer")
    assert a.notification_router is router


def test_inference_agent_not_started_as_task():
    mgr = _make_manager()
    mgr.register_inference_agent("infer", _RouterlessAgent)
    mgr.get_inference_agent("infer")
    # Registering/building an inference agent must not launch a loop task.
    assert "infer" not in mgr._running
