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
