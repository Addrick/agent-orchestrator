# tests/test_app_manager.py

import asyncio
import pytest
from unittest.mock import patch

from src.app_manager import AppManager
from src.agents.base import Agent


class StubAgent(Agent):
    """Minimal agent for AppManager tests."""

    def __init__(self):
        # Skip Agent.__init__ to avoid ChatSystem dependency
        self._stopping = False
        self.poll_count = 0
        self.on_start_called = False

    async def on_start(self):
        self.on_start_called = True

    async def poll(self):
        self.poll_count += 1


class FailingAgent(Agent):
    """Agent whose poll raises an exception."""

    def __init__(self):
        self._stopping = False

    async def poll(self):
        raise RuntimeError("Simulated poll failure")


class TestAppManagerRegistration:
    def test_register_agent(self):
        app = AppManager()
        agent = StubAgent()
        app.register_agent("test", agent, 10)
        assert "test" in app._agents
        assert app._agents["test"] is agent

    def test_register_task(self):
        app = AppManager()

        async def dummy():
            pass

        app.register_task("dummy", dummy())
        assert len(app._pending_tasks) == 1
        assert app._pending_tasks[0][0] == "dummy"


class TestAppManagerStart:
    @pytest.mark.asyncio
    async def test_on_start_called_for_agents(self):
        app = AppManager()
        agent = StubAgent()
        app.register_agent("test", agent, 60)

        # Register a task that completes immediately so start() returns
        async def quick_task():
            pass

        app.register_task("quick", quick_task())
        await app.start()
        assert agent.on_start_called

    @pytest.mark.asyncio
    async def test_on_start_error_does_not_crash(self):
        app = AppManager()
        agent = StubAgent()

        async def bad_start():
            raise RuntimeError("Startup failed")

        agent.on_start = bad_start  # type: ignore[assignment]
        app.register_agent("bad", agent, 60)

        async def quick_task():
            pass

        app.register_task("quick", quick_task())
        # Should not raise
        await app.start()

    @pytest.mark.asyncio
    async def test_no_tasks_or_agents_returns(self):
        app = AppManager()
        # Should return immediately with a warning
        await app.start()


class TestAppManagerShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_signals_agents(self):
        app = AppManager()
        agent = StubAgent()
        app.register_agent("test", agent, 60)
        assert agent.stopping is False
        await app.shutdown()
        assert agent.stopping is True

    @pytest.mark.asyncio
    async def test_shutdown_with_running_scheduler_does_not_raise(self):
        app = AppManager()
        agent = StubAgent()
        app.register_agent("test", agent, 60)

        # Start the scheduler, then verify shutdown completes cleanly
        app._scheduler.start()
        assert app._scheduler.running is True
        await app.shutdown()
        # Scheduler may not be fully stopped synchronously with wait=False,
        # but shutdown should not raise
        assert agent.stopping is True

    @pytest.mark.asyncio
    async def test_shutdown_safe_when_scheduler_not_started(self):
        app = AppManager()
        # Should not raise even if scheduler never started
        await app.shutdown()


class TestSafePoll:
    @pytest.mark.asyncio
    async def test_safe_poll_calls_agent(self):
        agent = StubAgent()
        await AppManager._safe_poll(agent, "test")
        assert agent.poll_count == 1

    @pytest.mark.asyncio
    async def test_safe_poll_catches_exception(self):
        agent = FailingAgent()
        # Should not raise
        await AppManager._safe_poll(agent, "failing")
