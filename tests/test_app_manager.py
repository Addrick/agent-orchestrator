# tests/test_app_manager.py

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.app_manager import AppManager


@pytest.fixture
def mock_agent_manager():
    mgr = MagicMock()
    mgr.auto_start = AsyncMock()
    mgr.shutdown_all = AsyncMock()
    mgr.get_running.return_value = []
    return mgr


class TestAppManagerRegistration:
    def test_register_task(self):
        app = AppManager()

        async def dummy():
            pass

        app.register_task("dummy", dummy())
        assert len(app._pending_tasks) == 1
        assert app._pending_tasks[0][0] == "dummy"

    def test_init_with_agent_manager(self, mock_agent_manager):
        app = AppManager(agent_manager=mock_agent_manager)
        assert app._agent_manager is mock_agent_manager

    def test_init_without_agent_manager(self):
        app = AppManager()
        assert app._agent_manager is None


class TestAppManagerStart:
    @pytest.mark.asyncio
    async def test_auto_starts_agents(self, mock_agent_manager):
        app = AppManager(agent_manager=mock_agent_manager)

        async def quick_task():
            pass

        app.register_task("quick", quick_task())
        await app.start()
        mock_agent_manager.auto_start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_tasks_or_agents_returns(self):
        app = AppManager()
        # Should return immediately with a warning
        await app.start()

    @pytest.mark.asyncio
    async def test_no_tasks_with_running_agents_blocks(self, mock_agent_manager):
        """When agents are running but no tasks, start() should keep alive."""
        mock_agent_manager.get_running.return_value = ["dispatch"]
        app = AppManager(agent_manager=mock_agent_manager)

        # start() would block forever in the stop_event.wait(), so we timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(app.start(), timeout=0.1)


class TestAppManagerShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_calls_agent_manager(self, mock_agent_manager):
        app = AppManager(agent_manager=mock_agent_manager)
        await app.shutdown()
        mock_agent_manager.shutdown_all.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_safe_without_agent_manager(self):
        app = AppManager()
        # Should not raise even without agent manager
        await app.shutdown()
