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


class TestAppManagerTaskShutdown:
    """DP-304: shutdown() used to ignore _running_tasks entirely, so a
    registered daemon was never stopped — it outlived shutdown until the event
    loop went away."""

    @pytest.mark.asyncio
    async def test_stop_callback_lets_task_drain(self):
        app = AppManager()
        stop_event = asyncio.Event()
        completed = False

        async def cooperative():
            nonlocal completed
            await stop_event.wait()
            completed = True

        app.register_task("cooperative", cooperative(), stop=stop_event.set)
        app._running_tasks = [
            asyncio.create_task(coro, name=name) for name, coro in app._pending_tasks
        ]

        await app.shutdown()

        # Drained to completion, not cancelled mid-flight.
        assert completed is True
        assert app._running_tasks[0].done()
        assert not app._running_tasks[0].cancelled()

    @pytest.mark.asyncio
    async def test_task_without_stop_callback_is_cancelled(self):
        app = AppManager()

        async def uncooperative():
            await asyncio.Event().wait()  # never completes on its own

        app.register_task("uncooperative", uncooperative())
        app._running_tasks = [
            asyncio.create_task(coro, name=name) for name, coro in app._pending_tasks
        ]
        await asyncio.sleep(0)  # let it reach the await

        app.GRACEFUL_SHUTDOWN_SECONDS = 0.05
        await app.shutdown()

        assert app._running_tasks[0].cancelled()

    @pytest.mark.asyncio
    async def test_ignored_stop_signal_is_cancelled_after_grace(self):
        """A task that is signalled but refuses to exit still gets cancelled."""
        app = AppManager()
        signalled = False

        def stop():
            nonlocal signalled
            signalled = True

        async def stubborn():
            await asyncio.Event().wait()

        app.register_task("stubborn", stubborn(), stop=stop)
        app._running_tasks = [
            asyncio.create_task(coro, name=name) for name, coro in app._pending_tasks
        ]
        await asyncio.sleep(0)

        app.GRACEFUL_SHUTDOWN_SECONDS = 0.05
        await app.shutdown()

        assert signalled is True
        assert app._running_tasks[0].cancelled()

    @pytest.mark.asyncio
    async def test_task_that_refuses_to_unwind_is_abandoned(self):
        """Cancellation is a request, not a guarantee. `discord.py`'s
        `Client.close()` does network teardown on the way out, so awaiting a
        cancelled task unbounded hangs shutdown — this pins the bound."""
        app = AppManager()

        async def swallows_cancellation():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Slow teardown that outlives the grace window.
                await asyncio.sleep(30)

        app.register_task("slow_teardown", swallows_cancellation())
        app._running_tasks = [
            asyncio.create_task(coro, name=name) for name, coro in app._pending_tasks
        ]
        await asyncio.sleep(0)

        app.GRACEFUL_SHUTDOWN_SECONDS = 0.05
        app.CANCEL_GRACE_SECONDS = 0.05

        # Must return, not hang.
        await asyncio.wait_for(app.shutdown(), timeout=2.0)

        # Abandoned, still pending — shutdown did not wait on it.
        assert not app._running_tasks[0].done()
        app._running_tasks[0].cancel()

    @pytest.mark.asyncio
    async def test_failing_stop_callback_does_not_block_shutdown(self):
        app = AppManager()

        def boom():
            raise RuntimeError("stop callback exploded")

        async def cooperative():
            await asyncio.Event().wait()

        app.register_task("boom", cooperative(), stop=boom)
        app._running_tasks = [
            asyncio.create_task(coro, name=name) for name, coro in app._pending_tasks
        ]
        await asyncio.sleep(0)

        app.GRACEFUL_SHUTDOWN_SECONDS = 0.05
        await app.shutdown()  # must not propagate the callback's error

        assert app._running_tasks[0].cancelled()
