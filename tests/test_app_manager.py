# tests/test_app_manager.py

import asyncio
import contextlib
import logging
import signal
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.app_manager import AppManager


@pytest.fixture
def mock_agent_manager():
    mgr = MagicMock()
    mgr.auto_start = AsyncMock()
    mgr.shutdown_all = AsyncMock()
    mgr.signal_stop_all = MagicMock()
    mgr.get_running.return_value = []
    return mgr


def _launch(app):
    """Start the registered coroutines the way start() does, without start()."""
    app._running_tasks = [
        asyncio.create_task(coro, name=name) for name, coro in app._pending_tasks
    ]
    return app._running_tasks


async def _elapsed(coro):
    """Run `coro`, return (result, seconds). Timing is the assertion for the
    'cancel the uncooperative immediately' contract — the *only* observable
    difference between cancelling at once and cancelling after the grace window
    is how long it took, so a test that doesn't measure it passes either way."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    result = await coro
    return result, loop.time() - started


class TestAppManagerRegistration:
    def test_register_task(self):
        app = AppManager()

        async def dummy():
            pass

        coro = dummy()
        app.register_task("dummy", coro)
        assert len(app._pending_tasks) == 1
        assert app._pending_tasks[0][0] == "dummy"
        coro.close()

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
        _launch(app)

        await app.shutdown()

        # Drained to completion, not cancelled mid-flight.
        assert completed is True
        assert app._running_tasks[0].done()
        assert not app._running_tasks[0].cancelled()

    @pytest.mark.asyncio
    async def test_task_without_stop_callback_is_cancelled_immediately(self):
        """A task nobody can signal must be cancelled at once, not waited out.

        The grace window is deliberately left at its production value: shrinking
        it is exactly what hides this bug, because `.cancelled()` is true either
        way and only the elapsed time tells the two implementations apart.
        """
        app = AppManager()

        async def uncooperative():
            await asyncio.Event().wait()  # never completes on its own

        app.register_task("uncooperative", uncooperative())
        _launch(app)
        await asyncio.sleep(0)  # let it reach the await

        assert app.GRACEFUL_SHUTDOWN_SECONDS >= 30.0
        _, elapsed = await _elapsed(app.shutdown())

        assert app._running_tasks[0].cancelled()
        assert elapsed < 1.0, (
            f"shutdown waited {elapsed:.2f}s on a task that was never signalled; "
            f"uncooperative tasks must be cancelled without draining"
        )

    @pytest.mark.asyncio
    async def test_uncooperative_task_does_not_delay_a_cooperative_one(self):
        """The production shape: one draining task alongside three that have no
        stop callback (discord, gmail, kobold_engine_api)."""
        app = AppManager()
        stop_event = asyncio.Event()

        async def cooperative():
            await stop_event.wait()

        async def interface():
            await asyncio.Event().wait()

        app.register_task("consolidator", cooperative(), stop=stop_event.set)
        for name in ("discord", "gmail", "kobold_engine_api"):
            app.register_task(name, interface())
        _launch(app)
        await asyncio.sleep(0)

        _, elapsed = await _elapsed(app.shutdown())

        assert elapsed < 1.0, f"shutdown took {elapsed:.2f}s; expected ~0s"
        assert app._running_tasks[0].done() and not app._running_tasks[0].cancelled()
        assert all(t.cancelled() for t in app._running_tasks[1:])

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
        _launch(app)
        await asyncio.sleep(0)

        app.GRACEFUL_SHUTDOWN_SECONDS = 0.05
        app._drain_budgets["stubborn"] = 0.05
        await app.shutdown()

        assert signalled is True
        assert app._running_tasks[0].cancelled()

    @pytest.mark.asyncio
    async def test_async_stop_callback_is_awaited(self):
        """`discord.py`'s Client.close() is `async def`. Calling it and dropping
        the coroutine never signals the task — and leaves a 'coroutine was never
        awaited' warning behind."""
        app = AppManager()
        stop_event = asyncio.Event()
        drained = False

        async def async_stop():
            stop_event.set()

        async def cooperative():
            nonlocal drained
            await stop_event.wait()
            drained = True

        app.register_task("async_stop", cooperative(), stop=async_stop)
        _launch(app)
        await asyncio.sleep(0)

        _, elapsed = await _elapsed(app.shutdown())

        assert drained is True
        assert not app._running_tasks[0].cancelled()
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_drain_timeout_overrides_the_default(self):
        """A task whose unit of work outlasts the default window registers its
        own budget; without it the drain expires mid-work-item."""
        app = AppManager()
        app.GRACEFUL_SHUTDOWN_SECONDS = 0.01
        stop_event = asyncio.Event()

        async def slow_last_item():
            await stop_event.wait()
            await asyncio.sleep(0.15)  # the in-flight work item

        app.register_task(
            "slow", slow_last_item(), stop=stop_event.set, drain_timeout=5.0
        )
        _launch(app)
        await asyncio.sleep(0)

        await app.shutdown()

        assert app._running_tasks[0].done()
        assert not app._running_tasks[0].cancelled()

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
        _launch(app)
        await asyncio.sleep(0)

        app.CANCEL_GRACE_SECONDS = 0.05

        # Must return, not hang.
        await asyncio.wait_for(app.shutdown(), timeout=2.0)

        # Abandoned, still pending — shutdown did not wait on it.
        assert not app._running_tasks[0].done()

        # Do not leak a task parked in sleep(30) into the loop teardown: cancel
        # it *and* yield, so the second cancellation is actually delivered.
        app._running_tasks[0].cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app._running_tasks[0]

    @pytest.mark.asyncio
    async def test_failing_stop_callback_does_not_block_shutdown(self):
        """A stop callback that raised never signalled its task, so there is
        nothing to drain — cancel at once rather than wait out the window."""
        app = AppManager()

        def boom():
            raise RuntimeError("stop callback exploded")

        async def cooperative():
            await asyncio.Event().wait()

        app.register_task("boom", cooperative(), stop=boom)
        _launch(app)
        await asyncio.sleep(0)

        _, elapsed = await _elapsed(app.shutdown())  # must not propagate

        assert app._running_tasks[0].cancelled()
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, mock_agent_manager):
        """A SIGTERM handler and start()'s finally both call shutdown(). A second
        pass would re-fire every stop callback and re-pay the grace windows on
        any task the first pass abandoned."""
        app = AppManager(agent_manager=mock_agent_manager)
        calls = []

        async def swallows_cancellation():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(30)

        app.register_task(
            "abandoned", swallows_cancellation(), stop=lambda: calls.append("stop")
        )
        _launch(app)
        await asyncio.sleep(0)

        app.GRACEFUL_SHUTDOWN_SECONDS = 0.05
        app._drain_budgets["abandoned"] = 0.05
        app.CANCEL_GRACE_SECONDS = 0.05
        await app.shutdown()
        assert calls == ["stop"]

        _, elapsed = await _elapsed(app.shutdown())

        assert calls == ["stop"], "second shutdown re-fired the stop callback"
        assert elapsed < 0.05
        mock_agent_manager.shutdown_all.assert_awaited_once()

        app._running_tasks[0].cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app._running_tasks[0]

    @pytest.mark.asyncio
    async def test_agents_are_signalled_before_tasks_drain(self, mock_agent_manager):
        """Agents and the consolidator share MemoryManager._lock. An agent left
        deploying for the whole drain window races the work item the drain
        exists to let finish."""
        order = []
        mock_agent_manager.signal_stop_all.side_effect = lambda: order.append("agents")
        app = AppManager(agent_manager=mock_agent_manager)

        async def cooperative():
            await asyncio.Event().wait()

        app.register_task("task", cooperative(), stop=lambda: order.append("task_stop"))
        _launch(app)
        await asyncio.sleep(0)

        app.GRACEFUL_SHUTDOWN_SECONDS = 0.05
        app._drain_budgets["task"] = 0.05
        await app.shutdown()

        assert order == ["agents", "task_stop"]

    @pytest.mark.asyncio
    async def test_crashed_task_exception_is_reported(self, caplog):
        """shutdown() holds the only reference to a task that died on its own.
        Unretrieved, its exception resurfaces as an unattributed
        'Task exception was never retrieved' at GC time, if at all."""
        app = AppManager()

        async def dies():
            raise RuntimeError("daemon exploded")

        app.register_task("dies", dies())
        _launch(app)
        await asyncio.sleep(0)

        with caplog.at_level(logging.ERROR, logger="src.app_manager"):
            await app.shutdown()

        assert "daemon exploded" in caplog.text
        assert "dies" in caplog.text


class TestAppManagerRegistrationValidation:
    def test_duplicate_task_name_rejected(self):
        """Names key the stop callbacks and drain budgets — a silent collision
        would drop one task's stop callback."""
        app = AppManager()

        async def dummy():
            pass

        coros = [dummy(), dummy()]
        app.register_task("dupe", coros[0])
        with pytest.raises(ValueError, match="already registered"):
            app.register_task("dupe", coros[1])
        for c in coros:
            c.close()

    def test_drain_timeout_without_stop_rejected(self):
        app = AppManager()

        async def dummy():
            pass

        coro = dummy()
        with pytest.raises(ValueError, match="no stop callback"):
            app.register_task("nostop", coro, drain_timeout=5.0)
        coro.close()


class TestAppManagerSignalledShutdown:
    """DP-304: the drain has to be reachable from the shutdown the process
    actually performs. Nothing used to call shutdown() except start()'s finally,
    which asyncio.run() only reaches *after* cancelling every task."""

    @pytest.mark.asyncio
    async def test_request_shutdown_returns_from_start_and_drains(self):
        app = AppManager()
        stop_event = asyncio.Event()
        drained = False

        async def cooperative():
            nonlocal drained
            await stop_event.wait()
            drained = True

        async def interface():
            await asyncio.Event().wait()

        app.register_task("cooperative", cooperative(), stop=stop_event.set)
        app.register_task("interface", interface())

        async def signaller():
            await asyncio.sleep(0.05)
            app.request_shutdown()

        signal_task = asyncio.create_task(signaller())
        # start() blocks on tasks that never finish; only the signal ends it.
        await asyncio.wait_for(app.start(), timeout=5.0)
        await signal_task

        assert drained is True, "cooperative task was cancelled instead of drained"
        assert app._running_tasks[1].cancelled()

    @pytest.mark.asyncio
    async def test_request_shutdown_is_idempotent(self):
        app = AppManager()
        app.request_shutdown()
        app.request_shutdown()
        assert app._shutdown_requested.is_set()

    def test_signal_handlers_route_to_request_shutdown(self):
        """Nothing used to install these, so the drain was unreachable."""
        from src.main import _install_shutdown_handlers

        app = AppManager()
        loop = MagicMock()
        _install_shutdown_handlers(app, loop=loop)

        installed = {call.args[0] for call in loop.add_signal_handler.call_args_list}
        assert {signal.SIGINT, signal.SIGTERM} <= installed
        for call in loop.add_signal_handler.call_args_list:
            assert call.args[1] == app.request_shutdown

    def test_signal_handlers_fall_back_when_loop_cannot_take_them(self):
        """Windows' ProactorEventLoop has no add_signal_handler."""
        from unittest.mock import patch
        from src.main import _install_shutdown_handlers

        app = AppManager()
        loop = MagicMock()
        loop.add_signal_handler.side_effect = NotImplementedError

        with patch("src.main.signal.signal") as sig_signal:
            _install_shutdown_handlers(app, loop=loop)

        assert {call.args[0] for call in sig_signal.call_args_list} == {
            signal.SIGINT, signal.SIGTERM
        }
        # The installed handler must hop back onto the loop, not touch the
        # Event from whatever thread the signal landed on.
        sig_signal.call_args_list[0].args[1](signal.SIGINT, None)
        loop.call_soon_threadsafe.assert_called_once_with(app.request_shutdown)

    @pytest.mark.asyncio
    async def test_crashing_one_shot_does_not_kill_the_interfaces(self):
        """A transient failure in a one-shot (model_update, fixr_orphan_notify)
        used to propagate out of gather() and tear down Discord with it."""
        app = AppManager()
        still_up = asyncio.Event()

        async def one_shot():
            raise RuntimeError("transient provider error")

        async def interface():
            still_up.set()
            await asyncio.Event().wait()

        app.register_task("one_shot", one_shot())
        app.register_task("interface", interface())

        async def signaller():
            await still_up.wait()
            await asyncio.sleep(0.05)
            app.request_shutdown()

        signal_task = asyncio.create_task(signaller())
        await asyncio.wait_for(app.start(), timeout=5.0)
        await signal_task

        # The interface ran to the signal; it was not torn down by the crash.
        assert still_up.is_set()
        assert app._running_tasks[1].cancelled()
