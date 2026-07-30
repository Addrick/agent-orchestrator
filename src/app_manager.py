# src/app_manager.py

import asyncio
import contextlib
import inspect
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.agent_manager import AgentManager

logger = logging.getLogger(__name__)


class AppManager:
    """
    Central lifecycle manager for long-running interface tasks (Discord, Gmail).

    Agent lifecycle is managed by AgentManager. AppManager coordinates
    startup and shutdown between interfaces and the agent subsystem.
    """

    #: Default window a cooperative task gets to drain after being signalled,
    #: before it is cancelled outright. Overridable per task via
    #: `register_task(..., drain_timeout=)` — a task whose unit of work is an
    #: LLM call needs a budget longer than that call (DP-304).
    GRACEFUL_SHUTDOWN_SECONDS: float = 30.0

    #: How long a cancelled task gets to actually unwind before it is abandoned.
    #: Cancellation is a request, not a guarantee: `discord.py`'s `Client.close()`
    #: does real network teardown on its way out, so awaiting a cancelled task
    #: unbounded can hang shutdown forever. Bounded, then abandoned (DP-304).
    CANCEL_GRACE_SECONDS: float = 5.0

    def __init__(self, agent_manager: Optional["AgentManager"] = None) -> None:
        self._agent_manager = agent_manager
        self._pending_tasks: List[Tuple[str, Coroutine[Any, Any, Any]]] = []
        self._running_tasks: List[asyncio.Task[Any]] = []
        self._stop_callbacks: Dict[str, Callable[[], Any]] = {}
        self._drain_budgets: Dict[str, float] = {}
        self._shutdown_requested = asyncio.Event()
        self._shutdown_done = False

    def register_task(
        self,
        name: str,
        coro: Coroutine[Any, Any, Any],
        stop: Optional[Callable[[], Any]] = None,
        drain_timeout: Optional[float] = None,
    ) -> None:
        """Register a long-running async task (e.g. Discord bot, Gmail bot).

        `stop` is an optional callable — sync or async — that signals the task
        to exit at its next checkpoint. Tasks that supply one get to drain on
        shutdown; tasks that don't are cancelled outright, immediately, rather
        than stalling shutdown for a signal they were never sent (DP-304).

        `drain_timeout` overrides GRACEFUL_SHUTDOWN_SECONDS for this task. Set
        it when one unit of the task's work can outlast the default window;
        otherwise the drain expires mid-work-item and the task is cancelled
        anyway, which is the outcome the drain exists to prevent.
        """
        if name in self._stop_callbacks or any(n == name for n, _ in self._pending_tasks):
            raise ValueError(
                f"Task '{name}' is already registered. Task names key the stop "
                f"callbacks and drain budgets, so they must be unique."
            )
        self._pending_tasks.append((name, coro))
        if stop is not None:
            self._stop_callbacks[name] = stop
            self._drain_budgets[name] = (
                drain_timeout if drain_timeout is not None else self.GRACEFUL_SHUTDOWN_SECONDS
            )
        elif drain_timeout is not None:
            raise ValueError(
                f"Task '{name}' was given a drain_timeout but no stop callback. "
                f"Without a stop callback there is nothing to drain."
            )
        logger.info(f"Registered task '{name}'")

    def request_shutdown(self) -> None:
        """Ask start() to stop and run the graceful shutdown path.

        Signal-safe and idempotent: this is what SIGINT/SIGTERM handlers call.
        Without it the only shutdown path is asyncio.run()'s own teardown,
        which cancels every task *before* start()'s finally block runs — so the
        cooperative drain never happens (DP-304).
        """
        if self._shutdown_requested.is_set():
            logger.warning("Shutdown already requested; still draining.")
            return
        logger.info("Shutdown requested.")
        self._shutdown_requested.set()

    async def start(self) -> None:
        """Start all interfaces and agents. Blocks until all tasks complete or shutdown."""
        # Auto-start agents via AgentManager
        if self._agent_manager:
            await self._agent_manager.auto_start()

        # Launch long-running tasks
        for task_name, coro in self._pending_tasks:
            self._running_tasks.append(
                asyncio.create_task(coro, name=task_name)
            )

        has_agents = self._agent_manager and self._agent_manager.get_running()
        if not self._running_tasks and not has_agents:
            logger.warning("No interfaces or agents registered. Exiting.")
            return

        if self._running_tasks:
            # return_exceptions: a crashing task is reported, not propagated. A
            # transient failure in a one-shot (model_update, fixr_orphan_notify)
            # used to tear down Discord and the web portal with it (DP-304).
            runner: asyncio.Future[Any] = asyncio.gather(
                *self._running_tasks, return_exceptions=True
            )
        else:
            # Only agents, no long-running tasks — keep alive until signalled.
            runner = asyncio.ensure_future(asyncio.Event().wait())

        signal_waiter = asyncio.ensure_future(self._shutdown_requested.wait())
        try:
            # Task exceptions are surfaced by shutdown()'s
            # _report_finished_task_errors(), which sees every finished task
            # rather than only the ones this gather covers.
            await asyncio.wait(
                {runner, signal_waiter}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            signal_waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await signal_waiter
            # shutdown() first: it drains and cancels the children, which is
            # what lets the gather below complete instead of hanging.
            await self.shutdown()
            if not runner.done():
                runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await runner

    async def shutdown(self) -> None:
        """Shut down all agents and stop interface tasks.

        Three phases (DP-304):

        1. Signal everything that knows how to stop itself — agents first, then
           cooperative tasks — without waiting on any of it. Agents and the
           memory consolidator share `MemoryManager._lock`, so an agent left
           running during the drain contends with the very task trying to
           finish its last work item cleanly.
        2. Cancel every task that has no stop callback, immediately. Waiting out
           a grace window for a task that was never signalled only turns a ~0s
           shutdown into a 30s one.
        3. Drain the cooperative tasks against their own budgets, then cancel
           and bounded-await the stragglers.

        Before DP-304 this method ignored `_running_tasks` entirely, so a
        registered daemon was never stopped at all — it outlived shutdown until
        the event loop itself went away.
        """
        if self._shutdown_done:
            logger.debug("AppManager.shutdown() already ran; ignoring repeat call.")
            return
        self._shutdown_done = True
        self._shutdown_requested.set()

        logger.info("Shutting down AppManager...")

        # Phase 1 — signal, don't wait.
        if self._agent_manager:
            self._agent_manager.signal_stop_all()
        cooperative, cancelled = await self._signal_tasks()

        # Phase 2 — cancel what cannot stop itself, now rather than in 30s.
        for task in cancelled:
            logger.info(f"Task '{task.get_name()}' has no usable stop callback; cancelling.")
            task.cancel()

        # Phase 3 — drain the cooperative tasks concurrently, each against its
        # own budget, then fold the ones that overran into the cancelled set.
        if cooperative:
            overran = await asyncio.gather(*(self._drain(t) for t in cooperative))
            cancelled.extend(t for t in overran if t is not None)

        if cancelled:
            # Bounded: a cancelled task still runs its own teardown, and that
            # teardown can block (see CANCEL_GRACE_SECONDS). Shutdown must
            # finish either way, so abandon whatever refuses to unwind.
            _, abandoned = await asyncio.wait(
                cancelled, timeout=self.CANCEL_GRACE_SECONDS
            )
            for task in abandoned:
                logger.error(
                    f"Task '{task.get_name()}' did not unwind within "
                    f"{self.CANCEL_GRACE_SECONDS}s of cancellation; abandoning it."
                )

        self._report_finished_task_errors()

        if self._agent_manager:
            await self._agent_manager.shutdown_all()

    async def _signal_tasks(
        self,
    ) -> Tuple[List["asyncio.Task[Any]"], List["asyncio.Task[Any]"]]:
        """Fire every still-running task's stop callback.

        Returns (drainable, undrainable). A task is only drainable if it was
        actually signalled — no callback, or a callback that raised, means the
        signal never landed and there is nothing to wait for.
        """
        drainable: List[asyncio.Task[Any]] = []
        undrainable: List[asyncio.Task[Any]] = []
        for task in [t for t in self._running_tasks if not t.done()]:
            stop = self._stop_callbacks.get(task.get_name())
            if stop is None:
                undrainable.append(task)
                continue
            try:
                result = stop()
                if inspect.isawaitable(result):
                    # An async stop (discord.py's Client.close is `async def`)
                    # returns a coroutine; dropping it never signals the task.
                    await result
            except Exception as e:
                logger.error(
                    f"Stop callback for task '{task.get_name()}' failed: {e}", exc_info=True
                )
                undrainable.append(task)
                continue
            drainable.append(task)
        return drainable, undrainable

    async def _drain(self, task: "asyncio.Task[Any]") -> Optional["asyncio.Task[Any]"]:
        """Wait out one cooperative task's drain budget. Returns the task if it
        overran (and has now been cancelled), else None."""
        budget = self._drain_budgets.get(task.get_name(), self.GRACEFUL_SHUTDOWN_SECONDS)
        _, not_done = await asyncio.wait([task], timeout=budget)
        if not not_done:
            return None
        logger.warning(
            f"Task '{task.get_name()}' did not stop within {budget}s; cancelling."
        )
        task.cancel()
        return task

    def _report_finished_task_errors(self) -> None:
        """Surface exceptions from tasks that died on their own before shutdown.

        shutdown() is the one place still holding those task objects; if nobody
        calls .exception() they resurface (if at all) as an unattributed
        'Task exception was never retrieved' at GC time.
        """
        for task in self._running_tasks:
            if not task.done() or task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                logger.error(
                    f"Task '{task.get_name()}' exited with an exception: {exc}", exc_info=exc
                )
