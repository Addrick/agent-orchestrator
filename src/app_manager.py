# src/app_manager.py

import asyncio
import logging
from typing import Any, Callable, Coroutine, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.agent_manager import AgentManager

logger = logging.getLogger(__name__)


class AppManager:
    """
    Central lifecycle manager for long-running interface tasks (Discord, Gmail).

    Agent lifecycle is managed by AgentManager. AppManager coordinates
    startup and shutdown between interfaces and the agent subsystem.
    """

    #: How long cooperative tasks get to drain after being signalled, before
    #: they are cancelled outright (DP-304).
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
        self._stop_callbacks: List[Tuple[str, Callable[[], None]]] = []

    def register_task(
        self,
        name: str,
        coro: Coroutine[Any, Any, Any],
        stop: Optional[Callable[[], None]] = None,
    ) -> None:
        """Register a long-running async task (e.g. Discord bot, Gmail bot).

        `stop` is an optional synchronous callable that signals the task to exit
        at its next checkpoint. Tasks that supply one get to drain on shutdown;
        tasks that don't are cancelled outright (DP-304).
        """
        self._pending_tasks.append((name, coro))
        if stop is not None:
            self._stop_callbacks.append((name, stop))
        logger.info(f"Registered task '{name}'")

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

        try:
            if self._running_tasks:
                await asyncio.gather(*self._running_tasks)
            else:
                # Only agents, no long-running tasks — keep alive
                stop_event = asyncio.Event()
                await stop_event.wait()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Shut down all agents and stop interface tasks.

        Two phases (DP-304): signal every cooperative task and give them a
        bounded window to drain, then cancel whatever is still running. Before
        DP-304 this method ignored `_running_tasks` entirely, so a registered
        daemon was never stopped at all — it outlived shutdown until the event
        loop itself went away.
        """
        logger.info("Shutting down AppManager...")

        # Phase 1 — signal cooperative tasks so they exit at their next checkpoint.
        for task_name, stop in self._stop_callbacks:
            try:
                stop()
            except Exception as e:
                logger.error(f"Stop callback for task '{task_name}' failed: {e}", exc_info=True)

        # Phase 2 — bounded drain, then cancel the stragglers.
        pending = [t for t in self._running_tasks if not t.done()]
        if pending:
            _, still_running = await asyncio.wait(
                pending, timeout=self.GRACEFUL_SHUTDOWN_SECONDS
            )
            for task in still_running:
                logger.warning(
                    f"Task '{task.get_name()}' did not stop within "
                    f"{self.GRACEFUL_SHUTDOWN_SECONDS}s; cancelling."
                )
                task.cancel()
            if still_running:
                # Bounded: a cancelled task still runs its own teardown, and that
                # teardown can block (see CANCEL_GRACE_SECONDS). Shutdown must
                # finish either way, so abandon whatever refuses to unwind.
                _, abandoned = await asyncio.wait(
                    still_running, timeout=self.CANCEL_GRACE_SECONDS
                )
                for task in abandoned:
                    logger.error(
                        f"Task '{task.get_name()}' did not unwind within "
                        f"{self.CANCEL_GRACE_SECONDS}s of cancellation; abandoning it."
                    )

        if self._agent_manager:
            await self._agent_manager.shutdown_all()
