# src/app_manager.py

import asyncio
import logging
from typing import Any, Coroutine, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.agent_manager import AgentManager

logger = logging.getLogger(__name__)


class AppManager:
    """
    Central lifecycle manager for long-running interface tasks (Discord, Gmail).

    Agent lifecycle is managed by AgentManager. AppManager coordinates
    startup and shutdown between interfaces and the agent subsystem.
    """

    def __init__(self, agent_manager: Optional["AgentManager"] = None) -> None:
        self._agent_manager = agent_manager
        self._pending_tasks: List[Tuple[str, Coroutine[Any, Any, Any]]] = []
        self._running_tasks: List[asyncio.Task[Any]] = []

    def register_task(self, name: str, coro: Coroutine[Any, Any, Any]) -> None:
        """Register a long-running async task (e.g. Discord bot, Gmail bot)."""
        self._pending_tasks.append((name, coro))
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
        """Shut down all agents and cancel interface tasks."""
        logger.info("Shutting down AppManager...")

        if self._agent_manager:
            await self._agent_manager.shutdown_all()
