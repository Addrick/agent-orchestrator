# src/app_manager.py

import asyncio
import logging
from typing import Any, Coroutine, Dict, List, Tuple

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.agents.base import Agent

logger = logging.getLogger(__name__)


class AppManager:
    """
    Central lifecycle manager for agents and long-running interface tasks.

    Agents are scheduled via APScheduler (interval-based polling).
    Interfaces (Discord, Gmail) run as plain asyncio tasks.
    """

    def __init__(self) -> None:
        self._scheduler = AsyncIOScheduler()
        self._agents: Dict[str, Agent] = {}
        self._pending_tasks: List[Tuple[str, Coroutine[Any, Any, Any]]] = []
        self._running_tasks: List[asyncio.Task[Any]] = []

    def register_agent(self, name: str, agent: Agent, interval: float) -> None:
        """Register a polling agent to be scheduled at the given interval (seconds)."""
        self._agents[name] = agent
        self._scheduler.add_job(
            self._safe_poll,
            'interval',
            seconds=interval,
            id=name,
            name=name,
            kwargs={'agent': agent, 'name': name},
            max_instances=1,
        )
        logger.info(f"Registered agent '{name}' (interval={interval}s)")

    def register_task(self, name: str, coro: Coroutine[Any, Any, Any]) -> None:
        """Register a long-running async task (e.g. Discord bot, Gmail bot)."""
        self._pending_tasks.append((name, coro))
        logger.info(f"Registered task '{name}'")

    async def start(self) -> None:
        """Start all agents and tasks. Blocks until all tasks complete or shutdown."""
        # Run on_start for all agents
        for name, agent in self._agents.items():
            try:
                await agent.on_start()
                logger.info(f"Agent '{name}' started.")
            except Exception:
                logger.error(f"Error starting agent '{name}':", exc_info=True)

        # Start the scheduler (agents begin polling)
        if self._agents:
            self._scheduler.start()

        # Launch long-running tasks
        for task_name, coro in self._pending_tasks:
            self._running_tasks.append(
                asyncio.create_task(coro, name=task_name)
            )

        if not self._running_tasks and not self._agents:
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
        """Signal all agents to stop and shut down the scheduler."""
        logger.info("Shutting down AppManager...")
        for name, agent in self._agents.items():
            agent.request_stop()
            logger.info(f"Requested stop for agent '{name}'")

        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    @staticmethod
    async def _safe_poll(agent: Agent, name: str) -> None:
        """Wrapper that catches exceptions so one bad poll doesn't crash the scheduler."""
        try:
            await agent.poll()
        except Exception:
            logger.error(f"Error in '{name}' poll:", exc_info=True)
