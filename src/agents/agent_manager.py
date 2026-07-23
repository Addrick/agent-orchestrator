# src/agents/agent_manager.py

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from src.agents.base import Agent

logger = logging.getLogger(__name__)

# Default path for agent configuration
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "agents.json"


@dataclass
class AgentRegistration:
    """Blueprint for an agent that can be instantiated."""
    agent_class: Type[Agent]
    default_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunningAgent:
    """Tracks a live agent instance and its asyncio task."""
    instance: Agent
    task: asyncio.Task[None]
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AgentManager:
    """
    Agent lifecycle manager — registry, instantiation, and status tracking.

    Manages the full lifecycle of Agent subclasses:
    - Registration: associate agent names with classes and default configs
    - Instantiation: create agents on demand with config overrides
    - Status: query running state, deploy counts, error rates
    - Shutdown: graceful stop of all running agents

    Designed to be held by ChatSystem as a thin reference and
    queried by agent tools (get_agent_status, manage_agent, etc.).
    """

    def __init__(
        self,
        chat_system: Any,
        memory_manager: Any,
        notification_router: Optional[Any] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        self._chat_system = chat_system
        self._memory_manager = memory_manager
        self._notification_router = notification_router

        self._registry: Dict[str, AgentRegistration] = {}
        self._running: Dict[str, RunningAgent] = {}
        # Single-shot inference agents (DP-292): DI-built, cached, never looped.
        self._inference_registry: Dict[str, AgentRegistration] = {}
        self._inference_instances: Dict[str, Any] = {}

        # Load file-based config (recipients, agent defaults)
        self._config: Dict[str, Any] = {}
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._load_config()

    def _load_config(self) -> None:
        """Load agent configuration from agents.json."""
        if self._config_path.exists():
            try:
                with open(self._config_path) as f:
                    self._config = json.load(f)
                logger.info(f"Loaded agent config from {self._config_path}")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load agent config: {e}")
        else:
            logger.info(f"No agent config file at {self._config_path}")

    @property
    def config(self) -> Dict[str, Any]:
        """Expose loaded config for tool handlers and agents."""
        return self._config

    @property
    def notification_router(self) -> Optional[Any]:
        """Expose notification router for agent construction."""
        return self._notification_router

    @notification_router.setter
    def notification_router(self, value: Any) -> None:
        """Allow lazy-setting the notification router after construction."""
        self._notification_router = value

    def register(
        self,
        name: str,
        agent_class: Type[Agent],
        default_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register an agent class so it can be started by name.

        Args:
            name: Unique agent identifier (e.g. "dispatch").
            agent_class: The Agent subclass.
            default_config: Default kwargs merged with file config and start-time overrides.
        """
        self._registry[name] = AgentRegistration(
            agent_class=agent_class,
            default_config=default_config or {},
        )
        logger.info(f"Registered agent class: {name} -> {agent_class.__name__}")

    def get_registered(self) -> List[str]:
        """List all registered agent names."""
        return list(self._registry.keys())

    def get_running(self) -> List[str]:
        """List names of currently running agents."""
        return [name for name, ra in self._running.items() if not ra.task.done()]

    async def start_agent(
        self,
        name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Instantiate and start a registered agent.

        Args:
            name: The registered agent name.
            config_overrides: Runtime overrides merged on top of defaults.

        Returns:
            Status message string.

        Raises:
            ValueError: If agent is not registered or already running.
        """
        if name not in self._registry:
            raise ValueError(f"Agent '{name}' is not registered. Available: {self.get_registered()}")

        # Check if already running
        if name in self._running and not self._running[name].task.done():
            raise ValueError(f"Agent '{name}' is already running.")

        registration = self._registry[name]

        # Merge config: file defaults < registration defaults < runtime overrides
        file_config = self._config.get("agents", {}).get(name, {})
        merged = {**file_config, **registration.default_config, **(config_overrides or {})}

        # Build constructor kwargs — the agent class determines what it needs
        instance = self._build_agent_instance(name, registration.agent_class, merged)

        # Apply runtime config to instance attributes
        if "schedule" in merged:
            instance.schedule = merged["schedule"]
        if "action_history_limit" in merged:
            instance.action_history_limit = int(merged["action_history_limit"])

        # Launch as asyncio task
        task: asyncio.Task[None] = asyncio.create_task(
            instance.start(),
            name=f"agent:{name}",
        )

        self._running[name] = RunningAgent(instance=instance, task=task)
        logger.info(f"Agent '{name}' started.")
        return f"Agent '{name}' started successfully."

    def _di_kwargs(
        self, name: str, agent_class: Type[Any], config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build constructor kwargs via convention-based dependency injection.

        If the class's __init__ accepts 'zammad_client', 'notification_router',
        or 'agent_config', supply them from the manager's held references.
        Shared by scheduled-loop agents (`_build_agent_instance`) and
        single-shot inference agents (`get_inference_agent`).
        """
        import inspect
        sig = inspect.signature(agent_class.__init__)
        params = list(sig.parameters.keys())

        kwargs: Dict[str, Any] = {"chat_system": self._chat_system}

        if "zammad_client" in params:
            from src.clients.zammad_service import ZammadIntegration
            zammad_service = self._chat_system.get_service("zammad")
            if not isinstance(zammad_service, ZammadIntegration):
                raise ValueError(
                    f"Agent '{name}' requires zammad_client but no Zammad service is registered."
                )
            kwargs["zammad_client"] = zammad_service.client

        if "notification_router" in params:
            if self._notification_router is None:
                raise ValueError(
                    f"Agent '{name}' requires notification_router but none is configured."
                )
            kwargs["notification_router"] = self._notification_router

        if "agent_config" in params:
            # Merge agent-specific config with global recipients
            agent_cfg = self._config.get("agents", {}).get(name, {})
            agent_cfg["_recipients"] = self._config.get("recipients", {})
            kwargs["agent_config"] = agent_cfg

        return kwargs

    def _build_agent_instance(
        self, name: str, agent_class: Type[Agent], config: Dict[str, Any],
    ) -> Agent:
        """Construct a scheduled-loop agent instance with injected dependencies."""
        return agent_class(**self._di_kwargs(name, agent_class, config))

    # ---------- single-shot inference agents (DP-292) ----------
    # A second agent shape: synchronous, caller-invoked, returns a verdict
    # (e.g. content classification, date extraction). These are still agents —
    # persona + LLM call + tool schema — but they do NOT fit the scheduled-loop
    # `Agent`/`start_agent` path (nothing to schedule). They register here so
    # they get the SAME convention-DI (notification_router, zammad_client, …) as
    # scheduled agents and a single lookup point, instead of being constructed
    # ad-hoc from `chat_system` by their callers. Built lazily (deps may be set
    # after registration) and cached.

    def register_inference_agent(
        self,
        name: str,
        agent_class: Type[Any],
        default_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a single-shot inference-agent class for DI-built lookup."""
        self._inference_registry[name] = AgentRegistration(
            agent_class=agent_class,
            default_config=default_config or {},
        )
        logger.info(f"Registered inference agent: {name} -> {agent_class.__name__}")

    def get_inference_agent(self, name: str) -> Optional[Any]:
        """Return the DI-built singleton for a registered inference agent, or
        None if `name` was never registered. Built on first access (so deps set
        after registration are still injected) and cached thereafter."""
        if name not in self._inference_registry:
            return None
        inst = self._inference_instances.get(name)
        if inst is None:
            reg = self._inference_registry[name]
            file_config = self._config.get("agents", {}).get(name, {})
            merged = {**file_config, **reg.default_config}
            inst = reg.agent_class(**self._di_kwargs(name, reg.agent_class, merged))
            self._inference_instances[name] = inst
        return inst

    async def stop_agent(self, name: str) -> str:
        """Stop a running agent gracefully.

        Returns:
            Status message string.

        Raises:
            ValueError: If agent is not running.
        """
        if name not in self._running or self._running[name].task.done():
            raise ValueError(f"Agent '{name}' is not running.")

        running = self._running[name]
        running.instance.stop()

        try:
            await asyncio.wait_for(running.task, timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(f"Agent '{name}' did not stop within 30s, cancelling task.")
            running.task.cancel()

        logger.info(f"Agent '{name}' stopped.")
        return f"Agent '{name}' stopped successfully."

    async def restart_agent(
        self,
        name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Stop and re-start an agent.

        Returns:
            Status message string.
        """
        if name in self._running and not self._running[name].task.done():
            await self.stop_agent(name)
        return await self.start_agent(name, config_overrides)

    def get_status(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get status for one or all agents.

        Args:
            name: If provided, return status for just this agent.
                  If None, return status for all registered agents.

        Returns:
            Dict with agent status information.
        """
        if name:
            return {"agents": {name: self._single_status(name)}}
        return {"agents": {n: self._single_status(n) for n in self._registry}}

    def _single_status(self, name: str) -> Dict[str, Any]:
        """Build status dict for a single agent."""
        status: Dict[str, Any] = {"registered": name in self._registry}

        if name not in self._registry:
            return status

        reg = self._registry[name]
        status["class"] = reg.agent_class.__name__

        if name in self._running:
            ra = self._running[name]
            instance = ra.instance
            task_done = ra.task.done()

            status["running"] = not task_done
            status["started_at"] = (
                instance.started_at.isoformat() if instance.started_at else None
            )
            status["last_deploy_time"] = (
                instance.last_deploy_time.isoformat() if instance.last_deploy_time else None
            )
            status["deploy_count"] = instance.deploy_count
            status["schedule"] = instance.schedule
            status["error_count"] = instance.error_count
            status["consecutive_errors"] = instance.consecutive_errors
            status["last_error"] = instance.last_error

            if task_done and ra.task.exception():
                status["crash_error"] = str(ra.task.exception())
        else:
            status["running"] = False

        return status

    async def shutdown_all(self) -> None:
        """Stop all running agents. Called during application shutdown."""
        running_names = self.get_running()
        if not running_names:
            return

        logger.info(f"Shutting down {len(running_names)} agent(s): {running_names}")
        for name in running_names:
            try:
                await self.stop_agent(name)
            except Exception as e:
                logger.error(f"Error stopping agent '{name}': {e}")

    async def auto_start(self) -> None:
        """Start agents that have auto_start=true in config."""
        agents_config = self._config.get("agents", {})
        for name, agent_cfg in agents_config.items():
            if agent_cfg.get("auto_start", False) and name in self._registry:
                try:
                    await self.start_agent(name)
                    logger.info(f"Auto-started agent: {name}")
                except Exception as e:
                    logger.error(f"Failed to auto-start agent '{name}': {e}")
