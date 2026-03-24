# src/agents/agent_service.py

"""
ServiceIntegration for the agent management subsystem.

Exposes agent tools (get_agent_status, get_agent_history, manage_agent)
to personas that opt in via service_bindings: ["agents"].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.clients.service_integration import ServiceIntegration

if TYPE_CHECKING:
    from src.agents.agent_manager import AgentManager
    from src.database.memory_manager import MemoryManager
    from src.tools.tool_manager import ToolManager


class AgentServiceIntegration(ServiceIntegration):
    """
    Service integration that gates agent management tools behind
    the 'agents' service binding.

    Personas must declare service_bindings: ["agents"] to access
    get_agent_status, get_agent_history, and manage_agent tools.
    """

    def __init__(
        self,
        agent_manager: "AgentManager",
        memory_manager: "MemoryManager",
    ) -> None:
        self._agent_manager = agent_manager
        self._memory_manager = memory_manager

    @property
    def name(self) -> str:
        return "agents"

    def register_tools(self, tool_manager: "ToolManager") -> None:
        from src.tools.agent_tool_handler import AgentToolHandler
        handler = AgentToolHandler(self._agent_manager, self._memory_manager)
        handler.register(tool_manager)
