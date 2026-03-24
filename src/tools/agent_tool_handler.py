# src/tools/agent_tool_handler.py

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.agent_manager import AgentManager
    from src.database.memory_manager import MemoryManager
    from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)


class AgentToolHandler:
    """
    Registers agent management tools with a ToolManager.

    Provides three tools:
    - get_agent_status: read-only status query
    - get_agent_history: read-only action history query
    - manage_agent: start/stop/restart agents (write operation)
    """

    def __init__(
        self,
        agent_manager: "AgentManager",
        memory_manager: "MemoryManager",
    ) -> None:
        self._agent_manager = agent_manager
        self._memory_manager = memory_manager

    def register(self, manager: "ToolManager") -> None:
        """Register all agent tools with the tool manager."""
        manager.register("get_agent_status", self._get_agent_status)
        manager.register("get_agent_history", self._get_agent_history)
        manager.register("manage_agent", self._manage_agent)

    async def _get_agent_status(
        self, agent_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return status for one or all agents."""
        logger.info(f"Executing tool: get_agent_status for agent={agent_name or 'all'}")
        return self._agent_manager.get_status(agent_name)

    async def _get_agent_history(
        self,
        agent_name: str,
        limit: int = 10,
        ticket_id: Optional[str] = None,
        customer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return recent action history for an agent, with optional context filters."""
        logger.info(
            f"Executing tool: get_agent_history for agent={agent_name}, "
            f"limit={limit}, ticket={ticket_id}, customer={customer}"
        )

        # Build match_contexts from optional filters
        match_contexts: Optional[List[tuple[str, str]]] = None
        if ticket_id or customer:
            match_contexts = []
            if ticket_id:
                match_contexts.append(("ticket", ticket_id))
            if customer:
                match_contexts.append(("customer", customer))

        actions = self._memory_manager.get_relevant_agent_actions(
            agent_name=agent_name,
            match_contexts=match_contexts,
            limit=limit,
        )

        # Format actions for tool response
        formatted: List[Dict[str, Any]] = []
        for action in actions:
            entry: Dict[str, Any] = {
                "id": action.get("id"),
                "action_type": action.get("action_type"),
                "trigger_context": action.get("trigger_context"),
                "outcome": action.get("outcome"),
                "timestamp": str(action.get("timestamp", "")),
            }

            # Include outcome payload summary
            payload = action.get("outcome_payload")
            if payload:
                try:
                    entry["outcome_details"] = json.loads(payload)
                except (json.JSONDecodeError, TypeError):
                    entry["outcome_details"] = payload

            # Include child steps for failed actions
            if action.get("outcome") in ("failed", "error", "notification_failed"):
                steps = self._memory_manager.get_action_steps(action.get("id", 0))
                if steps:
                    entry["steps"] = [
                        {
                            "action_type": s.get("action_type"),
                            "outcome": s.get("outcome"),
                        }
                        for s in steps
                    ]

            formatted.append(entry)

        return {
            "agent_name": agent_name,
            "action_count": len(formatted),
            "actions": formatted,
        }

    async def _manage_agent(
        self, agent_name: str, action: str,
    ) -> Dict[str, str]:
        """Start, stop, or restart an agent."""
        logger.info(f"Executing tool: manage_agent action={action} for agent={agent_name}")

        if action == "start":
            message = await self._agent_manager.start_agent(agent_name)
        elif action == "stop":
            message = await self._agent_manager.stop_agent(agent_name)
        elif action == "restart":
            message = await self._agent_manager.restart_agent(agent_name)
        else:
            raise ValueError(f"Unknown action '{action}'. Must be start, stop, or restart.")

        return {"status": "success", "message": message}
