# src/proposals/agent_call.py
"""Execute an approved subagent tool call (DP-240).

The MCP bridge never executes a gated tool itself — it queues a
``call_derpr_tool`` proposal row and returns. This module is the other half:
the component that turns an *approved* row into a real ``ToolManager`` call.

The load-bearing property is the **execute-time policy re-check**. A row can sit
in the queue indefinitely; between queueing and approval the operator may have
narrowed ``MCP_BRIDGE_TOOLS``, a persona's policy may have changed, or the tool
may have been unregistered entirely. So the stored ``tool_name`` is treated as a
*request*, never as evidence of permission: the runner re-derives the exposed
set from the live policy on every execution and refuses anything not currently
in it.

Both lookups are closures rather than bound references, matching
ConfirmationManager's reasoning: the engine rebinds ``tool_manager`` after
construction, and a copy captured at __init__ would go stale — here that would
mean executing against a manager the operator thought they had replaced.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

from src.tool_policy import ToolPolicy
from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)


class AgentCallRunner:
    """Runs approved ``call_derpr_tool`` proposals against the live ToolManager."""

    def __init__(
        self,
        tool_manager_lookup: Callable[[], ToolManager],
        policy_lookup: Callable[[], ToolPolicy],
    ) -> None:
        self._tool_manager_lookup = tool_manager_lookup
        self._policy_lookup = policy_lookup

    def exposed_tool_definitions(self) -> List[Dict[str, Any]]:
        """The tool definitions currently exposed over the bridge.

        Intersection of two things, both live: what the ToolManager actually has
        a registered handler for, and what the bridge ToolPolicy allows. Used by
        the MCP server for ``tools/list`` and by ``run`` for the re-check, so
        listing and execution cannot disagree about what is exposed.
        """
        registered = self._tool_manager_lookup().get_tool_definitions()
        return self._policy_lookup().filter_tools(registered)

    def exposed_tool_names(self) -> List[str]:
        return [
            name for name in (
                t.get("function", {}).get("name")
                for t in self.exposed_tool_definitions()
            ) if name
        ]

    def is_exposed(self, tool_name: str) -> bool:
        return tool_name in self.exposed_tool_names()

    async def execute_ungated(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a non-gated tool immediately, returning the raw ToolManager
        envelope ({'result': …} / {'error': …}).

        Callers must have established that the tool is ungated; the exposure
        check still applies and is re-run here rather than trusted from the
        caller.
        """
        if not self.is_exposed(tool_name):
            return {"error": f"tool '{tool_name}' is not exposed by the current bridge policy"}
        return await self._tool_manager_lookup().execute_tool(tool_name, **tool_args)

    async def run(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute one approved call. Returns (success, result message).

        Refuses — loudly and without executing — any tool the live policy no
        longer exposes. This is the check that makes a stale approved row safe.
        """
        if not self.is_exposed(tool_name):
            logger.warning(
                "Refusing approved agent call to '%s': not exposed by the current "
                "bridge policy (policy narrowed or tool unregistered since queueing).",
                tool_name,
            )
            return False, (
                f"tool '{tool_name}' is not exposed by the current bridge policy; "
                "refused at execution time"
            )

        outcome = await self._tool_manager_lookup().execute_tool(tool_name, **tool_args)
        # execute_tool never raises — it returns {'result': ...} or {'error': ...}.
        if "error" in outcome:
            return False, str(outcome["error"])
        return True, str(outcome.get("result"))
