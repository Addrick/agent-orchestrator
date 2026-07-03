"""ServiceIntegration for the MCP client management tools (DP-268).

Registration-only, like every ServiceIntegration: the session lifecycle lives
in ``MCPClientManager`` (owned by ``main.py``, voice precedent). This wrapper
registers the three management tools behind the ``mcp`` binding and hands the
``ToolManager`` to the manager so live discovery can register handlers later.

Always registers — even when ``MCP_ENABLED`` is false — so the startup-wiring
contract (every ``mcp`` tool has a handler) holds; the manager short-circuits
disabled calls with a clear error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from src.clients.service_integration import ServiceIntegration
from src.tools.mcp_client import MCPClientManager

if TYPE_CHECKING:
    from src.tools.tool_manager import ToolManager


class MCPIntegration(ServiceIntegration):
    def __init__(self, manager: MCPClientManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "mcp"

    def register_tools(self, tool_manager: "ToolManager") -> None:
        self._manager.attach_tool_manager(tool_manager)
        tool_manager.register("add_mcp_server", self._add_mcp_server)
        tool_manager.register("remove_mcp_server", self._remove_mcp_server)
        tool_manager.register("list_mcp_servers", self._list_mcp_servers)

    async def _add_mcp_server(self, name: str, url: str) -> Dict[str, Any]:
        return await self._manager.add_server(name, url)

    async def _remove_mcp_server(self, name: str) -> Dict[str, Any]:
        return await self._manager.remove_server(name)

    async def _list_mcp_servers(self) -> List[Dict[str, Any]]:
        return await self._manager.list_servers()
