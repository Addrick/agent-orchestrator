"""ServiceIntegration for the stagr setup-VM tools (DP-456).

Registration-only, like the other ServiceIntegrations. Personas opt in via
``service_bindings: ["stagr"]``. Always registers, so the startup-wiring
contract holds; with ``STAGR_SERVER_URL`` unset the shell and transfer tools
answer with an error, and the web-access tools need ``PVE_TOOLS_ENABLED`` (they
ride the proxmox node gate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.clients.service_integration import ServiceIntegration
from src.proxmox.ssh import SSHRunner
from src.stagr.handler import StagrToolHandler

if TYPE_CHECKING:
    from src.tools.tool_manager import ToolManager


class StagrIntegration(ServiceIntegration):
    def __init__(self, runner: SSHRunner | None = None) -> None:
        self._handler = StagrToolHandler(runner)

    @property
    def name(self) -> str:
        return "stagr"

    def register_tools(self, tool_manager: "ToolManager") -> None:
        self._handler.register(tool_manager)
