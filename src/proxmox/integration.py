"""ServiceIntegration for the Proxmox management tools (DP-262).

Registration-only, like the other ServiceIntegrations. Owns a single
``SSHRunner`` (config-driven) that every tool call shares. Personas opt in via
``service_bindings: ["proxmox"]``.

The service always registers — even when ``PVE_TOOLS_ENABLED`` is false — so the
startup-wiring contract (every ``proxmox`` tool has a handler) holds; the handler
short-circuits disabled calls with a clear error instead of attempting SSH.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.clients.service_integration import ServiceIntegration
from src.proxmox.handler import ProxmoxToolHandler
from src.proxmox.ssh import SSHRunner

if TYPE_CHECKING:
    from src.tools.tool_manager import ToolManager


class ProxmoxIntegration(ServiceIntegration):
    def __init__(self, runner: SSHRunner | None = None) -> None:
        self._handler = ProxmoxToolHandler(runner)

    @property
    def name(self) -> str:
        return "proxmox"

    def register_tools(self, tool_manager: "ToolManager") -> None:
        self._handler.register(tool_manager)
