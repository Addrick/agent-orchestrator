"""ServiceIntegration for the HuggingFace model-provisioning tools (DP-265).

Registration-only, like the other ServiceIntegrations. Personas opt in via
``service_bindings: ["huggingface"]``.

The service **always** registers — even when ``HF_TOOLS_ENABLED`` is false — so
the startup-wiring contract (every ``huggingface`` tool has a handler) holds; the
handler short-circuits disabled calls with a clear error instead of reaching the
Hub or the node.

It shares the proxmox ``SSHRunner`` config (``PVE_SSH_*``) rather than owning a
second key or a second host: ``install_model`` drives the same pve node the
proxmox tools drive, through the same forced-command entry. A second transport
here would be a second thing to lock down.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.clients.service_integration import ServiceIntegration
from src.huggingface.client import HFClient
from src.huggingface.handler import HuggingFaceToolHandler
from src.proxmox.ssh import SSHRunner

if TYPE_CHECKING:
    from src.tools.tool_manager import ToolManager


class HuggingFaceIntegration(ServiceIntegration):
    def __init__(
        self,
        client: Optional[HFClient] = None,
        runner: Optional[SSHRunner] = None,
    ) -> None:
        self._handler = HuggingFaceToolHandler(client, runner)

    @property
    def name(self) -> str:
        return "huggingface"

    def register_tools(self, tool_manager: "ToolManager") -> None:
        self._handler.register(tool_manager)
