# src/clients/zammad_service.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.clients.service_integration import ServiceIntegration
from src.clients.zammad_client import ZammadClient

if TYPE_CHECKING:
    from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)


class ZammadIntegration(ServiceIntegration):
    """
    Service integration for Zammad ticketing.

    Registers Zammad CRUD tool handlers with the ToolManager.
    Personas with service_bindings: ["zammad"] gain access to these tools.
    """

    def __init__(self, zammad_client: ZammadClient) -> None:
        self._client = zammad_client

    @property
    def name(self) -> str:
        return "zammad"

    def register_tools(self, tool_manager: "ToolManager") -> None:
        """Register all Zammad CRUD tools with the ToolManager."""
        from src.tools.tool_manager import ZammadToolHandler
        ZammadToolHandler(self._client).register(tool_manager)
