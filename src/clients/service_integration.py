# src/clients/service_integration.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.tools.tool_manager import ToolManager


class ServiceIntegration(ABC):
    """
    Pluggable service that contributes tools to the ChatSystem.

    Each integration is identified by a unique ``name`` (e.g. "zammad").
    Personas declare which integrations they use via ``service_bindings``.
    Tool definitions declare which service they belong to via ``service_binding``.

    Registration (called once by ChatSystem.register_service):
      - register_tools  — register service-specific tool handlers
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique service identifier used in persona service_bindings and tool definitions."""
        ...

    def register_tools(self, tool_manager: "ToolManager") -> None:
        """
        Register service-specific tool handlers with the ToolManager.

        Called once when the service is registered with ChatSystem.
        Override this to register async callables for each tool your service provides.
        """
        pass
