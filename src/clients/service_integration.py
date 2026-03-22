# src/clients/service_integration.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.tools.tool_manager import ToolManager


class ServiceIntegration(ABC):
    """
    Pluggable service that contributes context and lifecycle hooks
    to the ChatSystem request pipeline.

    Each integration is identified by a unique ``name`` (e.g. "zammad").
    Personas declare which integrations they use via ``service_bindings``.
    Tool definitions declare which service they belong to via ``service_binding``.

    Registration (called once by ChatSystem.register_service):
      - register_tools      — register service-specific tool handlers

    Lifecycle (called by ChatSystem during generate_response):
      1. resolve_context     — populate service-specific state for this request
      2. on_message          — called when user message received / LLM response generated
      3. prepare_tool_args   — modify tool arguments before execution
      4. on_tool_result      — process tool results after execution
      5. get_system_messages — inject system messages into conversation history
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

    def get_tracking_id(self, service_data: Dict[str, Any]) -> Optional[int]:
        """
        Return an external tracking/ticket ID for memory logging, or None.

        ChatSystem uses this to associate memory records with external entities
        (e.g. a Zammad ticket ID) without knowing about any specific service.
        """
        return None

    async def resolve_context(
        self,
        user_identifier: str,
        channel: str,
        message: str,
        user_display_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resolve service-specific context for this request.

        Called once during _prepare_request for each bound service.
        Returns a dict of service state that flows through the pipeline
        in RequestContext.service_data[self.name].
        """
        return {}

    async def on_message(self, service_data: Dict[str, Any], message: str) -> None:
        """
        Called when a user message is received or an LLM response is generated.

        Use for mirroring messages to external systems (e.g. posting to a ticket).
        """
        pass

    def prepare_tool_args(
        self,
        tool_name: str,
        args: Dict[str, Any],
        service_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Modify tool arguments before execution.

        Use for injecting service context (e.g. customer_id into create_ticket).
        Return the (possibly modified) args dict.
        """
        return args

    def on_tool_result(
        self,
        tool_name: str,
        result: Dict[str, Any],
        service_data: Dict[str, Any],
    ) -> None:
        """
        Process a tool result after execution.

        Use for capturing state changes (e.g. storing a newly created ticket ID).
        May mutate service_data in place.
        """
        pass

    def get_system_messages(self, service_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Return system messages to prepend to conversation history.

        Use for adding context like "This conversation is part of ticket #X".
        """
        return []
