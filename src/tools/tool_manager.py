# src/tool_manager.py

import asyncio
import json
import logging
from typing import Any, Coroutine, Dict, List, Callable, Optional

from src.clients.zammad_client import ZammadClient
from src.tools.definitions import ALL_TOOL_DEFINITIONS

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Manages the execution of tools defined in src.tools.definitions.

    This class acts as a bridge between the tool schemas (the "what") and
    their actual implementation (the "how"). It uses a ZammadClient instance
    to perform the actions requested by the LLM.
    """

    def __init__(self, zammad_client: ZammadClient) -> None:
        """
        Initializes the ToolManager.

        Args:
            zammad_client: An authenticated instance of the ZammadClient.
        """
        self.zammad_client = zammad_client
        self.tool_functions: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
            "get_ticket_details": self._get_ticket_details,
            "update_ticket": self._update_ticket,
            "add_note_to_ticket": self._add_note_to_ticket,
            "search_tickets": self._search_tickets,
        }

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns the list of all available tool definitions."""
        return ALL_TOOL_DEFINITIONS

    async def execute_tool(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes a specified tool with the given arguments.

        Args:
            tool_name: The name of the tool to execute.
            **kwargs: The arguments for the tool, matching its definition.

        Returns:
            A dictionary containing either the 'result' of the successful
            execution or an 'error' message.
        """
        if tool_name not in self.tool_functions:
            return {"error": f"Tool '{tool_name}' not found."}

        func = self.tool_functions[tool_name]
        try:
            # Execute the corresponding async function and get the result
            result = await func(**kwargs)
            # Ensure the result is JSON serializable for the LLM
            # The client methods return dicts, which are fine.
            return {"result": result}
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}' with args {kwargs}: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred while executing {tool_name}: {str(e)}"}

    # --- Tool Implementation Methods ---

    async def _get_ticket_details(self, ticket_id: int) -> Dict[str, Any]:
        """
        Implementation for the 'get_ticket_details' tool.

        Note: The current ZammadClient can only retrieve a ticket summary via search,
        not the full article history. This implementation provides the best available
        data from the existing client.
        """
        search_result = await asyncio.to_thread(
            self.zammad_client.search_tickets, query=f"id:{ticket_id}"
        )
        if not search_result:
            raise ValueError(f"Ticket with ID {ticket_id} not found.")
        return search_result[0]

    async def _update_ticket(self, ticket_id: int, **kwargs: Any) -> Dict[str, Any]:
        """Implementation for the 'update_ticket' tool."""
        payload: Dict[str, Any] = {}
        valid_args = ["state", "priority", "owner_id", "tags"]

        for key, value in kwargs.items():
            if key in valid_args:
                if key == "tags" and isinstance(value, list):
                    payload[key] = ",".join(value)
                else:
                    payload[key] = value

        if not payload:
            raise ValueError("No valid update parameters provided for update_ticket.")

        return await asyncio.to_thread(
            self.zammad_client.update_ticket, ticket_id=ticket_id, payload=payload
        )

    async def _add_note_to_ticket(self, ticket_id: int, body: str, internal: bool = False) -> Dict[str, Any]:
        """Implementation for the 'add_note_to_ticket' tool."""
        return await asyncio.to_thread(
            self.zammad_client.add_article_to_ticket,
            ticket_id=ticket_id,
            body=body,
            internal=internal
        )

    async def _search_tickets(self, query: str) -> List[Dict[str, Any]]:
        """Implementation for the 'search_tickets' tool."""
        return await asyncio.to_thread(
            self.zammad_client.search_tickets, query=query
        )
