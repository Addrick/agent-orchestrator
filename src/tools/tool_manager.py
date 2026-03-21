# src/tools/tool_manager.py

import asyncio
import logging
from typing import Any, Coroutine, Dict, List, Callable, Optional

from src.tools.definitions import ALL_TOOL_DEFINITIONS

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Generic registry for tool implementations.

    Tool handlers register their async callables via `register()`.
    The manager provides a unified interface for definition listing and execution.
    Only definitions for registered tools (plus non-callable flags like google_grounding)
    are exposed to the LLM.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}

    def register(self, name: str, handler: Callable[..., Coroutine[Any, Any, Any]]) -> None:
        """Register an async handler for a named tool."""
        self._handlers[name] = handler

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns definitions for registered tools plus non-callable tool flags."""
        return [
            t for t in ALL_TOOL_DEFINITIONS
            if t.get('function', {}).get('name') in self._handlers
            or t.get('type') != 'function'
        ]

    async def execute_tool(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes a registered tool by name.

        Returns:
            A dictionary containing either the 'result' or an 'error' message.
        """
        if tool_name not in self._handlers:
            return {"error": f"Tool '{tool_name}' not found."}

        try:
            result = await self._handlers[tool_name](**kwargs)
            return {"result": result}
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}' with args {kwargs}: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred while executing {tool_name}: {str(e)}"}


class ZammadToolHandler:
    """Registers all Zammad CRUD tools with a ToolManager."""

    def __init__(self, zammad_client: Any) -> None:
        self.zammad_client = zammad_client

    def register(self, manager: ToolManager) -> None:
        manager.register("get_ticket_details", self._get_ticket_details)
        manager.register("update_ticket", self._update_ticket)
        manager.register("add_note_to_ticket", self._add_note_to_ticket)
        manager.register("search_tickets", self._search_tickets)
        manager.register("create_ticket", self._create_ticket)
        manager.register("search_user", self._search_user)
        manager.register("create_user", self._create_user)
        manager.register("update_user", self._update_user)
        manager.register("delete_user", self._delete_user)

    async def _get_ticket_details(self, ticket_number: int) -> Dict[str, Any]:
        """Translates user-facing ticket number to internal ID, then fetches details."""
        logger.info(f"Executing tool: get_ticket_details for ticket_number={ticket_number}")
        search_results = await asyncio.to_thread(
            self.zammad_client.search_tickets, query=f"number:{ticket_number}"
        )
        if not search_results:
            raise ValueError(f"No ticket found with number {ticket_number}.")

        ticket_id = search_results[0]['id']
        logger.info(f"Found ticket ID {ticket_id} for ticket number {ticket_number}.")

        ticket: Dict[str, Any] = await asyncio.to_thread(
            self.zammad_client.get_ticket, ticket_id=ticket_id
        )
        if not ticket:
            raise ValueError(f"Could not retrieve details for ticket ID {ticket_id} after finding it.")
        return ticket

    async def _update_ticket(self, ticket_id: int, **kwargs: Any) -> Dict[str, Any]:
        logger.info(f"Executing tool: update_ticket on ticket_id={ticket_id} with args={kwargs}")
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
        logger.info(f"Executing tool: add_note_to_ticket on ticket_id={ticket_id}")
        return await asyncio.to_thread(
            self.zammad_client.add_article_to_ticket,
            ticket_id=ticket_id, body=body, internal=internal
        )

    async def _search_tickets(self, query: str) -> List[Dict[str, Any]]:
        logger.info(f"Executing tool: search_tickets with query='{query}'")
        return await asyncio.to_thread(
            self.zammad_client.search_tickets, query=query
        )

    async def _create_ticket(self, title: str, body: str, customer_id: Optional[int] = None) -> Dict[str, Any]:
        logger.info(f"Executing tool: create_ticket with title='{title}' for customer_id={customer_id}")
        if not customer_id:
            raise ValueError("create_ticket requires a customer_id, which was not provided by the system.")
        return await asyncio.to_thread(
            self.zammad_client.create_ticket,
            title=title, group='Users', customer_id=customer_id, article_body=body
        )

    async def _search_user(self, query: str) -> List[Dict[str, Any]]:
        logger.info(f"Executing tool: search_user with query='{query}'")
        return await asyncio.to_thread(
            self.zammad_client.search_user, query=query
        )

    async def _create_user(self, firstname: str, lastname: str, email: str,
                           note: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"Executing tool: create_user with email='{email}'")
        return await asyncio.to_thread(
            self.zammad_client.create_user,
            firstname=firstname, lastname=lastname, email=email, note=note
        )

    async def _update_user(self, user_id: int, **kwargs: Any) -> Dict[str, Any]:
        logger.info(f"Executing tool: update_user on user_id={user_id} with args={kwargs}")
        valid_args = ["firstname", "lastname", "email", "active", "note"]
        payload = {k: v for k, v in kwargs.items() if k in valid_args}
        if not payload:
            raise ValueError("No valid update parameters provided for update_user.")
        return await asyncio.to_thread(
            self.zammad_client.update_user, user_id=user_id, payload=payload
        )

    async def _delete_user(self, user_id: int) -> Dict[str, str]:
        logger.info(f"Executing tool: delete_user on user_id={user_id}")
        await asyncio.to_thread(self.zammad_client.delete_user, user_id=user_id)
        return {"status": "success", "message": f"User {user_id} deleted."}


class WebSearchHandler:
    """Registers web search tools with a ToolManager."""

    def register(self, manager: ToolManager) -> None:
        manager.register("web_search", self._web_search)

    async def _web_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"Executing tool: web_search with query='{query}', max_results={max_results}")
        from duckduckgo_search import DDGS

        def _sync_search() -> List[Dict[str, Any]]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        raw = await asyncio.to_thread(_sync_search)
        return [{"title": r["title"], "url": r["href"], "summary": r["body"]} for r in raw]
