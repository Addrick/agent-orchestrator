# src/clients/zammad_service.py

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse

from src.clients.service_integration import ServiceIntegration
from src.clients.zammad_client import ZammadClient

if TYPE_CHECKING:
    from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)


@dataclass
class ZammadContext:
    """Internal state bundle for Zammad context resolution."""
    customer_id: Optional[int] = None
    zammad_email: Optional[str] = None
    ticket_id: Optional[int] = None
    user_facing_ticket_number: Optional[int] = None


class ZammadIntegration(ServiceIntegration):
    """
    Service integration for Zammad ticketing.

    Handles user resolution, ticket lookup, message mirroring,
    tool registration, and tool argument injection for the Zammad API.
    """

    def __init__(self, zammad_client: ZammadClient) -> None:
        self._client = zammad_client

    @property
    def name(self) -> str:
        return "zammad"

    # --- Registration hooks ---

    def register_tools(self, tool_manager: "ToolManager") -> None:
        """Register all Zammad CRUD tools with the ToolManager."""
        from src.tools.tool_manager import ZammadToolHandler
        ZammadToolHandler(self._client).register(tool_manager)

    def get_tracking_id(self, service_data: Dict[str, Any]) -> Optional[int]:
        """Return the Zammad ticket ID for memory tracking."""
        return service_data.get("ticket_id")

    # --- ServiceIntegration lifecycle hooks ---

    async def resolve_context(
        self,
        user_identifier: str,
        channel: str,
        message: str,
        user_display_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve Zammad user, ticket number, and active ticket."""
        ctx = ZammadContext()

        ctx.customer_id, ctx.zammad_email = await self._get_or_create_user(
            user_identifier, channel, user_display_name
        )
        if not ctx.customer_id:
            return self._ctx_to_dict(ctx)

        ctx.user_facing_ticket_number = self._find_ticket_number_in_message(message)
        if ctx.user_facing_ticket_number:
            ctx.ticket_id = await self._get_ticket_id_from_number(ctx.user_facing_ticket_number)
            if not ctx.ticket_id:
                logger.warning(
                    f"User mentioned ticket number {ctx.user_facing_ticket_number}, but it was not found.")
        else:
            ctx.ticket_id = await self._find_active_ticket_for_user(ctx.customer_id)

        return self._ctx_to_dict(ctx)

    async def on_message(self, service_data: Dict[str, Any], message: str) -> None:
        """Mirror a message to the associated Zammad ticket."""
        ticket_id = service_data.get("ticket_id")
        zammad_email = service_data.get("zammad_email")
        if ticket_id:
            try:
                await asyncio.to_thread(
                    self._client.add_article_to_ticket,
                    ticket_id=ticket_id,
                    body=message,
                    impersonate_email=zammad_email,
                )
            except Exception as e:
                logger.error(f"Failed to mirror message to Zammad ticket {ticket_id}: {e}")

    def prepare_tool_args(
        self,
        tool_name: str,
        args: Dict[str, Any],
        service_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Inject customer_id into create_ticket calls."""
        if tool_name == "create_ticket":
            if "customer_id" not in args and service_data.get("customer_id"):
                args["customer_id"] = service_data["customer_id"]
        return args

    def on_tool_result(
        self,
        tool_name: str,
        result: Dict[str, Any],
        service_data: Dict[str, Any],
    ) -> None:
        """Capture newly created ticket IDs."""
        if tool_name == "create_ticket" and result.get("result", {}).get("id"):
            service_data["ticket_id"] = result["result"]["id"]

    def get_system_messages(self, service_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Add ticket context to conversation history."""
        ticket_id = service_data.get("ticket_id")
        if not ticket_id:
            return []

        display_number = service_data.get("user_facing_ticket_number") or ticket_id
        return [{"role": "system", "content": f"This conversation is part of Zammad ticket #{display_number}."}]

    # --- Internal Zammad methods (extracted from ChatSystem) ---

    @staticmethod
    def _find_ticket_number_in_message(message: str) -> Optional[int]:
        """Finds a ticket NUMBER pattern like [Ticket#12345] in a message."""
        match = re.search(r'\[Ticket#(\d+)\]', message, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    async def _get_ticket_id_from_number(self, ticket_number: int) -> Optional[int]:
        """Translates a user-facing ticket number to an internal Zammad ticket ID."""
        try:
            search_results = await asyncio.to_thread(
                self._client.search_tickets, query=f"number:{ticket_number}"
            )
            if search_results:
                ticket_id: int = search_results[0]['id']
                return ticket_id
            return None
        except Exception as e:
            logger.error(f"Error searching for ticket number {ticket_number}: {e}", exc_info=True)
            return None

    async def _find_active_ticket_for_user(self, customer_id: int) -> Optional[int]:
        """Finds the most recently updated open or new ticket for a given Zammad user ID."""
        try:
            query = f"customer_id:{customer_id} AND state.name:(open OR new)"
            search_results: List[Dict[str, Any]] = await asyncio.to_thread(
                self._client.search_tickets,
                query=query,
                sort_by='updated_at',
                order_by='desc'
            )
            if search_results:
                ticket_id: int = search_results[0]['id']
                return ticket_id
            return None
        except Exception as e:
            logger.error(f"Error searching for active tickets for customer {customer_id}: {e}", exc_info=True)
            return None

    async def _get_or_create_user(
        self,
        user_identifier: str,
        channel: str,
        user_display_name: Optional[str] = None,
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Finds a Zammad user by email or creates one.
        Returns (Zammad user ID, Zammad-registered email).
        """
        email_match: Optional[re.Match[str]] = re.search(r'<(.+?)>', user_identifier)

        email: str
        firstname: str
        lastname: str
        note: Optional[str]

        if email_match:
            email = email_match.group(1)
            name_part: str = user_display_name if user_display_name else user_identifier.split('<')[0].strip()
            name_parts: List[str] = name_part.split()
            firstname = name_parts[0] if name_parts else "Unknown"
            lastname = ' '.join(name_parts[1:]) if len(name_parts) > 1 else "User"
            note = None
        else:
            domain = urlparse(self._client.api_url or "").hostname or "local.host"
            email = f"{channel.lower()}-{user_identifier}@{domain}"
            name_parts = user_display_name.split() if user_display_name else [f"{channel.capitalize()} User",
                                                                              user_identifier]
            firstname = name_parts[0]
            lastname = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ""
            note = f"Auto-generated user from {channel.capitalize()}. Original identifier: {user_identifier}"

        try:
            search_results = await asyncio.to_thread(self._client.search_user, email)
            if search_results:
                return search_results[0]['id'], search_results[0]['email']

            logger.info(f"Creating new Zammad user for identifier '{user_identifier}' with email: {email}")
            new_user = await asyncio.to_thread(
                self._client.create_user,
                email=email, firstname=firstname, lastname=lastname, note=note
            )
            return new_user['id'], new_user['email']

        except Exception as e:
            logger.error(f"Error getting or creating Zammad user for '{user_identifier}': {e}", exc_info=True)
            return None, None

    @staticmethod
    def _ctx_to_dict(ctx: ZammadContext) -> Dict[str, Any]:
        """Convert internal ZammadContext to the dict returned by resolve_context."""
        return {
            "customer_id": ctx.customer_id,
            "zammad_email": ctx.zammad_email,
            "ticket_id": ctx.ticket_id,
            "user_facing_ticket_number": ctx.user_facing_ticket_number,
        }
