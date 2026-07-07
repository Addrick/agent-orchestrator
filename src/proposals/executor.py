# src/proposals/executor.py

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

from src.clients.zammad_client import ZammadClient
from src.proposals.schemas import validate_proposal_args

logger = logging.getLogger(__name__)


class ProposalExecutor:
    """
    Executes human-approved proposals against Zammad (DP-282).

    Separate from the proposing agent by design (ADR 2026-07-04): managr can
    only write proposal rows; this executor is the sole component that turns
    an approved row into an external write, and it re-validates the stored
    args against the whitelist schema before every dispatch.
    """

    def __init__(self, zammad_client: ZammadClient) -> None:
        self._client = zammad_client

    async def execute(self, proposal: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute one approved proposal. Returns (success, result message)."""
        action_type = proposal.get("action_type", "")
        args = proposal.get("action_args") or {}

        errors = validate_proposal_args(action_type, args)
        if errors:
            return False, f"args failed re-validation: {'; '.join(errors)}"

        try:
            ticket = await self._resolve_ticket(args["ticket_number"])
        except ValueError as e:
            return False, str(e)
        except Exception as e:
            logger.error(f"Ticket resolution failed ({action_type} #{args['ticket_number']}): {e}")
            return False, f"ticket resolution failed: {e}"
        ticket_id = ticket.get("id")
        if not ticket_id:
            return False, f"ticket #{args['ticket_number']} search hit has no id"
        number = ticket.get("number", args["ticket_number"])

        try:
            result = await self._dispatch(action_type, ticket_id, number, args)
        except Exception as e:
            logger.error(f"Proposal execution failed ({action_type} on #{number}): {e}")
            return False, f"zammad call failed: {e}"
        if result is not None:
            return True, result
        # Unreachable while validate_proposal_args shares PROPOSAL_ACTIONS
        return False, f"no executor dispatch for action_type '{action_type}'"

    async def _dispatch(self, action_type: str, ticket_id: int, number: Any,
                        args: Dict[str, Any]) -> Optional[str]:
        """Run one whitelisted action; returns the success message, or None
        for an action_type with no dispatch branch."""
        if action_type == "add_note":
            # internal=True is hard-forced: Phase 1 proposals are never
            # customer-visible regardless of what the row says.
            await asyncio.to_thread(
                self._client.add_article_to_ticket,
                ticket_id=ticket_id, body=args["body"], internal=True,
            )
            return f"internal note added to ticket #{number}"
        if action_type == "set_priority":
            await asyncio.to_thread(
                self._client.update_ticket,
                ticket_id=ticket_id, payload={"priority": args["priority"]},
            )
            return f"ticket #{number} priority set to {args['priority']}"
        if action_type == "remind":
            await asyncio.to_thread(
                self._client.update_ticket,
                ticket_id=ticket_id,
                payload={"state": "pending reminder",
                         "pending_time": f"{args['pending_until']}T09:00:00Z"},
            )
            return f"ticket #{number} parked until {args['pending_until']}"
        return None

    async def _resolve_ticket(self, ticket_number: int) -> Dict[str, Any]:
        """Resolve a user-facing ticket number to the ticket dict (internal id)."""
        results = await asyncio.to_thread(
            self._client.search_tickets, query=f"number:{ticket_number}", limit=1,
        )
        if not results:
            raise ValueError(f"ticket #{ticket_number} not found")
        return dict(results[0])
