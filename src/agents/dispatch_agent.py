# src/agents/dispatch_agent.py

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from config.global_config import (
    DISPATCH_POLL_INTERVAL,
    DISPATCH_TRIAGE_TAG,
    DISPATCH_DISPATCHED_TAG,
    DISPATCH_PERSONA_NAME,
)
from src.agents.base import AgentLoop
from src.chat_system import ChatSystem
from src.clients.notification import NotificationRouter
from src.clients.zammad_client import ZammadClient

logger = logging.getLogger(__name__)


class DispatchAgent(AgentLoop):
    """
    Polls for triaged tickets and dispatches notifications based on LLM decisions.

    Pipeline (per ticket):
      1. [Hardcoded] Fetch ticket + triage note
      2. [LLM]       Decide priority, channel, and message (with action history)
      3. [Hardcoded] Send notification via NotificationRouter
      4. [Hardcoded] Tag ticket as dispatched
      5. [Hardcoded] Log each step to Agent_Actions table
    """

    poll_interval: float = DISPATCH_POLL_INTERVAL
    agent_name: str = "dispatch"
    action_history_limit: int = 10

    def __init__(
        self,
        chat_system: ChatSystem,
        zammad_client: ZammadClient,
        notification_router: NotificationRouter,
        agent_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(chat_system)
        self.zammad_client = zammad_client
        self.notification_router = notification_router
        self._agent_config = agent_config or {}
        self._notification_defaults = self._agent_config.get("notification_defaults", {})
        self._recipients = self._agent_config.get("_recipients", {})

    async def _poll(self) -> None:
        """Find triaged-but-not-dispatched tickets and process each."""
        query = f"tags:{DISPATCH_TRIAGE_TAG} AND NOT tags:{DISPATCH_DISPATCHED_TAG} AND state.name:new"
        try:
            tickets: List[Dict[str, Any]] = await asyncio.to_thread(
                self.zammad_client.search_tickets, query=query, limit=10
            )
        except Exception as e:
            logger.error(f"Failed to search for dispatchable tickets: {e}")
            return

        for ticket in tickets:
            if self._shutdown_event.is_set():
                break
            await self._dispatch_ticket(ticket['id'])

    async def _dispatch_ticket(self, ticket_id: int) -> None:
        """Run the full dispatch pipeline for a single ticket."""
        action_id = self.memory_manager.log_agent_action(
            agent_name=self.agent_name,
            action_type="dispatch",
            trigger_context=f"ticket:{ticket_id}",
            outcome="pending",
        )

        try:
            # 1. Fetch ticket
            ticket = await asyncio.to_thread(self.zammad_client.get_ticket, ticket_id=ticket_id)
            title = ticket.get('title', 'No Title')
            self._log_step(action_id, "fetch_ticket",
                           action_payload=json.dumps({"ticket_id": ticket_id}),
                           outcome_payload=json.dumps({
                               "title": title,
                               "customer": ticket.get("customer"),
                               "number": ticket.get("number"),
                           }))

            # Tag with multi-dimensional contexts
            contexts = [("ticket", str(ticket_id))]
            customer = ticket.get("customer")
            if customer:
                contexts.append(("customer", str(customer)))
            self.memory_manager.add_action_contexts(action_id, contexts)

            # 2. Fetch articles + extract triage note
            articles = await asyncio.to_thread(
                self.zammad_client.get_ticket_articles, ticket_id=ticket_id
            )
            triage_note = self._extract_triage_note(articles)
            self._log_step(action_id, "fetch_articles",
                           action_payload=json.dumps({"ticket_id": ticket_id}),
                           outcome_payload=json.dumps({"article_count": len(articles)}))

            # 3. LLM dispatch decision (with action history in context)
            decision = await self._get_dispatch_decision(title, triage_note, ticket_id, customer)
            self._log_step(action_id, "llm_decision",
                           action_payload=json.dumps({"title": title}),
                           outcome="success" if decision else "failed",
                           outcome_payload=json.dumps(decision) if decision else "no decision returned")
            if decision is None:
                self.memory_manager.update_agent_action_outcome(
                    action_id, "failed", "llm_decision step failed"
                )
                return

            # 4. Send notification
            notify_channel = decision.get("notify_channel", "zammad")
            summary = decision.get("summary", title)
            priority = decision.get("priority", "medium")

            notification_body = (
                f"Priority: {priority.upper()}\n"
                f"Ticket: #{ticket.get('number', ticket_id)}\n"
                f"Issue: {summary}\n"
                f"Reasoning: {decision.get('reasoning', 'N/A')}"
            )

            if notify_channel == "zammad":
                recipient = str(ticket_id)
            else:
                resolved = self._resolve_recipient(notify_channel)
                if resolved is None:
                    logger.warning(
                        f"Could not resolve recipient for channel '{notify_channel}'. "
                        f"Falling back to zammad note."
                    )
                    notify_channel = "zammad"
                    recipient = str(ticket_id)
                else:
                    recipient = resolved

            subject = f"[{priority.upper()}] {title}"
            sent = await self.notification_router.send(
                channel=notify_channel,
                recipient=recipient,
                subject=subject,
                body=notification_body,
            )
            self._log_step(action_id, "send_notification",
                           action_payload=json.dumps({
                               "channel": notify_channel,
                               "recipient": recipient,
                               "subject": subject,
                           }),
                           outcome="success" if sent else "failed")

            # 5. Tag ticket as dispatched
            await asyncio.to_thread(
                self.zammad_client.add_tag,
                ticket_id=ticket_id,
                tag=DISPATCH_DISPATCHED_TAG,
            )
            self._log_step(action_id, "tag_ticket",
                           action_payload=json.dumps({
                               "ticket_id": ticket_id,
                               "tag": DISPATCH_DISPATCHED_TAG,
                           }))

            # 6. Update parent task outcome
            outcome_payload = json.dumps({
                "priority": priority,
                "channel": notify_channel,
                "sent": sent,
                "decision": decision,
            })
            self.memory_manager.update_agent_action_outcome(
                action_id, "success" if sent else "notification_failed", outcome_payload
            )
            logger.info(f"Ticket {ticket_id} dispatched: priority={priority}, channel={notify_channel}")

        except Exception as e:
            logger.error(f"Error dispatching ticket {ticket_id}: {e}", exc_info=True)
            self.memory_manager.update_agent_action_outcome(
                action_id, "error", str(e)
            )

    def _resolve_recipient(self, channel: str) -> Optional[str]:
        """Resolve a platform-specific recipient ID for the given channel.

        Uses notification_defaults.recipient from agent config to find the
        logical recipient, then maps it to a platform ID via the recipients table.

        Returns None if the channel isn't available or no mapping exists.
        """
        if channel not in self.notification_router.available_channels:
            return None

        # Look up logical recipient name from agent config
        logical_name = self._notification_defaults.get("recipient")
        if not logical_name:
            logger.warning("No default recipient configured for dispatch agent.")
            return None

        # Map logical name to platform-specific ID
        recipient_info = self._recipients.get(logical_name, {})

        # Channel → recipient field mapping
        channel_to_field = {
            "discord_dm": "discord_user_id",
            "email": "email",
        }

        field = channel_to_field.get(channel)
        if not field:
            logger.warning(f"No recipient field mapping for channel '{channel}'.")
            return None

        resolved: Optional[str] = recipient_info.get(field)
        if not resolved:
            logger.warning(
                f"Recipient '{logical_name}' has no '{field}' configured. "
                f"Update the recipients section in agents.json."
            )
            return None

        return str(resolved)

    def _extract_triage_note(self, articles: List[Dict[str, Any]]) -> str:
        """Extract the AI triage note from ticket articles (last internal note)."""
        for article in reversed(articles):
            if article.get('internal', False):
                body: str = article.get('body', '')
                if 'AI TRIAGE CONTEXT DUMP' in body or 'Recommended Action' in body:
                    return body
        # Fallback: return the last article body
        if articles:
            result: str = articles[-1].get('body', 'No content')
            return result
        return 'No content'

    async def _get_dispatch_decision(
        self, title: str, triage_note: str,
        ticket_id: int, customer: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Call the dispatch_analyst persona to decide routing."""
        persona = self.chat_system.personas.get(DISPATCH_PERSONA_NAME)
        if not persona:
            logger.error(f"System persona '{DISPATCH_PERSONA_NAME}' not found. Cannot dispatch.")
            return None

        prompt = (
            f"TICKET TITLE: {title}\n\n"
            f"TRIAGE NOTE:\n{triage_note[:4000]}\n\n"
            f"AVAILABLE NOTIFICATION CHANNELS: {', '.join(self.notification_router.available_channels) or 'zammad'}\n\n"
            f"Decide how to dispatch this ticket."
        )

        # Build task_data for multi-dimensional context matching
        match_contexts = [("ticket", str(ticket_id))]
        if customer:
            match_contexts.append(("customer", str(customer)))
        task_data: Dict[str, Any] = {"match_contexts": match_contexts}

        try:
            response, _ = await self.text_engine.generate_response(
                persona_config=persona.get_config_for_engine(),
                context_object=self._build_llm_context(persona, prompt, task_data=task_data),
                tools=None,
            )

            if response.get('type') != 'text':
                return None

            content = response.get('content', '').strip()
            # Parse JSON from the LLM response
            parsed: Dict[str, Any] = json.loads(content)
            return parsed

        except json.JSONDecodeError as e:
            logger.warning(f"Dispatch LLM returned invalid JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"Dispatch LLM call failed: {e}")
            return None
