# src/agents/dispatch_agent.py

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from config.global_config import (
    DISPATCH_TRIAGE_TAG,
    DISPATCH_DISPATCHED_TAG,
    DISPATCH_PERSONA_NAME,
)
from src.agents.base import Agent
from src.chat_system import ChatSystem
from src.clients.notification import NotificationRouter
from src.clients.zammad_client import ZammadClient

logger = logging.getLogger(__name__)


class DispatchAgent(Agent):
    """
    Dispatches notifications for triaged tickets based on LLM decisions.

    Pipeline (per ticket):
      1. [Hardcoded] Fetch ticket + triage note
      2. [LLM]       Decide priority and summarize
      3. [Hardcoded] Send notification via config-defined channel/recipient
      4. [Hardcoded] Tag ticket as dispatched
      5. [Hardcoded] Log action to Agent_Actions table
    """

    agent_name: str = "dispatch"
    # Mingle agent series into the dispatch_analyst persona bank so reflect
    # surfaces past series alongside chat-prose extractions.
    experience_bank: Optional[str] = DISPATCH_PERSONA_NAME
    experience_persona: Optional[str] = DISPATCH_PERSONA_NAME

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
        self.agent_config = agent_config or {}

    async def deploy(self) -> None:
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
        action_id = self._log_task_root(
            action_type="dispatch",
            trigger_context=f"ticket:{ticket_id}",
            action_payload={"ticket_id": ticket_id},
            contexts=[("ticket_id", ticket_id), ("persona", DISPATCH_PERSONA_NAME)],
        )

        try:
            # 1. Fetch ticket and triage note
            ticket = await asyncio.to_thread(self.zammad_client.get_ticket, ticket_id=ticket_id)
            title = ticket.get('title', 'No Title')
            number = ticket.get('number', ticket_id)
            self._log_step(
                action_id, "tool:zammad.get_ticket",
                action_payload={"ticket_id": ticket_id},
                outcome_payload={"number": number, "title": title},
            )

            articles = await asyncio.to_thread(
                self.zammad_client.get_ticket_articles, ticket_id=ticket_id
            )
            triage_note = self._extract_triage_note(articles)
            triage_excerpt = triage_note[:600]
            # Raw articles live in Zammad (fully auditable there). Log a ref
            # + short excerpt instead of the bodies so the series stays small
            # and memory extraction sees signal, not bulk article text.
            self._log_step(
                action_id, "tool:zammad.get_ticket_articles",
                action_payload={"ticket_id": ticket_id},
                outcome_payload={
                    "article_count": len(articles),
                    "ref": f"zammad.ticket({ticket_id}).articles",
                    "triage_excerpt": triage_excerpt,
                },
            )

            # 2. LLM dispatch decision (priority + summary only)
            llm_step_id = self._log_step(
                action_id, "llm_step",
                action_payload={
                    "persona": DISPATCH_PERSONA_NAME,
                    "title": title,
                    "triage_excerpt": triage_excerpt,
                },
                outcome="pending",
            )
            decision = await self._get_dispatch_decision(title, triage_note)
            if decision is None:
                self._finalize_action(
                    llm_step_id, "failed",
                    {"reason": "LLM returned no dispatch decision"},
                )
                self._finalize_action(
                    action_id, "failed",
                    {"reason": "LLM returned no dispatch decision",
                     "title": title, "ticket_id": ticket_id},
                )
                await self._retain_action_series(action_id)
                return
            self._finalize_action(llm_step_id, "success", decision)

            # 3. Send notification via config-defined channel/recipient
            defaults = self.agent_config.get("notification_defaults", {})
            notify_channel = defaults.get("channel", "zammad")
            summary = decision.get("summary", title)
            priority = decision.get("priority", "medium")

            notification_body = (
                f"Priority: {priority.upper()}\n"
                f"Ticket: #{number}\n"
                f"Issue: {summary}\n"
                f"Reasoning: {decision.get('reasoning', 'N/A')}"
            )

            if notify_channel == "zammad":
                recipient = str(ticket_id)
            else:
                recipient = self._resolve_recipient(notify_channel, ticket_id)

            self._add_contexts(action_id, [
                ("priority", priority),
                ("channel", notify_channel),
                ("recipient", recipient),
            ])

            subject = f"[{priority.upper()}] {title}"
            try:
                sent = await self.notification_router.send(
                    channel=notify_channel,
                    recipient=recipient,
                    subject=subject,
                    body=notification_body,
                )
                self._log_step(
                    action_id, "tool:notification.send",
                    action_payload={
                        "channel": notify_channel,
                        "recipient": recipient,
                        "subject": subject,
                        "body_excerpt": notification_body[:400],
                    },
                    outcome="success" if sent else "failed",
                    outcome_payload={"sent": sent},
                )
            except Exception as send_exc:
                sent = False
                self._log_step(
                    action_id, "tool:notification.send",
                    action_payload={
                        "channel": notify_channel,
                        "recipient": recipient,
                        "subject": subject,
                    },
                    outcome="error",
                    outcome_payload={"error": str(send_exc)},
                )

            # 4. Tag ticket as dispatched
            await asyncio.to_thread(
                self.zammad_client.add_tag,
                ticket_id=ticket_id,
                tag=DISPATCH_DISPATCHED_TAG,
            )
            self._log_step(
                action_id, "tool:zammad.add_tag",
                action_payload={"ticket_id": ticket_id, "tag": DISPATCH_DISPATCHED_TAG},
                outcome_payload={"tagged": True},
            )

            # 5. Log outcome
            self._finalize_action(
                action_id,
                "success" if sent else "notification_failed",
                {
                    "ticket_id": ticket_id,
                    "number": number,
                    "title": title,
                    "priority": priority,
                    "channel": notify_channel,
                    "recipient": recipient,
                    "sent": sent,
                    "decision": decision,
                },
            )
            logger.info(f"Ticket {ticket_id} dispatched: priority={priority}, channel={notify_channel}")
            await self._retain_action_series(action_id)

        except Exception as e:
            logger.error(f"Error dispatching ticket {ticket_id}: {e}", exc_info=True)
            self._finalize_action(action_id, "error", {"error": str(e)})
            await self._retain_action_series(action_id)

    def _resolve_recipient(self, channel: str, ticket_id: int) -> str:
        """Resolve the notification recipient from agent_config or fall back to zammad."""
        defaults = self.agent_config.get("notification_defaults", {})
        recipient_name = defaults.get("recipient")
        recipients = self.agent_config.get("_recipients", {})

        if recipient_name and recipient_name in recipients:
            recipient_info = recipients[recipient_name]
            if channel == "discord_channel" and recipient_info.get("discord_channel_id"):
                return str(recipient_info["discord_channel_id"])
            if channel == "discord_dm" and recipient_info.get("discord_user_id"):
                return str(recipient_info["discord_user_id"])
            if "email" in channel and recipient_info.get("email"):
                return str(recipient_info["email"])

        # No mapping found — fall back to the ticket ID (works for zammad notifier)
        logger.warning(
            f"No recipient mapping for channel '{channel}'. "
            f"Falling back to ticket ID {ticket_id}."
        )
        return str(ticket_id)

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

    async def _get_dispatch_decision(self, title: str, triage_note: str) -> Optional[Dict[str, Any]]:
        """Call the dispatch_analyst persona to assess priority and summarize."""
        persona = self.chat_system.personas.get(DISPATCH_PERSONA_NAME)
        if not persona:
            logger.error(f"System persona '{DISPATCH_PERSONA_NAME}' not found. Cannot dispatch.")
            return None

        prompt = (
            f"TICKET TITLE: {title}\n\n"
            f"TRIAGE NOTE:\n{triage_note[:4000]}\n\n"
            f"Assess this ticket's priority and summarize it for dispatch."
        )

        try:
            response, _ = await self.text_engine.generate_response(
                persona_config=persona.get_config_for_engine(),
                history_object=self._build_history_object(persona, prompt),
                tools=None,
            )

            if response.get('type') != 'text':
                return None

            content = response.get('content', '').strip()
            parsed: Dict[str, Any] = self._parse_json_response(content)
            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Dispatch LLM returned invalid JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"Dispatch LLM call failed: {e}")
            return None

    @staticmethod
    def _parse_json_response(content: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling markdown fences."""
        # Try bare JSON first
        try:
            result: Dict[str, Any] = json.loads(content)
            return result
        except json.JSONDecodeError:
            pass

        # Try extracting from ```json ... ``` fences
        import re
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', content, re.DOTALL)
        if match:
            result = json.loads(match.group(1).strip())
            return result

        raise ValueError(f"No valid JSON found in response: {content[:200]}")
