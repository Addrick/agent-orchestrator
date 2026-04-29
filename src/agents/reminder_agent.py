# src/agents/reminder_agent.py

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base import Agent
from src.chat_system import ChatSystem
from src.clients.notification import NotificationRouter
from src.clients.zammad_client import ZammadClient

logger = logging.getLogger(__name__)

class ReminderAgent(Agent):
    """
    Monitors Zammad for ticketing updates and sends daily summaries.
    - Startup: Sends a DM to adrich.
    - Daily: Sends a summary to configured targets (azarvand-tix channel).
    """

    agent_name: str = "reminder"

    def __init__(
        self,
        chat_system: ChatSystem,
        zammad_client: ZammadClient,
        notification_router: NotificationRouter,
        agent_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(chat_system, inject_personas=False)
        self.zammad_client = zammad_client
        self.notification_router = notification_router
        self.agent_config = agent_config or {}

    async def deploy(self) -> None:
        """Send the daily ticket summary. This is called once per day based on the agent schedule."""
        # Special case: On the very first run (startup), send a DM to adrich as a "bot online" signal.
        if self.deploy_count == 0:
            logger.info("Agent 'reminder' startup run: Sending summary DM to adrich.")
            startup_target = {"channel": "discord_dm", "recipient": "adrich"}
            await self._send_batch_summary(target_override=startup_target)
            return

        logger.info("Starting daily ticket summary...")
        await self._send_batch_summary()

    async def _send_batch_summary(self, target_override: Optional[Dict[str, str]] = None) -> None:
        """Find all new and open tickets and send a formatted batch summary."""
        query = "state.name:(new OR open)"
        
        try:
            tickets = await asyncio.to_thread(self.zammad_client.search_tickets, query=query, limit=50)
            if not tickets:
                logger.info("No open or new tickets found for summary.")
                return

            reminder_lines = []
            ticket_ids = []
            now_utc = datetime.now(timezone.utc)

            for ticket in tickets:
                requester_name, user_link = await self._get_user_info(ticket.get('customer_id'))
                time_str = self._calculate_time_diff(ticket.get('updated_at'), now_utc)
                
                base_url = self.zammad_client.api_url.rstrip('/')
                ticket_link = f"{base_url}/#ticket/zoom/{ticket['id']}"
                
                line = f"• [**#{ticket.get('number')}** {ticket.get('title')}]({ticket_link}) — [{requester_name}]({user_link}) (Updated {time_str})"
                reminder_lines.append(line)
                ticket_ids.append(ticket['id'])

            if reminder_lines:
                body = "The following tickets are currently new or open:\n\n" + "\n\n".join(reminder_lines)
                
                # Use override if provided, otherwise use all configured targets
                targets = [target_override] if target_override else self.agent_config.get("notification_targets", [])
                
                success_count = 0
                for target in targets:
                    sent = await self._dispatch_to_target(
                        target=target,
                        subject="Daily Ticket Summary",
                        body=body,
                        ticket_id=ticket_ids[0]
                    )
                    if sent:
                        success_count += 1

                if success_count > 0:
                    logger.info(f"Successfully sent summary to {success_count} targets.")
                else:
                    logger.error("Failed to send summary to any targets.")

        except Exception as e:
            logger.error(f"Failed to process summary: {e}", exc_info=True)

    async def _dispatch_to_target(self, target: Dict[str, str], subject: str, body: str, ticket_id: int) -> bool:
        """Helper to send notification to a specific target (channel + recipient)."""
        channel = target.get("channel", "discord_dm")
        recipient_key = target.get("recipient", "adrich")
        
        recipient = self._resolve_recipient(channel, recipient_key, ticket_id)

        action_id = self.memory_manager.log_agent_action(
            agent_name=self.agent_name,
            action_type="daily_summary",
            trigger_context=f"target:{channel}:{recipient}",
            outcome="pending",
        )
        
        try:
            sent = await self.notification_router.send(
                channel=channel,
                recipient=recipient,
                subject=subject,
                body=body,
            )
            self.memory_manager.update_agent_action_outcome(action_id, "success" if sent else "failed")
            return sent
        except Exception as e:
            self.memory_manager.update_agent_action_outcome(action_id, "error", str(e))
            return False

    def _resolve_recipient(self, channel: str, recipient_key: str, ticket_id: int) -> str:
        """Resolve a recipient key to an ID, or return the key if it's already an ID."""
        if recipient_key.isdigit():
            return recipient_key

        recipients = self.agent_config.get("_recipients", {})
        if recipient_key in recipients:
            recipient_info = recipients[recipient_key]
            if channel == "discord_channel" and recipient_info.get("discord_channel_id"):
                return str(recipient_info["discord_channel_id"])
            if channel == "discord_dm" and recipient_info.get("discord_user_id"):
                return str(recipient_info["discord_user_id"])
            if "email" in channel and recipient_info.get("email"):
                return str(recipient_info["email"])

        return str(recipient_key)

    async def _get_user_info(self, customer_id: Optional[int]) -> Tuple[str, str]:
        """Fetch human-readable name and profile link for a user."""
        name = "Unknown"
        link = "#"
        if not customer_id:
            return name, link

        base_url = self.zammad_client.api_url.rstrip('/')
        try:
            customer = await asyncio.to_thread(self.zammad_client.get_user, user_id=customer_id)
            firstname = customer.get('firstname', '')
            lastname = customer.get('lastname', '')
            name = f"{firstname} {lastname}".strip() or customer.get('login', 'Unknown')
            link = f"{base_url}/#user/profile/{customer_id}"
        except Exception:
            pass
        return name, link

    def _calculate_time_diff(self, updated_at_str: Optional[str], now: datetime) -> str:
        """Calculate relative time string."""
        if not updated_at_str:
            return "Unknown"
        try:
            updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
            diff = now - updated_at
            if diff.days > 0: return f"{diff.days}d ago"
            if diff.seconds // 3600 > 0: return f"{diff.seconds // 3600}h ago"
            return f"{diff.seconds // 60}m ago"
        except:
            return "Unknown"

def create_reminder_agent(
    chat_system: ChatSystem, 
    zammad_client: ZammadClient, 
    notification_router: NotificationRouter,
    agent_config: Optional[Dict[str, Any]] = None
) -> ReminderAgent:
    return ReminderAgent(chat_system, zammad_client, notification_router, agent_config)
