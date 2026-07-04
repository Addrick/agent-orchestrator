# src/agents/managr_agent.py

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from config.global_config import (
    MANAGR_PLANNER_NAME,
    MANAGR_STALE_ANALYST_NAME,
    MANAGR_PATTERN_ANALYST_NAME,
    MANAGR_BOARD_TICKET_LIMIT,
    MANAGR_MAX_BOARD_CHARS,
    MANAGR_MAX_BRIEF_CHARS,
    MANAGR_PEER_AGENTS,
    MANAGR_PEER_ACTION_LIMIT,
)
from src.agents.base import Agent
from src.chat_system import ChatSystem
from src.clients.notification import NotificationRouter
from src.clients.zammad_client import ZammadClient
from src.persona import Persona

logger = logging.getLogger(__name__)


class ManagrAgent(Agent):
    """
    Board-level planning agent (DP-280, Phase 0: read-only).

    Where triage/dispatch react to a single ticket, managr reviews the whole
    board on a slow cadence and posts a "Manager's Report" digest. It is
    deliberately neutered: it holds no write path to Zammad or anywhere else —
    its only externally visible output is the digest sent via the
    NotificationRouter. Proposal/approval infrastructure is Phase 1.

    Pipeline (per cycle):
      1. [Hardcoded] Snapshot the board: open tickets + recent peer-agent activity
      2. [LLM fleet]  Read-only analyst briefs (stale tickets, cross-ticket patterns)
      3. [LLM]        Planner persona produces the Manager's Report
      4. [Hardcoded] Send digest to configured notification targets
    """

    agent_name: str = "managr"
    action_history_limit: int = 5

    # (label, persona-name) pairs fanned out in step 2. Class-level so tests
    # and future config can narrow/extend the fleet.
    analysts: List[Tuple[str, str]] = [
        ("stale", MANAGR_STALE_ANALYST_NAME),
        ("patterns", MANAGR_PATTERN_ANALYST_NAME),
    ]

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
        """Run one full report cycle."""
        action_id = self._log_task_root(
            action_type="manager_report",
            trigger_context="board_sweep",
            contexts=[("persona", MANAGR_PLANNER_NAME)],
        )
        try:
            board = await self._snapshot_board(action_id)
            if board is None:
                self._finalize_action(
                    action_id, "skipped", {"reason": "no open tickets on the board"},
                )
                await self._retain_action_series(action_id)
                return

            briefs = await self._gather_briefs(action_id, board)
            plan = await self._make_plan(action_id, board, briefs)
            if plan is None:
                self._finalize_action(
                    action_id, "failed", {"reason": "planner returned no report"},
                )
                await self._retain_action_series(action_id)
                return

            sent_count = await self._send_digest(action_id, plan)
            self._finalize_action(
                action_id,
                "success" if sent_count > 0 else "notification_failed",
                {
                    "sent_targets": sent_count,
                    "briefs": sorted(briefs.keys()),
                    # Excerpt feeds next cycle's action-history injection, so
                    # the planner sees what it reported last time.
                    "plan_excerpt": plan[:1500],
                },
            )
            logger.info(f"Manager's report cycle complete: sent to {sent_count} targets.")
        except Exception as e:
            logger.error(f"Manager's report cycle failed: {e}", exc_info=True)
            self._finalize_action(action_id, "error", {"error": str(e)})
        await self._retain_action_series(action_id)

    # --- 1. Observe ---

    async def _snapshot_board(self, action_id: int) -> Optional[str]:
        """Build the board snapshot: open tickets + recent peer-agent activity.

        Returns None when the board has no open tickets (nothing to report on).
        """
        query = "state.name:(new OR open)"
        tickets: List[Dict[str, Any]] = await asyncio.to_thread(
            self.zammad_client.search_tickets,
            query=query, limit=MANAGR_BOARD_TICKET_LIMIT,
        )
        self._log_step(
            action_id, "tool:zammad.search_tickets",
            action_payload={"query": query, "limit": MANAGR_BOARD_TICKET_LIMIT},
            outcome_payload={"ticket_count": len(tickets)},
        )
        if not tickets:
            return None

        now = datetime.now(timezone.utc)
        lines = [f"OPEN TICKETS ({len(tickets)}):"]
        for t in tickets:
            lines.append(self._format_ticket_line(t, now))

        peer_lines = self._recent_peer_activity()
        if peer_lines:
            lines.append("")
            lines.append("RECENT AUTOMATION ACTIVITY:")
            lines.extend(peer_lines)

        board = "\n".join(lines)
        if len(board) > MANAGR_MAX_BOARD_CHARS:
            board = board[:MANAGR_MAX_BOARD_CHARS] + "\n...[snapshot truncated]"
        return board

    def _format_ticket_line(self, ticket: Dict[str, Any], now: datetime) -> str:
        """One compact line per ticket. Search results only guarantee
        id/number/title/timestamps; state/priority appear only on expanded
        payloads, so both are optional here."""
        number = ticket.get("number", ticket.get("id", "?"))
        title = str(ticket.get("title", "No Title"))[:120]
        parts = [f"#{number} {title}"]

        state = ticket.get("state")
        if isinstance(state, str):
            parts.append(f"state={state}")
        priority = ticket.get("priority")
        if isinstance(priority, str):
            parts.append(f"priority={priority}")

        created = self._age_str(ticket.get("created_at"), now)
        updated = self._age_str(ticket.get("updated_at"), now)
        if created:
            parts.append(f"age={created}")
        if updated:
            parts.append(f"last_update={updated}")
        return "- " + " | ".join(parts)

    @staticmethod
    def _age_str(timestamp_str: Optional[str], now: datetime) -> str:
        if not timestamp_str:
            return ""
        try:
            ts = datetime.fromisoformat(str(timestamp_str).replace("Z", "+00:00"))
            diff = now - ts
            if diff.days > 0:
                return f"{diff.days}d"
            hours = diff.seconds // 3600
            if hours > 0:
                return f"{hours}h"
            return f"{diff.seconds // 60}m"
        except ValueError:
            return ""

    def _recent_peer_activity(self) -> List[str]:
        """Compact lines for what the other agents did recently."""
        lines: List[str] = []
        for peer in MANAGR_PEER_AGENTS:
            try:
                actions = self.memory_manager.get_relevant_agent_actions(
                    agent_name=peer, limit=MANAGR_PEER_ACTION_LIMIT,
                )
            except Exception as e:
                logger.warning(f"Could not fetch actions for peer agent '{peer}': {e}")
                continue
            for a in actions or []:
                ts = a.get("timestamp", "?")
                if hasattr(ts, "strftime"):
                    ts = ts.strftime("%Y-%m-%d %H:%M")
                else:
                    ts = str(ts)[:16]
                lines.append(
                    f"- [{ts}] {peer}: {a.get('action_type', '?')} "
                    f"{a.get('trigger_context') or ''} -> {a.get('outcome', '?')}"
                )
        return lines

    # --- 2. Orient ---

    async def _gather_briefs(self, action_id: int, board: str) -> Dict[str, str]:
        """Fan out the board snapshot to the read-only analyst personas."""
        briefs: Dict[str, str] = {}
        prompt = f"BOARD SNAPSHOT:\n{board}\n\nProduce your analysis brief."
        for label, persona_name in self.analysts:
            step_id = self._log_step(
                action_id, f"brief:{label}",
                action_payload={"persona": persona_name},
                outcome="pending",
            )
            persona = self.chat_system.personas.get(persona_name)
            if not persona:
                logger.error(f"System persona '{persona_name}' not found; skipping brief.")
                self._finalize_action(step_id, "failed", {"reason": "persona not found"})
                continue
            text = await self._call_persona(persona, prompt)
            if text:
                briefs[label] = text[:MANAGR_MAX_BRIEF_CHARS]
                self._finalize_action(step_id, "success", {"brief_excerpt": text[:400]})
            else:
                self._finalize_action(step_id, "failed", {"reason": "no text response"})
        return briefs

    # --- 3. Decide ---

    async def _make_plan(
        self, action_id: int, board: str, briefs: Dict[str, str],
    ) -> Optional[str]:
        """Single planning call over the snapshot + briefs."""
        persona = self.chat_system.personas.get(MANAGR_PLANNER_NAME)
        if not persona:
            logger.error(f"System persona '{MANAGR_PLANNER_NAME}' not found. Cannot plan.")
            self._log_step(
                action_id, "llm_step",
                action_payload={"persona": MANAGR_PLANNER_NAME},
                outcome="failed",
                outcome_payload={"reason": "persona not found"},
            )
            return None

        sections = [f"BOARD SNAPSHOT:\n{board}"]
        for label, text in sorted(briefs.items()):
            sections.append(f"ANALYST BRIEF ({label}):\n{text}")
        sections.append("Produce today's Manager's Report.")
        prompt = "\n\n".join(sections)

        step_id = self._log_step(
            action_id, "llm_step",
            action_payload={"persona": MANAGR_PLANNER_NAME,
                            "briefs": sorted(briefs.keys())},
            outcome="pending",
        )
        plan = await self._call_persona(persona, prompt, with_history=True)
        if plan is None:
            self._finalize_action(step_id, "failed", {"reason": "no text response"})
            return None
        self._finalize_action(step_id, "success", {"plan_excerpt": plan[:400]})
        return plan

    async def _call_persona(
        self, persona: Persona, prompt: str, with_history: bool = False,
    ) -> Optional[str]:
        """Single-shot, tool-less LLM call. Returns text content or None.

        with_history=True injects recent manager_report actions (last plans +
        outcomes) via the base-class history builder, giving the planner
        continuity across cycles.
        """
        try:
            if with_history:
                history_object = self._build_history_object(persona, prompt)
            else:
                history_object = {
                    "persona_prompt": persona.get_prompt(),
                    "message_history": [{"role": "user", "content": prompt}],
                    "history": [{"role": "user", "content": prompt}],
                    "current_message": {"text": prompt, "image_url": None},
                }
            response, _ = await self.text_engine.generate_response(
                persona_config=persona.get_config_for_engine(),
                history_object=history_object,
                tools=None,
            )
            if response.get("type") != "text":
                return None
            content = str(response.get("content", "")).strip()
            return content or None
        except Exception as e:
            logger.warning(f"Persona call failed ({persona.get_name()}): {e}")
            return None

    # --- 4. Report ---

    async def _send_digest(self, action_id: int, plan: str) -> int:
        """Send the report to every configured notification target."""
        targets = self.agent_config.get("notification_targets", [])
        if not targets:
            logger.warning("Managr has no notification_targets configured; report not sent.")
            return 0

        subject = f"Manager's Report — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        sent_count = 0
        for target in targets:
            channel = target.get("channel", "discord_dm")
            recipient = self._resolve_recipient(channel, str(target.get("recipient", "")))
            try:
                sent = await self.notification_router.send(
                    channel=channel,
                    recipient=recipient,
                    subject=subject,
                    body=plan,
                )
            except Exception as e:
                sent = False
                logger.error(f"Failed to send manager's report to {channel}:{recipient}: {e}")
            self._log_step(
                action_id, "tool:notification.send",
                action_payload={"channel": channel, "recipient": recipient,
                                "subject": subject},
                outcome="success" if sent else "failed",
                outcome_payload={"sent": sent},
            )
            if sent:
                sent_count += 1
        return sent_count

    def _resolve_recipient(self, channel: str, recipient_key: str) -> str:
        """Resolve a recipient key from agents.json recipients, or pass through."""
        if recipient_key.isdigit():
            return recipient_key
        recipients = self.agent_config.get("_recipients", {})
        info = recipients.get(recipient_key)
        if info:
            if channel == "discord_channel" and info.get("discord_channel_id"):
                return str(info["discord_channel_id"])
            if channel == "discord_dm" and info.get("discord_user_id"):
                return str(info["discord_user_id"])
            if "email" in channel and info.get("email"):
                return str(info["email"])
        return recipient_key
