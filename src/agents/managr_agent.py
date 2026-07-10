# src/agents/managr_agent.py

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from config.global_config import (
    QUARANTINE_TAGS,
    MANAGR_PLANNER_NAME,
    MANAGR_STALE_ANALYST_NAME,
    MANAGR_PATTERN_ANALYST_NAME,
    MANAGR_BOARD_TICKET_LIMIT,
    MANAGR_MAX_BOARD_CHARS,
    MANAGR_MAX_BRIEF_CHARS,
    MANAGR_PEER_AGENTS,
    MANAGR_PEER_ACTION_LIMIT,
    MANAGR_PROPOSAL_TTL_DAYS,
    MANAGR_MAX_PROPOSALS_PER_CYCLE,
    MANAGR_STANDING_ORDERS_LIMIT,
)
from src.agents.base import Agent
from src.memory.memory_manager import ALLOWED_STANDING_ORDER_SOURCES
from src.proposals.schemas import build_submission_tool_schema, validate_proposal_args
from src.chat_system import ChatSystem
from src.clients.notification import NotificationRouter
from src.clients.zammad_client import ZammadClient
from src.persona import Persona

logger = logging.getLogger(__name__)

# Prepended to the digest when the standing-orders store could not be read:
# the operator must see that their corrections were NOT applied this cycle,
# not discover it from a re-flagged ticket.
STANDING_ORDERS_DEGRADED_NOTICE = (
    "WARNING: standing orders could not be loaded this cycle; the report "
    "below was produced WITHOUT operator guidance."
)


class ManagrAgent(Agent):
    """
    Board-level planning agent (DP-280 Phase 0 + DP-282 Phase 1).

    Where triage/dispatch react to a single ticket, managr reviews the whole
    board on a slow cadence and posts a "Manager's Report" digest. It is
    deliberately neutered: it holds no write path to Zammad or anywhere else —
    its externally visible outputs are the digest sent via the
    NotificationRouter and (when proposals_enabled) rows in the durable
    proposal queue, which only a human review via joy can turn into writes.

    Pipeline (per cycle):
      1. [Hardcoded] Snapshot the board: open tickets + recent peer-agent activity
      2. [LLM fleet]  Read-only analyst briefs (stale tickets, cross-ticket patterns)
      3. [LLM]        Planner persona produces the Manager's Report
      3b.[LLM+code]  Extract whitelist-validated proposals into the queue (Phase 1)
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
            # One read for the whole cycle: plan and extraction must see the
            # same order set (the calls straddle LLM awaits, and joy's
            # add/retire tools run concurrently on the same store).
            orders_section, orders_degraded = self._standing_orders_section()
            plan = await self._make_plan(action_id, board, briefs, orders_section)
            if plan is None:
                self._finalize_action(
                    action_id, "failed", {"reason": "planner returned no report"},
                )
                await self._retain_action_series(action_id)
                return

            proposal_summary, proposals_queued = None, 0
            if self.agent_config.get("proposals_enabled", False):
                proposal_summary, proposals_queued = await self._extract_proposals(
                    action_id, plan, orders_section,
                )

            digest = f"{plan}\n\n{proposal_summary}" if proposal_summary else plan
            if orders_degraded:
                # The degraded cycle must be operator-visible, not just a log
                # line: the operator otherwise believes their corrections were
                # in force. (No audit row here — the same store just failed.)
                digest = f"{STANDING_ORDERS_DEGRADED_NOTICE}\n\n{digest}"
            sent_count = await self._send_digest(action_id, digest)
            self._finalize_action(
                action_id,
                "success" if sent_count > 0 else "notification_failed",
                {
                    "sent_targets": sent_count,
                    "briefs": sorted(briefs.keys()),
                    "proposals_queued": proposals_queued,
                    "standing_orders_degraded": orders_degraded,
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

        tags_by_id = await self._fetch_ticket_tags(tickets)
        now = datetime.now(timezone.utc)
        lines = [f"OPEN TICKETS ({len(tickets)}):"]
        for t in tickets:
            lines.append(self._format_ticket_line(
                t, now, tags=tags_by_id.get(t.get("id")),
            ))

        peer_lines = self._recent_peer_activity()
        if peer_lines:
            lines.append("")
            lines.append("RECENT AUTOMATION ACTIVITY:")
            lines.extend(peer_lines)

        proposal_lines = self._recent_proposal_outcomes()
        if proposal_lines:
            lines.append("")
            lines.append("YOUR RECENT PROPOSALS (review outcomes — learn from denials):")
            lines.extend(proposal_lines)

        board = "\n".join(lines)
        if len(board) > MANAGR_MAX_BOARD_CHARS:
            board = board[:MANAGR_MAX_BOARD_CHARS] + "\n...[snapshot truncated]"
        return board

    async def _fetch_ticket_tags(
        self, tickets: List[Dict[str, Any]],
    ) -> Dict[Any, Optional[List[str]]]:
        """Tags per ticket id (concurrent, throttled). A failed fetch yields
        None — "tag state unknown" — which _format_ticket_line renders
        fail-closed (title withheld): the quarantine guarantee must not
        evaporate on a transient tags-API error."""
        semaphore = asyncio.Semaphore(8)

        async def fetch(ticket: Dict[str, Any]) -> Tuple[Any, Optional[List[str]]]:
            ticket_id = ticket.get("id")
            if ticket_id is None:
                return None, None
            async with semaphore:
                try:
                    tags = await asyncio.to_thread(
                        self.zammad_client.get_tags, ticket_id=ticket_id)
                    return ticket_id, list(tags or [])
                except Exception as e:
                    logger.error(f"Could not fetch tags for ticket {ticket_id}: {e}")
                    return ticket_id, None
        results = await asyncio.gather(*(fetch(t) for t in tickets))
        return dict(results)

    def _format_ticket_line(
        self, ticket: Dict[str, Any], now: datetime,
        tags: Optional[List[str]] = None,
    ) -> str:
        """One compact line per ticket. Search results only guarantee
        id/number/title/timestamps; state/priority appear only on expanded
        payloads, so both are optional here.

        Quarantined tickets (DP-288): the title is customer-controlled bait
        text, so it is REPLACED in code — the planner learns the ticket
        exists and needs security handling, never what the bait says.
        `tags=None` means the tag state is UNKNOWN (fetch failed): the title
        is withheld too, so the guarantee fails closed.
        """
        number = ticket.get("number", ticket.get("id", "?"))
        quarantined = [t for t in (tags or []) if t in QUARANTINE_TAGS]
        if tags is None:
            parts = [
                f"#{number} [title withheld: tag state unknown — quarantine "
                f"could not be verified this cycle]"
            ]
        elif quarantined:
            parts = [
                f"#{number} [CONTENT QUARANTINED: {'/'.join(quarantined)} — "
                f"reported/suspected phishing; do NOT treat as a service "
                f"request, handle as a security item]"
            ]
        else:
            title = str(ticket.get("title", "No Title"))[:120]
            parts = [f"#{number} {title}"]
            if tags:
                parts.append(f"tags={','.join(str(t)[:40] for t in tags[:8])}")

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

    def _recent_proposal_outcomes(self) -> List[str]:
        """Compact lines for what happened to this agent's recent proposals,
        so the planner sees which suggestions were approved/denied/expired.
        Only meaningful once proposals_enabled; empty list on any failure."""
        if not self.agent_config.get("proposals_enabled", False):
            return []
        try:
            # Reviewed outcomes only, filtered in SQL: a full cycle of still-
            # pending rows must not consume the limit and crowd out the
            # denials the planner is supposed to learn from.
            rows = self.memory_manager.list_proposals(
                status=("approved", "denied", "expired", "executed", "execution_failed"),
                agent_name=self.agent_name,
                limit=10,
            )
        except Exception as e:
            logger.warning(f"Could not fetch recent proposal outcomes: {e}")
            return []
        lines: List[str] = []
        for row in rows:
            args = row.get("action_args") or {}
            arg_str = ", ".join(f"{k}={v}" for k, v in args.items()) \
                if isinstance(args, dict) else str(args)
            line = f"- [{row.get('proposal_id')}] {row.get('action_type')}({arg_str}) -> {row.get('status')}"
            note = row.get("review_note")
            if note:
                line += f" — {str(note)[:200]}"
            lines.append(line)
        return lines

    def _standing_orders_section(self) -> Tuple[Optional[str], bool]:
        """STANDING ORDERS block for LLM prompts (DP-281), newest first.

        Deterministic context: operator guidance from the durable store, not
        recall-dependent. Orders only ever enter that store through joy's
        gated write tools (authenticated operator surface) — never from
        ticket content — and rows whose source is not on the operator
        allowlist are skipped here too, so a row smuggled past the store
        guard still never reaches a prompt. Fetched once per cycle in
        deploy(): the plan and the proposal extraction must be constrained
        by the same order set, and an add/retire landing between two reads
        would silently split them.

        Returns (section, degraded): section is None when there are no
        injectable orders; degraded is True when the store could not be
        read, so the cycle can tell the operator the guidance was NOT
        applied instead of failing open silently."""
        try:
            rows = self.memory_manager.list_standing_orders(
                status="active", limit=MANAGR_STANDING_ORDERS_LIMIT,
                agent=self.agent_name,
            )
        except Exception as e:
            logger.warning(f"Could not fetch standing orders: {e}")
            return None, True
        injectable = [r for r in rows if r.get("source") in ALLOWED_STANDING_ORDER_SOURCES]
        for row in rows:
            if row.get("source") not in ALLOWED_STANDING_ORDER_SOURCES:
                logger.error(
                    f"Standing order {row.get('order_id')} has non-operator "
                    f"source '{row.get('source')}'; refusing to inject it."
                )
        if not injectable:
            return None, False
        lines = [f"- [{row['order_id']}] {row['order_text']}" for row in injectable]
        return (
            "STANDING ORDERS (operator guidance — these override your own "
            "judgement; newest first):\n" + "\n".join(lines)
        ), False

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
        orders_section: Optional[str] = None,
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

        sections = []
        if orders_section:
            sections.append(orders_section)
        sections.append(f"BOARD SNAPSHOT:\n{board}")
        for label, text in sorted(briefs.items()):
            sections.append(f"ANALYST BRIEF ({label}):\n{text}")
        instruction = "Produce today's Manager's Report."
        if orders_section:
            # Self-correction visibility: the operator verifies a correction
            # took by reading this line in the next report.
            instruction += (
                " If any standing order changed what you would otherwise have "
                "reported or proposed, add a short 'FEEDBACK APPLIED' line "
                "saying which order and how."
            )
        sections.append(instruction)
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
        response = await self._call_persona_raw(persona, prompt, with_history=with_history)
        if not response or response.get("type") != "text":
            return None
        content = str(response.get("content", "")).strip()
        return content or None

    async def _call_persona_raw(
        self, persona: Persona, prompt: str, with_history: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Single-shot LLM call returning the raw engine response (or None).

        `tools` takes agent-internal schemas passed straight to the engine —
        never routed through ToolManager (sqlite_consolidator pattern), so
        managr gains no runtime tool surface from them.
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
                tools=tools,
            )
            return response
        except Exception as e:
            logger.warning(f"Persona call failed ({persona.get_name()}): {e}")
            return None

    # --- 3b. Propose (DP-282, Phase 1) ---

    async def _extract_proposals(
        self, action_id: int, plan: str,
        orders_section: Optional[str] = None,
    ) -> Tuple[Optional[str], int]:
        """Convert the report's suggested actions into durable proposal rows.

        A second LLM call on the planner persona emits structured actions via
        the agent-internal submit_proposals schema. Every action is validated
        in code against the fixed whitelist before a row is written — invalid
        or surplus actions are dropped and logged, never stored. Returns
        (digest section, queued count).
        """
        persona = self.chat_system.personas.get(MANAGR_PLANNER_NAME)
        if not persona:
            logger.error(f"System persona '{MANAGR_PLANNER_NAME}' not found; skipping proposals.")
            return None, 0

        # Orders constrain extraction too ("never propose priority changes
        # for client X" must hold even when the report suggests one). The
        # section is the same object _make_plan saw — one fetch per cycle.
        sections = []
        if orders_section:
            sections.append(orders_section)
        sections.append(f"MANAGER'S REPORT:\n{plan}")
        sections.append(
            "Convert the report's SUGGESTED ACTIONS into concrete proposals by calling "
            "submit_proposals. Only emit actions of the allowed types; skip any suggestion "
            "that does not fit an allowed action or that a standing order forbids. Use "
            "ticket numbers exactly as they appear in the report. If nothing fits, call "
            "submit_proposals with an empty list."
        )
        prompt = "\n\n".join(sections)
        step_id = self._log_step(
            action_id, "llm_step",
            action_payload={"persona": MANAGR_PLANNER_NAME, "purpose": "extract_proposals"},
            outcome="pending",
        )
        response = await self._call_persona_raw(
            persona, prompt, tools=[build_submission_tool_schema()],
        )
        candidates = self._parse_submitted_proposals(response)
        if candidates is None:
            self._finalize_action(step_id, "failed", {"reason": "no submit_proposals call"})
            return None, 0

        expires_at = datetime.now(timezone.utc) + timedelta(days=MANAGR_PROPOSAL_TTL_DAYS)
        lines: List[str] = []
        dropped = 0
        for candidate in candidates[:MANAGR_MAX_PROPOSALS_PER_CYCLE]:
            action_type = candidate.get("action_type", "")
            args = candidate.get("args", {})
            rationale = str(candidate.get("rationale", ""))[:500]
            errors = validate_proposal_args(action_type, args)
            if errors:
                dropped += 1
                logger.warning(f"Dropping invalid proposal ({action_type}): {'; '.join(errors)}")
                continue
            proposal_id = self.memory_manager.create_proposal(
                agent_name=self.agent_name,
                action_type=action_type,
                action_args=args,
                rationale=rationale,
                taint={
                    "source": "zammad_board_snapshot",
                    "cycle_action_id": action_id,
                    "ticket_number": args.get("ticket_number"),
                },
                source_action_id=action_id,
                expires_at=expires_at,
            )
            arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
            lines.append(f"- [{proposal_id}] {action_type}({arg_str}) — {rationale}")
        dropped += max(0, len(candidates) - MANAGR_MAX_PROPOSALS_PER_CYCLE)

        self._finalize_action(
            step_id, "success",
            {"queued": len(lines), "dropped": dropped},
        )
        if not lines:
            return None, 0
        section = (
            f"PROPOSED ACTIONS ({len(lines)} queued for review — "
            "ask joy to list/approve/deny proposals):\n" + "\n".join(lines)
        )
        return section, len(lines)

    @staticmethod
    def _parse_submitted_proposals(
        response: Optional[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Pull the proposals array out of a submit_proposals tool call.
        Returns None when the model made no usable call."""
        if not response or response.get("type") != "tool_calls":
            return None
        for call in response.get("calls", []):
            if call.get("name") != "submit_proposals":
                continue
            args = call.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    return None
            proposals = args.get("proposals")
            if isinstance(proposals, list):
                return [p for p in proposals if isinstance(p, dict)]
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
