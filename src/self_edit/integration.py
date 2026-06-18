# src/self_edit/integration.py
"""ServiceIntegration for the fixr self-improvement supervisor (DP-227 + DP-230).

Owns the dispatch subsystem (registry + Dispatcher). Two event paths leave the
per-agent bridge:

- **on_event** (DP-230) — fires for EVERY event, streaming the agent's parsed
  ``AgentEvent``s into a per-agent Discord *thread* (the transcript): ``progress``
  coalesced/muted, ``question`` highlighted, ``done/error`` summarized.
- **on_wake** (DP-227) — fires only for {question, done, error}. A ``question``
  is now answered by a human *directly in the thread* (no fixr LLM turn) when the
  direct channel is active; fixr is only woken as an idle fallback. ``done/error``
  still wake fixr for its report.

When the direct channel is OFF (no Discord client attached, or no parent channel
configured) every wake falls back to the original DP-227 fixr turn, so behaviour
is unchanged on deployments that don't set ``CC_FIXR_AGENTS_CHANNEL_ID``.

Personas opt in via ``service_bindings: ["fixr"]``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional, Protocol, TYPE_CHECKING

from config import global_config
from src.clients.service_integration import ServiceIntegration
from src.self_edit.dispatcher import Dispatcher
from src.self_edit.events import (
    DONE,
    ERROR,
    PROGRESS,
    QUESTION,
    STARTED,
    AgentEvent,
)
from src.self_edit.registry import AgentRecord, AgentRegistry
from src.self_edit.store import AgentStore

if TYPE_CHECKING:
    from src.chat_system import ChatSystem
    from src.clients.notification import NotificationRouter
    from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)


class DiscordThreadClient(Protocol):
    """The slice of the Discord client the direct channel needs (DP-230).

    ``CustomDiscordBot`` satisfies it; tests inject a fake."""

    async def create_agent_thread(self, parent_channel_id: int, name: str) -> Optional[int]:
        ...

    async def send_to_channel(self, channel_id: int, content: str) -> bool:
        ...


def _wake_message(record: AgentRecord, event: AgentEvent) -> str:
    """Render the event into the user-message text that wakes fixr."""
    lines = [
        f"[fixr-bridge] agent `{record.agent_id}` (bug {record.bug_id}) "
        f"emitted a `{event.type}` event.",
    ]
    text = event.payload.get("text") or event.payload.get("summary")
    if text:
        lines.append(text)
    if event.payload.get("pr_url"):
        lines.append(f"PR: {event.payload['pr_url']}")
    if event.payload.get("detail"):
        lines.append(f"(detail: {event.payload['detail']})")
    lines.append(
        f"Branch `{record.branch}`. Use inspect_agents/answer_agent/"
        f"kill_agent/send_discord as appropriate. Decide and act."
    )
    return "\n".join(lines)


def _format_event(record: AgentRecord, ev: AgentEvent) -> str:
    """Render a single event for the agent's Discord thread (the transcript)."""
    if ev.type == STARTED:
        return f"🟢 dispatched agent `{record.agent_id}` on branch `{record.branch}`."
    if ev.type == QUESTION:
        return (
            "❓ **Question** — reply in this thread to answer "
            "(prefix `//` for a note-to-self):\n"
            f"{ev.payload.get('text', '').strip()}"
        )
    if ev.type == DONE:
        summary = (ev.payload.get("summary") or "agent finished").strip()
        out = f"✅ **Done** — {summary}"
        if ev.payload.get("pr_url"):
            out += f"\nPR: {ev.payload['pr_url']}"
        return out
    if ev.type == ERROR:
        body = (ev.payload.get("text") or "agent errored").strip()
        out = f"⚠️ **Error** — {body}"
        if ev.payload.get("detail"):
            out += f"\n(detail: {ev.payload['detail']})"
        return out
    return str(ev.payload.get("text", "")).strip()


class FixrIntegration(ServiceIntegration):
    def __init__(
        self,
        chat_system: "ChatSystem",
        notification_router: "NotificationRouter",
        *,
        registry: AgentRegistry | None = None,
        dispatcher: Dispatcher | None = None,
    ) -> None:
        self._chat_system = chat_system
        self._notifier = notification_router
        self.registry = registry or AgentRegistry(
            store=AgentStore(global_config.CC_FIXR_REGISTRY_DB)
        )
        self.dispatcher = dispatcher or Dispatcher(
            self.registry,
            on_wake=self._on_wake,
            on_event=self._on_event,
            model_arg=global_config.CC_FIXR_MODEL_ARG,
            clone_dir=global_config.CC_FIXR_CLONE_DIR,
        )
        # DP-230 direct-channel state (all late-bound / lazy).
        self._discord: Optional[DiscordThreadClient] = None
        self._progress_buffers: Dict[str, list[str]] = {}
        self._flush_tasks: Dict[str, "asyncio.Task[None]"] = {}
        self._idle_timers: Dict[str, "asyncio.Task[None]"] = {}

    @property
    def name(self) -> str:
        return "fixr"

    def register_tools(self, tool_manager: "ToolManager") -> None:
        from src.self_edit.fixr_tools import FixrToolHandler
        handler = FixrToolHandler(self.dispatcher, self.registry, self._notifier)
        handler.register(tool_manager)

    # -- DP-230 wiring -------------------------------------------------------

    def attach_discord(self, discord_client: DiscordThreadClient) -> None:
        """Late-bind the Discord client (created after this service at startup)."""
        self._discord = discord_client

    def _direct_channel_active(self) -> bool:
        """True when agent Q&A should go straight to a human thread, not fixr."""
        return self._discord is not None and bool(global_config.CC_FIXR_AGENTS_CHANNEL_ID)

    # -- transcript sink (every event) ---------------------------------------

    async def _on_event(self, record: AgentRecord, ev: AgentEvent) -> None:
        """Stream one event into the agent's Discord thread (the transcript)."""
        # Any event means the agent is active again → drop a pending idle wake.
        self._cancel_idle(record.agent_id)
        if not self._direct_channel_active():
            return
        await self._ensure_thread(record)
        if record.discord_thread_id is None:
            return  # thread create failed; nothing to post to this event
        if ev.type == PROGRESS:
            self._buffer_progress(record, ev)
            return
        # Non-progress: flush any buffered progress first so order is preserved.
        await self._flush_progress(record, cancel_pending=True)
        await self._post_thread(record, _format_event(record, ev))

    async def _ensure_thread(self, record: AgentRecord) -> None:
        if record.discord_thread_id or self._discord is None:
            return
        try:
            parent = int(global_config.CC_FIXR_AGENTS_CHANNEL_ID)
        except (TypeError, ValueError):
            return
        name = f"{record.bug_id} · {record.agent_id}"[:100]
        try:
            thread_id = await self._discord.create_agent_thread(parent, name)
        except Exception:  # noqa: BLE001 — thread create must never kill the bridge
            logger.exception("agent thread create failed for %s", record.agent_id)
            return
        if thread_id:
            await self.registry.update(record.agent_id, discord_thread_id=str(thread_id))
            record.discord_thread_id = str(thread_id)

    def _buffer_progress(self, record: AgentRecord, ev: AgentEvent) -> None:
        text = (ev.payload.get("text") or "").strip()
        if text:
            self._progress_buffers.setdefault(record.agent_id, []).append(text)
        task = self._flush_tasks.get(record.agent_id)
        if task is None or task.done():
            self._flush_tasks[record.agent_id] = asyncio.create_task(
                self._debounced_flush(record)
            )

    async def _debounced_flush(self, record: AgentRecord) -> None:
        try:
            await asyncio.sleep(global_config.CC_FIXR_PROGRESS_DEBOUNCE_SECONDS)
        except asyncio.CancelledError:
            return
        await self._flush_progress(record, cancel_pending=False)

    async def _flush_progress(self, record: AgentRecord, *, cancel_pending: bool) -> None:
        if cancel_pending:
            task = self._flush_tasks.pop(record.agent_id, None)
            if task is not None and not task.done():
                task.cancel()
        buffered = self._progress_buffers.pop(record.agent_id, [])
        if buffered:
            await self._post_thread(record, "… " + " ".join(buffered))

    async def _post_thread(self, record: AgentRecord, content: str) -> None:
        if self._discord is None or not record.discord_thread_id or not content:
            return
        try:
            await self._discord.send_to_channel(int(record.discord_thread_id), content)
        except Exception:  # noqa: BLE001 — a failed post must not kill the bridge
            logger.exception("thread post failed for agent %s", record.agent_id)

    # -- wake path (fixr) ----------------------------------------------------

    async def _on_wake(self, record: AgentRecord, event: AgentEvent) -> None:
        """Decide fixr-vs-human for a wake event.

        With the direct channel active a ``question`` is answered by a human in
        the thread (no fixr turn); fixr is only woken if the human doesn't answer
        within the idle window. ``done/error`` always wake fixr for its report."""
        if event.type == QUESTION and self._direct_channel_active():
            self._arm_idle(record, event)
            return
        self._cancel_idle(record.agent_id)
        await self._wake_fixr(record, event)

    async def _wake_fixr(self, record: AgentRecord, event: AgentEvent) -> None:
        """Enqueue a fixr turn for one event via the normal pipeline."""
        message = _wake_message(record, event)
        try:
            await self._chat_system.generate_response(
                persona_name=global_config.CC_FIXR_PERSONA,
                user_identifier=f"agent:{record.agent_id}",
                channel=global_config.CC_FIXR_CHANNEL,
                message=message,
            )
        except Exception:  # noqa: BLE001 — a failed fixr turn must not kill the bridge
            logger.exception(
                "fixr wake turn failed for agent %s (event %s)",
                record.agent_id, event.type,
            )

    # -- idle fallback -------------------------------------------------------

    def _arm_idle(self, record: AgentRecord, event: AgentEvent) -> None:
        self._cancel_idle(record.agent_id)
        self._idle_timers[record.agent_id] = asyncio.create_task(
            self._idle_fallback(record, event)
        )

    def _cancel_idle(self, agent_id: str) -> None:
        task = self._idle_timers.pop(agent_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def _idle_fallback(self, record: AgentRecord, event: AgentEvent) -> None:
        try:
            await asyncio.sleep(global_config.CC_FIXR_IDLE_MINUTES * 60)
        except asyncio.CancelledError:
            return
        logger.info(
            "agent %s question unanswered after %.0f min — waking fixr",
            record.agent_id, global_config.CC_FIXR_IDLE_MINUTES,
        )
        await self._wake_fixr(record, event)
