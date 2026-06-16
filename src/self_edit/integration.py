# src/self_edit/integration.py
"""ServiceIntegration for the fixr self-improvement supervisor (DP-227).

Owns the dispatch subsystem (registry + Dispatcher) and wires the per-agent
event bridge's wake callback to ``chat_system.generate_response("fixr", …)`` —
so a {question, done, error} event from any dispatched agent enqueues a fixr
turn carrying that event, exactly as a human message would. fixr then reasons
and acts with its tools (answer_agent / send_discord / dispatch_fix / …).

Personas opt in via ``service_bindings: ["fixr"]``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config import global_config
from src.clients.service_integration import ServiceIntegration
from src.self_edit.dispatcher import Dispatcher
from src.self_edit.events import AgentEvent
from src.self_edit.registry import AgentRecord, AgentRegistry

if TYPE_CHECKING:
    from src.chat_system import ChatSystem
    from src.clients.notification import NotificationRouter
    from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)


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
        self.registry = registry or AgentRegistry()
        self.dispatcher = dispatcher or Dispatcher(
            self.registry,
            on_wake=self._on_wake,
            model_arg=global_config.CC_FIXR_MODEL_ARG,
            clone_dir=global_config.CC_FIXR_CLONE_DIR,
        )

    @property
    def name(self) -> str:
        return "fixr"

    def register_tools(self, tool_manager: "ToolManager") -> None:
        from src.self_edit.fixr_tools import FixrToolHandler
        handler = FixrToolHandler(self.dispatcher, self.registry, self._notifier)
        handler.register(tool_manager)

    async def _on_wake(self, record: AgentRecord, event: AgentEvent) -> None:
        """Enqueue a fixr turn for one wake event via the normal pipeline."""
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
