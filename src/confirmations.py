# src/confirmations.py
"""Write-confirmation parking for the universal audit gate (DP-200 slice B).

Extracted from ChatSystem: the orchestrator owns *when* a turn parks and
resumes, but the confirmation store, eviction/decision audit logging, and the
approved/denied write execution live here. ToolLoop stays stateless across the
park — it surfaces pending writes and this manager carries them to the resume.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.memory.memory_manager import MemoryManager
from src.tools.definitions import get_tool_capabilities
from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)


@dataclass
class PendingConfirmation:
    """Stores state for a tool call awaiting user approval.

    All write-tool calls are parked here for audit before execution,
    regardless of execution mode. The audit_info dict carries structured
    metadata (irreversibility flags, taint sources, model reasoning) so
    the approval surface can present an informed review.
    """
    write_calls: List[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    persona_name: str
    tools_for_llm: List[Dict[str, Any]]
    image_url: Optional[str]
    channel: str = ""
    server_id: Optional[str] = None
    turn_tainted: bool = False
    audit_info: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    # Index into conversation_history where this turn's tool messages start.
    # Passed to the resumed tool loop as history_start_override so the approved
    # write and its result are captured into tool_context_json (and thus
    # replayed on later turns) instead of being dropped.
    tool_context_start: int = 0
    # DP-130 history contract: stable handle for the rendered-but-unpersisted
    # confirmation chunk. Surfaced as `ephemeral_chunk_id` in the SSE id-frame
    # and the transcript projection, and as the correlation token an interactive
    # surface (portal) sends back on approve/deny so a resume can be matched to
    # *this* park. (Reconciles with the DP-127-engine confirm-modal `token`.)
    token: str = field(default_factory=lambda: uuid.uuid4().hex)
    # The parked confirmation's rendered text — projected as the ephemeral
    # chunk's content so a fresh page load (transcript) can render the pending
    # approval without a DB row.
    confirmation_text: str = ""
    # Retry linkage: when the parked turn was itself a portal retry, this is
    # the archived assistant row the resumed continuation must UPDATE in
    # place. Without it the resume would INSERT a fresh assistant row and
    # strand the archived one with its pre-retry content.
    retry_assistant_id: Optional[int] = None


class ConfirmationManager:
    """Orchestrator-owned store + lifecycle for parked write confirmations.

    Keyed (user_identifier, persona_name) — at most one pending write per
    conversation pair; a newer park supersedes (and audit-logs the eviction
    of) the old one.
    """

    def __init__(self, tool_manager_lookup: Callable[[], ToolManager],
                 memory_manager: MemoryManager) -> None:
        # A lookup closure (mirrors RequestBuilder.persona_lookup) rather than
        # a bound reference: ToolLoop reads chat_system.tool_manager per call,
        # so a post-init swap must be visible here too or approved writes
        # would execute against the stale manager.
        self._tool_manager_lookup = tool_manager_lookup
        self.memory_manager = memory_manager
        self.pending: Dict[Tuple[str, str], PendingConfirmation] = {}

    def park(self, user_identifier: str, persona_name: str,
             parked: PendingConfirmation) -> None:
        """Store a parked write and log the audit_parked event.

        A still-pending write for this (user, persona) is silently superseded
        by this one — the operator's eventual approve/deny would resolve the
        wrong intent. We can't queue both (the key is 1:1), so at minimum the
        eviction must leave an audit trail.
        """
        key = (user_identifier, persona_name)
        evicted = self.pending.get(key)
        if evicted is not None:
            self.memory_manager.log_audit_event(
                event_type="audit_parked_evicted",
                operator_id=user_identifier,
                prior_state="pending",
                new_state="evicted",
                reason="Superseded by a newer pending write for the same persona",
                metadata=evicted.audit_info,
            )
        self.pending[key] = parked
        # Phase 7: Log audit parking
        self.memory_manager.log_audit_event(
            event_type="audit_parked",
            operator_id=user_identifier,
            new_state="pending",
            reason="Universal write-audit gate triggered",
            metadata=parked.audit_info,
        )

    async def execute_write_calls(
            self,
            write_calls: List[Dict[str, Any]],
            conversation_history: List[Dict[str, Any]]
    ) -> None:
        """Execute write tool calls and append results to history."""
        # Resolve once per batch — all calls in one decision execute against
        # the same manager, even if it is swapped mid-batch.
        tool_manager = self._tool_manager_lookup()
        for call_item in write_calls:
            tool_name: str = call_item.get("name", "")
            tool_args = call_item.get("arguments", {})
            tool_result = await tool_manager.execute_tool(tool_name, **tool_args)
            conversation_history.append({
                "role": "tool",
                "tool_call_id": call_item.get("id"),
                "name": tool_name,
                "content": json.dumps(tool_result)
            })

    @staticmethod
    def append_denied_tool_results(
            write_calls: List[Dict[str, Any]],
            conversation_history: List[Dict[str, Any]]
    ) -> None:
        """Appends denial results for write tools the user rejected."""
        for call_item in write_calls:
            conversation_history.append({
                "role": "tool",
                "tool_call_id": call_item.get("id"),
                "name": call_item.get("name"),
                "content": json.dumps({"error": "Tool call denied by user"})
            })

    async def apply_resume_decision(
            self,
            pending: PendingConfirmation,
            approved: bool,
            conversation_history: List[Dict[str, Any]],
            *,
            operator_id: str,
            turn_tainted: bool,
    ) -> bool:
        """Apply an approve/deny decision to the parked turn before continuation.

        On approval the write calls execute and their results are appended to
        the parked history (so they precede the model's continuation); denial
        appends synthetic denial results instead. Either way the audit decision
        is logged. Read-side taint is recomputed from the executed write calls
        and folded into the returned `turn_tainted` so the continuation loop
        inherits it.
        """
        if approved:
            await self.execute_write_calls(pending.write_calls, conversation_history)
            for wc in pending.write_calls:
                wc_name = wc.get("name") or "unknown"
                if get_tool_capabilities(wc_name).get("produces_untrusted"):
                    turn_tainted = True
            decision_state = "approved"
            decision_reason = "Human approved tool execution"
        else:
            self.append_denied_tool_results(pending.write_calls, conversation_history)
            decision_state = "denied"
            decision_reason = "Human denied tool execution"

        # Phase 7: Log audit decision
        self.memory_manager.log_audit_event(
            event_type="audit_decision",
            operator_id=operator_id,
            prior_state="pending",
            new_state=decision_state,
            reason=decision_reason,
            metadata={
                "write_calls": pending.write_calls,
                "audit_info": pending.audit_info,
                "turn_tainted": turn_tainted,
            },
        )
        return turn_tainted
