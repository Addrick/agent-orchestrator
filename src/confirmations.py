# src/confirmations.py
"""Token-keyed store + lifecycle for writes gated on human approval.

DP-200 slice B extracted this from ChatSystem; DP-297 made it *non-blocking*.
A parked write no longer ends the turn, so one turn can queue several, each
with its own token, each resolvable independently and out of order.

Division of labour: this module owns the pending set, execution of an approved
call, the in-place patch of the parked history entry, expiry, and the audit
trail. The orchestrator (`ChatSystem`) owns the continuation turn that runs
afterwards, because that needs the whole turn lifecycle.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from config.global_config import PENDING_ACTION_TTL
from src.memory.memory_manager import MemoryManager
from src.security.scrubber import get_scrubber
from src.tools.definitions import get_tool_capabilities
from src.tools.tool_loop import (
    PARK_STATUS_APPROVED, PARK_STATUS_AWAITING, PARK_STATUS_DENIED,
    PARK_STATUS_EXPIRED, PARK_STATUS_FAILED,
)
from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)

ConversationKey = Tuple[str, str]

# What a denied write reports back to the model, for as long as the entry
# survives in history. Deliberately more than a verdict: it names the state the
# model should now be in ("wait"), because the verdict alone reads as a
# recoverable tool failure and invites a retry.
DENIAL_INSTRUCTION = (
    "Tool call denied by operator. Wait for corrections or further instruction."
)


@dataclass
class ParkedWrite:
    """One write tool call awaiting an operator decision.

    Exactly one call — not a list. A turn that proposes three writes creates
    three of these, so each can be approved or denied on its own.

    Deliberately carries NO conversation snapshot. The continuation rebuilds
    live history from the DB, which is what lets several parks from one turn be
    resolved in any order without forking the conversation: there is no stale
    copy to replay.
    """
    token: str
    write_call: Dict[str, Any]
    audit_info: Dict[str, Any]
    confirmation_text: str
    user_identifier: str
    persona_name: str
    channel: str = ""
    server_id: Optional[str] = None
    turn_tainted: bool = False
    # The assistant row whose sealed tool_context holds this call's
    # `awaiting_human_approval` entry — the row patched when it resolves.
    parked_assistant_id: Optional[int] = None
    # `(row_id, call_id)` for every write the duplicate guard suppressed in
    # favour of this park. Each left a `duplicate_of_pending` entry saying the
    # action is "still awaiting the operator", so each has to be patched too or
    # history claims something is queued after it was decided. A list because a
    # model can re-propose across several turns, each landing in its own row —
    # which is why one `parked_assistant_id` cannot cover them.
    duplicate_refs: List[Tuple[int, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def key(self) -> ConversationKey:
        return (self.user_identifier, self.persona_name)

    @property
    def call_id(self) -> Optional[str]:
        cid = self.write_call.get("id")
        return str(cid) if cid is not None else None


@dataclass
class Decision:
    """An operator's answer to one park, plus the outcome of acting on it."""
    park: ParkedWrite
    approved: bool
    note: Optional[str] = None
    result: Any = None
    ok: bool = False
    # False when `patch_parked_entry` could not rewrite the history entry, so
    # durable history still reads `awaiting_human_approval` for a write that
    # already ran. `apply()` used to discard that return, leaving only a
    # WARNING — the continuation then read its own proposal as still pending
    # and summarized the wrong outcome, which is precisely what the
    # execute-then-patch ordering exists to prevent.
    patched: bool = True

    @property
    def status(self) -> str:
        """The outcome as durable history records it.

        `approved` and `ok` are different axes: `approved` is what the operator
        decided, `ok` is whether the tool actually ran. Deriving this from
        `approved` alone wrote an approved-then-failed write into history as a
        plain `approved`, so every consumer that keys off the status — and not
        the `error` buried in `result` — read a failure as a success. That is
        the same defect `DENIAL_INSTRUCTION` fixes one branch over: a verdict
        whose real outcome outlives the only place that states it.
        """
        if not self.approved:
            return PARK_STATUS_DENIED
        return PARK_STATUS_APPROVED if self.ok else PARK_STATUS_FAILED


class ConfirmationManager:
    """Orchestrator-owned store for gated writes, keyed by token.

    Was keyed `(user_identifier, persona_name)` and held at most one park —
    a second park for the same pair evicted the first. DP-297 replaced that
    with a token key plus a per-conversation index, so a burst survives intact.
    """

    def __init__(self, tool_manager_lookup: Callable[[], ToolManager],
                 memory_manager: MemoryManager) -> None:
        # A lookup closure (mirrors RequestBuilder.persona_lookup) rather than
        # a bound reference: ToolLoop reads chat_system.tool_manager per call,
        # so a post-init swap must be visible here too or approved writes
        # would execute against the stale manager.
        self._tool_manager_lookup = tool_manager_lookup
        self.memory_manager = memory_manager
        self.pending: Dict[str, ParkedWrite] = {}
        # Insertion-ordered token list per conversation — drives the portal's
        # pending list and Discord's ordering.
        self._by_key: Dict[ConversationKey, List[str]] = {}
        # Decisions acted on but not yet folded into a continuation turn.
        self._queued: Dict[ConversationKey, List[Decision]] = {}
        # One lock per conversation. Serializes execute -> patch -> continue, so
        # two fast approvals cannot run two tool loops over the same history.
        self._locks: Dict[ConversationKey, asyncio.Lock] = {}
        # In-flight off-loop expiry sweeps, held so they are not GC'd.
        self._sweep_tasks: Set["asyncio.Task[None]"] = set()

    # ---- store -----------------------------------------------------------

    def park(self, parked: ParkedWrite) -> None:
        """Store a gated write and log the audit_parked event.

        Nothing is evicted: since DP-297 a second park for the same
        conversation is a sibling, not a replacement, so the
        `audit_parked_evicted` event this used to emit no longer exists.
        """
        self._sweep_off_thread()
        self.pending[parked.token] = parked
        self._by_key.setdefault(parked.key, []).append(parked.token)
        self.memory_manager.log_audit_event(
            event_type="audit_parked",
            operator_id=parked.user_identifier,
            new_state="pending",
            reason="Universal write-audit gate triggered",
            metadata=parked.audit_info,
        )

    def take(self, token: str) -> Optional[ParkedWrite]:
        """Remove and return a park, or None if it is already gone.

        Pure synchronous — no `await` anywhere in it. That is what makes it
        atomic under asyncio and what stops a double-click (or a retried POST)
        from executing the same write twice: only one caller can win the pop.
        """
        parked = self.pending.pop(token, None)
        if parked is None:
            return None
        tokens = self._by_key.get(parked.key)
        if tokens and token in tokens:
            tokens.remove(token)
            if not tokens:
                self._by_key.pop(parked.key, None)
        return parked

    def restore(self, parked: ParkedWrite) -> None:
        """Put a taken park back (a claim that turned out to be invalid)."""
        self.pending[parked.token] = parked
        self._by_key.setdefault(parked.key, []).append(parked.token)

    def list_for(self, user_identifier: str,
                 persona_name: str) -> List[ParkedWrite]:
        """Live parks for one conversation, oldest first."""
        self._sweep_off_thread()
        return [
            self.pending[t]
            for t in self._by_key.get((user_identifier, persona_name), [])
            if t in self.pending
        ]

    def lock_for(self, key: ConversationKey) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())

    def enqueue(self, decision: Decision) -> None:
        self._queued.setdefault(decision.park.key, []).append(decision)

    def drain(self, key: ConversationKey) -> List[Decision]:
        """Take every decision queued for this conversation.

        Called by whichever caller holds the lock. Decisions that arrived while
        it was waiting get folded into its continuation instead of spawning a
        second one — which is why rapid-fire approvals produce one summary and
        deliberate, spaced approvals produce one each.
        """
        return self._queued.pop(key, [])

    # ---- resolution ------------------------------------------------------

    async def apply(self, decision: Decision) -> None:
        """Execute (or refuse) one decided write, then patch its history entry.

        Ordering matters: the patch must land before the continuation rebuilds
        history, or the model reads its own proposal as still pending and
        summarizes the wrong thing.
        """
        park = decision.park
        tool_name = park.write_call.get("name") or "unknown"

        if decision.approved:
            tool_manager = self._tool_manager_lookup()
            try:
                decision.result = await tool_manager.execute_tool(
                    tool_name, **(park.write_call.get("arguments") or {}),
                )
                decision.ok = True
            except Exception as e:
                logger.error(
                    f"Approved write {tool_name} (token {park.token}) "
                    f"failed: {e}", exc_info=True,
                )
                decision.result = {"error": f"Tool execution failed: {e}"}
                decision.ok = False
            if get_tool_capabilities(tool_name).get("produces_untrusted"):
                park.turn_tainted = True
        else:
            # The standing instruction lives HERE, in the patched entry, not in
            # the continuation nudge. The nudge is ephemeral by design, so a
            # denial framed only there decays into a bare `error` one turn
            # later — and a bare `error` is the shape this loop uses everywhere
            # else to mean "the tool failed, adapt and retry". The verdict and
            # what to do about it have the same lifetime as the proposal they
            # describe, because they are the same fact.
            decision.result = {"error": DENIAL_INSTRUCTION,
                               "note": decision.note}
            decision.ok = False

        self.memory_manager.log_audit_event(
            event_type="audit_decision",
            operator_id=park.user_identifier,
            prior_state="pending",
            new_state=decision.status,
            reason=(decision.note or
                    ("Human approved tool execution" if decision.approved
                     else "Human denied tool execution")),
            # No raw `write_call` here. It carried the tool name and arguments
            # a second time — `audit_info["actions"][0]` already has both, plus
            # the irreversibility / sensitivity / enrichment / taint flags that
            # make the row reviewable. The only field the raw copy added was the
            # provider call id, kept below as `call_id` for correlation with the
            # patched tool_context entry.
            #
            # It existed because it was the *execution* payload, not because the
            # audit needed it. The sink scrubs now either way, so this is
            # defence in depth rather than the fix — but a field whose only
            # distinguishing property was "unredacted" should not be written to
            # a permanent store at all.
            metadata={
                "audit_info": park.audit_info,
                "turn_tainted": park.turn_tainted,
                "token": park.token,
                "call_id": park.call_id,
                "executed_ok": decision.ok,
            },
        )
        decision.patched = self.patch_parked_entry(
            park, decision.status, decision.result,
        )
        if not decision.patched:
            logger.error(
                "History entry for %s (token %s) could not be patched; the "
                "write already ran and durable history still reads pending. "
                "Audit row carries the real outcome (executed_ok=%s).",
                tool_name, park.token, decision.ok,
            )

    def patch_parked_entry(self, park: ParkedWrite, status: str,
                           result: Any) -> bool:
        """Rewrite this call's entry inside an already-committed row's
        tool_context, flipping `awaiting_human_approval` to the real outcome.

        Safe to do in place because the park appended a *real* synthetic tool
        result when it was created, so the sealed blob contains that entry
        verbatim — there is no synthesized placeholder to collide with and the
        target is guaranteed present. (Before DP-297 the seal invented the
        entry at write time, which is why this could not be done then.)

        Also patches every suppressed duplicate of this park. Those entries say
        the action is "still awaiting the operator", and nothing else would ever
        correct them — leaving history asserting a decided action is queued.
        They live in whichever row the re-proposal landed in, which is why this
        walks `duplicate_refs` rather than one row.
        """
        primary_ok = False
        row_id = park.parked_assistant_id
        call_id = park.call_id
        if row_id is not None and call_id is not None:
            primary_ok = self._patch_one(
                park, row_id, call_id, status, result, duplicate=False)

        for dup_row_id, dup_call_id in park.duplicate_refs:
            # Best-effort: a stale duplicate is a cosmetic history wart, while a
            # failed primary patch is the real defect. Never let one shadow the
            # other in the return value.
            self._patch_one(park, dup_row_id, dup_call_id, status, result,
                            duplicate=True)

        return primary_ok

    def _patch_one(self, park: ParkedWrite, row_id: int, call_id: str,
                   status: str, result: Any, *, duplicate: bool) -> bool:
        """Rewrite a single tool entry in a single row's tool_context."""
        blob = self.memory_manager.get_tool_context(row_id)
        if not blob:
            logger.warning(
                "park %s: assistant row %s has no tool_context to patch",
                park.token, row_id,
            )
            return False
        try:
            msgs = json.loads(blob)
        except (ValueError, TypeError):
            logger.error("park %s: row %s tool_context is not valid JSON",
                         park.token, row_id)
            return False

        for msg in msgs:
            if msg.get("role") == "tool" and msg.get("tool_call_id") == call_id:
                entry: Dict[str, Any] = {
                    "status": status,
                    "token": park.token,
                    # Egress scrub (DP-225 boundary 1): this result reaches the
                    # model's replayed history and the portal transcript, so it
                    # is redacted exactly like a live tool result.
                    "result": get_scrubber().scrub(result),
                }
                if duplicate:
                    # Marked so the transcript explains why one outcome appears
                    # against two call ids, rather than reading as the action
                    # having happened twice.
                    entry["duplicate_of"] = park.call_id
                msg["content"] = json.dumps(entry)
                break
        else:
            logger.warning(
                "park %s: no tool entry %s found in row %s — history will keep "
                "showing it as pending", park.token, call_id, row_id,
            )
            return False

        return self.memory_manager.set_tool_context(row_id, json.dumps(msgs))

    # ---- expiry ----------------------------------------------------------

    def sweep_expired(self, now: Optional[float] = None) -> int:
        """Drop and patch every park past its TTL. Returns how many.

        Swept lazily from `park`, `list_for` and the resolve path rather than
        by a background task — a periodic loop here would re-introduce exactly
        the shutdown-contract problem DP-304 just fixed, for a deadline that
        does not need second-level precision.

        An expiry fires NO continuation: an unprompted summary hours after the
        operator walked away is noise. The patched entry is enough — the model
        sees the outcome next time it speaks.
        """
        stale = self._take_expired(now)
        for parked in stale:
            self.expire(parked, f"No decision within {PENDING_ACTION_TTL}s")
        return len(stale)

    def _take_expired(self, now: Optional[float] = None) -> List[ParkedWrite]:
        """Remove every past-TTL park from the store. Pure in-memory.

        Split out from the DB half so the hot paths (`park`, `list_for`) can
        evict synchronously — which must stay atomic, like `take` — without
        also running SELECT + UPDATE + INSERT inline. Those calls sit inside
        the token stream and the SSE routes, where a single day-old park was
        enough to stall every other stream on the loop.
        """
        now = time.time() if now is None else now
        tokens = [t for t, p in self.pending.items()
                  if now - p.created_at > PENDING_ACTION_TTL]
        stale = [p for p in (self.take(t) for t in tokens) if p is not None]
        if stale:
            logger.info("Expired %d unanswered gated write(s)", len(stale))
        return stale

    def _sweep_off_thread(self) -> None:
        """Evict expired parks now; do their DB writes off the event loop.

        Fire-and-forget by design — an expiry fires no continuation, so nothing
        downstream waits on the patch. Falls back to inline when there is no
        running loop (sync callers, tests).
        """
        stale = self._take_expired()
        if not stale:
            return
        reason = f"No decision within {PENDING_ACTION_TTL}s"

        def _finish() -> None:
            for parked in stale:
                self.expire(parked, reason)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _finish()
            return
        task = loop.create_task(asyncio.to_thread(_finish))
        # Hold a reference: a bare create_task can be GC'd mid-flight.
        self._sweep_tasks.add(task)
        task.add_done_callback(self._sweep_tasks.discard)

    def expire(self, parked: ParkedWrite, reason: str) -> None:
        """Terminate an already-taken park as expired: patch, then audit.

        Shared by the lazy sweep and the resolve path. The click path used to
        inline the patch with a hardcoded "expired" and log nothing, which made
        it the only park-terminating path leaving no audit trail — so the fact
        that a human actually tried to approve an expired irreversible action
        was recorded nowhere, and its writer was decoupled from the constant
        every other consumer keys off.
        """
        self.patch_parked_entry(parked, PARK_STATUS_EXPIRED,
                                {"reason": "expired before review"})
        self.memory_manager.log_audit_event(
            event_type="audit_park_expired",
            operator_id=parked.user_identifier,
            prior_state="pending",
            new_state=PARK_STATUS_EXPIRED,
            reason=reason,
            metadata=parked.audit_info,
        )

    def is_expired(self, parked: ParkedWrite,
                   now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        return now - parked.created_at > PENDING_ACTION_TTL


def new_token() -> str:
    """Stable per-park handle, surfaced as `ephemeral_chunk_id` to surfaces."""
    return uuid.uuid4().hex


__all__ = [
    "ConfirmationManager", "ParkedWrite", "Decision", "new_token",
    "PARK_STATUS_AWAITING", "PARK_STATUS_FAILED", "DENIAL_INSTRUCTION",
]
