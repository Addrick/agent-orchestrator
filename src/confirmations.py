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

from config.global_config import (
    PARK_REEXECUTION_GUARD_WINDOW, PARK_ROW_RETENTION, PENDING_ACTION_TTL,
)
from src.memory.memory_manager import MemoryManager
from src.security.scrubber import get_scrubber
from src.tools.definitions import get_tool_capabilities
from src.tools.tool_loop import (
    PARK_STATUS_APPROVED, PARK_STATUS_AWAITING, PARK_STATUS_DENIED,
    PARK_STATUS_EXPIRED, PARK_STATUS_FAILED, PARK_STATUS_INTERRUPTED,
    write_call_identity,
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

# The same shape for a park whose resolution was cut short by a restart. It says
# "unknown", not "failed", on purpose: a bare failure invites the retry every
# other `error` in this loop invites, and here a retry is a possible second
# execution of an irreversible action.
INTERRUPTED_INSTRUCTION = (
    "The service restarted while this action was being decided, so whether it "
    "ran is unknown. Do NOT assume either outcome — check the current state "
    "before proposing it again."
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

    @property
    def identity_hash(self) -> str:
        """Storage form of this call's duplicate-detection identity."""
        name, args = write_call_identity(self.write_call)
        return MemoryManager.parked_write_identity(name, args)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ParkedWrite":
        """Rebuild a park from its durable row (DP-319 restart path).

        `duplicate_refs` comes back from JSON as lists, not tuples — converted
        here because `_patch_one` unpacks them positionally and a silent shape
        drift would only surface as an unpatched history entry hours later.
        """
        refs = row.get("duplicate_refs") or []
        return cls(
            token=str(row["token"]),
            write_call=row.get("write_call") or {},
            audit_info=row.get("audit_info") or {},
            confirmation_text=row.get("confirmation_text") or "",
            user_identifier=str(row["user_identifier"]),
            persona_name=str(row["persona_name"]),
            channel=row.get("channel") or "",
            server_id=row.get("server_id"),
            turn_tainted=bool(row.get("turn_tainted")),
            parked_assistant_id=row.get("parked_assistant_id"),
            duplicate_refs=[(int(r[0]), str(r[1])) for r in refs if len(r) == 2],
            created_at=float(row["created_at"]),
        )


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
        self._persist_new(parked)
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
        # DP-319: mark the durable row claimed. The in-memory pop above is
        # still the authority within this process — the DB claim is what stops
        # a park that survived a restart being resolved twice, and what tells a
        # later boot that a decision was in flight when the process died.
        self.memory_manager.claim_parked_write(token)
        return parked

    def restore(self, parked: ParkedWrite) -> None:
        """Put a taken park back (a claim that turned out to be invalid)."""
        self.pending[parked.token] = parked
        self._by_key.setdefault(parked.key, []).append(parked.token)
        if not self.memory_manager.release_parked_write(parked.token):
            # The row is gone or already terminal, so the restored in-memory
            # park would outlive its durable record and vanish on the next
            # restart. Re-insert rather than leave the two stores disagreeing.
            self._persist_new(parked)

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

    # ---- durability (DP-319) ---------------------------------------------
    #
    # The in-memory structures above stay the live index: `take` must remain a
    # pure synchronous pop to keep its atomicity, and the per-conversation locks
    # cannot be persisted at all. The DB is written through on every mutation
    # and read back once, at boot. Single-process by assumption — a second
    # process would need the DB to become the authority, and the locks to move
    # with it.

    def _persist_new(self, parked: ParkedWrite) -> None:
        """Write-through for a park entering (or re-entering) the pending set."""
        self.memory_manager.insert_parked_write(
            token=parked.token,
            created_at=parked.created_at,
            user_identifier=parked.user_identifier,
            persona_name=parked.persona_name,
            channel=parked.channel,
            server_id=parked.server_id,
            write_call=parked.write_call,
            call_identity=parked.identity_hash,
            audit_info=parked.audit_info,
            confirmation_text=parked.confirmation_text,
            turn_tainted=parked.turn_tainted,
            parked_assistant_id=parked.parked_assistant_id,
            duplicate_refs=[list(r) for r in parked.duplicate_refs],
        )

    def note_duplicate_ref(self, parked: ParkedWrite,
                           row_id: int, call_id: str) -> None:
        """Record a suppressed duplicate against a live park, durably.

        The caller used to append straight to `parked.duplicate_refs`, which
        after DP-319 would leave the durable row stale: a restart would reload
        the park without the reference, and the duplicate's history entry would
        keep claiming the action is still awaiting an operator forever.
        """
        parked.duplicate_refs.append((row_id, call_id))
        self.memory_manager.update_parked_write_refs(
            parked.token, duplicate_refs=[list(r) for r in parked.duplicate_refs],
        )

    def already_resolved(self, key: ConversationKey,
                         write_call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The recently-EXECUTED park matching this proposal, if there is one.

        Closes the hole the in-memory store could not: during a continuation the
        park being resolved has already been `take`n, so `list_for` no longer
        contains it and the pending-duplicate guard is blind at exactly the
        moment the model is most likely to re-propose — it is re-reading its own
        tool span. A fresh park would then be created, and approving it would run
        the write a SECOND time.

        Narrow on both axes on purpose. Only `PARK_STATUS_APPROVED` counts (the
        tool actually ran); a denial executed nothing, and DP-297 deliberately
        supports the operator asking for a denied action again. And only inside
        `PARK_REEXECUTION_GUARD_WINDOW`, sized for the continuation turn rather
        than for the park's whole 24h TTL — a day-wide guard would silently
        refuse a legitimate repeat of the same action.
        """
        name, args = write_call_identity(write_call)
        # Typed `Any` deliberately: the isinstance check below is dead code
        # against the declared return type, and it is kept anyway because what
        # it guards is a write the operator never gets offered. A truthy
        # non-row here (a test double, a swapped store) would suppress every
        # park silently, which looks exactly like the gate working.
        row: Any = self.memory_manager.find_resolved_parked_write(
            key[0], key[1], MemoryManager.parked_write_identity(name, args),
            time.time() - PARK_REEXECUTION_GUARD_WINDOW,
            (PARK_STATUS_APPROVED,),
        )
        if row is None:
            return None
        if not isinstance(row, dict):
            # A non-row here would suppress the park — and a suppressed park is
            # a write the operator is never offered, i.e. this guard silently
            # disabling the gate's only affordance. Refuse to act on a shape
            # the store does not promise.
            logger.warning(
                "Resolved-park lookup returned %s, not a row; ignoring it",
                type(row).__name__,
            )
            return None
        return row

    def rebuild_from_store(self) -> Dict[str, int]:
        """Reload durable parks at boot; returns a counts summary.

        Three populations, three different answers:

        - `pending` and still inside its TTL — reinstated, resolvable exactly as
          before the restart.
        - `pending` and past its TTL — expired properly, which means patching the
          history entry and writing the audit row. This is the half the lazy
          sweep can never do after a restart: `sweep_expired` only walks
          `self.pending`, so a park it never loaded is a park it never expires,
          and the model would read `awaiting_human_approval` on every subsequent
          turn and wait forever for a result no code path can produce.
        - `claimed` — a decision was in flight when the process died. NOT
          re-executed: the write may or may not have run, and re-running an
          irreversible call on a guess is worse than either outcome. Terminated
          as `interrupted_by_restart` so the model re-checks state instead.
        """
        counts = {"restored": 0, "expired": 0, "interrupted": 0}
        try:
            rows = self.memory_manager.load_parked_writes(("pending", "claimed"))
        except Exception as e:
            logger.error("Could not reload parked writes at boot: %s", e,
                         exc_info=True)
            return counts

        now = time.time()
        for row in rows:
            try:
                parked = ParkedWrite.from_row(row)
            except (KeyError, TypeError, ValueError) as e:
                logger.error("Skipping unreadable parked write row %s: %s",
                             row.get("token"), e)
                continue

            if row.get("status") == "claimed":
                self._terminate_interrupted(parked)
                counts["interrupted"] += 1
            elif self.is_expired(parked, now):
                self.expire(parked, "Expired while the process was down")
                counts["expired"] += 1
            else:
                self.pending[parked.token] = parked
                self._by_key.setdefault(parked.key, []).append(parked.token)
                counts["restored"] += 1

        if any(counts.values()):
            logger.info(
                "Parked writes reloaded: %d restored, %d expired, %d "
                "interrupted by the restart", counts["restored"],
                counts["expired"], counts["interrupted"],
            )
        try:
            self.memory_manager.purge_parked_writes(now - PARK_ROW_RETENTION)
        except Exception as e:
            logger.warning("Could not purge old parked-write rows: %s", e)
        return counts

    def _terminate_interrupted(self, parked: ParkedWrite) -> None:
        """Close out a park whose resolution died with the process."""
        self.patch_parked_entry(
            parked, PARK_STATUS_INTERRUPTED,
            {"error": INTERRUPTED_INSTRUCTION},
        )
        self.memory_manager.finalize_parked_write(
            parked.token, "interrupted", "Process restarted mid-resolution",
        )
        self.memory_manager.log_audit_event(
            event_type="audit_park_interrupted",
            operator_id=parked.user_identifier,
            prior_state="claimed",
            new_state=PARK_STATUS_INTERRUPTED,
            reason="Process restarted after the decision was claimed; the "
                   "write was NOT re-executed",
            metadata=parked.audit_info,
        )

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
                # `ok` is "did the tool succeed", NOT "did the call return".
                # `ToolManager.execute_tool` never raises — it catches every
                # handler exception and RETURNS `{"error": ...}` — so deriving
                # `ok` from reaching this line recorded a Zammad 500 that fired
                # *after* the ticket was created as a plain `approved`.
                # `PARK_STATUS_FAILED` was therefore unreachable in production,
                # and with it every consumer keyed off it: the "approved but
                # FAILED" continuation line, and the `executed_ok` flag in the
                # audit row. The unit tests missed it because they mock a raise
                # the real ToolManager cannot emit.
                decision.ok = not (isinstance(decision.result, dict)
                                   and "error" in decision.result)
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
        # Terminal, durably: the row survives (the duplicate guard reads it to
        # recognize a re-proposal of an action that already ran) but its payload
        # columns are erased, so the arguments stop living on disk the moment
        # they stop being needed to execute.
        self.memory_manager.finalize_parked_write(
            park.token, "resolved", decision.status,
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
        self.memory_manager.finalize_parked_write(
            parked.token, "expired", reason,
        )
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
    "INTERRUPTED_INSTRUCTION",
]
