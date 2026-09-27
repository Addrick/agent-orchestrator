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
from typing import AbstractSet, Any, Callable, Dict, List, Optional, Set, Tuple

from config.global_config import (
    PARK_PURGE_INTERVAL, PARK_REEXECUTION_GUARD_WINDOW, PARK_ROW_RETENTION,
    PENDING_ACTION_TTL,
)
from src.deferral_kinds import (
    DEFERRAL_DECLARATION_KEY, DEFERRAL_KIND_APPROVAL, declared_deferral,
)
from src.memory.memory_manager import (
    PARK_DB_CLAIMED, PARK_DB_EXPIRED, PARK_DB_INTERRUPTED, PARK_DB_LIVE,
    PARK_DB_PENDING, PARK_DB_QUARANTINED, PARK_DB_RESOLVED, PARK_DB_UNKNOWN,
    MemoryManager,
)
from src.security.scrubber import get_scrubber
from src.tools.definitions import get_tool_capabilities
from src.tools.tool_loop import (
    PARK_STATUS_APPROVED, PARK_STATUS_AWAITING, PARK_STATUS_DENIED,
    PARK_STATUS_EXPIRED, PARK_STATUS_FAILED, PARK_STATUS_INTERRUPTED,
    PARK_STATUS_QUARANTINED, awaiting_status, write_call_identity_hash,
)
from src.tools.tool_manager import ToolManager, tool_error

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

# The same shape for a non-approval deferral caught mid-settle by a restart.
# Nothing was being *decided* — the work had already run elsewhere and the
# process died while recording what it did — so the approval wording above would
# describe a review that never happened. The instruction is the same, and for
# the same reason: the outcome is genuinely unknown to us, and a bare failure
# invites a retry that would re-run whatever the authority already did.
#
# This is where a per-kind boot reconciler would go once a kind can be
# re-queried (a node job can: `job_status` still answers). Until a kind actually
# supplies one, telling the model to re-check is the honest answer rather than a
# hook with no implementation behind it.
INTERRUPTED_DEFERRAL_INSTRUCTION = (
    "The service restarted while this action's result was being recorded, so "
    "whether it completed is unknown. Do NOT assume either outcome — check the "
    "current state before proposing it again."
)


@dataclass
class ParkedWrite:
    """One deferred tool call — a call whose real result arrives after the turn.

    Exactly one call — not a list. A turn that proposes three writes creates
    three of these, so each can be approved or denied on its own.

    Deliberately carries NO conversation snapshot. The continuation rebuilds
    live history from the DB, which is what lets several parks from one turn be
    resolved in any order without forking the conversation: there is no stale
    copy to replay.

    **The coordinates below are the point of DP-345.** `user_identifier`,
    `persona_name`, `channel` and `server_id` are taken from the turn that
    raised the deferral, so resolving it needs no configuration at all — the row
    already knows where to answer. Both subsystems that re-derived this
    mechanism (fixr, DP-343) discarded the turn and re-asserted those same four
    facts from env vars, which is how five settings came to exist naming a
    persona, a channel and a user. A deferral kind that has to be *told* where
    to reply is a deferral kind that threw away state it was handed.
    """
    token: str
    write_call: Dict[str, Any]
    audit_info: Dict[str, Any]
    confirmation_text: str
    user_identifier: str
    persona_name: str
    channel: str = ""
    server_id: Optional[str] = None
    # Which external event answers this call. `approval` is a human clicking a
    # token; other kinds are answered by whatever authority owns the work.
    #
    # There is no companion `handle` field naming the pending thing the way
    # that authority knows it, because there is no second identifier: derpr
    # mints the token and hands that same string outward as the job id (the
    # node validates it against `^[a-z0-9][a-z0-9-]{0,63}$`, which a 32-char
    # lowercase-hex token satisfies). The handle IS the token, so every kind
    # is claimed with plain `take(token)`.
    kind: str = DEFERRAL_KIND_APPROVAL
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
    # False when the durable row was refused (unserializable arguments, a store
    # error). The park is still live in memory — pre-DP-319 behaviour — but its
    # re-execution guard cannot read a row that does not exist, so `apply` has
    # to remember the outcome in-process instead.
    persisted: bool = True

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
        return write_call_identity_hash(self.write_call)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ParkedWrite":
        """Rebuild a park from its durable row (DP-319 restart path).

        `duplicate_refs` comes back from JSON as lists, not tuples — converted
        here because `_patch_one` unpacks them positionally and a silent shape
        drift would only surface as an unpatched history entry hours later.

        A missing or undecodable `write_call` is rejected outright rather than
        defaulted to `{}`. An empty call still renders a perfectly normal-looking
        approve/deny affordance, but approving it executes tool "unknown" and its
        `call_id` is None — so `patch_parked_entry` cannot patch anything and the
        history entry reads `awaiting_human_approval` forever. That orphan state
        is exactly what the boot-time expiry branch exists to eliminate; a row
        whose payload cannot be read must be quarantined, not restored.

        An unreadable `duplicate_refs` is NOT fatal, deliberately: quarantining
        over it would destroy a perfectly executable park to protect a cosmetic
        pointer list. But it is not silent either. Every reference lost here is
        a `duplicate_of_pending` entry saying the action is "still awaiting the
        operator" that nothing will ever correct — permanently wrong history,
        and this log line is the only signal it happened.
        """
        undecodable = row.get("_undecodable") or []
        if "write_call" in undecodable:
            raise ValueError("write_call did not decode")
        write_call = row.get("write_call")
        if not isinstance(write_call, dict) or not write_call:
            raise ValueError("write_call is missing or empty")
        refs = row.get("duplicate_refs") or []
        wrong_shape = not isinstance(refs, list)
        if wrong_shape:
            # JSON that decoded to a scalar or an object: not iterable as pairs,
            # and iterating it would raise straight into the quarantine branch.
            refs = []
        # Each ref is converted defensively rather than in a comprehension.
        # `len(r)` raises TypeError on a bare int and `int(r[0])` raises
        # ValueError on a non-numeric row id — both escaped `from_row` and were
        # caught by `_reconcile_row`'s (KeyError, TypeError, ValueError) handler,
        # which QUARANTINES the park. That is the exact opposite of the rule
        # stated two paragraphs up: a decodable-but-malformed pointer list would
        # have destroyed a perfectly executable irreversible write to protect a
        # cosmetic list. Malformed entries are dropped and reported instead.
        kept: List[Tuple[int, str]] = []
        for r in refs:
            try:
                if len(r) == 2:
                    kept.append((int(r[0]), str(r[1])))
            except (TypeError, ValueError):
                continue
        if ("duplicate_refs" in undecodable or wrong_shape
                or len(kept) != len(refs)):
            logger.error(
                "Parked write %s lost suppressed-duplicate pointers at load "
                "(%d of %d usable%s). Their history entries will keep claiming "
                "the action is awaiting an operator; nothing else corrects "
                "them.",
                row.get("token"), len(kept), len(refs),
                "; the column did not decode"
                if "duplicate_refs" in undecodable else "",
            )
        return cls(
            token=str(row["token"]),
            write_call=write_call,
            audit_info=row.get("audit_info") or {},
            confirmation_text=row.get("confirmation_text") or "",
            user_identifier=str(row["user_identifier"]),
            persona_name=str(row["persona_name"]),
            channel=row.get("channel") or "",
            server_id=row.get("server_id"),
            # Defaulted, not `row["kind"]`: rows written before the DP-345
            # migration have no such column, and the whole population they
            # represent is approvals.
            kind=str(row.get("kind") or DEFERRAL_KIND_APPROVAL),
            turn_tainted=bool(row.get("turn_tainted")),
            parked_assistant_id=row.get("parked_assistant_id"),
            duplicate_refs=kept,
            created_at=float(row["created_at"]),
        )


@dataclass
class Decision:
    """One deferral's answer, plus the outcome of acting on it.

    Named for the approval kind because that is the only kind whose answer is a
    *decision*. For every other kind the answer is a report: the work already
    ran somewhere else, and `outcome_status` / `result` carry what it did.
    """
    park: ParkedWrite
    approved: bool
    note: Optional[str] = None
    result: Any = None
    ok: bool = False
    # DP-345: the outcome as supplied by a non-approval kind's authority. When
    # set it wins over the derivation below, because there was no operator whose
    # verdict could be derived from — the node said `done` or `failed` and that
    # is the whole of it. Must never be an `awaiting:` value: this is the status
    # that REPLACES the placeholder, so writing another awaiting status here
    # would leave the call pending forever with its row already terminal.
    outcome_status: Optional[str] = None
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
        if self.outcome_status is not None:
            return self.outcome_status
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
                 memory_manager: MemoryManager,
                 offered_lookup: Callable[[str], AbstractSet[str]]) -> None:
        # A lookup closure (mirrors RequestBuilder.persona_lookup) rather than
        # a bound reference: ToolLoop reads chat_system.tool_manager per call,
        # so a post-init swap must be visible here too or approved writes
        # would execute against the stale manager.
        self._tool_manager_lookup = tool_manager_lookup
        # DP-404: persona name -> the tool names that persona is offered NOW.
        # An approved park re-derives it at execution time rather than trusting
        # the turn that parked it, so a binding or allowlist narrowed while the
        # park waited (or a durable row restored after a restart) cannot run.
        self._offered_lookup = offered_lookup
        self.memory_manager = memory_manager
        self.pending: Dict[str, ParkedWrite] = {}
        # Insertion-ordered token list per conversation — drives the portal's
        # pending list and Discord's ordering.
        self._by_key: Dict[ConversationKey, List[str]] = {}
        # No second index keyed by the external event's own name for the work:
        # the token IS that name (see `ParkedWrite.kind`), so a ping resolves
        # through `self.pending` like a click does. DP-343's in-process
        # `OrderedDict` of job ids is what that replaces, and it was lost on
        # restart; `rebuild_from_store` reloads `pending` from the rows.

        # Decisions acted on but not yet folded into a continuation turn.
        self._queued: Dict[ConversationKey, List[Decision]] = {}
        # One lock per conversation. Serializes execute -> patch -> continue, so
        # two fast approvals cannot run two tool loops over the same history.
        self._locks: Dict[ConversationKey, asyncio.Lock] = {}
        # In-flight off-loop expiry sweeps, held so they are not GC'd.
        self._sweep_tasks: Set["asyncio.Task[None]"] = set()
        # When the retention purge last ran. Starts at "never", so the first
        # sweep after boot does one — a process that is restarted often would
        # otherwise never reach the interval.
        self._last_purge: float = 0.0
        # Outcomes of parks whose durable row never existed, keyed
        # `(conversation, identity hash) -> (resolved_at, resolution)`. The
        # DB-backed duplicate guard cannot see those, and a guard that fails
        # open here means an approved irreversible write can be re-proposed and
        # executed a SECOND time — the one outcome this subsystem exists to
        # prevent. Bounded by the same guard window as the durable lookup, and
        # process-lifetime by definition: a park that was never persisted has
        # nothing to survive a restart with anyway.
        self._resolved_fallback: Dict[Tuple[ConversationKey, str],
                                      Tuple[float, str]] = {}

    # ---- store -----------------------------------------------------------

    def park(self, parked: ParkedWrite) -> None:
        """Store a gated write and log the audit_parked event.

        Nothing is evicted: since DP-297 a second park for the same
        conversation is a sibling, not a replacement, so the
        `audit_parked_evicted` event this used to emit no longer exists.

        A refused durable write does NOT refuse the park — the operator still
        gets the affordance, exactly as before DP-319. It is recorded on the
        park instead, because the consequence is not "it will not survive a
        restart": with no row, `finalize_parked_write` matches nothing and
        `already_resolved` finds nothing, so the re-execution guard is silently
        off for this call. `apply` closes that with an in-process fallback.
        """
        self._sweep_off_thread()
        # Through `_reinstate`, not a bare append: `_by_key` holding one token
        # twice makes `list_for` yield the same park twice (it filters on
        # membership, not uniqueness), which the portal renders as two pending
        # chunks sharing an `ephemeral_chunk_id`. `_reinstate` was made
        # idempotent for exactly that reason and this path was left unguarded.
        self._reinstate(parked)
        parked.persisted = self._persist_new(parked)
        if not parked.persisted:
            logger.error(
                "Parked write %s (%s) has no durable row: it will not survive a "
                "restart, and its double-execution guard falls back to "
                "in-process memory only.",
                parked.token, parked.write_call.get("name"),
            )
        self.memory_manager.log_audit_event(
            event_type="audit_parked",
            operator_id=parked.user_identifier,
            new_state="pending",
            reason="Universal write-audit gate triggered",
            metadata=parked.audit_info,
        )

    def _pop(self, token: str) -> Optional[ParkedWrite]:
        """Remove a park from the in-memory index only. No DB, no await.

        The atomic half of `take`. Split out because the expiry sweep must be
        able to evict synchronously on the hot paths without also running a
        commit inline — and because a sweep that marks rows `claimed` before
        its off-thread half finishes turns a crash in that window into a park
        that reboots as "a decision was in flight" when nobody ever saw it.
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

    def take(self, token: str) -> Optional[ParkedWrite]:
        """Remove and return a park, or None if it is already gone.

        Synchronous — no `await` anywhere in it. That is what makes it atomic
        under asyncio and what stops a double-click (or a retried POST) from
        executing the same write twice: only one caller can win the pop.

        The DB claim that follows the pop is what stops a park that survived a
        restart being resolved twice, and what tells a later boot that a
        decision was in flight when the process died. It cannot raise —
        `claim_parked_write` swallows `sqlite3.Error` — because by the time it
        runs the pop has already happened, so an exception here would destroy
        the park in memory while its row stayed `pending`.
        """
        parked = self._pop(token)
        if parked is None:
            return None
        if not self.memory_manager.claim_parked_write(token):
            # Not fatal — the in-memory pop is the authority in this process —
            # but it means the durable row is missing or already terminal, so
            # the restart path will not agree with what happens next.
            logger.warning(
                "park %s: durable row could not be claimed (missing or already "
                "terminal); resolving from memory only", token,
            )
        return parked

    def restore(self, parked: ParkedWrite) -> None:
        """Put a taken park back (a claim that turned out to be invalid).

        The durable half is deliberately not a blind re-INSERT. `release`
        returning False is ambiguous between "no row" and "row already
        terminal", and `INSERT OR REPLACE` writes `status='pending'` over the
        whole row — so treating the two alike rewound `resolved_at` and
        `resolution` to NULL and resurrected a decided, already-executed write
        as an approvable affordance that survived the next restart. That is the
        precise outcome durable parks exist to prevent, so a terminal row wins
        over the in-memory restore instead of the other way round.

        Three answers, and "could not tell" fails CLOSED. A transient
        `database is locked` makes `release_parked_write` answer False and the
        status read answer nothing — and if that is read as "no row", the
        re-INSERT rewrites `status='pending'` and NULLs `resolved_at`, which is
        the resurrection this branch exists to forbid, reached by a route that
        looks like the safe one. Dropping a park that may have been live is
        recoverable by a human; re-offering an executed irreversible write is
        not.
        """
        if self.memory_manager.release_parked_write(parked.token):
            self._reinstate(parked)
            return

        status = self.memory_manager.get_parked_write_status(parked.token)
        if status == PARK_DB_UNKNOWN:
            logger.error(
                "park %s: the store could not say what state its row is in, so "
                "it is NOT being restored. Re-inserting on a failed read can "
                "resurrect an already-executed write as approvable.",
                parked.token,
            )
        elif status is None:
            # Genuinely gone: re-insert, or the restored park would outlive its
            # durable record and vanish on the next restart.
            self._reinstate(parked)
            parked.persisted = self._persist_new(parked)
        elif status in PARK_DB_LIVE:
            # Already `pending` (a concurrent release, or it was never claimed).
            self._reinstate(parked)
        else:
            logger.error(
                "park %s: refusing to restore — its durable row is already %s. "
                "The decision stands; it must not become approvable again.",
                parked.token, status,
            )

    def _reinstate(self, parked: ParkedWrite) -> None:
        """Put a park back into the in-memory index, without touching the DB.

        Idempotent. `self.pending[token] = parked` overwrites harmlessly, but an
        unconditional append to `_by_key` does not: a second `rebuild_from_store`
        (or a rebuild over a manager that already holds the park) left the token
        in the list twice, and `list_for` filters on membership rather than
        uniqueness — so it yielded the same park twice, and the portal rendered
        two pending chunks sharing one `ephemeral_chunk_id`.
        """
        self.pending[parked.token] = parked
        tokens = self._by_key.setdefault(parked.key, [])
        if parked.token not in tokens:
            tokens.append(parked.token)

    def list_for(self, user_identifier: str, persona_name: str,
                 kind: str = DEFERRAL_KIND_APPROVAL) -> List[ParkedWrite]:
        """Live deferrals of one kind for one conversation, oldest first.

        Defaults to `approval` rather than to "everything", and that default is
        the fail-closed one. Every caller of this method renders an approve/deny
        affordance — the portal's pending list, Discord's re-post of unanswered
        proposals, the kobold adapter's transcript. A node job or an agent
        dispatch has no decision for a human to make, so returning it here would
        put a button in front of the operator that resolves a deferral nobody
        was asked about, and `apply()` would then execute `install_model` a
        second time. A new kind becomes clickable only when a surface asks for
        it by name.
        """
        self._sweep_off_thread()
        return [
            self.pending[t]
            for t in self._by_key.get((user_identifier, persona_name), [])
            if t in self.pending and self.pending[t].kind == kind
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

    def _persist_new(self, parked: ParkedWrite) -> bool:
        """Write-through for a park entering (or re-entering) the pending set.

        Returns whether the row actually landed. The caller must not discard
        that: `insert_parked_write` refuses a call it cannot serialize
        losslessly, and a park with no row is one the durable duplicate guard
        cannot see.
        """
        return self.memory_manager.insert_parked_write(
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
            kind=parked.kind,
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
        if not self.memory_manager.update_parked_write_duplicate_refs(
                parked.token, [list(r) for r in parked.duplicate_refs]):
            # Discarding this answer defeated the entire reason the caller was
            # routed through the manager instead of appending to the list: the
            # in-memory park now carries a pointer its durable row does not, so
            # a restart reloads the park without it and the duplicate's history
            # entry claims the action is awaiting an operator forever — the
            # failure this method exists to prevent, silently.
            logger.error(
                "Parked write %s: suppressed-duplicate pointer (row %s, call "
                "%s) did not reach the durable row. It will be lost on a "
                "restart and that entry will keep reading as pending.",
                parked.token, row_id, call_id,
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

        Narrow on purpose, on three axes.

        The outcome must be one where the tool RAN. That is both
        `PARK_STATUS_APPROVED` and `PARK_STATUS_FAILED`: `apply()` records
        `approved_but_failed` whenever `execute_tool` *raises*, which covers the
        ticket that was created before the API returned 500 and the write that
        landed before the client timed out. Treating that as "nothing happened"
        is what re-opened the double-execution hole for the worst case — the
        operator sees "it failed", approves the re-proposal, and gets two
        tickets. A denial is genuinely different: nothing ran, and DP-297
        deliberately supports asking for a denied action again.

        Only inside `PARK_REEXECUTION_GUARD_WINDOW`, sized for the continuation
        turn rather than for the park's whole 24h TTL.

        And only ON a continuation turn — see `ChatSystem._already_resolved`,
        which is where that scoping lives, because this manager cannot see what
        kind of turn is running.
        """
        identity = write_call_identity_hash(write_call)
        # Typed `Any` deliberately: the isinstance check below is dead code
        # against the declared return type, and it is kept anyway because what
        # it guards is a write the operator never gets offered. A truthy
        # non-row here (a test double, a swapped store) would suppress every
        # park silently, which looks exactly like the gate working.
        since = time.time() - PARK_REEXECUTION_GUARD_WINDOW
        row: Any = self.memory_manager.find_resolved_parked_write(
            key[0], key[1], identity, since,
            (PARK_STATUS_APPROVED, PARK_STATUS_FAILED),
        )
        if row is None:
            return self._resolved_fallback_row(key, identity, since)
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

    def _remember_unpersisted_outcome(self, park: ParkedWrite, status: str,
                                      when: float) -> None:
        """Record a decided park that has no durable row to be found in.

        `finalize_parked_write` answering False means the row is missing (or was
        never written), so `find_resolved_parked_write` will answer None for
        this call forever. Without this the model can re-propose the same
        irreversible write on the continuation turn, get a fresh park, and an
        approval executes it twice — the failure DP-319 exists to close, reached
        by the one path where durability was never achieved.

        Prunes as it writes. Entries were only ever dropped when the SAME key
        was read back and found stale, so a park that was never re-proposed —
        the common case — left its entry for the life of the process, and the
        map grew without the bound its own comment claims.
        """
        cutoff = when - PARK_REEXECUTION_GUARD_WINDOW
        for stale_key in [k for k, (at, _) in self._resolved_fallback.items()
                          if at < cutoff]:
            self._resolved_fallback.pop(stale_key, None)
        self._resolved_fallback[(park.key, park.identity_hash)] = (when, status)

    def _resolved_fallback_row(self, key: ConversationKey, identity: str,
                               since: float) -> Optional[Dict[str, Any]]:
        """The in-process twin of `find_resolved_parked_write`, row-shaped.

        Same window and same shape as the durable lookup, so the caller cannot
        tell which one answered. Expired entries are dropped as they are read —
        the map is tiny (only parks whose row was refused) and bounded by the
        guard window, so nothing else needs to sweep it.
        """
        entry = self._resolved_fallback.get((key, identity))
        if entry is None:
            return None
        resolved_at, resolution = entry
        if resolved_at < since:
            self._resolved_fallback.pop((key, identity), None)
            return None
        return {
            "token": None,
            "status": PARK_DB_RESOLVED,
            "resolution": resolution,
            "resolution_reason": "Decided in this process; the park had no "
                                 "durable row",
            "resolved_at": resolved_at,
        }

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
          Only the *resolve* path ever claims a row; the expiry sweep pops in
          memory and does its DB work afterwards, so a crash mid-sweep leaves a
          `pending` row that expires normally here rather than a `claimed` one
          that would fabricate a decision nobody made.

        Every per-row step is wrapped. Boot is the wrong place to be strict: an
        unreadable or unpatchable row raising out of here takes
        `create_chat_system` with it, and a durable store that stops the bot
        booting is worse than the one that reloads nothing.
        """
        counts = {"restored": 0, "expired": 0, "interrupted": 0,
                  "quarantined": 0}
        try:
            rows = self.memory_manager.load_parked_writes(PARK_DB_LIVE)
        except Exception as e:
            logger.error("Could not reload parked writes at boot: %s", e,
                         exc_info=True)
            return counts

        now = time.time()
        for row in rows:
            outcome = self._reconcile_row(row, now)
            if outcome is not None:
                counts[outcome] += 1

        if any(counts.values()):
            logger.info(
                "Parked writes reloaded: %d restored, %d expired, %d "
                "interrupted by the restart, %d quarantined",
                counts["restored"], counts["expired"], counts["interrupted"],
                counts["quarantined"],
            )
        # Mark the clock as well as running the purge. `_last_purge` starts at
        # "never", and only `_purge_due` assigns it — so a boot purge that
        # bypassed it left the very next `park()` or `list_for()` scheduling a
        # second, identical DELETE milliseconds later.
        self._last_purge = now
        self._purge_old_rows(now)
        return counts

    def _reconcile_row(self, row: Dict[str, Any],
                       now: float) -> Optional[str]:
        """Decide one durable row's fate at boot. Returns the counter to bump.

        Every step is inside a `try`, including the terminal work. Both
        `_terminate_interrupted` and `expire` reach into `tool_context` and the
        store, and an exception from either used to travel straight out of
        `rebuild_from_store` into `create_chat_system` — one bad row and the bot
        does not start, discarding every park reconciled before it. A durable
        store that blocks boot is worse than one that reloads nothing.

        A row that could not be reconciled is quarantined rather than skipped,
        for the same reason an unreadable one is. Returning without finalizing
        left it `pending`/`claimed` forever — never loaded, never expired, and
        never purged, since `purge_parked_writes` only deletes terminal rows —
        so it failed identically on every subsequent boot and its unscrubbed
        `write_call` arguments stayed on disk permanently. That is precisely the
        trap `_quarantine` was written to eliminate, reached through the
        reconcile branch instead of the decode branch.
        """
        try:
            parked = ParkedWrite.from_row(row)
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Quarantining unreadable parked write row %s: %s",
                         row.get("token"), e)
            return "quarantined" if self._quarantine(row, str(e)) else None

        try:
            if row.get("status") == PARK_DB_CLAIMED:
                self._terminate_interrupted(parked)
                return "interrupted"
            if self.is_expired(parked, now):
                self.expire(parked, PARK_STATUS_EXPIRED,
                            "Expired while the process was down")
                return "expired"
            self._reinstate(parked)
            return "restored"
        except Exception as e:
            logger.error("Could not reconcile parked write %s at boot: %s",
                         parked.token, e, exc_info=True)
            # Best-effort terminal close-out. If the failure came *after* the
            # row was already finalized (an audit write, say), this answers
            # False and the row is correctly left alone.
            if self._quarantine(row, f"could not be reconciled at boot: {e}"):
                return "quarantined"
            return None

    def _quarantine(self, row: Dict[str, Any], why: str) -> bool:
        """Force an unreadable row terminal so it stops being reloaded.

        Skipping it instead — which is what this did — leaves it `pending`
        forever: never loaded, never expired, and never purged, since
        `purge_parked_writes` only deletes terminal rows. That is not merely an
        ERROR line on every boot for the life of the database. `write_call` on
        a pending row holds the REAL argument values (it has to; an approved
        call executes with them), and `finalize_parked_write` is the only thing
        that ever erases them — so one malformed row parks whatever secret that
        call carried on disk permanently.

        Its own terminal state, and its own audit event. Writing
        `PARK_STATUS_INTERRUPTED` here made a corrupt row indistinguishable from
        a genuine decision-in-flight in the one column a query filters on, and
        every other park-terminating path (`park`, `apply`, `expire`,
        `_terminate_interrupted`) logs an audit row while this one logged none —
        so after `PARK_ROW_RETENTION` the only durable trace that the action was
        ever proposed said nothing about it being terminated.
        """
        token = row.get("token")
        if not isinstance(token, str) or not token:
            logger.error(
                "Unreadable parked write has no usable token; its payload "
                "cannot be erased and it will be re-read on every boot.",
            )
            return False
        try:
            closed = self.memory_manager.finalize_parked_write(
                token, PARK_DB_QUARANTINED, PARK_STATUS_QUARANTINED,
                f"Row could not be used at boot: {why}",
            )
        except Exception as e:
            logger.error("Could not quarantine parked write %s: %s", token, e)
            return False
        if closed:
            self.memory_manager.log_audit_event(
                event_type="audit_park_quarantined",
                operator_id=str(row.get("user_identifier") or "unknown"),
                prior_state=str(row.get("status") or PARK_DB_PENDING),
                new_state=PARK_STATUS_QUARANTINED,
                reason=f"Durable park row could not be used at boot: {why}. "
                       f"The write was NOT executed and its payload was erased.",
                metadata={"token": token},
            )
        return closed

    def _purge_old_rows(self, now: Optional[float] = None) -> int:
        """Drop terminal park rows past their retention. Returns how many."""
        now = time.time() if now is None else now
        try:
            return self.memory_manager.purge_parked_writes(
                now - PARK_ROW_RETENTION)
        except Exception as e:
            logger.warning("Could not purge old parked-write rows: %s", e)
            return 0

    def _terminate_interrupted(self, parked: ParkedWrite) -> None:
        """Close out a deferral whose resolution died with the process."""
        self.patch_parked_entry(
            parked, PARK_STATUS_INTERRUPTED,
            {"error": INTERRUPTED_INSTRUCTION
                if parked.kind == DEFERRAL_KIND_APPROVAL
                else INTERRUPTED_DEFERRAL_INSTRUCTION},
        )
        self.memory_manager.finalize_parked_write(
            parked.token, PARK_DB_INTERRUPTED, PARK_STATUS_INTERRUPTED,
            "Process restarted mid-resolution",
        )
        self.memory_manager.log_audit_event(
            event_type="audit_park_interrupted",
            operator_id=parked.user_identifier,
            prior_state=PARK_DB_CLAIMED,
            new_state=PARK_STATUS_INTERRUPTED,
            reason=("Process restarted after the decision was claimed; the "
                    "write was NOT re-executed"
                    if parked.kind == DEFERRAL_KIND_APPROVAL else
                    f"Process restarted while settling a {parked.kind} "
                    f"deferral; nothing was re-executed"),
            metadata=parked.audit_info,
        )

    # ---- resolution ------------------------------------------------------

    @staticmethod
    def _default_reason(decision: Decision) -> str:
        """The audit sentence when the caller supplied no note.

        One helper rather than the expression repeated at the audit row and the
        finalize call. Those two copies had to agree — `resolution_reason` on the
        row and `reason` in `Audit_Log` are the pair a forensic query joins on —
        and once a third kind existed, "approved" versus "denied" stopped being
        an exhaustive answer at both sites simultaneously.
        """
        park = decision.park
        if park.kind != DEFERRAL_KIND_APPROVAL:
            return (f"{park.kind} deferral settled by its authority as "
                    f"{decision.status}")
        return ("Human approved tool execution" if decision.approved
                else "Human denied tool execution")

    async def apply(self, decision: Decision) -> None:
        """Settle one deferral: produce its real result, then patch history.

        Ordering matters: the patch must land before the continuation rebuilds
        history, or the model reads its own proposal as still pending and
        summarizes the wrong thing.

        The head branches on kind; everything from the audit row down is shared.
        That split is the point of DP-345 — the *outcome* of a deferral is
        kind-specific (a human's verdict plus an execution, versus a report from
        the authority that already did the work), but recording it is not, and
        each of the three re-derivations rebuilt the recording half too.

        A non-approval kind executes NOTHING here. Its tool already ran on the
        turn that deferred it — that is why there is a job to wait on — so
        running it again would be a second irreversible action, reached through
        the one path in this module whose entire job is to prevent that. Its
        caller re-reads the outcome from the authority and hands it in.
        """
        park = decision.park
        tool_name = park.write_call.get("name") or "unknown"

        if park.kind != DEFERRAL_KIND_APPROVAL:
            if decision.outcome_status is None:
                raise ValueError(
                    f"a {park.kind!r} deferral must be settled with an "
                    f"outcome_status; there is no operator verdict to derive "
                    f"one from"
                )
            # Same taint rule as an approved execution, and for the same reason:
            # what gets patched into history here is a payload from outside the
            # system (a node's job document, an agent's report), so if this
            # tool's output is untrusted then this turn is tainted. Set in this
            # branch rather than in the shared tail on purpose — hoisting it
            # would newly taint DENIALS, which execute nothing and read no
            # external bytes at all.
            if get_tool_capabilities(tool_name).get("produces_untrusted"):
                park.turn_tainted = True
        elif decision.approved:
            tool_manager = self._tool_manager_lookup()
            try:
                decision.result = await tool_manager.execute_tool(
                    tool_name, self._offered_lookup(park.persona_name),
                    **(park.write_call.get("arguments") or {}),
                )
                # `ok` is "did the tool succeed", NOT "did the call return"
                # and NOT "did the envelope carry an error". Two ways a write
                # fails without raising, and DP-322 only closed the first:
                #   - `execute_tool` catches every handler exception and
                #     RETURNS `{"error": ...}`, so a Zammad 500 that fired
                #     *after* the ticket was created was a plain `approved`;
                #   - seven gated writes never raise at all. The proxmox tools
                #     return `{"status": "error", ...}` and `approve_proposal`
                #     returns `{"executed": False, ...}` — both nested under
                #     the envelope's `result` key, where an envelope-level
                #     check cannot see them.
                # Either way `PARK_STATUS_FAILED` was unreachable, and with it
                # everything keyed off it: the DP-319 re-execution guard's
                # `approved_but_failed` arm, the "whether it took effect is
                # unknown" instruction in `tool_loop`, the user_guide's promise
                # that an errored write is treated as having run, the "approved
                # but FAILED" continuation line, and `executed_ok` in the audit
                # row.
                decision.ok = tool_error(decision.result) is None
            except Exception as e:
                # Narrower than it looks, and NOT the handler-failure path:
                # `execute_tool` swallows everything the handler raises, so the
                # only way to land here is the call itself failing to be made —
                # `**arguments` not unpacking because the model (or a patched
                # history entry) supplied something that is not a string-keyed
                # mapping. Reading this as "handler exceptions are covered here"
                # is exactly the inference that let the defect above survive
                # four reviews; it covers the frame, not the callee.
                logger.error(
                    f"Approved write {tool_name} (token {park.token}) "
                    f"could not be invoked: {e}", exc_info=True,
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
            reason=decision.note or self._default_reason(decision),
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
        # `patched` starts False so an exception out of the patch leaves it
        # False rather than at its optimistic default — `_render_resolution_nudge`
        # keys off it to tell the model not to trust the tool context.
        decision.patched = False
        redeferred = self._maybe_redefer(decision)
        try:
            decision.patched = self.patch_parked_entry(
                park,
                awaiting_status(redeferred) if redeferred else decision.status,
                decision.result,
            )
        finally:
            # Terminal, durably, in a `finally`: the row survives (the duplicate
            # guard reads it to recognize a re-proposal of an action that
            # already ran) but its payload columns are erased, so the arguments
            # stop living on disk the moment they stop being needed to execute.
            #
            # `patch_parked_entry` can raise — `_patch_one` json-encodes an
            # arbitrary tool result — and `stream_resolve_park` catches that and
            # continues into the continuation turn anyway. Finalizing outside
            # the `finally` meant such a raise left the row `claimed` with the
            # write ALREADY EXECUTED: `find_resolved_parked_write` filters on
            # `resolved`, so the re-execution guard went blind on exactly the
            # turn it exists for, and the in-process fallback below was skipped
            # too. The next boot then reported a known outcome as `interrupted`.
            finalized = self.memory_manager.finalize_parked_write(
                park.token, PARK_DB_RESOLVED, decision.status,
                decision.note or self._default_reason(decision),
            )
            if not finalized and decision.status in (PARK_STATUS_APPROVED,
                                                     PARK_STATUS_FAILED):
                # No durable row to find later, and the tool RAN. The duplicate
                # guard reads the store, so without an in-process record the
                # model's re-proposal on the continuation turn parks a fresh
                # copy and an approval executes the write a second time.
                logger.error(
                    "Parked write %s (%s) resolved as %s with no durable row "
                    "to finalize; falling back to an in-process re-execution "
                    "guard.", park.token, tool_name, decision.status,
                )
                self._remember_unpersisted_outcome(park, decision.status,
                                                   time.time())
        if not decision.patched:
            logger.error(
                "History entry for %s (token %s) could not be patched; the "
                "write already ran and durable history still reads pending. "
                "Audit row carries the real outcome (executed_ok=%s).",
                tool_name, park.token, decision.ok,
            )

    def _maybe_redefer(self, decision: Decision) -> Optional[str]:
        """Re-park an approved write that started work instead of finishing it.

        `install_model` and the cold-tier promotion both hand the node a job id
        and return immediately, so the honest history entry is
        `awaiting:<kind>` rather than a completed call. Returns the kind that
        now owns this call, or None to patch it as done.

        Only on the approval path, and only when the call actually succeeded: a
        denial ran nothing, and a failed start has no job to wait for. Strips
        the declaration itself, which is plumbing between the handler and this
        store — the model gets the handler's own note about what it started.
        """
        park = decision.park
        if not (park.kind == DEFERRAL_KIND_APPROVAL
                and decision.approved and decision.ok):
            return None
        declared = declared_deferral(decision.result)
        if declared is None or not self._redefer(park, *declared):
            return None
        decision.result = {k: v for k, v in decision.result.items()
                           if k != DEFERRAL_DECLARATION_KEY}
        return declared[0]

    def _redefer(self, park: "ParkedWrite", kind: str, token: str) -> bool:
        """Re-park an executed write that is waiting on an external event.

        The approval and the deferral are two parks, not one, and they must be:
        the approval is already `resolved` by the time this runs (the write DID
        run — it started the job), which is what the DP-319 re-execution guard
        reads to recognize a re-proposal. Rewriting that row instead would make
        an executed irreversible call look pending again.

        Everything else is copied off the approval park, because the approval
        park is the turn that raised it. That is what lets the node's ping
        answer in the operator's own channel with nothing configured — see
        `deferral_kinds`.

        Returns False if the row could not be stored, in which case the caller
        patches the entry as complete instead: a deferral nobody can resolve
        would leave history reading `awaiting:<kind>` forever.
        """
        deferred = ParkedWrite(
            token=token,
            # The SAME write_call, so `call_id` still identifies the tool entry
            # this deferral has to patch when it settles — it is the same call.
            write_call=park.write_call,
            audit_info=park.audit_info,
            confirmation_text="",
            user_identifier=park.user_identifier,
            persona_name=park.persona_name,
            channel=park.channel,
            server_id=park.server_id,
            kind=kind,
            turn_tainted=park.turn_tainted,
            parked_assistant_id=park.parked_assistant_id,
        )
        if not self._persist_new(deferred):
            logger.error(
                "%s deferral %s for %s could not be stored; reporting the "
                "call as complete instead of leaving history awaiting an "
                "answer that can never be claimed.", kind, token,
                park.write_call.get("name") or "unknown",
            )
            return False
        self._reinstate(deferred)
        return True

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
            self.expire(parked, PARK_STATUS_EXPIRED,
                        f"No decision within {PENDING_ACTION_TTL}s")
        return len(stale)

    def _take_expired(self, now: Optional[float] = None) -> List[ParkedWrite]:
        """Remove every past-TTL park from the store. Pure in-memory.

        Split out from the DB half so the hot paths (`park`, `list_for`) can
        evict synchronously — which must stay atomic, like `take` — without
        also running SELECT + UPDATE + INSERT inline. Those calls sit inside
        the token stream and the SSE routes, where a single day-old park was
        enough to stall every other stream on the loop.

        `_pop`, not `take`: `take` commits a claim, which would put that split
        straight back (one fsync per stale token, on the loop thread, while the
        previous sweep's worker may already hold the store lock) and would also
        make a crash mid-sweep look like an operator decision that was in
        flight. Nothing needs these rows claimed — the very next thing that
        happens to them is `expire`, which is terminal either way.
        """
        now = time.time() if now is None else now
        tokens = [t for t, p in self.pending.items()
                  if now - p.created_at > PENDING_ACTION_TTL]
        stale = [p for p in (self._pop(t) for t in tokens) if p is not None]
        if stale:
            logger.info("Expired %d unanswered gated write(s)", len(stale))
        return stale

    def _purge_due(self, now: Optional[float] = None) -> bool:
        """True at most once per interval; marks the clock as it answers.

        The mark happens here, synchronously, rather than after the delete —
        the purge itself runs off-thread, so checking-then-marking later would
        let every read in the interim schedule its own redundant DELETE.
        """
        now = time.time() if now is None else now
        if now - self._last_purge < PARK_PURGE_INTERVAL:
            return False
        self._last_purge = now
        return True

    def _sweep_off_thread(self) -> None:
        """Evict expired parks now; do their DB writes off the event loop.

        Also carries the retention purge. That used to run only in
        `rebuild_from_store`, i.e. once at boot — so the process durability was
        added for (one that stays up for months) was the one process that never
        purged anything, and `PARK_ROW_RETENTION` went unenforced for as long as
        the bot kept running. Hanging it off the lazy sweep matches how
        `expire_stale_proposals` is driven.

        Fire-and-forget by design — an expiry fires no continuation, so nothing
        downstream waits on the patch. Falls back to inline when there is no
        running loop (sync callers, tests).
        """
        stale = self._take_expired()
        purge = self._purge_due()
        if not stale and not purge:
            return
        reason = f"No decision within {PENDING_ACTION_TTL}s"

        def _finish() -> None:
            for parked in stale:
                self.expire(parked, PARK_STATUS_EXPIRED, reason)
            if purge:
                self._purge_old_rows()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _finish()
            return
        task = loop.create_task(asyncio.to_thread(_finish))
        # Hold a reference: a bare create_task can be GC'd mid-flight.
        self._sweep_tasks.add(task)
        task.add_done_callback(self._sweep_tasks.discard)

    def expire(self, parked: ParkedWrite, resolution: str,
               reason: str) -> None:
        """Terminate an already-taken park as expired: patch, then audit.

        Shared by the lazy sweep, the boot reload and the resolve path. The
        click path used to inline the patch with a hardcoded "expired" and log
        nothing, which made it the only park-terminating path leaving no audit
        trail — so the fact that a human actually tried to approve an expired
        irreversible action was recorded nowhere, and its writer was decoupled
        from the constant every other consumer keys off.

        `resolution` is the filterable outcome, `reason` the sentence that says
        which of the three ways it got here. Passing the sentence as the
        resolution — the original shape — made every future query over expired
        rows match nothing.
        """
        self.patch_parked_entry(parked, PARK_STATUS_EXPIRED,
                                {"reason": "expired before review"})
        self.memory_manager.finalize_parked_write(
            parked.token, PARK_DB_EXPIRED, resolution, reason,
        )
        self.memory_manager.log_audit_event(
            event_type="audit_park_expired",
            operator_id=parked.user_identifier,
            prior_state=PARK_DB_PENDING,
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
    "INTERRUPTED_INSTRUCTION", "INTERRUPTED_DEFERRAL_INSTRUCTION",
    "DEFERRAL_KIND_APPROVAL",
]
