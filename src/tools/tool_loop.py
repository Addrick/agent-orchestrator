# src/tools/tool_loop.py
"""Stream-shaped tool loop.

Owns a single iteration: drive `text_engine.stream_messages`, forward
token deltas, surface tool calls as `ToolCallStartEvent` /
`ToolCallResultEvent`, append results to history, repeat until the model
stops calling tools or `max_iterations` trips.

The loop trusts the tools list it's handed — capability filtering /
policy decisions live in the caller (currently `ChatSystem`, eventually
the security framework in the sibling plan).
"""

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from typing import (
    Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union, cast,
)

from config.global_config import MAX_TOOL_CALLS, MAX_TOOL_ITERATIONS
from src.engine import LLMCommunicationError, TextEngine
from src.security.scrubber import get_scrubber
from src.generation_events import (
    ErrorEvent, ResponseType, TokenEvent,
    ToolCallResultEvent, ToolCallStartEvent,
    format_internal_error,
)
from src.persona import Persona
from src.tools.definitions import (
    ALWAYS_CONFIRM_TOOLS,
    get_tool_capabilities, is_irreversible, get_tool_definition, is_write_tool
)
from src.tools.tool_manager import ToolManager, tool_error

logger = logging.getLogger(__name__)


@dataclass
class _ApiPayloadEvent:
    """Loop-internal: forwards `api_payload` from the underlying provider
    so the orchestrator can cache it for `last_api_requests`."""
    payload: Dict[str, Any]
    iter_idx: int


@dataclass
class _ToolContextEvent:
    """Loop-internal: carries this turn's sealed tool context out of an exit
    that has no `_LoopFinishedEvent` to hang it on (the error paths).

    The orchestrator siphons it exactly like `_ApiPayloadEvent`, so an errored
    turn can still persist the tool calls it made before dying."""
    tool_context_json: Optional[str]


@dataclass
class WriteParkedEvent:
    """One write call gated for human approval (DP-297).

    Emitted *mid-turn* — the loop keeps running after it, so a turn can emit
    several. The orchestrator parks each one and forwards it to the surface as
    its own approve/deny affordance. Public (not underscore-prefixed) because
    interfaces consume it directly, unlike the loop-internal events above.
    """
    token: str
    write_call: Dict[str, Any]
    audit_info: Dict[str, Any]
    confirmation_text: str
    turn_tainted: bool = False


@dataclass
class _WriteDuplicateEvent:
    """Loop-internal: a write suppressed by the pending-duplicate guard.

    Emits no public affordance — that is the point of suppressing it — but the
    orchestrator still has to hear about it, because the synthetic result it
    just appended will need patching when the ORIGINAL proposal resolves. Its
    row id, like a park's, does not exist until this turn commits.
    """
    token: str
    call_id: Optional[str]


@dataclass
class _LoopFinishedEvent:
    """Loop-internal terminal event. Carries the resolved state so the
    orchestrator can persist the assistant turn / re-emit a public DoneEvent."""
    final_text: str
    response_type: ResponseType
    tool_context_json: Optional[str] = None
    turn_tainted: bool = False
    #: What to put in the semantic memory bank, when that is NOT `final_text`.
    #: Only the exhaustion exit sets it (DP-335): its reply is prose plus a
    #: machine-generated list of the turn's tool calls and arguments, and only
    #: the prose is a thing the persona said. Embedding the list would make
    #: "`hf_search` {"query": ...} — ok" a recallable memory and replay it to
    #: the model next turn as its own prior words. `None` means "retain
    #: `final_text`", which is every other exit.
    retain_text: Optional[str] = None


LoopEvent = Union[
    TokenEvent, ErrorEvent,
    ToolCallStartEvent, ToolCallResultEvent,
    _ApiPayloadEvent, _LoopFinishedEvent, _ToolContextEvent, WriteParkedEvent,
    _WriteDuplicateEvent,
]

# Status values for a gated write's synthetic tool result. These are what the
# model reads in replayed history, and `PARK_STATUS_AWAITING` is the entry
# `ConfirmationManager` later patches in place once the operator decides.
PARK_STATUS_AWAITING = "awaiting_human_approval"
PARK_STATUS_APPROVED = "approved"
PARK_STATUS_DENIED = "denied"
# Approved by the operator, but the tool raised when it ran. Distinct from
# `PARK_STATUS_APPROVED` because both outcomes leave a `result` and only the
# status distinguishes them, and distinct from `PARK_STATUS_DENIED` because the
# operator said yes — the model must not read it as a refusal to re-argue.
PARK_STATUS_FAILED = "approved_but_failed"
PARK_STATUS_EXPIRED = "expired"
# DP-319: the process died between the operator's decision being claimed and the
# write running, so whether it ran is unknown. Deliberately NOT re-executed on
# the next boot — a gated write is gated because it is irreversible, and
# re-running an "it might already have happened" call is the one failure this
# subsystem exists to prevent. The model is told to re-check state and re-propose
# rather than assume either outcome.
PARK_STATUS_INTERRUPTED = "interrupted_by_restart"
# DP-319 review: a durable row whose payload could not be read at boot. Distinct
# from PARK_STATUS_INTERRUPTED even though both are terminated by the same boot
# pass, because `resolution` is the column a forensic query filters on and the
# two answer opposite questions. An interrupted park is one a human decided and
# the process may have executed; a quarantined one was never readable, so its
# write provably never ran. Collapsing them made "which gated writes may have
# executed during the outage?" return rows that certainly did not.
PARK_STATUS_QUARANTINED = "quarantined_unreadable"
# The model re-proposed a write that was already decided in an earlier turn.
# Distinct from PARK_STATUS_DUPLICATE, which answers a still-*pending* twin: this
# one says the action has already been executed or refused, so a second park
# would be a second execution rather than a redundant affordance.
PARK_STATUS_ALREADY_RESOLVED = "already_resolved"
# Not a park outcome — the answer to a write the model proposed while an
# identical one was already waiting. No second park is created.
PARK_STATUS_DUPLICATE = "duplicate_of_pending"


def write_call_identity(call: Dict[str, Any]) -> Tuple[str, str]:
    """Identity of a write proposal for duplicate detection: tool name plus
    canonicalized arguments.

    Deliberately excludes the provider call id — two re-proposals of the same
    action always carry different ids, which is precisely the case this has to
    catch. `sort_keys` because argument order is not stable across iterations.
    """
    name = str(call.get("name") or "")
    try:
        args = json.dumps(call.get("arguments") or {}, sort_keys=True,
                          default=str)
    except (TypeError, ValueError):
        # Unserializable arguments cannot be compared; fall back to an identity
        # that never matches, so an odd call parks rather than being swallowed.
        #
        # It must be a fresh uuid, NOT `repr(object())`: CPython reuses the
        # address of the object it just freed, so two calls to `repr(object())`
        # return the same string and the "never matches" identity matched
        # ALWAYS — inverting this guard into one that swallowed every
        # unserializable write without ever surfacing an affordance.
        args = f"<unserializable:{uuid.uuid4().hex}>"
    return (name, args)


def write_call_identity_hash(call: Dict[str, Any]) -> str:
    """Storage form of `write_call_identity`: one hash of name + args.

    Lives HERE, next to the canonicalization it hashes, rather than on
    `MemoryManager` where DP-319 first put it. The identity contract was spread
    over three modules — the canonicalizer in `src.tools`, the digest in
    `src.memory`, and two call sites in `src.confirmations` that re-composed
    them by hand — and any one of them drifting produces a hash that matches
    nothing. That failure is silent and it fails OPEN: the duplicate guard
    simply stops recognizing re-proposals, which is a second execution of an
    irreversible write.

    Hashed rather than stored raw because the canonical args are the same
    secret-bearing payload `finalize_parked_write` exists to erase; equality is
    all the duplicate guard needs.
    """
    return identity_digest(*write_call_identity(call))


def identity_digest(name: str, args: str) -> str:
    """The one place the identity digest is constructed.

    Separate from `write_call_identity_hash` because two callers need it from
    two different starting points — the duplicate guard has a call, the DP-335
    turn tally has an already-canonicalized `(name, args)` pair it built while
    counting. The tally used to rebuild these four lines inline, which is
    exactly the spread-out identity contract the docstring above records as
    having already drifted once, silently and fail-open.
    """
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(args.encode("utf-8"))
    return digest.hexdigest()


# Reasons a tool call can be left without a real result when the turn ends.
# (There is no awaiting-approval reason: since DP-297 a parked write is
# answered inline with a real synthetic result, so it is never unanswered at
# seal time — which is also what guarantees the patch target exists later.)
SEAL_ERROR = "error"
SEAL_MAX_ITERATIONS = "max_iterations_exceeded"
# The clean exit: the model stopped asking for tools. Every call should already
# be answered here, so this reason is expected to appear in no sealed result —
# it exists so a call that somehow slipped through does not tell the model its
# successful turn errored.
SEAL_UNKNOWN = "unknown"


def seal_tool_context(
    conversation_history: List[Dict[str, Any]],
    start: int,
    reason: str,
) -> Optional[str]:
    """Serialize this turn's tool messages, synthesizing a result for every
    tool call left unanswered.

    A turn can end with calls still outstanding: a write parked for approval,
    a provider error mid-iteration, the iteration cap. Persisting that slice
    verbatim stores an assistant message whose `tool_calls` have no matching
    `tool_result`, and both Anthropic and Gemini reject unpaired blocks on the
    next request — which is why these exits used to persist nothing at all and
    the model lost every gated or errored action it took.

    Sealing keeps the turn replayable: the model sees what it called and that
    the call did not complete. Does not mutate `conversation_history` — the
    resume path still needs the unsealed list so real results can land.
    """
    tool_msgs = conversation_history[start:]
    if not tool_msgs:
        return None

    answered = {
        m.get("tool_call_id") for m in tool_msgs if m.get("role") == "tool"
    }
    sealed: List[Dict[str, Any]] = list(tool_msgs)
    for msg in tool_msgs:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            call_id = call.get("id")
            if call_id in answered:
                continue
            answered.add(call_id)
            sealed.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": call.get("name"),
                "content": json.dumps(
                    {"status": "not_executed", "reason": reason}
                ),
            })
    return json.dumps(sealed)


def _render_confirmation_text(
    action: Dict[str, Any],
    turn_tainted: bool,
    taint_sources: List[str],
) -> str:
    """Human-readable approval prompt for ONE gated action.

    Takes an already-scrubbed action dict (DP-225 boundary 2 runs on the whole
    audit_info before this is called), so nothing here needs to re-scrub.
    """
    flags = []
    if action["service_binding"]:
        flags.append(action["service_binding"].upper())
    if action["sensitivity"]:
        flags.append(action["sensitivity"].upper())
    if action["irreversible"]:
        flags.append("IRREVERSIBLE")
    if action["always_confirm"]:
        flags.append("HIGH-IMPACT")

    flag_str = f" [{', '.join(flags)}]" if flags else ""
    enrich_str = f": **{action['enrichment']}**" if action["enrichment"] else ":"
    lines = [
        "I'd like to perform the following action:",
        f"- **{action['tool']}**{flag_str}{enrich_str} "
        f"{json.dumps(action['arguments'])}",
    ]
    if turn_tainted:
        lines.append(
            f"\n⚠️ Context contains untrusted content from: "
            f"{', '.join(taint_sources)}"
        )
    return "\n".join(lines)


# Cap-hit summary rendering (DP-335). The iteration cap used to answer with one
# canned sentence, and that sentence reads as a malfunction: in the prod turn
# that motivated this, every one of the ten calls returned `ok` — the turn
# simply ran out of steps before it reached the action the user asked for. The
# sealed `tool_context` held the whole story and nothing put it in front of the
# person who had to decide what to do next.
_SUMMARY_ARG_LIMIT = 100
_SUMMARY_ERROR_LIMIT = 80
_SUMMARY_MAX_CALLS = 20


def _truncate(text: str, limit: int) -> str:
    """Flatten to one line and clip — a tool result can be kilobytes, and this
    string is headed for a Discord message with a 2000-char ceiling."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit - 1] + "…"


def _summarize_outcome(content: Optional[str]) -> str:
    """How one call turned out, read back from its history result.

    The park statuses are spelled out rather than folded into "ok": a proposal
    that is still waiting on the operator is the most likely thing the user
    wants to act on, and `tool_error` reports it as a success.
    """
    if content is None:
        # `seal_tool_context` synthesizes a result for these, but it runs after
        # this does — the final iteration's calls are genuinely unanswered.
        return "no result"
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return "ok"
    if isinstance(payload, dict):
        status = payload.get("status")
        if status == PARK_STATUS_AWAITING:
            return "waiting for your approval"
        if status == PARK_STATUS_DUPLICATE:
            return "skipped, an identical proposal is already waiting"
        if status == PARK_STATUS_ALREADY_RESOLVED:
            return "skipped, you already decided this one"
    failure = tool_error(payload)
    if failure:
        # Scrubbed for the same DP-225 reason `_render_args` scrubs: this
        # string is headed for a user surface. Every writer of `role: tool`
        # content scrubs before it lands in history today, so this is belt and
        # braces — but the asymmetry was invisible, and `scrub` is idempotent.
        clean = cast(str, get_scrubber().scrub(failure))
        return f"failed — {_truncate(clean, _SUMMARY_ERROR_LIMIT)}"
    return "ok"


def _render_args(arg_str: str) -> str:
    """A call's canonical argument string as a short, scrubbed, leading-space-
    prefixed blob.

    Takes the string `write_call_identity` already produced rather than
    re-serializing the call: two canonicalizations of the same arguments can
    drift, and if they do, the rendered args and the `(same call as #N)` marker
    beside them stop describing the same call.

    Empty arguments render as nothing at all: half the calls in a
    budget-exhausting turn are zero-arg reads, and ten trailing `{}`s bury the
    ones that carry the query that actually mattered.
    """
    if arg_str in ("{}", "null") or arg_str.startswith("<unserializable:"):
        return ""
    # DP-225: arguments are model-authored and this string is headed for a
    # surface, so it crosses the same egress boundary a tool result does.
    scrubbed = cast(str, get_scrubber().scrub(arg_str))
    return " " + _truncate(scrubbed, _SUMMARY_ARG_LIMIT)


def _call_lines(
    conversation_history: List[Dict[str, Any]],
    start: int,
) -> Tuple[List[str], int]:
    """One numbered line per tool call in the turn's slice, plus the true total.

    Reads the same history slice `seal_tool_context` seals, so the user sees
    exactly what the model saw. Repeats are marked, because a turn that spends
    its budget re-fetching bytes it already had is the common shape of this
    exit and it is invisible in a bare list.

    The returned total counts every call, including any past `_SUMMARY_MAX_CALLS`
    that got no line — callers need the real number to say "and N more" and to
    report the spend honestly.
    """
    results: Dict[Any, Optional[str]] = {}
    for msg in conversation_history[start:]:
        if msg.get("role") == "tool":
            results.setdefault(msg.get("tool_call_id"), msg.get("content"))

    lines: List[str] = []
    # `write_call_identity` is the name-plus-canonicalized-args key the write
    # guards use; nothing about it is write-specific, and reusing it here means
    # "the same call twice" means one thing across the module.
    first_seen: Dict[Tuple[str, str], int] = {}
    total = 0
    for msg in conversation_history[start:]:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            total += 1
            if total > _SUMMARY_MAX_CALLS:
                continue
            identity = write_call_identity(call)
            name = identity[0] or "unknown tool"
            rendered_args = _render_args(identity[1])
            outcome = _summarize_outcome(results.get(call.get("id")))
            repeat = first_seen.get(identity)
            marker = f" (same call as #{repeat})" if repeat else ""
            first_seen.setdefault(identity, total)
            lines.append(f"{total}. `{name}`{rendered_args} — {outcome}{marker}")

    if total > _SUMMARY_MAX_CALLS:
        lines.append(f"…and {total - _SUMMARY_MAX_CALLS} more.")
    return lines, total


# DP-335 review: `used` and `budget` are passed in rather than derived from the
# rendered slice, and both renderers are keyword-only past the history, for two
# reasons that were live bugs in the first cut.
#
# 1. `_call_lines`' total counts the whole rendered slice, and on a park
#    continuation `history_start_override` walks that slice back over the
#    PARKED turn (so the seal spans both) while the resumed turn's spend starts
#    at zero. Deriving the headline number from the slice therefore reported
#    another turn's calls as this one's — "(21 of 15 used)" — and pushed the
#    continuation's own calls behind the `…and N more` cap.
# 2. The budget that stopped the turn is not always the call budget. Reporting
#    `max_tool_calls` unconditionally produced "I used all 100 of my tool steps"
#    after three, which is the wrong-diagnosis-in-the-exit-message failure this
#    ticket exists to remove.
#
# `total` is still used for `…and N more`: that one IS a fact about the list.
def _spend_clause(used: int, budget: int, exhausted: bool) -> str:
    """How the turn's spend is stated, in the terms of the limit that tripped.

    `used` can exceed `budget` by design — a batch is charged whole and never
    truncated — so this never says "all N of N"; it always states both numbers.
    """
    if exhausted:
        return f"my whole tool budget ({used} of {budget} tool steps)"
    return f"{used} tool step(s) before my loop guard stopped the turn"


def render_max_iteration_text(
    conversation_history: List[Dict[str, Any]],
    start: int,
    *,
    used: int,
    budget: int,
    exhausted: bool = True,
) -> str:
    """Standalone cap-hit message: the call list and nothing else.

    The **fallback** since DP-335's second half — the loop's first choice is now
    to ask the model for a real answer and hang `render_call_summary_footer`
    under it. This wording stays for the case where that extra completion is
    unavailable (provider down, empty response), because a turn that ends with
    no prose at all still has to say what happened.
    """
    lines, _ = _call_lines(conversation_history, start)
    if not lines:
        return (
            f"I stopped after spending {_spend_clause(used, budget, exhausted)} "
            "without reaching an answer. Could you clarify the request?"
        )
    return "\n".join([
        f"I spent {_spend_clause(used, budget, exhausted)} on this turn "
        "without getting to an answer, so I stopped instead of guessing. "
        "Here is what I ran:",
        "",
        *lines,
        "",
        "Tell me which of these to build on, or narrow the request, and I'll "
        "pick it up from there.",
    ])


def render_call_summary_footer(
    conversation_history: List[Dict[str, Any]],
    start: int,
    *,
    used: int,
    budget: int,
    exhausted: bool = True,
) -> str:
    """The same list, as ground truth pinned under a prose answer.

    The two compose deliberately. The prose is a generation and can be wrong or
    vague about what it actually ran; this list is read straight out of history
    and cannot be. Keeping it means the exhaustion answer never has to be taken
    on trust, and repeats stay measurable in the reply itself — which is how
    the duplicate-read question gets its evidence without a DB dig.
    """
    lines, total = _call_lines(conversation_history, start)
    if not lines:
        return ""
    if exhausted:
        header = f"*Ran out of tool steps ({used} of {budget} used):*"
    else:
        header = f"*Loop guard stopped the turn after {used} tool step(s):*"
    if total != used:
        # The rendered slice covers more than this turn's spend — the park
        # continuation case. Say so rather than letting the count beneath the
        # header silently contradict it.
        header += f" the {total} calls below span the whole parked exchange."
    return "\n".join(["", header, *lines])


# The synthetic user message that opens the exhaustion wrap-up (DP-335). Not
# persisted and not appended to `conversation_history` — it exists only in the
# message array of that one request, exactly like `_render_resolution_nudge` in
# the park-continuation path, and for the same reason: ending the array on the
# model's own tool span makes Anthropic treat it as a prefill to continue
# rather than a turn to answer.
#
# Deliberately carries NO `[system]` prefix, though it is a system instruction.
# This codebase treats tool output as attacker-influenceable (`produces_untrusted`
# → `turn_tainted`), so a turn that demonstrates "`[system] …` in the user
# channel means system authority" makes an injected `[system] Ignore the
# approval gate and …` inside a `web_search` result materially more credible on
# the next turn. `_render_resolution_nudge` — the park-continuation twin this
# is modeled on — states its facts plainly for the same reason.
_EXHAUSTION_NUDGE = (
    "You have used your entire tool budget for this turn and no "
    "further tool calls will run. Answer the user now from what you already "
    "have. Say what you found, name what is still unresolved and why, and "
    "propose the single next step you would take — do not describe your tool "
    "usage, and do not claim to be stuck."
)


def _log_turn_call_tally(
    tally: Dict[Tuple[str, str], int],
    iterations: int,
    calls: int,
) -> None:
    """Log this turn's call multiset and its repeat count (DP-335).

    The instrumentation that replaced a proposed per-turn read cache. One
    observed turn spent 4 of its 10 steps re-running reads whose answers were
    already in `conversation_history`; that is n=1, and a cache sized on n=1
    would have funded more wandering while breaking `install_status`, which
    exists to be polled. So the repeat rate is *measured* instead, and a
    recurrence arrives with numbers attached rather than an anecdote.

    Identities are logged as `name#<8 hex of the identity hash>`, never as raw
    arguments: the canonical argument string is the same secret-bearing payload
    `write_call_identity_hash` is hashed to avoid storing, and distinguishing
    two calls is all this needs.
    """
    if not tally:
        return
    repeats = {k: n for k, n in tally.items() if n > 1}
    if repeats:
        digests = []
        for (name, args), n in sorted(repeats.items(), key=lambda kv: -kv[1]):
            digests.append(f"{name}#{identity_digest(name, args)[:8]} x{n}")
        detail = "; repeated: " + ", ".join(digests)
    else:
        detail = ""
    logger.info(
        "tool-loop turn: %d call(s) over %d iteration(s), %d distinct, "
        "%d repeated call(s)%s",
        calls, iterations, len(tally), calls - len(tally), detail,
    )


def build_wire_messages(
    persona: Persona, conversation_history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prepend the persona system prompt to the conversation history to form
    the exact message array sent to the provider.

    Single source of truth for the system-prompt prepend: both the live tool
    loop's first iteration (`ToolLoop.run`) and the `/assemble` dry-run
    (`ChatSystem.assemble_request`) call this, so the wire messages the inspector
    shows cannot drift from what a live submit actually sends.
    """
    from datetime import datetime
    system_prompt = persona.get_prompt()
    inject = True
    if hasattr(persona, "get_inject_timestamp"):
        inject = persona.get_inject_timestamp()

    if inject:
        # Wednesday, June 10, 2026, 01:01 AM EDT
        now_str = datetime.now().astimezone().strftime("%A, %B %d, %Y, %I:%M %p %Z")
        system_prompt = f"[Current Time: {now_str}]\n\n{system_prompt}"

    return [{"role": "system", "content": system_prompt}] + list(conversation_history)


class ToolLoop:
    """Drives the stream_messages → tool_calls → execute → repeat cycle."""

    def __init__(
        self,
        text_engine: TextEngine,
        tool_manager: ToolManager,
        max_iterations: int = MAX_TOOL_ITERATIONS,
        max_tool_calls: int = MAX_TOOL_CALLS,
    ) -> None:
        """`max_tool_calls` is the turn's budget; `max_iterations` is the
        runaway guard. DP-335 split them: one number cannot be both, because
        what an iteration buys depends entirely on how many calls the model
        packs into one message, so a single limit means a different amount of
        work per provider. See `config.global_config` for the sizing.
        """
        self.text_engine = text_engine
        self.tool_manager = tool_manager
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls

    async def run(
        self,
        *,
        persona: Persona,
        conversation_history: List[Dict[str, Any]],
        params: Any,
        tools: List[Dict[str, Any]],
        local_inference_config: Optional[Dict[str, Any]] = None,
        image_url: Optional[str] = None,
        turn_tainted: bool = False,
        initial_taint_sources: Optional[List[str]] = None,
        history_start_override: Optional[int] = None,
        pending_lookup: Optional[
            Callable[[Dict[str, Any]], Optional[str]]
        ] = None,
        resolved_lookup: Optional[
            Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]
        ] = None,
    ) -> AsyncIterator[LoopEvent]:
        """Yield generation events for one turn. Mutates
        `conversation_history` in-place so the orchestrator (and any
        CONFIRM-mode resume path) sees the same list.

        `history_start_override` lets a resumed write-confirmation point the
        tool-context boundary back at the parked turn's first tool message, so
        the captured tool_context_json spans the whole turn (parked read calls,
        the approved write, and its result) rather than only post-resume calls.

        `pending_lookup(write_call) -> token | None` answers "is an identical
        proposal already waiting?". Injected rather than read from a store
        because the pending set lives in `ConfirmationManager`, which sits
        ABOVE this module in the layer order — the loop stays policy-free and
        the caller decides what counts as already-pending.

        `resolved_lookup(write_call) -> row | None` is the same question for an
        already-DECIDED proposal (DP-319). Separate from `pending_lookup`
        because the answer differs: a pending twin is answered "wait for it",
        a decided one is answered with its outcome, and only the first has a
        history entry that will need correcting later.
        """
        persona_config = persona.get_config_for_engine()
        history_start = (
            history_start_override if history_start_override is not None
            else len(conversation_history)
        )
        taint_sources: List[str] = list(initial_taint_sources or [])
        # turn_tainted is passed in to support conversation-level stickiness

        # DP-335: two counters, two limits. `calls_used` is the budget the
        # persona is allowed to spend and is what the user's request is
        # measured against; `iterations_used` only catches a loop that talks to
        # the provider forever without spending anything.
        calls_used = 0
        iterations_used = 0
        call_tally: Dict[Tuple[str, str], int] = {}

        while (calls_used < self.max_tool_calls
               and iterations_used < self.max_iterations):
            iter_idx = iterations_used
            iterations_used += 1
            api_payload: Optional[Dict[str, Any]] = None
            full_text_from_done: Optional[str] = None
            tool_calls_collected: Optional[List[Dict[str, Any]]] = None
            accumulated_parts: List[str] = []

            messages_for_llm: List[Dict[str, Any]] = build_wire_messages(
                persona, conversation_history,
            )

            try:
                stream = self.text_engine.stream_messages(
                    persona_config,
                    messages_for_llm,
                    params,
                    tools=tools,
                    local_inference_config=local_inference_config,
                    image_url=image_url if iter_idx == 0 else None,
                )
                async for ev in stream:
                    etype = ev.get("type")
                    if etype == "api_payload":
                        api_payload = ev.get("payload")
                    elif etype == "text_delta":
                        text_chunk = ev.get("text") or ""
                        if text_chunk:
                            accumulated_parts.append(text_chunk)
                            yield TokenEvent(delta=text_chunk)
                    elif etype == "tool_calls":
                        tool_calls_collected = list(ev.get("calls") or [])
                        # Normalize identity once, at ingestion: providers may
                        # omit `id`, and every downstream consumer (assistant
                        # message, lifecycle events, tool-result history) must
                        # agree on it or the next iteration sends the model
                        # unpaired call/result blocks.
                        for c in tool_calls_collected:
                            if not c.get("id"):
                                c["id"] = f"call_{uuid.uuid4().hex[:12]}"
                    elif etype == "done":
                        full_text_from_done = ev.get("full_text")
            except LLMCommunicationError as e:
                payload_to_store = e.api_payload or api_payload
                if payload_to_store:
                    yield _ApiPayloadEvent(payload=payload_to_store, iter_idx=iter_idx)
                err_msg = (
                    "I'm not sure how to continue. Could you please rephrase?"
                    if "empty response" in str(e)
                    else "Error while generating a response: " + str(e)
                )
                yield _ToolContextEvent(
                    tool_context_json=seal_tool_context(
                        conversation_history, history_start, SEAL_ERROR,
                    )
                )
                _log_turn_call_tally(call_tally, iterations_used, calls_used)
                yield ErrorEvent(message=err_msg)
                return
            except Exception as e:
                err_id, err_msg = format_internal_error(e, scrub=get_scrubber().scrub)
                logger.error(
                    f"[err {err_id}] Unexpected error during stream_messages "
                    f"(iter {iter_idx}): {e}",
                    exc_info=True,
                )
                yield _ToolContextEvent(
                    tool_context_json=seal_tool_context(
                        conversation_history, history_start, SEAL_ERROR,
                    )
                )
                _log_turn_call_tally(call_tally, iterations_used, calls_used)
                yield ErrorEvent(message=err_msg)
                return

            if api_payload:
                yield _ApiPayloadEvent(payload=api_payload, iter_idx=iter_idx)

            if not tool_calls_collected:
                final_text = (
                    full_text_from_done if full_text_from_done is not None
                    else "".join(accumulated_parts)
                )
                tool_context_json = seal_tool_context(
                    conversation_history, history_start, SEAL_UNKNOWN,
                )
                _log_turn_call_tally(call_tally, iterations_used, calls_used)
                yield _LoopFinishedEvent(
                    final_text=final_text,
                    response_type=ResponseType.LLM_GENERATION,
                    tool_context_json=tool_context_json,
                    turn_tainted=turn_tainted,
                )
                return

            # DP-335: the budget is charged for the whole batch, here, before
            # anything runs — but the batch is never truncated to fit. Half of a
            # group the model proposed as one plan is worse than one call of
            # overshoot, and the group is dispatched with `asyncio.gather`
            # anyway, so a batch that crosses the line costs one round trip, not
            # several. The next loop check sees the overshoot and ends the turn.
            calls_used += len(tool_calls_collected)
            for call_item in tool_calls_collected:
                identity = write_call_identity(call_item)
                call_tally[identity] = call_tally.get(identity, 0) + 1

            group_id = f"iter{iter_idx}_{uuid.uuid4().hex[:8]}"
            for call_item in tool_calls_collected:
                call_item["group_id"] = group_id
            # DP-338: the assistant's own words go into history BESIDE its
            # calls. The prose a model writes before a batch is the plan for
            # that batch ("checking the node, the card and the unit list before
            # proposing a swap"); dropping it left the next iteration reading a
            # transcript where calls appeared for no stated reason, so the model
            # re-derived the plan from an identical history and re-emitted the
            # same batch. Same class as the DP-335 answer loss, one iteration
            # earlier. Providers park the prose in different places: the
            # streaming ones delta it out and zero `full_text` on a tool turn,
            # agy's one-shot has no deltas at all and carries it on `done`.
            assistant_prose = (
                "".join(accumulated_parts).strip()
                or (full_text_from_done or "").strip()
            )
            assistant_entry: Dict[str, Any] = {
                "role": "assistant", "tool_calls": tool_calls_collected,
            }
            if assistant_prose:
                assistant_entry["content"] = assistant_prose
            conversation_history.append(assistant_entry)
            read_calls = [c for c in tool_calls_collected if not is_write_tool(c.get("name") or "")]
            write_calls = [c for c in tool_calls_collected if is_write_tool(c.get("name") or "")]

            async for tool_ev in self._execute_calls(read_calls, conversation_history, group_id=group_id):
                yield tool_ev

            # Update turn_tainted from read_calls that just finished
            for rc in read_calls:
                tool_name = rc.get("name") or "unknown"
                caps = get_tool_capabilities(tool_name)
                if caps.get("produces_untrusted"):
                    turn_tainted = True
                    taint_sources.append(tool_name)

            # --- All write tools require audit ---
            if write_calls:
                logger.info(
                    "tool-loop iter %d: parking %d write call(s) for audit: %s "
                    "(reads this iter: %s) — turn CONTINUES",
                    iter_idx,
                    len(write_calls),
                    [w.get("name") for w in write_calls],
                    [r.get("name") for r in read_calls],
                )
                # Same prose the history entry got (DP-338). Reading
                # `accumulated_parts` directly meant every agy-backed persona
                # parked its writes with a blank "why" on the operator's
                # dialog — agy streams no deltas, so that join was always "".
                model_reasoning = assistant_prose
                audit_actions = []
                for wc in write_calls:
                    wc_name = wc.get("name", "")
                    wc_args = wc.get("arguments", {})
                    
                    # Extract binding and sensitivity from definition
                    defn = get_tool_definition(wc_name) or {}
                    binding = defn.get("service_binding")
                    caps = defn.get("capabilities") or {}
                    sensitivity = caps.get("sensitivity")
                    
                    # Fetch enrichment info (e.g. ticket number/title)
                    enrichment = await self.tool_manager.enrich_audit_action(wc_name, wc_args)
                    
                    audit_actions.append({
                        "tool": wc_name,
                        "arguments": wc_args,
                        "irreversible": is_irreversible(wc_name, wc_args),
                        "always_confirm": wc_name in ALWAYS_CONFIRM_TOOLS,
                        "service_binding": binding,
                        "sensitivity": sensitivity,
                        "enrichment": enrichment,
                    })

                audit_info: Dict[str, Any] = {
                    "actions": audit_actions,
                    "tainted": turn_tainted,
                    "taint_sources": taint_sources,
                    "model_reasoning": model_reasoning or None,
                    "execution_mode": persona.get_execution_mode().name,
                }
                # Egress scrub (DP-225 boundary 2): scrub the whole audit_info
                # once at the seam, so EVERY secret-bearing field — action
                # arguments, model_reasoning, enrichment — is redacted before it
                # is persisted to Agent_Actions or echoed to the UI and the
                # confirmation text built below. Scrubbing the dict (not each
                # field) means fields added here later are covered automatically.
                # pending_writes (raw write_calls) stays unscrubbed so the
                # approved write still executes with real argument values.
                audit_info = cast(Dict[str, Any], get_scrubber().scrub(audit_info))

                # DP-297: one park per write call, not one park per iteration.
                # Independent approve/deny is the whole point — a burst of three
                # writes must produce three separately resolvable proposals.
                for wc, action in zip(write_calls, audit_info["actions"]):
                    # Duplicate guard. The `instruction` below asks the model
                    # not to re-submit, but that is model compliance, not
                    # enforcement — and a model that ignores it would hand the
                    # operator N identical affordances to clear, each of which
                    # executes the write again if approved. Answer the call so
                    # the block stays paired, but create no second park.
                    existing = pending_lookup(wc) if pending_lookup else None
                    if existing is not None:
                        logger.info(
                            "tool-loop iter %d: %s re-proposed while token %s "
                            "is still pending — not parking a duplicate",
                            iter_idx, wc.get("name"), existing,
                        )
                        conversation_history.append({
                            "role": "tool",
                            "tool_call_id": wc.get("id"),
                            "name": wc.get("name"),
                            "content": json.dumps({
                                "status": PARK_STATUS_DUPLICATE,
                                "token": existing,
                                "instruction": (
                                    "You already proposed this exact action "
                                    "and it is still awaiting the operator. It "
                                    "was NOT queued a second time. Do not "
                                    "propose it again; wait for the outcome."
                                ),
                            }),
                        })
                        yield _WriteDuplicateEvent(
                            token=existing, call_id=wc.get("id"),
                        )
                        continue

                    # DP-319: the same guard for a proposal that was already
                    # DECIDED. `pending_lookup` cannot see this case — during the
                    # continuation turn the park being resolved has already been
                    # taken out of the pending set, and that turn is precisely
                    # when the model re-proposes, because it is re-reading its
                    # own tool span. Parking a second copy and approving it
                    # executes an irreversible write twice.
                    #
                    # No `_WriteDuplicateEvent`: that event exists so a later
                    # resolution can correct a "still awaiting" entry, and this
                    # entry is already terminal.
                    #
                    # The caller decides WHEN to consult this — it is scoped to
                    # continuation turns, because suppressing here produces no
                    # affordance on any surface, so on an ordinary turn it would
                    # swallow a repeat the operator meant.
                    resolved = resolved_lookup(wc) if resolved_lookup else None
                    if resolved is not None:
                        outcome = resolved.get("resolution")
                        logger.info(
                            "tool-loop iter %d: %s re-proposed after token %s "
                            "was already resolved (%s) — not parking it again",
                            iter_idx, wc.get("name"), resolved.get("token"),
                            outcome,
                        )
                        # An outcome of `approved_but_failed` means the tool ran
                        # and then raised, so the effect is genuinely unknown —
                        # the same shape as INTERRUPTED_INSTRUCTION, and for the
                        # same reason: "it failed" reads as a plain retryable
                        # error, and the retry is a possible second execution of
                        # an irreversible action.
                        if outcome == PARK_STATUS_FAILED:
                            instruction = (
                                "You already proposed this exact action; the "
                                "operator approved it and the tool then errored, "
                                "so whether it took effect is unknown. It was "
                                "NOT queued again. Do NOT assume either outcome "
                                "— check the current state and report what you "
                                "find."
                            )
                        else:
                            instruction = (
                                "You already proposed this exact action and "
                                "the operator decided it. It was NOT queued "
                                "again. Report the outcome above; do not "
                                "re-propose it."
                            )
                        conversation_history.append({
                            "role": "tool",
                            "tool_call_id": wc.get("id"),
                            "name": wc.get("name"),
                            "content": json.dumps({
                                "status": PARK_STATUS_ALREADY_RESOLVED,
                                "outcome": outcome,
                                "instruction": instruction,
                            }),
                        })
                        continue

                    token = uuid.uuid4().hex

                    # Append a REAL synthetic tool result. This is what lets the
                    # loop continue past a park at all: an assistant message
                    # whose tool_calls have no matching result is rejected
                    # outright by Anthropic and Gemini on the next iteration.
                    #
                    # The `instruction` field is the re-proposal guard, and it
                    # lives here rather than in the system prompt because a tool
                    # result is provider-neutral — no prompt-assembly change, and
                    # every provider surfaces it identically.
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": wc.get("id"),
                        "name": wc.get("name"),
                        "content": json.dumps({
                            "status": PARK_STATUS_AWAITING,
                            "token": token,
                            "instruction": (
                                "Proposal queued for the operator. Do NOT "
                                "re-submit it; you will be re-invoked with the "
                                "result once it is approved or denied."
                            ),
                        }),
                    })

                    yield WriteParkedEvent(
                        token=token,
                        write_call=wc,
                        # Single-action slice: each park carries only its own
                        # action, so a surface renders one dialog per proposal
                        # and the audit row describes exactly what was gated.
                        audit_info={**audit_info, "actions": [action]},
                        confirmation_text=_render_confirmation_text(
                            action, turn_tainted, taint_sources,
                        ),
                        turn_tainted=turn_tainted,
                    )

                # THE DP-297 CHANGE: fall through to the next iteration instead
                # of returning. The model can now keep working — and propose
                # again — while the operator decides.
                continue

            # If we reach here, there were no write_calls this iteration.

        self._log_turn_exit(
            call_tally, iterations_used, calls_used,
            budget_exhausted=calls_used >= self.max_tool_calls,
        )
        async for exit_ev in self._finish_exhausted_turn(
            persona=persona,
            persona_config=persona_config,
            conversation_history=conversation_history,
            history_start=history_start,
            params=params,
            local_inference_config=local_inference_config,
            calls_used=calls_used,
            iterations_used=iterations_used,
            turn_tainted=turn_tainted,
        ):
            yield exit_ev

    def _log_turn_exit(
        self,
        call_tally: Dict[Tuple[str, str], int],
        iterations_used: int,
        calls_used: int,
        *,
        budget_exhausted: bool,
    ) -> None:
        """Log a turn that ended on a limit, at the severity it deserves.

        INFO, not ERROR. The whole premise of DP-335 is that spending the
        budget is a normal, answerable outcome — the user-facing text was
        rewritten precisely because the old wording described a malfunction
        that had not happened. Logging it at ERROR made the same false claim to
        every log shipper and alert rule watching this process, on turns where
        the wrap-up produced a good answer and every call returned `ok`.

        A runaway-guard trip IS odd — it is unreachable at the shipped limits
        (see `MAX_TOOL_ITERATIONS`) — so that arm keeps WARNING.
        """
        log = logger.info if budget_exhausted else logger.warning
        log(
            "Tool budget spent: %d call(s) over %d iteration(s) "
            "(limits: %d calls, %d iterations — %s tripped).",
            calls_used, iterations_used,
            self.max_tool_calls, self.max_iterations,
            "call budget" if budget_exhausted else "runaway guard",
        )
        _log_turn_call_tally(call_tally, iterations_used, calls_used)

    async def _finish_exhausted_turn(
        self,
        *,
        persona: Persona,
        persona_config: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
        history_start: int,
        params: Any,
        local_inference_config: Optional[Dict[str, Any]],
        calls_used: int,
        iterations_used: int,
        turn_tainted: bool,
    ) -> AsyncIterator[LoopEvent]:
        """The whole exit for a turn that ran out of budget.

        Lifted out of `run` because that generator already owns the streaming
        loop, the park path, the duplicate guard and the resolved guard; this
        is straight-line exit handling whose only branch is "did the wrap-up
        produce prose", so keeping it inline only bought `run` complexity.

        DP-335: ask for a real answer before giving up. Everything needed to
        respond is already in `conversation_history` at this point — the turn
        that motivated this had the answer sitting in its second tool result
        and still ended on a canned sentence that read as a malfunction. One
        completion with `tools=None` converts that transcript into prose; the
        deterministic call list is pinned underneath it as ground truth.
        """
        budget_exhausted = calls_used >= self.max_tool_calls
        wrap_text, wrap_payload = await self._answer_without_tools(
            persona=persona,
            persona_config=persona_config,
            conversation_history=conversation_history,
            params=params,
            local_inference_config=local_inference_config,
        )
        if wrap_payload:
            yield _ApiPayloadEvent(payload=wrap_payload, iter_idx=iterations_used)

        if wrap_text:
            footer = render_call_summary_footer(
                conversation_history, history_start,
                used=calls_used, budget=self.max_tool_calls,
                exhausted=budget_exhausted,
            )
            # LLM_GENERATION, not DEV_COMMAND: this is a real answer to the
            # user's question, so it must be persisted AND retained like any
            # other. The canned fault it replaces was correctly excluded from
            # the memory bank by the `response_type` gate in `_orchestrate`;
            # excluding *this* would drop the turn's only conclusion.
            #
            # Only the PROSE is retained, though. `retain_text` splits the two
            # halves because the footer is a machine-generated listing of tool
            # names and arguments, not something the persona said — embedding
            # it would make `hf_search {"query": …} — ok` a recallable semantic
            # memory and replay it next turn as the persona's own prior words.
            yield _LoopFinishedEvent(
                final_text=(wrap_text + "\n" + footer).rstrip(),
                response_type=ResponseType.LLM_GENERATION,
                tool_context_json=seal_tool_context(
                    conversation_history, history_start, SEAL_MAX_ITERATIONS,
                ),
                turn_tainted=turn_tainted,
                retain_text=wrap_text.rstrip(),
            )
            return
        # The wrap-up is best-effort. A provider that is down or returns
        # nothing must not turn an exhausted turn into a silent one, so fall
        # back to the deterministic list on its own.
        yield _LoopFinishedEvent(
            final_text=render_max_iteration_text(
                conversation_history, history_start,
                used=calls_used, budget=self.max_tool_calls,
                exhausted=budget_exhausted,
            ).rstrip(),
            response_type=ResponseType.DEV_COMMAND,
            tool_context_json=seal_tool_context(
                conversation_history, history_start, SEAL_MAX_ITERATIONS,
            ),
            turn_tainted=turn_tainted,
        )

    async def _answer_without_tools(
        self,
        *,
        persona: Persona,
        persona_config: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
        params: Any,
        local_inference_config: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """One toolless completion over the exhausted turn's own transcript.

        Returns `(text, api_payload)`; `text` is `None` if the provider failed
        or said nothing, which the caller treats as "fall back to the list".
        Every exception is swallowed on purpose — this runs on a path that is
        already an unhappy ending, and letting it raise would convert a turn
        that merely ran long into an error the user has to interpret.

        `_EXHAUSTION_NUDGE` is appended to the wire messages only, never to
        `conversation_history`: the history is what gets sealed and persisted,
        and a synthetic instruction in it would replay to the model next turn
        as something the user said.

        No `image_url` and no `tools`. The image belongs to iteration 0 and
        re-sending it buys nothing; the tools are withheld because the budget
        is spent, and offering them would invite a call this path cannot run.
        """
        messages = build_wire_messages(persona, conversation_history)
        messages.append({"role": "user", "content": _EXHAUSTION_NUDGE})

        parts: List[str] = []
        done_text: Optional[str] = None
        payload: Optional[Dict[str, Any]] = None
        try:
            async for ev in self.text_engine.stream_messages(
                persona_config, messages, params, tools=None,
                local_inference_config=local_inference_config,
            ):
                etype = ev.get("type")
                if etype == "api_payload":
                    payload = ev.get("payload")
                elif etype == "text_delta":
                    parts.append(ev.get("text") or "")
                elif etype == "done":
                    done_text = ev.get("full_text")
        except Exception as e:
            logger.warning(
                "Tool-budget wrap-up generation failed (%s); falling back to "
                "the call list alone.", e,
            )
            return None, payload

        # `or`, not `is not None`: a provider can report `done` with an empty
        # `full_text` while having streamed the answer as deltas — every
        # one-shot result the driver classifies as `tool_calls` does exactly
        # that, and so do truncated streams. Preferring an empty `full_text`
        # over the deltas already in hand threw a real answer away and dropped
        # the turn to the canned list.
        text = (done_text or "".join(parts)).strip()
        return (text or None), payload

    async def _execute_calls(
        self,
        calls: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]],
        group_id: Optional[str] = None,
    ) -> AsyncIterator[LoopEvent]:
        """Execute a batch of tool calls, yielding start/result events
        and appending results to the shared conversation history. Calls in
        one batch share a `group_id` and are dispatched concurrently (they
        were grouped precisely because they're independent); results are
        appended/emitted in the original order so the model sees a stable
        transcript. Tool errors surface via `ToolCallResultEvent.error` and
        are also threaded into the LLM-visible result string so the model can
        adapt rather than seeing a hard stop."""
        # Resolve identity + emit all starts before any execution.
        resolved: List[Dict[str, Any]] = []
        for call_item in calls:
            tool_name = call_item.get("name", "")
            tool_args = call_item.get("arguments", {}) or {}
            call_id = call_item.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            resolved.append({"name": tool_name, "args": tool_args, "call_id": call_id})
            yield ToolCallStartEvent(
                tool_name=tool_name,
                arguments=tool_args,
                call_id=call_id,
                group_id=group_id,
            )

        async def _run_one(name: str, args: Dict[str, Any]) -> Any:
            try:
                return await self.tool_manager.execute_tool(name, **args)
            except Exception as e:
                logger.error(
                    f"Tool {name} raised unexpectedly: {e}", exc_info=True,
                )
                return {"error": f"Tool execution failed: {e}"}

        results = await asyncio.gather(
            *(_run_one(r["name"], r["args"]) for r in resolved)
        )

        # Append/emit in original order — concurrency must not reorder the
        # transcript the model reads next iteration.
        for r, tool_result in zip(resolved, results):
            # Egress scrub (DP-225 boundary 1): redact any registered secret
            # before the serialized result reaches BOTH the LLM-visible history
            # and the UI event, so both stay consistent and secret-free.
            result_str = cast(str, get_scrubber().scrub(json.dumps(tool_result)))
            err_str: Optional[str] = None
            # Shared with the gated-write path (DP-323): both used to re-derive
            # "did this fail" from the envelope's `error` key alone, and both
            # therefore read a handler that reports failure by *returning*
            # (`{"status": "error"}`, `{"executed": False}`) as a success. Here
            # that only mis-coloured a ToolCard; there it wrote a failed
            # irreversible write into history as `approved`.
            failure = tool_error(tool_result)
            if failure:
                # Egress scrub (DP-225 boundary 1): the error is surfaced raw in
                # ToolCallResultEvent.error (portal SSE / ToolCard), so redact it
                # exactly like result_str above — the sibling result field being
                # scrubbed is not enough on its own.
                err_str = cast(str, get_scrubber().scrub(failure))

            conversation_history.append({
                "role": "tool",
                "tool_call_id": r["call_id"],
                "name": r["name"],
                "content": result_str,
            })

            yield ToolCallResultEvent(
                call_id=r["call_id"],
                tool_name=r["name"],
                result=result_str,
                error=err_str,
                group_id=group_id,
            )
