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
import json
import logging
import uuid
from dataclasses import dataclass
from typing import (
    Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, Union, cast,
)

from config.global_config import MAX_TOOL_CALLS
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
from src.tools.tool_manager import ToolManager

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
class _LoopFinishedEvent:
    """Loop-internal terminal event. Carries the resolved state so the
    orchestrator can persist the assistant turn / re-emit a public DoneEvent."""
    final_text: str
    response_type: ResponseType
    tool_context_json: Optional[str] = None
    turn_tainted: bool = False


LoopEvent = Union[
    TokenEvent, ErrorEvent,
    ToolCallStartEvent, ToolCallResultEvent,
    _ApiPayloadEvent, _LoopFinishedEvent, _ToolContextEvent, WriteParkedEvent,
]

# Status values for a gated write's synthetic tool result. These are what the
# model reads in replayed history, and `PARK_STATUS_AWAITING` is the entry
# `ConfirmationManager` later patches in place once the operator decides.
PARK_STATUS_AWAITING = "awaiting_human_approval"
PARK_STATUS_APPROVED = "approved"
PARK_STATUS_DENIED = "denied"
PARK_STATUS_EXPIRED = "expired"
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
        args = repr(object())
    return (name, args)

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
        max_iterations: int = MAX_TOOL_CALLS,
    ) -> None:
        self.text_engine = text_engine
        self.tool_manager = tool_manager
        self.max_iterations = max_iterations

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
        """
        persona_config = persona.get_config_for_engine()
        history_start = (
            history_start_override if history_start_override is not None
            else len(conversation_history)
        )
        taint_sources: List[str] = list(initial_taint_sources or [])
        # turn_tainted is passed in to support conversation-level stickiness

        for iter_idx in range(self.max_iterations):
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
                yield _LoopFinishedEvent(
                    final_text=final_text,
                    response_type=ResponseType.LLM_GENERATION,
                    tool_context_json=tool_context_json,
                    turn_tainted=turn_tainted,
                )
                return

            group_id = f"iter{iter_idx}_{uuid.uuid4().hex[:8]}"
            for call_item in tool_calls_collected:
                call_item["group_id"] = group_id
            conversation_history.append(
                {"role": "assistant", "tool_calls": tool_calls_collected}
            )
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
                model_reasoning = "".join(accumulated_parts).strip()
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

        logger.error(f"Exceeded max tool iterations ({self.max_iterations}).")
        yield _LoopFinishedEvent(
            final_text="I seem to be stuck in a loop. Could you please clarify your request?",
            response_type=ResponseType.DEV_COMMAND,
            tool_context_json=seal_tool_context(
                conversation_history, history_start, SEAL_MAX_ITERATIONS,
            ),
            turn_tainted=turn_tainted,
        )

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
            if isinstance(tool_result, dict) and tool_result.get("error"):
                # Egress scrub (DP-225 boundary 1): the error is surfaced raw in
                # ToolCallResultEvent.error (portal SSE / ToolCard), so redact it
                # exactly like result_str above — the sibling result field being
                # scrubbed is not enough on its own.
                err_str = cast(str, get_scrubber().scrub(str(tool_result["error"])))

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
