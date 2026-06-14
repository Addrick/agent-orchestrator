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
from typing import Any, AsyncIterator, Dict, List, Optional, Union, cast

from config.global_config import MAX_TOOL_CALLS
from src.engine import LLMCommunicationError, TextEngine
from src.security.scrubber import get_scrubber
from src.generation_events import (
    ErrorEvent, ResponseType, TokenEvent,
    ToolCallResultEvent, ToolCallStartEvent,
)
from src.persona import Persona
from src.tools.definitions import (
    WRITE_TOOLS, ALWAYS_CONFIRM_TOOLS,
    get_tool_capabilities, is_irreversible, get_tool_definition
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
class _LoopFinishedEvent:
    """Loop-internal terminal event. Carries the resolved state so the
    orchestrator can persist the assistant turn / park CONFIRM-mode
    confirmation / re-emit a public DoneEvent."""
    final_text: str
    response_type: ResponseType
    tool_context_json: Optional[str] = None
    pending_writes: Optional[List[Dict[str, Any]]] = None
    turn_tainted: bool = False
    audit_info: Optional[Dict[str, Any]] = None
    # Index into conversation_history marking where this turn's tool messages
    # begin. Carried on the pending-write event so a resumed continuation can
    # capture the parked tool calls (+ their results) into tool_context_json
    # rather than dropping them. See ChatSystem resume path.
    tool_context_start: int = 0


LoopEvent = Union[
    TokenEvent, ErrorEvent,
    ToolCallStartEvent, ToolCallResultEvent,
    _ApiPayloadEvent, _LoopFinishedEvent,
]


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
    ) -> AsyncIterator[LoopEvent]:
        """Yield generation events for one turn. Mutates
        `conversation_history` in-place so the orchestrator (and any
        CONFIRM-mode resume path) sees the same list.

        `history_start_override` lets a resumed write-confirmation point the
        tool-context boundary back at the parked turn's first tool message, so
        the captured tool_context_json spans the whole turn (parked read calls,
        the approved write, and its result) rather than only post-resume calls.
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
                yield ErrorEvent(message=err_msg)
                return
            except Exception as e:
                logger.error(
                    f"Unexpected error during stream_messages (iter {iter_idx}): {e}",
                    exc_info=True,
                )
                yield ErrorEvent(
                    message="An internal error occurred while processing your request."
                )
                return

            if api_payload:
                yield _ApiPayloadEvent(payload=api_payload, iter_idx=iter_idx)

            if not tool_calls_collected:
                final_text = (
                    full_text_from_done if full_text_from_done is not None
                    else "".join(accumulated_parts)
                )
                tool_msgs = conversation_history[history_start:]
                tool_context_json = json.dumps(tool_msgs) if tool_msgs else None
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
            read_calls = [c for c in tool_calls_collected if c.get("name") not in WRITE_TOOLS]
            write_calls = [c for c in tool_calls_collected if c.get("name") in WRITE_TOOLS]

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
                    "(reads this iter: %s) — turn ends PENDING_CONFIRMATION",
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

                # Build human-readable confirmation text from the scrubbed actions.
                lines = ["I'd like to perform the following actions:"]
                for a in audit_info["actions"]:
                    flags = []
                    if a["service_binding"]:
                        flags.append(a["service_binding"].upper())
                    if a["sensitivity"]:
                        flags.append(a["sensitivity"].upper())
                    if a["irreversible"]:
                        flags.append("IRREVERSIBLE")
                    if a["always_confirm"]:
                        flags.append("HIGH-IMPACT")
                    
                    flag_str = f" [{', '.join(flags)}]" if flags else ""
                    enrich_str = f": **{a['enrichment']}**" if a["enrichment"] else ":"
                    lines.append(f"- **{a['tool']}**{flag_str}{enrich_str} {json.dumps(a['arguments'])}")
                
                if turn_tainted:
                    lines.append(f"\n⚠️ Context contains untrusted content from: {', '.join(taint_sources)}")

                yield _LoopFinishedEvent(
                    final_text="\n".join(lines),
                    response_type=ResponseType.PENDING_CONFIRMATION,
                    tool_context_json=None,
                    pending_writes=write_calls,
                    turn_tainted=turn_tainted,
                    audit_info=audit_info,
                    tool_context_start=history_start,
                )
                return

            # If we reach here, there were no write_calls this iteration.
            # (All write_calls are parked above; this branch only runs for
            # read-only iterations that loop back for more LLM output.)

        logger.error(f"Exceeded max tool iterations ({self.max_iterations}).")
        yield _LoopFinishedEvent(
            final_text="I seem to be stuck in a loop. Could you please clarify your request?",
            response_type=ResponseType.DEV_COMMAND,
            tool_context_json=None,
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
