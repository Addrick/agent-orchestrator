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

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from config.global_config import MAX_TOOL_CALLS
from src.engine import LLMCommunicationError, TextEngine
from src.generation_events import (
    ErrorEvent, ResponseType, TokenEvent,
    ToolCallResultEvent, ToolCallStartEvent,
)
from src.persona import ExecutionMode, Persona
from src.tools.definitions import WRITE_TOOLS, ALWAYS_CONFIRM_TOOLS
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


LoopEvent = Union[
    TokenEvent, ErrorEvent,
    ToolCallStartEvent, ToolCallResultEvent,
    _ApiPayloadEvent, _LoopFinishedEvent,
]


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
    ) -> AsyncIterator[LoopEvent]:
        """Yield generation events for one turn. Mutates
        `conversation_history` in-place so the orchestrator (and any
        CONFIRM-mode resume path) sees the same list."""
        persona_config = persona.get_config_for_engine()
        history_start = len(conversation_history)

        for iter_idx in range(self.max_iterations):
            api_payload: Optional[Dict[str, Any]] = None
            full_text_from_done: Optional[str] = None
            tool_calls_collected: Optional[List[Dict[str, Any]]] = None
            accumulated_parts: List[str] = []

            messages_for_llm: List[Dict[str, Any]] = (
                [{"role": "system", "content": persona.get_prompt()}]
                + list(conversation_history)
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
                )
                return

            conversation_history.append(
                {"role": "assistant", "tool_calls": tool_calls_collected}
            )
            read_calls = [c for c in tool_calls_collected if c.get("name") not in WRITE_TOOLS]
            write_calls = [c for c in tool_calls_collected if c.get("name") in WRITE_TOOLS]

            async for tool_ev in self._execute_calls(read_calls, conversation_history):
                yield tool_ev

            # Determine if we need to halt for confirmation.
            # We halt if:
            # 1. The persona is in CONFIRM mode and there are ANY write calls.
            # 2. There are specific tools in write_calls that are in ALWAYS_CONFIRM_TOOLS.
            needs_confirmation = (persona.get_execution_mode() == ExecutionMode.CONFIRM and write_calls) or \
                                any(wc.get("name") in ALWAYS_CONFIRM_TOOLS for wc in write_calls)

            if needs_confirmation and write_calls:
                descriptions = [
                    f"- **{wc.get('name')}**: {json.dumps(wc.get('arguments', {}))}"
                    for wc in write_calls
                ]
                final_text = (
                    "I'd like to perform the following actions:\n"
                    + "\n".join(descriptions)
                )
                yield _LoopFinishedEvent(
                    final_text=final_text,
                    response_type=ResponseType.PENDING_CONFIRMATION,
                    tool_context_json=None,
                    pending_writes=write_calls,
                )
                return

            async for tool_ev in self._execute_calls(write_calls, conversation_history):
                yield tool_ev

        logger.error(f"Exceeded max tool iterations ({self.max_iterations}).")
        yield _LoopFinishedEvent(
            final_text="I seem to be stuck in a loop. Could you please clarify your request?",
            response_type=ResponseType.DEV_COMMAND,
            tool_context_json=None,
        )

    async def _execute_calls(
        self,
        calls: List[Dict[str, Any]],
        conversation_history: List[Dict[str, Any]],
    ) -> AsyncIterator[LoopEvent]:
        """Execute a batch of tool calls, yielding start/result events
        and appending results to the shared conversation history. Tool
        errors surface via `ToolCallResultEvent.error` and are also
        threaded into the LLM-visible result string so the model can
        adapt rather than seeing a hard stop."""
        for call_item in calls:
            tool_name = call_item.get("name", "")
            tool_args = call_item.get("arguments", {}) or {}
            call_id = call_item.get("id") or f"call_{uuid.uuid4().hex[:12]}"

            yield ToolCallStartEvent(
                tool_name=tool_name,
                arguments=tool_args,
                call_id=call_id,
            )

            try:
                tool_result = await self.tool_manager.execute_tool(
                    tool_name, **tool_args
                )
            except Exception as e:
                logger.error(
                    f"Tool {tool_name} raised unexpectedly: {e}", exc_info=True,
                )
                tool_result = {"error": f"Tool execution failed: {e}"}

            result_str = json.dumps(tool_result)
            err_str: Optional[str] = None
            if isinstance(tool_result, dict) and tool_result.get("error"):
                err_str = str(tool_result["error"])

            conversation_history.append({
                "role": "tool",
                "tool_call_id": call_item.get("id"),
                "name": tool_name,
                "content": result_str,
            })

            yield ToolCallResultEvent(
                call_id=call_id,
                tool_name=tool_name,
                result=result_str,
                error=err_str,
            )
