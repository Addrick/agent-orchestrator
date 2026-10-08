# src/engine/providers/openai.py
"""OpenAI provider (DP-244) — the first provider extracted end-to-end as the
proof-of-pattern for the Provider ABC family.

The request-building, dump-redaction, image-attach, client lazy-init, and the
canonical token-stream generator all live here. ``TextEngine`` retains a thin
``_stream_openai_response`` delegator (the seam the driver and the existing
driver-policy tests inject through); ``OpenAIProvider.stream`` routes back
through it so behaviour stays byte-identical.
"""

import json
import logging
from contextlib import aclosing
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Dict, List, Optional

from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI, APIStatusError, APITimeoutError

from config import global_config
from src.llm_errors import LLMCommunicationError
from src.security.vault import get_vault

from .base import Provider
from ._shared import extract_system_prompt, parse_openai_tool_calls

if TYPE_CHECKING:
    from src.engine.driver import TextEngine

logger = logging.getLogger(__name__)


async def get_openai_client(engine: "TextEngine") -> AsyncOpenAI:
    """Initializes and returns the (lazily cached) OpenAI client. The client is
    cached on the engine so a process reuses one connection pool."""
    if engine.openai_client is None:
        api_key = get_vault().get("OPENAI_API_KEY")
        if not api_key:
            raise LLMCommunicationError("OPENAI_API_KEY not set — skipping OpenAI provider.")
        engine.openai_client = AsyncOpenAI(api_key=api_key)
    return engine.openai_client


def attach_openai_image(messages: List[Dict[str, Any]], image_url: str) -> None:
    """Attaches an image URL to the last user message for OpenAI."""
    last_message = messages[-1]
    if last_message['role'] != 'user':
        return
    if isinstance(last_message['content'], str):
        last_message['content'] = [{"type": "text", "text": last_message['content']}]
    last_message['content'].append({"type": "image_url", "image_url": {"url": image_url}})


def to_openai_wire_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Translate one history turn into chat-completions wire shape (DP-417).

    The tool loop records a call as ``{"id", "name", "arguments": dict}`` and a
    result as ``{"role": "tool", "tool_call_id", "name", "content"}``. The wire
    wants ``{"id", "type": "function", "function": {"name", "arguments": str}}``
    and no ``name`` on a tool turn, so a tool turn sent as stored is malformed
    for this API. Before DP-417 only the kobold-native local path replayed tool
    turns, and it rendered them into its own prompt text. Every other turn is
    passed through unchanged."""
    role = msg.get("role")
    if role == "assistant" and msg.get("tool_calls"):
        calls = []
        for i, tc in enumerate(msg["tool_calls"]):
            if "function" in tc:  # already wire shape
                calls.append(tc)
                continue
            args = tc.get("arguments", {})
            calls.append({
                "id": tc.get("id") or f"call_{tc.get('name')}_{i}",
                "type": "function",
                "function": {
                    "name": tc.get("name"),
                    "arguments": args if isinstance(args, str) else json.dumps(args),
                },
            })
        return {"role": "assistant", "content": msg.get("content") or None, "tool_calls": calls}
    if role == "tool":
        content = msg.get("content")
        return {
            "role": "tool",
            "tool_call_id": msg.get("tool_call_id"),
            "content": content if isinstance(content, str) else json.dumps(content),
        }
    return msg


def build_openai_params(config: Dict[str, Any], history_object: Dict[str, Any],
                        tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Builds the chat.completions request kwargs. Single source of truth for
    the canonical streaming driver (wire-payload parity pinned by
    tests/test_engine_payload_parity.py)."""
    # DP-317: a leading system turn is *merged* onto the persona prompt, never
    # substituted for it. This used to inline its own split that dropped
    # `persona_prompt` whenever the history opened with a system turn — see
    # `extract_system_prompt` for why that was a transcription slip and not a
    # design choice.
    system_prompt, history_to_process = extract_system_prompt(history_object)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    # Add remaining history
    for msg in history_to_process:
        if msg["role"] == "system":
            continue
        messages.append(to_openai_wire_message(msg))

    if history_object["current_message"].get("image_url"):
        attach_openai_image(messages, history_object["current_message"]["image_url"])

    api_params: Dict[str, Any] = {
        "model": config["model_name"],
        "messages": messages,
        "max_tokens": config.get("max_output_tokens") or global_config.DEFAULT_TOKEN_LIMIT,
        "temperature": config.get("temperature"),
        "top_p": config.get("top_p")
    }
    if tools:
        api_params["tools"] = [
            {"type": "function", "function": t["function"]}
            for t in tools if "function" in t
        ]
        api_params["tool_choice"] = "auto"

    api_params = {k: v for k, v in api_params.items() if v is not None}
    return api_params


def openai_dump_params(api_params: Dict[str, Any]) -> Dict[str, Any]:
    """Log-safe copy of the request kwargs: tools listed by name."""
    dump = dict(api_params)
    if "tools" in dump:
        dump["tools"] = [tool.get("function", {}).get("name", "unknown")
                         for tool in dump["tools"]]
    return dump


async def stream_openai(
    engine: "TextEngine", config: Dict[str, Any], history_object: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Canonical OpenAI driver (DP-206): a true SDK token stream emitting the
    unified event shape (api_payload → text_delta* → [tool_calls] → done). The
    one-shot path is `collect_stream` over this generator."""
    client = await get_openai_client(engine)
    api_params = build_openai_params(config, history_object, tools)
    async with aclosing(stream_chat_completions(client, api_params, "OpenAI")) as events:
        async for ev in events:
            yield ev


_THINK_OPEN = "<think>\n"
_THINK_CLOSE = "\n</think>\n"


async def stream_chat_completions(
    client: AsyncOpenAI, api_params: Dict[str, Any], provider_label: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Run one streaming chat-completions request and emit the unified event
    shape. Shared by every server that speaks this API: OpenAI itself and,
    since DP-417, the local model server (KoboldCPP or Strata).

    - `reasoning_content` deltas (a local server's thinking) are folded into
      the text as one `<think>…</think>` block. That is the shape the
      kobold-native transport produced and everything downstream already
      handles: the portal renders it, retain strips it (DP-409).
    - Closing this generator early closes the HTTP stream, which is how a
      generation is cancelled: both local servers stop on disconnect. Every
      generator that delegates to this one must wrap it in `aclosing` —
      `async for` does not close an inner async generator when the outer one
      is closed, so without it the cancel never reaches the socket.
    """
    yield {"type": "api_payload", "payload": openai_dump_params(api_params)}

    text_parts: List[str] = []
    in_think = False
    # index → {"id", "name", "arguments"} accumulated across delta chunks
    raw_calls: Dict[int, Dict[str, Any]] = {}
    stream: Any = None
    try:
        stream = await client.chat.completions.create(**api_params, stream=True)
        async for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta
            extra = getattr(delta, "model_extra", None)
            reasoning = extra.get("reasoning_content") if isinstance(extra, dict) else None
            if isinstance(reasoning, str) and reasoning:
                piece = reasoning if in_think else _THINK_OPEN + reasoning
                in_think = True
                text_parts.append(piece)
                yield {"type": "text_delta", "text": piece}
            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                if in_think:
                    # KoboldCPP opens the answer with the blank line that
                    # followed `</think>` in the raw stream; drop it.
                    content = _THINK_CLOSE + content.lstrip()
                    in_think = False
                text_parts.append(content)
                yield {"type": "text_delta", "text": content}
            for tc in (getattr(delta, "tool_calls", None) or []):
                idx = getattr(tc, "index", 0) or 0
                slot = raw_calls.setdefault(idx, {"id": None, "name": "", "arguments": ""})
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    slot["arguments"] += getattr(fn, "arguments", None) or ""
    except (APIStatusError, APITimeoutError) as e:
        rate_limited = isinstance(e, APIStatusError) and e.status_code == 429
        is_server_error = isinstance(e, APIStatusError) and e.status_code >= 500
        logger.error(f"{provider_label} API error: {e}", exc_info=not is_server_error)
        raise LLMCommunicationError(f"{provider_label} API returned an error: {e}",
                                    api_payload=api_params, rate_limited=rate_limited) from e
    except Exception as e:
        logger.error(f"An unexpected {provider_label} error occurred: {e}", exc_info=True)
        raise LLMCommunicationError(f"An unexpected error occurred with the {provider_label} API.",
                                    api_payload=api_params) from e
    finally:
        if stream is not None:
            await stream.close()

    if in_think:
        # Thinking that ran straight into tool calls (or the end) never saw a
        # content delta to close it.
        text_parts.append(_THINK_CLOSE)
        yield {"type": "text_delta", "text": _THINK_CLOSE}

    if raw_calls:
        # Reuse the one parser: wrap accumulated fragments in the same
        # attribute shape the SDK's non-streaming message objects expose.
        wrapped = [
            SimpleNamespace(
                id=slot["id"] or f"call_{slot['name']}_{i}",
                function=SimpleNamespace(name=slot["name"], arguments=slot["arguments"]),
            )
            for i, (_, slot) in enumerate(sorted(raw_calls.items()))
        ]
        tool_calls = parse_openai_tool_calls(wrapped)
        yield {"type": "tool_calls", "calls": tool_calls}
        yield {"type": "done", "full_text": ""}
    else:
        yield {"type": "done", "full_text": "".join(text_parts)}


class OpenAIProvider(Provider):
    """OpenAI chat-completions provider (model names starting with ``gpt``)."""

    def __init__(self, engine: "TextEngine") -> None:
        self._engine = engine

    #: name of the engine seam method (back-compat for `_get_provider_route`).
    route_method_name = "_stream_openai_response"

    def matches(self, model_name: str) -> bool:
        return model_name.startswith("gpt")

    def limiters_for(self, model_name: str) -> List[AsyncLimiter]:
        return [self._engine._openai_limiter]

    async def stream(
        self,
        persona_config: Dict[str, Any],
        history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        local_inference_config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        # Route through the engine seam so instance/class patches of
        # `_stream_openai_response` (the driver-policy tests) still intercept.
        async for ev in self._engine._stream_openai_response(persona_config, history_object, tools):
            yield ev
