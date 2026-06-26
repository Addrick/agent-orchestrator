# src/engine/providers/openai.py
"""OpenAI provider (DP-244) — the first provider extracted end-to-end as the
proof-of-pattern for the Provider ABC family.

The request-building, dump-redaction, image-attach, client lazy-init, and the
canonical token-stream generator all live here. ``TextEngine`` retains a thin
``_stream_openai_response`` delegator (the seam the driver and the existing
driver-policy tests inject through); ``OpenAIProvider.stream`` routes back
through it so behaviour stays byte-identical.
"""

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI, APIStatusError, APITimeoutError

from config import global_config
from src.llm_errors import LLMCommunicationError
from src.security.vault import get_vault

from .base import Provider
from ._shared import parse_openai_tool_calls

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


def build_openai_params(config: Dict[str, Any], history_object: Dict[str, Any],
                        tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Builds the chat.completions request kwargs. Single source of truth for
    the canonical streaming driver (wire-payload parity pinned by
    tests/test_engine_payload_parity.py)."""
    messages: List[Dict[str, Any]] = []
    message_history = history_object.get("message_history", history_object.get("history", []))
    if message_history and message_history[0]["role"] == "system":
        messages.append(message_history[0])
        history_to_process = message_history[1:]
    else:
        messages.append({"role": "system", "content": history_object["persona_prompt"]})
        history_to_process = message_history

    # Add remaining history
    for msg in history_to_process:
        if msg["role"] == "system":
            continue
        messages.append(msg)

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
    yield {"type": "api_payload", "payload": openai_dump_params(api_params)}

    text_parts: List[str] = []
    # index → {"id", "name", "arguments"} accumulated across delta chunks
    raw_calls: Dict[int, Dict[str, Any]] = {}
    try:
        stream = await client.chat.completions.create(**api_params, stream=True)
        async for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta
            content = getattr(delta, "content", None)
            if content:
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
        logger.error(f"OpenAI API error: {e}", exc_info=not is_server_error)
        raise LLMCommunicationError(f"OpenAI API returned an error: {e}", api_payload=api_params,
                                    rate_limited=rate_limited) from e
    except Exception as e:
        logger.error(f"An unexpected OpenAI error occurred: {e}", exc_info=True)
        raise LLMCommunicationError("An unexpected error occurred with the OpenAI API.",
                                    api_payload=api_params) from e

    if raw_calls:
        # Reuse the one parser: wrap accumulated fragments in the same
        # attribute shape the SDK's non-streaming message objects expose.
        wrapped = [
            SimpleNamespace(
                id=slot["id"],
                function=SimpleNamespace(name=slot["name"], arguments=slot["arguments"]),
            )
            for _, slot in sorted(raw_calls.items())
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
