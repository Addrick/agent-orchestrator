# src/engine/providers/anthropic.py
"""Anthropic provider (DP-244) — second provider extracted end-to-end, mirroring
the openai pattern.

Request-building, dump-redaction, image-attach, client lazy-init, and the
canonical token-stream generator live here. ``TextEngine`` keeps a thin
``_stream_anthropic_response`` delegator (the seam the driver and the existing
driver-policy tests inject through); ``AnthropicProvider.stream`` routes back
through it so behaviour stays byte-identical.
"""

import base64
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

import aiohttp
import anthropic
from aiolimiter import AsyncLimiter

from config import global_config
from src.llm_errors import LLMCommunicationError
from src.security.vault import get_vault

from .base import Provider
from ._shared import extract_system_prompt

if TYPE_CHECKING:
    from src.engine.driver import TextEngine

logger = logging.getLogger(__name__)


def get_anthropic_client(engine: "TextEngine") -> "anthropic.AsyncAnthropic":
    """Initializes and returns the (lazily cached) async Anthropic client. The
    client is cached on the engine so a process reuses one connection pool."""
    if engine.anthropic_client is None:
        api_key = get_vault().get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMCommunicationError("ANTHROPIC_API_KEY not set — skipping Anthropic provider.")
        engine.anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
    return engine.anthropic_client


async def attach_anthropic_image(engine: "TextEngine", messages: List[Dict[str, Any]],
                                 image_url: str) -> None:
    """Downloads and attaches an image to the last user message for Anthropic.
    Routes the download through the engine seam (``_download_image``) so test
    patches of that method still intercept."""
    last_message = messages[-1]
    if last_message['role'] != 'user':
        return
    if isinstance(last_message['content'], str):
        last_message['content'] = [{"type": "text", "text": last_message['content']}]
    try:
        image_bytes, mime_type = await engine._download_image(image_url)
        if mime_type not in ['image/jpeg', 'image/png', 'image/webp', 'image/gif']:
            logger.warning(f"Unsupported image MIME type '{mime_type}' for Claude. Skipping image.")
        else:
            last_message['content'].append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64.b64encode(image_bytes).decode('utf-8'),
                },
            })
    except aiohttp.ClientError as e:
        logger.error(f"Failed to download image from {image_url}: {e}")


async def build_anthropic_params(engine: "TextEngine", config: Dict[str, Any],
                                 history_object: Dict[str, Any],
                                 tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Builds the messages request kwargs for Anthropic. Single source of truth
    for the canonical streaming driver (pinned by
    tests/test_engine_payload_parity.py)."""
    system_prompt, history = extract_system_prompt(history_object)

    if history_object["current_message"].get("image_url"):
        await attach_anthropic_image(engine, history, history_object["current_message"]["image_url"])

    api_params: Dict[str, Any] = {
        "model": config["model_name"],
        "system": system_prompt,
        "messages": history,
        "max_tokens": config.get("max_output_tokens") or global_config.DEFAULT_TOKEN_LIMIT,
        "temperature": config.get("temperature"),
        "top_p": config.get("top_p"),
        "top_k": config.get("top_k")
    }
    if tools:
        api_params["tools"] = [
            {"name": t["function"]["name"],
             "description": t["function"].get("description", ""),
             "input_schema": t["function"].get("parameters", {})}
            for t in tools if "function" in t
        ]

    return {k: v for k, v in api_params.items() if v is not None}


def anthropic_dump_params(api_params: Dict[str, Any]) -> Dict[str, Any]:
    """Log-safe copy of the request kwargs: tools listed by name."""
    dump = dict(api_params)
    if "tools" in dump:
        dump["tools"] = [tool.get("name", "unknown") for tool in dump["tools"]]
    return dump


async def stream_anthropic(
    engine: "TextEngine", config: Dict[str, Any], history_object: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Canonical Anthropic driver (DP-206): `messages.stream(...)` with the SDK's
    accumulator, emitting the unified event shape (api_payload → text_delta* →
    [tool_calls] → done). The one-shot path is `collect_stream` over this
    generator. Uses `AsyncAnthropic` (DP-211) so token iteration never blocks the
    event loop; request kwargs are identical to the sync client's (frozen payload
    goldens)."""
    client = get_anthropic_client(engine)
    api_params = await build_anthropic_params(engine, config, history_object, tools)
    yield {"type": "api_payload", "payload": anthropic_dump_params(api_params)}

    tool_calls: Optional[List[Dict[str, Any]]] = None
    response_content = ""
    try:
        async with client.messages.stream(**api_params) as stream:
            async for text_chunk in stream.text_stream:
                if text_chunk:
                    yield {"type": "text_delta", "text": text_chunk}
            response = await stream.get_final_message()

        if response.stop_reason == "tool_use":
            tool_calls = []
            for content_block in response.content:
                if content_block.type == 'tool_use':
                    tool_calls.append({
                        "id": content_block.id,
                        "name": content_block.name,
                        "arguments": content_block.input
                    })
        else:
            # content[0] is a TextBlock for a normal stop; .text is unnarrowed in
            # the SDK union (ignore). A non-text first block (e.g. a thinking
            # block) raises AttributeError -> caught below -> retried, matching
            # the pre-DP-244 behaviour.
            response_content = response.content[0].text or ""  # type: ignore[union-attr]

    except anthropic.APIError as e:
        status_code = getattr(e, 'status_code', None)
        rate_limited = status_code == 429
        is_server_error = isinstance(status_code, int) and status_code >= 500
        logger.error(f"Anthropic API error: {e}", exc_info=not is_server_error)
        raise LLMCommunicationError(f"Anthropic API returned an error: {e}", api_payload=api_params,
                                    rate_limited=rate_limited) from e
    except Exception as e:
        logger.error(f"An unexpected Anthropic error occurred: {e}", exc_info=True)
        raise LLMCommunicationError("An unexpected error occurred with the Anthropic API.",
                                    api_payload=api_params) from e

    if tool_calls is not None:
        yield {"type": "tool_calls", "calls": tool_calls}
        yield {"type": "done", "full_text": ""}
    else:
        yield {"type": "done", "full_text": response_content}


class AnthropicProvider(Provider):
    """Anthropic messages provider (model names containing ``claude``)."""

    def __init__(self, engine: "TextEngine") -> None:
        self._engine = engine

    #: name of the engine seam method (back-compat for `_get_provider_route`).
    route_method_name = "_stream_anthropic_response"

    def matches(self, model_name: str) -> bool:
        return "claude" in model_name

    def limiters_for(self, model_name: str) -> List[AsyncLimiter]:
        return [self._engine._anthropic_limiter]

    async def stream(
        self,
        persona_config: Dict[str, Any],
        history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        local_inference_config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        # Route through the engine seam so instance/class patches of
        # `_stream_anthropic_response` (the driver-policy tests) still intercept.
        async for ev in self._engine._stream_anthropic_response(persona_config, history_object, tools):
            yield ev
