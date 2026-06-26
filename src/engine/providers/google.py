# src/engine/providers/google.py
"""Google (Gemini / Gemma) provider (DP-244) — third provider extracted,
mirroring the openai/anthropic pattern.

History-building, tool-building, response-parsing, chunk-accumulation, client
lazy-init, and the canonical token-stream generator live here. ``TextEngine``
keeps thin ``_stream_google_response`` / ``_build_google_history`` /
``_parse_google_response`` delegators (the seams the driver routes through and
the existing tests call/patch directly); the provider routes back through the
stream seam so behaviour stays byte-identical.

NOTE: this module carries a whole-file ``ignore_errors`` in mypy.ini — the
google-genai SDK ships incomplete type stubs (``Part(**dict)``, ``Candidate``
attribute chains, ``genai.client``), the same legacy noise that kept
``[mypy-src.engine.driver]`` ignored before the extraction. The noise relocated
here with the code; everything else in ``src/engine`` stays strict-checked.
"""

import base64
import json
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple

import aiohttp
from aiolimiter import AsyncLimiter
from google import genai
from google.genai.types import (
    GenerateContentConfig, Tool, GoogleSearch, Candidate, FunctionDeclaration, Part, ThinkingConfig,
)

from config import global_config
from src.llm_errors import LLMCommunicationError
from src.security.vault import get_vault
from src.utils.google_utils import process_grounding_metadata

from .base import Provider
from ._shared import extract_system_prompt

if TYPE_CHECKING:
    from src.engine.driver import TextEngine

logger = logging.getLogger(__name__)


def initialize_google_client(engine: "TextEngine") -> None:
    """Initializes the Google client (cached on the engine) using the original
    project's method."""
    if engine.google_client is not None:
        return

    api_key = get_vault().get("GOOGLE_GENERATIVEAI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_GENERATIVEAI_API_KEY not found in environment.")

    client: genai.client.BaseApiClient = genai.client.BaseApiClient(api_key=api_key)
    engine.google_client = genai.client.AsyncClient(client)
    engine.google_search_tool = Tool(google_search=GoogleSearch())
    logger.info("Google AI Studio client initialized.")


async def build_google_history(
    engine: "TextEngine", system_prompt: str, history: List[Dict[str, Any]], image_url: Optional[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (history_for_api, serializable_history)."""
    history_for_api = []
    serializable_history = []
    if system_prompt:
        serializable_history.append({'role': 'system', 'parts': [{'text': system_prompt}]})

    for item in history:
        role = 'model' if item['role'] == 'assistant' else 'user'
        serializable_item = item.copy()

        if item['role'] == 'tool':
            part_dict = {'function_response': {'name': item['name'], 'response': json.loads(item['content'])}}
            if history_for_api and history_for_api[-1]['role'] == 'tool':
                history_for_api[-1]['parts'].append(Part(**part_dict))
                serializable_history[-1]['parts'].append(part_dict)
            else:
                history_for_api.append({'role': 'tool', 'parts': [Part(**part_dict)]})
                serializable_item['parts'] = [part_dict]
                serializable_history.append(serializable_item)
        elif item.get('tool_calls'):
            api_parts = []
            serializable_parts = []
            for call in item['tool_calls']:
                part_kwargs: Dict[str, Any] = {
                    'function_call': {'name': call['name'], 'args': call['arguments']}
                }
                ser_part: Dict[str, Any] = {'function_call': part_kwargs['function_call']}
                if call.get('thought_signature') is not None:
                    # Convert back from base64 string to bytes for the Google API
                    part_kwargs['thought_signature'] = base64.b64decode(call['thought_signature'])
                    ser_part['thought_signature'] = '...present...'
                api_parts.append(Part(**part_kwargs))
                serializable_parts.append(ser_part)
            if history_for_api and history_for_api[-1]['role'] == 'model':
                history_for_api[-1]['parts'].extend(api_parts)
                serializable_history[-1]['parts'].extend(serializable_parts)
            else:
                history_for_api.append({'role': 'model', 'parts': api_parts})
                serializable_item['parts'] = serializable_parts
                serializable_history.append(serializable_item)
        else:
            content_text = item['content']
            parts_for_api = [Part(text=content_text)]
            serializable_parts = [{'text': content_text}]

            if image_url and role == 'user' and item is history[-1]:
                try:
                    image_bytes, mime_type = await engine._download_image(image_url)
                    if mime_type not in ['image/jpeg', 'image/png', 'image/webp', 'image/heic', 'image/heif']:
                        logger.warning(f"Unsupported image MIME type '{mime_type}'. Skipping image.")
                    else:
                        parts_for_api.append(Part(inline_data={'data': image_bytes, 'mime_type': mime_type}))
                        serializable_parts.append({'inline_data': {'mime_type': mime_type, 'data': '...bytes...'}})
                except aiohttp.ClientError as e:
                    logger.error(f"Failed to download image from {image_url}: {e}")

            if history_for_api and history_for_api[-1]['role'] == role:
                history_for_api[-1]['parts'].extend(parts_for_api)
                serializable_history[-1]['parts'].extend(serializable_parts)
            else:
                history_for_api.append({'role': role, 'parts': parts_for_api})
                serializable_item['parts'] = serializable_parts
                serializable_history.append(serializable_item)
    return history_for_api, serializable_history


def build_google_tools(
    engine: "TextEngine", tools: Optional[List[Dict[str, Any]]]
) -> Tuple[List[Tool], Optional[Dict[str, Any]]]:
    """Returns (api_tools, tool_config_or_none)."""
    api_tools: List[Tool] = []
    tool_config = None
    if tools:
        if any(t.get('type') == 'google_grounding' for t in tools):
            api_tools.append(engine.google_search_tool)
        function_tools = [t for t in tools if t.get('type') == 'function' and t.get('function')]
        if function_tools:
            api_tools.extend([Tool(function_declarations=[FunctionDeclaration(**t['function'])])
                              for t in function_tools])
            tool_config = {"function_calling_config": {"mode": "AUTO"}}
    return api_tools, tool_config


def parse_google_response(
    response_obj: Any, api_params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parses Google response into standard result format.
    Raises LLMCommunicationError if response was blocked."""
    if response_obj.prompt_feedback and response_obj.prompt_feedback.block_reason:
        raise LLMCommunicationError(
            f"Response blocked by Google due to {response_obj.prompt_feedback.block_reason.name}.")

    candidate: Optional[Candidate] = response_obj.candidates[0] if response_obj.candidates else None
    if not candidate or not candidate.content or not candidate.content.parts:
        return {}, api_params

    tool_calls: List[Dict[str, Any]] = []
    for i, part in enumerate(candidate.content.parts):
        if part.function_call:
            arguments = {k: v for k, v in part.function_call.args.items()}
            call_dict: Dict[str, Any] = {
                "id": f"call_{part.function_call.name}_{i}",
                "name": part.function_call.name,
                "arguments": arguments,
            }
            thought_sig = getattr(part, 'thought_signature', None)
            if isinstance(thought_sig, (bytes, bytearray)):
                # thought_signature is bytes, must be serializable for our history storage
                call_dict["thought_signature"] = base64.b64encode(thought_sig).decode('utf-8')
            tool_calls.append(call_dict)
    if tool_calls:
        return {"type": "tool_calls", "calls": tool_calls}, api_params

    base_text_from_response = "".join(
        part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text)
    final_text_content, search_query_display, citations_display = process_grounding_metadata(
        base_text_from_response, candidate.grounding_metadata, logger
    )
    if search_query_display:
        final_text_content += search_query_display
    if citations_display:
        final_text_content += citations_display

    return {"type": "text", "content": final_text_content.strip()}, api_params


def build_google_dump_params(
    model_name: str, content_config: Dict[str, Any], serializable_history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Builds a serializable version of API params for logging."""
    dump_config = content_config.copy()
    if 'tools' in dump_config:
        tool_names = []
        for t in dump_config['tools']:
            if hasattr(t, 'function_declarations') and t.function_declarations:
                tool_names.extend([d.name for d in t.function_declarations])
            elif hasattr(t, 'google_search') and t.google_search is not None:
                tool_names.append("google_search")
        dump_config['tools'] = tool_names
    return {'model': model_name, 'contents': serializable_history, 'config': dump_config}


async def build_google_params(engine: "TextEngine", config: Dict[str, Any], history_object: Dict[str, Any],
                              tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[
        List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Builds the generate_content request pieces. Single source of truth for
    the canonical streaming driver (pinned by tests/test_engine_payload_parity.py).
    Returns (history_for_api, content_config_for_api, api_params_for_dumping)."""
    system_prompt, history_to_process = extract_system_prompt(history_object)
    image_url = history_object["current_message"].get("image_url")
    history_for_api, serializable_history = await build_google_history(
        engine, system_prompt, history_to_process, image_url
    )

    content_config_for_api: Dict[str, Any] = {"safety_settings": engine.google_safety_settings}
    if system_prompt:
        content_config_for_api['system_instruction'] = system_prompt

    api_tools, tool_config = build_google_tools(engine, tools)
    if tool_config:
        content_config_for_api['tool_config'] = tool_config
    if api_tools:
        content_config_for_api['tools'] = api_tools

    content_config_for_api['max_output_tokens'] = config.get(
        "max_output_tokens") or global_config.DEFAULT_TOKEN_LIMIT
    if isinstance(config.get("temperature"), (int, float)):
        content_config_for_api['temperature'] = config.get("temperature")
    if isinstance(config.get("top_p"), (int, float)):
        content_config_for_api['top_p'] = config.get("top_p")
    if isinstance(config.get("top_k"), (int, float)):
        content_config_for_api['top_k'] = config.get("top_k")

    if config.get("thinking_level"):
        content_config_for_api['thinking_config'] = ThinkingConfig(
            thinking_level=config["thinking_level"]
        )

    api_params_for_dumping = build_google_dump_params(
        config["model_name"], content_config_for_api, serializable_history
    )
    return history_for_api, content_config_for_api, api_params_for_dumping


def accumulate_google_chunks(chunks: List[Any]) -> Any:
    """Recombine streamed GenerateContentResponse chunks into one
    response-shaped object that `parse_google_response` understands:
    parts concatenate in arrival order (text may be split across chunks;
    joining them reproduces the non-streamed text), prompt_feedback comes
    from the first chunk that carries one, and grounding_metadata from the
    last chunk that carries one (the SDK attaches it to the final chunk)."""
    prompt_feedback = next(
        (c.prompt_feedback for c in chunks if getattr(c, "prompt_feedback", None)), None
    )
    parts: List[Any] = []
    grounding_metadata: Any = None
    for chunk in chunks:
        candidate = chunk.candidates[0] if getattr(chunk, "candidates", None) else None
        if candidate is None:
            continue
        if candidate.content and candidate.content.parts:
            parts.extend(candidate.content.parts)
        if getattr(candidate, "grounding_metadata", None) is not None:
            grounding_metadata = candidate.grounding_metadata
    candidates: List[Any] = []
    if parts:
        candidates = [SimpleNamespace(
            content=SimpleNamespace(parts=parts),
            grounding_metadata=grounding_metadata,
        )]
    return SimpleNamespace(prompt_feedback=prompt_feedback, candidates=candidates)


async def stream_google(
    engine: "TextEngine", config: Dict[str, Any], history_object: Dict[str, Any],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Canonical Google driver (DP-206): `generate_content_stream` chunks,
    emitting the unified event shape. Text parts stream as raw deltas; the
    terminal `done` carries the grounding-processed full text (citations /
    search queries appended), so collect_stream reproduces the pre-DP-206
    one-shot result exactly."""
    try:
        initialize_google_client(engine)
        assert engine.google_client is not None and engine.google_search_tool is not None
    except (ValueError, AssertionError) as e:
        raise LLMCommunicationError(f"Error: Google not configured: {e}") from e

    history_for_api, content_config_for_api, api_params_for_dumping = \
        await build_google_params(engine, config, history_object, tools)
    yield {"type": "api_payload", "payload": api_params_for_dumping}

    chunks: List[Any] = []
    try:
        stream = await engine.google_client.models.generate_content_stream(
            model=config["model_name"],
            contents=history_for_api,
            config=GenerateContentConfig(**content_config_for_api)
        )
        async for chunk in stream:
            chunks.append(chunk)
            candidate = chunk.candidates[0] if getattr(chunk, "candidates", None) else None
            if candidate is None or not candidate.content or not candidate.content.parts:
                continue
            for part in candidate.content.parts:
                if part.function_call:
                    continue
                if hasattr(part, 'text') and part.text:
                    yield {"type": "text_delta", "text": part.text}
    except Exception as e:
        err_str = str(e)
        rate_limited = '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str

        if rate_limited:
            logger.warning(f"Google API rate-limited ({config['model_name']}): retryable.")
        else:
            # Only emit the full traceback when verbose (DEBUG) logging is on;
            # by default a single-line error is plenty for transient API errors.
            logger.error(f"Google API error: {e}", exc_info=logger.isEnabledFor(logging.DEBUG))

        raise LLMCommunicationError(f"An error occurred with Google API: {e}",
                                    api_payload=api_params_for_dumping, rate_limited=rate_limited) from e

    result, _ = parse_google_response(
        accumulate_google_chunks(chunks), api_params_for_dumping
    )
    if result.get("type") == "tool_calls":
        yield {"type": "tool_calls", "calls": result["calls"]}
        yield {"type": "done", "full_text": ""}
    else:
        yield {"type": "done", "full_text": result.get("content", "")}


class GoogleProvider(Provider):
    """Google provider — Gemini and Gemma model families."""

    def __init__(self, engine: "TextEngine") -> None:
        self._engine = engine

    #: name of the engine seam method (back-compat for `_get_provider_route`).
    route_method_name = "_stream_google_response"

    def matches(self, model_name: str) -> bool:
        return "gemma" in model_name or "gemini" in model_name

    def limiters_for(self, model_name: str) -> List[AsyncLimiter]:
        """Google splits rate limits by model family (preserves the waterfall:
        gemma-4/gemma → gemma-4 RPM; gemini-3.1 → gemini-3 RPM; gemini → 2.5
        RPM+RPD)."""
        if "gemma" in model_name:
            return [self._engine._gemma_4_rpm_limiter]
        if "gemini-3.1" in model_name:
            return [self._engine._gemini_3_rpm_limiter]
        return [self._engine._gemini_25_rpm_limiter, self._engine._gemini_25_rpd_limiter]

    async def stream(
        self,
        persona_config: Dict[str, Any],
        history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        local_inference_config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        # Route through the engine seam so instance/class patches of
        # `_stream_google_response` still intercept.
        async for ev in self._engine._stream_google_response(persona_config, history_object, tools):
            yield ev
