# src/engine.py

import json
import logging
import os
import asyncio
import re
import shutil
import tempfile
import uuid
from types import SimpleNamespace
from typing import Dict, Any, Optional, Tuple, List, Callable, AsyncIterator
from contextlib import asynccontextmanager, AsyncExitStack

from dotenv import load_dotenv

from aiolimiter import AsyncLimiter

from config import global_config
from config.global_config import (
    EMPTY_RESPONSE_RETRIES, EMPTY_RESPONSE_RETRY_DELAY,
    RATE_LIMIT_GEMINI_25_RPM, RATE_LIMIT_GEMINI_25_RPD,
    RATE_LIMIT_GEMINI_3_RPM,
    RATE_LIMIT_GEMMA_3_RPM, RATE_LIMIT_GEMMA_4_RPM,
    RATE_LIMIT_OPENAI_RPM, RATE_LIMIT_ANTHROPIC_RPM,
    RATE_LIMIT_AGY_RPM, RATE_LIMIT_CC_RPM,
)
# --- Provider-specific imports ---
import base64
import aiohttp
import anthropic
from openai import AsyncOpenAI, APIStatusError, APITimeoutError
from google import genai
from google.genai.types import GenerateContentConfig, Tool, GoogleSearch, Candidate, \
    FunctionDeclaration, Part, ThinkingConfig
from src.utils.google_utils import process_grounding_metadata
from src.generation_params import GenerationParams
from src.llm_errors import LLMCommunicationError
from src.security.vault import get_vault
from src.stream_engine import StreamEngine
from src.text_tool_protocol import (
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    decode_tool_call_payload,
    extract_first_tool_call_block,
    render_tool_descriptions,
)

AGY_CALL_TIMEOUT_SECONDS = 120.0
# Claude Code runs a full agentic loop (its own tools) in one headless call, so
# it needs far more headroom than the one-shot agy text route.
CC_CALL_TIMEOUT_SECONDS = 600.0

logger = logging.getLogger(__name__)

__all__ = [
    "TextEngine", "LLMCommunicationError",
    "AGY_CALL_TIMEOUT_SECONDS", "CC_CALL_TIMEOUT_SECONDS",
]


class TextEngine:
    """A centralized engine for handling requests to various LLM APIs."""

    # Model fallback mapping: primary → fallback on 429.
    # Add entries here to enable automatic fallback for any model.
    # TODO: expand this to alert user of model change, add 'use_fallback_models' to persona config, probably more design warranted
    _FALLBACK_MODELS: Dict[str, str] = {
        "gemma-4-31b-it": "gemma-4-26b-a4b-it",
    }

    def __init__(self, stream_engine: Optional[Any] = None) -> None:
        # --- Lazy-loaded clients ---
        self.openai_client: Optional[AsyncOpenAI] = None
        self.anthropic_client: Optional[anthropic.AsyncAnthropic] = None
        # Kobold-native local provider (DP-206b: engine-owned — the engine is
        # the single entry; StreamEngine is its `local` transport component).
        # The parameter exists for tests to inject fakes.
        self.stream_engine: Any = stream_engine if stream_engine is not None else StreamEngine()

        # --- Google Client (matching original implementation) ---
        self.google_client: Optional[genai.client.AsyncClient] = None
        self.google_search_tool: Optional[Tool] = None
        # self.google_tool_config is now built dynamically
        self.google_safety_settings: List[Dict[str, str]] = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # --- Per-provider rate limiters ---
        self._gemini_25_rpm_limiter = AsyncLimiter(max_rate=RATE_LIMIT_GEMINI_25_RPM, time_period=60)
        self._gemini_25_rpd_limiter = AsyncLimiter(max_rate=RATE_LIMIT_GEMINI_25_RPD, time_period=86400)
        self._gemini_3_rpm_limiter  = AsyncLimiter(max_rate=RATE_LIMIT_GEMINI_3_RPM,  time_period=60)
        self._gemma_3_rpm_limiter   = AsyncLimiter(max_rate=RATE_LIMIT_GEMMA_3_RPM,   time_period=60)
        self._gemma_4_rpm_limiter   = AsyncLimiter(max_rate=RATE_LIMIT_GEMMA_4_RPM,   time_period=60)
        self._openai_limiter        = AsyncLimiter(max_rate=RATE_LIMIT_OPENAI_RPM,    time_period=60)
        self._anthropic_limiter     = AsyncLimiter(max_rate=RATE_LIMIT_ANTHROPIC_RPM, time_period=60)
        self._agy_limiter           = AsyncLimiter(max_rate=RATE_LIMIT_AGY_RPM,       time_period=60)
        self._cc_limiter            = AsyncLimiter(max_rate=RATE_LIMIT_CC_RPM,        time_period=60)
        # Persistent agy/cc workspaces are shared across calls; a per-workspace
        # lock serializes them so one call's CLI state can't clobber another's.
        self._agy_workspace_locks: Dict[str, asyncio.Lock] = {}
        self._cc_workspace_locks: Dict[str, asyncio.Lock] = {}
        logger.info(
            f"Rate limiters initialised — "
            f"Gemini 2.5: {RATE_LIMIT_GEMINI_25_RPM} RPM / {RATE_LIMIT_GEMINI_25_RPD} RPD | "
            f"Gemini 3.1: {RATE_LIMIT_GEMINI_3_RPM} RPM | "
            f"Gemma 3: {RATE_LIMIT_GEMMA_3_RPM} RPM | Gemma 4: {RATE_LIMIT_GEMMA_4_RPM} RPM | "
            f"OpenAI: {RATE_LIMIT_OPENAI_RPM} RPM | "
            f"Anthropic: {RATE_LIMIT_ANTHROPIC_RPM} RPM | "
            f"AGY: {RATE_LIMIT_AGY_RPM} RPM | "
            f"CC: {RATE_LIMIT_CC_RPM} RPM"
        )

        self._initialize_env()

    async def aclose(self) -> None:
        """Release transport resources (the kobold-native HTTP client)."""
        await self.stream_engine.aclose()

    def _initialize_env(self) -> None:
        """Load API keys from .env file."""
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
        else:
            logger.warning(".env file not found, API keys must be in environment.")

    def model_supports_images(self, model_name: str) -> bool:
        """Checks if a model is known to support image inputs."""
        model_name = model_name.lower()
        # OpenAI: gpt-4, gpt-4o, o1, etc.
        if 'gpt-4' in model_name or model_name.startswith('o1'):
            return True
        # Anthropic: claude-3, claude-4, etc.
        if 'claude-3' in model_name or 'claude-4' in model_name:
            return True
        # Google: gemini and gemma models
        if 'gemini' in model_name or 'gemma' in model_name:
            return True
        return False

    async def _get_openai_client(self) -> AsyncOpenAI:
        """Initializes and returns the OpenAI client."""
        if self.openai_client is None:
            api_key = get_vault().get("OPENAI_API_KEY")
            if not api_key:
                raise LLMCommunicationError("OPENAI_API_KEY not set — skipping OpenAI provider.")
            self.openai_client = AsyncOpenAI(api_key=api_key)
        return self.openai_client

    def _get_anthropic_client(self) -> anthropic.AsyncAnthropic:
        """Initializes and returns the async Anthropic client."""
        if self.anthropic_client is None:
            api_key = get_vault().get("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMCommunicationError("ANTHROPIC_API_KEY not set — skipping Anthropic provider.")
            self.anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
        return self.anthropic_client

    def _initialize_google_client(self) -> None:
        """Initializes the Google client using the original project's method."""
        if self.google_client is not None:
            return

        api_key = get_vault().get("GOOGLE_GENERATIVEAI_API_KEY")
        if not api_key: raise ValueError("GOOGLE_GENERATIVEAI_API_KEY not found in environment.")

        client: genai.client.BaseApiClient = genai.client.BaseApiClient(api_key=api_key)
        self.google_client = genai.client.AsyncClient(client)
        self.google_search_tool = Tool(google_search=GoogleSearch())
        logger.info("Google AI Studio client initialized.")

    @classmethod
    def _get_fallback_model(cls, model_name: str) -> Optional[str]:
        """Returns a fallback model for rate-limited requests, or None."""
        return cls._FALLBACK_MODELS.get(model_name)

    def _get_provider_route(self, model_name: str) -> Tuple[Callable, List[AsyncLimiter]]:
        """Returns (stream_factory, [limiters]) for the model name (DP-206b).
        Every factory is an async generator emitting the unified event shape
        (api_payload → text_delta* → [tool_calls] → done); the `local` factory
        additionally accepts `local_inference_config`.
        Raises LLMCommunicationError for unsupported models."""
        # cc-* must be checked before the `"claude" in model_name` branch below,
        # which would otherwise capture it and route to the Anthropic API.
        if model_name.startswith("cc-"):
            self._ensure_cc_supported()
            return self._stream_cc_response, [self._cc_limiter]
        if model_name.startswith("gpt"):
            return self._stream_openai_response, [self._openai_limiter]
        if "claude" in model_name:
            return self._stream_anthropic_response, [self._anthropic_limiter]
        if "gemma-4" in model_name:
            return self._stream_google_response, [self._gemma_4_rpm_limiter]
        if "gemma" in model_name:
            return self._stream_google_response, [self._gemma_4_rpm_limiter]
        if "gemini-3.1" in model_name:
            return self._stream_google_response, [self._gemini_3_rpm_limiter]
        if "gemini" in model_name:
            return self._stream_google_response, [self._gemini_25_rpm_limiter, self._gemini_25_rpd_limiter]
        if model_name.startswith("agy"):
            self._ensure_agy_supported()
            return self._stream_agy_response, [self._agy_limiter]
        if model_name == 'local':
            return self._stream_local_response, []
        raise LLMCommunicationError(f"Error: Model '{model_name}' is not supported.")

    @staticmethod
    @asynccontextmanager
    async def _rate_limited(limiters: List[AsyncLimiter]) -> AsyncIterator[None]:
        """Acquires all limiters in sequence via AsyncExitStack."""
        async with AsyncExitStack() as stack:
            for limiter in limiters:
                await stack.enter_async_context(limiter)
            yield

    async def generate_response(self, persona_config: Dict[str, Any], history_object: Dict[str, Any],
                                tools: Optional[List[Dict[str, Any]]] = None,
                                local_inference_config: Optional[Dict[str, Any]] = None) -> Tuple[
        Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        One-shot entry: drains the policy-driven stream (DP-206b cutover —
        one-shot = collect(stream); `_stream_response` owns routing, rate
        limiting, the empty-response retry loop, and 429 fallback).
        Returns: A tuple containing:
                 1. A structured dictionary:
                    - {'type': 'text', 'content': '...'} for a text response.
                    - {'type': 'tool_calls', 'calls': [{'id': '...', 'name': '...', 'arguments': {...}}]} for a tool call.
                 2. The API payload dictionary for debugging, or None.
        Raises: LLMCommunicationError if all retries fail or produce empty/invalid responses.
        """
        result, api_payload = await self.collect_stream(
            self._stream_response(persona_config, history_object, tools, local_inference_config,
                                  one_shot=True)
        )
        # one_shot mode validates each complete attempt before emitting, so
        # every invalid shape — including content-then-zero-parseable-calls —
        # was retried inside the driver. This check is the final guard for
        # the historical contract that generate_response never returns an
        # empty/invalid result.
        if result.get('type') == 'text' and result.get('content', '').strip():
            return result, api_payload
        if result.get('type') == 'tool_calls' and result.get('calls'):
            return result, api_payload
        raise LLMCommunicationError("LLM provider returned an empty or invalid response after all retries.")

    async def _stream_response(
        self, persona_config: Dict[str, Any], history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        local_inference_config: Optional[Dict[str, Any]] = None,
        *, one_shot: bool = False,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Single driving layer for every provider (DP-206b).

        Owns the request *policy* around the canonical per-provider streams:
        image-support handling, provider routing, rate limiting, the
        empty-response retry loop, and 429 model fallback — emitting the
        unified event shape (api_payload → text_delta* → [tool_calls] → done).

        Events pass through as true token deltas with one safeguard: nothing
        is emitted for an attempt until it produces real content (cumulative
        non-whitespace text, a non-empty tool_calls, or a non-empty done), so
        an empty/invalid attempt can be retried invisibly — mirroring the
        pre-cutover generate_response validation. A tool_calls event with zero
        parseable calls is dropped (the attempt stays invalid, exactly like
        the old `{"type": "tool_calls", "calls": []}` one-shot result). Once
        content has been emitted ("committed"), errors propagate instead of
        retrying: streamed output cannot be retracted.

        With ``one_shot=True`` (the generate_response = collect(stream) path,
        DP-210), no output ever reaches a user mid-attempt, so retraction is a
        non-issue: the whole attempt is buffered and validated complete before
        anything is emitted. This restores the pre-cutover one-shot retry for
        the shape where the model streams text and then produces zero
        parseable tool calls (e.g. a local model emitting prose followed by a
        malformed ``<tool_call>`` block), and retries mid-stream errors that
        the true-streaming path must propagate.
        """
        model_name: str = persona_config.get("model_name", "")

        if history_object["current_message"].get("image_url") and not self.model_supports_images(model_name):
            logger.info(f"Model {model_name} does not support images. Modifying prompt.")
            history_object["persona_prompt"] += (
                "\n\n[System note: The user has attached an image that you cannot see."
                " Please inform them of this fact in your response.]"
            )
            history_object["current_message"]["image_url"] = None

        stream_factory, limiters = self._get_provider_route(model_name)

        for attempt in range(EMPTY_RESPONSE_RETRIES + 1):
            committed = False
            pending: List[Dict[str, Any]] = []
            pending_text: List[str] = []
            # one_shot validation state: mirrors collect_stream's result
            # shaping (a tool_calls event means the model chose the tool
            # path; last event wins; done full_text beats concatenation).
            attempt_calls: Optional[List[Dict[str, Any]]] = None
            attempt_done_text: Optional[str] = None

            try:
                async with self._rate_limited(limiters):
                    if model_name == 'local':
                        stream = stream_factory(persona_config, history_object, tools, local_inference_config)
                    else:
                        stream = stream_factory(persona_config, history_object, tools)
                    async for ev in stream:
                        if committed:
                            yield ev
                            continue
                        etype = ev.get("type")
                        if one_shot:
                            # Buffer the entire attempt; validity is judged
                            # on the complete result after the stream ends.
                            pending.append(ev)
                            if etype == "text_delta":
                                pending_text.append(ev.get("text", "") or "")
                            elif etype == "tool_calls":
                                attempt_calls = list(ev.get("calls", []))
                            elif etype == "done":
                                attempt_done_text = ev.get("full_text")
                            continue
                        if etype == "text_delta":
                            pending.append(ev)
                            pending_text.append(ev.get("text", "") or "")
                            if "".join(pending_text).strip():
                                committed = True
                                for held in pending:
                                    yield held
                                pending = []
                        elif etype == "tool_calls":
                            if ev.get("calls"):
                                committed = True
                                for held in pending:
                                    yield held
                                pending = []
                                yield ev
                            # else: zero parseable calls — drop the event and
                            # leave the attempt invalid so it gets retried.
                        elif etype == "done":
                            if (ev.get("full_text") or "").strip():
                                committed = True
                                for held in pending:
                                    yield held
                                pending = []
                                yield ev
                            else:
                                pending.append(ev)
                        else:
                            # api_payload (and any future event types) are held
                            # until the attempt proves valid.
                            pending.append(ev)

                if one_shot:
                    if attempt_calls is not None:
                        attempt_valid = bool(attempt_calls)
                    else:
                        full = (attempt_done_text if attempt_done_text is not None
                                else "".join(pending_text))
                        attempt_valid = bool((full or "").strip())
                    if attempt_valid:
                        for held in pending:
                            yield held
                        return
                elif committed:
                    return
                # Attempt completed without real content → retry below.

            except LLMCommunicationError as e:
                if committed:
                    raise
                if e.rate_limited:
                    fallback = self._get_fallback_model(model_name)
                    if fallback:
                        logger.warning(
                            f"Rate limit (429) hit for '{model_name}', "
                            f"falling back to '{fallback}'."
                        )
                        persona_config = {**persona_config, "model_name": fallback}
                        model_name = fallback
                        stream_factory, limiters = self._get_provider_route(model_name)
                        continue
                    logger.warning(f"Rate limit (429) hit for model '{model_name}'. Aborting retries.")
                    raise
                if attempt >= EMPTY_RESPONSE_RETRIES:
                    raise
                logger.warning(f"LLM communication error (Attempt {attempt + 1}). Retrying... Error: {e}")

            if attempt < EMPTY_RESPONSE_RETRIES:
                logger.warning(f"LLM returned an empty or invalid response (Attempt {attempt + 1}). Retrying...")
                await asyncio.sleep(EMPTY_RESPONSE_RETRY_DELAY)

        logger.error(f"LLM returned an empty or invalid response after {EMPTY_RESPONSE_RETRIES + 1} attempts.")
        raise LLMCommunicationError("LLM provider returned an empty or invalid response after all retries.")

    @staticmethod
    def _parse_openai_tool_calls(raw_calls: list) -> List[Dict[str, Any]]:
        """Parses OpenAI tool call objects into standardized dicts."""
        tool_calls: List[Dict[str, Any]] = []
        for call in raw_calls:
            try:
                arguments = json.loads(call.function.arguments)
                tool_calls.append({"id": call.id, "name": call.function.name, "arguments": arguments})
            except json.JSONDecodeError:
                logger.error(f"Failed to parse tool call arguments: {call.function.arguments}")
                continue
        return tool_calls

    @staticmethod
    def _attach_openai_image(messages: List[Dict[str, Any]], image_url: str) -> None:
        """Attaches an image URL to the last user message for OpenAI."""
        last_message = messages[-1]
        if last_message['role'] != 'user':
            return
        if isinstance(last_message['content'], str):
            last_message['content'] = [{"type": "text", "text": last_message['content']}]
        last_message['content'].append({"type": "image_url", "image_url": {"url": image_url}})

    def _build_openai_params(self, config: Dict[str, Any], history_object: Dict[str, Any],
                             tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Builds the chat.completions request kwargs. Single source of truth
        shared by the canonical streaming driver and the local one-shot path
        — wire-payload parity between them holds by construction
        (pinned by tests/test_engine_payload_parity.py)."""
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
            self._attach_openai_image(messages, history_object["current_message"]["image_url"])

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

    @staticmethod
    def _openai_dump_params(api_params: Dict[str, Any]) -> Dict[str, Any]:
        """Log-safe copy of the request kwargs: tools listed by name."""
        dump = dict(api_params)
        if "tools" in dump:
            dump["tools"] = [tool.get("function", {}).get("name", "unknown")
                             for tool in dump["tools"]]
        return dump

    async def _stream_openai_response(
        self, config: Dict[str, Any], history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Canonical OpenAI driver (DP-206): a true SDK token stream emitting
        the unified event shape (api_payload → text_delta* → [tool_calls] →
        done). The one-shot path is `collect_stream` over this generator."""
        client = await self._get_openai_client()
        api_params = self._build_openai_params(config, history_object, tools)
        yield {"type": "api_payload", "payload": self._openai_dump_params(api_params)}

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
            tool_calls = self._parse_openai_tool_calls(wrapped)
            yield {"type": "tool_calls", "calls": tool_calls}
            yield {"type": "done", "full_text": ""}
        else:
            yield {"type": "done", "full_text": "".join(text_parts)}

    async def _attach_anthropic_image(self, messages: List[Dict[str, Any]], image_url: str) -> None:
        """Downloads and attaches an image to the last user message for Anthropic."""
        last_message = messages[-1]
        if last_message['role'] != 'user':
            return
        if isinstance(last_message['content'], str):
            last_message['content'] = [{"type": "text", "text": last_message['content']}]
        try:
            image_bytes, mime_type = await self._download_image(image_url)
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

    async def _build_anthropic_params(self, config: Dict[str, Any], history_object: Dict[str, Any],
                                      tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Builds the messages request kwargs for Anthropic. Single source of
        truth for the canonical streaming driver (pinned by
        tests/test_engine_payload_parity.py)."""
        system_prompt, history = self._extract_system_prompt(history_object)

        if history_object["current_message"].get("image_url"):
            await self._attach_anthropic_image(history, history_object["current_message"]["image_url"])

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

    @staticmethod
    def _anthropic_dump_params(api_params: Dict[str, Any]) -> Dict[str, Any]:
        """Log-safe copy of the request kwargs: tools listed by name."""
        dump = dict(api_params)
        if "tools" in dump:
            dump["tools"] = [tool.get("name", "unknown") for tool in dump["tools"]]
        return dump

    async def _stream_anthropic_response(
        self, config: Dict[str, Any], history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Canonical Anthropic driver (DP-206): `messages.stream(...)` with the
        SDK's accumulator, emitting the unified event shape. The one-shot path
        is `collect_stream` over this generator. Uses `AsyncAnthropic`
        (DP-211) so token iteration never blocks the event loop; the request
        kwargs are identical to the sync client's (frozen payload goldens)."""
        client = self._get_anthropic_client()
        api_params = await self._build_anthropic_params(config, history_object, tools)
        yield {"type": "api_payload", "payload": self._anthropic_dump_params(api_params)}

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
                response_content = response.content[0].text or ""

        except anthropic.APIError as e:
            rate_limited = hasattr(e, 'status_code') and e.status_code == 429
            is_server_error = hasattr(e, 'status_code') and e.status_code >= 500
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

    @staticmethod
    def _extract_system_prompt(history_object: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        """Returns (merged_system_prompt, remaining_history)."""
        system_prompt = history_object["persona_prompt"]
        history = history_object.get("message_history", history_object.get("history", []))
        if history and history[0]["role"] == "system":
            system_prompt = f"{system_prompt}\n\n{history[0]['content']}"
            history = history[1:]
        return system_prompt, history

    @staticmethod
    def _render_agy_prompt(history: List[Dict[str, Any]]) -> str:
        """Flatten a message history into a single role-tagged transcript for the
        `agy` route.

        The Antigravity SDK's `chat()` accepts only one user turn and offers no
        API to seed prior assistant turns, while the engine is stateless and
        rebuilds the full context on every call. We therefore render the entire
        `history` — which already ends with the current user turn (see
        `_extract_system_prompt`) — into one deterministic, auditable transcript
        so `agy` contributes nothing of its own. This is also what lets the
        engine's multi-turn tool loop work: a `tool`-role result from a prior
        iteration is just another rendered line.

        The system prompt is delivered separately via `CustomSystemInstructions`
        and is intentionally not included here. `current_message["text"]` is a
        duplicate of the final user turn already present in `history`, so it is
        not appended (doing so would duplicate the last message).
        """
        lines: List[str] = []
        for item in history:
            role = item.get("role")
            if role == "tool":
                lines.append(f"Tool({item.get('name', 'unknown')}): {item.get('content', '')}")
            elif role == "assistant":
                if item.get("content"):
                    lines.append(f"Assistant: {item['content']}")
                for call in item.get("tool_calls", []) or []:
                    args = json.dumps(call.get("arguments", {}), ensure_ascii=False)
                    lines.append(f"Assistant (tool call {call.get('name', 'unknown')}): {args}")
            else:  # user (and any unlabeled turn) renders as the user
                lines.append(f"User: {item.get('content', '')}")
        return "\n\n".join(lines)

    async def _download_image(self, image_url: str) -> Tuple[bytes, str]:
        """Downloads image, returns (raw_bytes, mime_type).
        Raises aiohttp.ClientError on failure."""
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                resp.raise_for_status()
                image_bytes = await resp.read()
                mime_type = resp.content_type
        return image_bytes, mime_type

    async def _build_google_history(
        self, system_prompt: str, history: List[Dict[str, Any]], image_url: Optional[str]
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
                        image_bytes, mime_type = await self._download_image(image_url)
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

    def _build_google_tools(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Tuple[List[Tool], Optional[Dict[str, Any]]]:
        """Returns (api_tools, tool_config_or_none)."""
        api_tools: List[Tool] = []
        tool_config = None
        if tools:
            if any(t.get('type') == 'google_grounding' for t in tools):
                api_tools.append(self.google_search_tool)
            function_tools = [t for t in tools if t.get('type') == 'function' and t.get('function')]
            if function_tools:
                api_tools.extend([Tool(function_declarations=[FunctionDeclaration(**t['function'])])
                                  for t in function_tools])
                tool_config = {"function_calling_config": {"mode": "AUTO"}}
        return api_tools, tool_config

    @staticmethod
    def _parse_google_response(
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

    @staticmethod
    def _build_google_dump_params(
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

    async def _build_google_params(self, config: Dict[str, Any], history_object: Dict[str, Any],
                                   tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[
        List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """Builds the generate_content request pieces. Single source of truth
        for the canonical streaming driver (pinned by
        tests/test_engine_payload_parity.py). Returns
        (history_for_api, content_config_for_api, api_params_for_dumping)."""
        system_prompt, history_to_process = self._extract_system_prompt(history_object)
        image_url = history_object["current_message"].get("image_url")
        history_for_api, serializable_history = await self._build_google_history(
            system_prompt, history_to_process, image_url
        )

        content_config_for_api: Dict[str, Any] = {"safety_settings": self.google_safety_settings}
        if system_prompt:
            content_config_for_api['system_instruction'] = system_prompt

        api_tools, tool_config = self._build_google_tools(tools)
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

        api_params_for_dumping = self._build_google_dump_params(
            config["model_name"], content_config_for_api, serializable_history
        )
        return history_for_api, content_config_for_api, api_params_for_dumping

    @staticmethod
    def _accumulate_google_chunks(chunks: List[Any]) -> Any:
        """Recombine streamed GenerateContentResponse chunks into one
        response-shaped object that `_parse_google_response` understands:
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

    async def _stream_google_response(
        self, config: Dict[str, Any], history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Canonical Google driver (DP-206): `generate_content_stream` chunks,
        emitting the unified event shape. Text parts stream as raw deltas; the
        terminal `done` carries the grounding-processed full text (citations /
        search queries appended), so collect_stream reproduces the pre-DP-206
        one-shot result exactly."""
        try:
            self._initialize_google_client()
            assert self.google_client is not None and self.google_search_tool is not None
        except (ValueError, AssertionError) as e:
            raise LLMCommunicationError(f"Error: Google not configured: {e}") from e

        history_for_api, content_config_for_api, api_params_for_dumping = \
            await self._build_google_params(config, history_object, tools)
        yield {"type": "api_payload", "payload": api_params_for_dumping}

        chunks: List[Any] = []
        try:
            stream = await self.google_client.models.generate_content_stream(
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

        result, _ = self._parse_google_response(
            self._accumulate_google_chunks(chunks), api_params_for_dumping
        )
        if result.get("type") == "tool_calls":
            yield {"type": "tool_calls", "calls": result["calls"]}
            yield {"type": "done", "full_text": ""}
        else:
            yield {"type": "done", "full_text": result.get("content", "")}

    async def _stream_local_response(
        self, config: Dict[str, Any], history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        local_inference_config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Canonical local driver (DP-206b): the kobold-native token stream.
        StreamEngine renders the chat template, folds the tool list into the
        system prompt as the `<tool_call>` protocol, and parses tool-call
        blocks out of the token stream — so the local one-shot path
        (collect over this) uses the exact same transport and tool protocol
        as the streaming portal path. This replaced the OpenAI-compat
        `/v1/chat/completions` one-shot transport."""
        async for ev in self.stream_engine.stream_local(
            config, history_object, tools, local_inference_config
        ):
            yield ev

    @staticmethod
    async def _events_from_one_shot(
        result: Dict[str, Any], api_payload: Optional[Dict[str, Any]],
    ) -> AsyncIterator[Dict[str, Any]]:
        """Synthesize the unified event shape from a one-shot
        (result, api_payload) pair — the inverse of `collect_stream`."""
        yield {"type": "api_payload", "payload": api_payload or {}}
        if result.get("type") == "tool_calls":
            yield {"type": "tool_calls", "calls": list(result.get("calls", []))}
            yield {"type": "done", "full_text": ""}
        else:
            text = result.get("content", "") or ""
            if text:
                yield {"type": "text_delta", "text": text}
            yield {"type": "done", "full_text": text}

    @staticmethod
    def _render_agy_tool_protocol(tools: Optional[List[Dict[str, Any]]]) -> str:
        if not tools:
            return ""

        protocol_desc = (
            "You may request a tool by emitting EXACTLY "
            f"{TOOL_CALL_OPEN}{{\"name\": \"<tool_name>\", \"arguments\": "
            f"{{<json args>}}}}{TOOL_CALL_CLOSE} "
            "as the last thing. Answer in plain text otherwise, and use no other tools/files/shell/web."
        )

        # Shared renderer keeps the agy and streaming paths from drifting on
        # how a tool's name/description/parameters are formatted.
        lines = [protocol_desc, *render_tool_descriptions(tools)]
        return "\n".join(lines)

    @staticmethod
    def _parse_agy_tool_call(text: str) -> Optional[List[Dict[str, Any]]]:
        if not text:
            return None
        cleaned = re.sub(r"<SYSTEM_MESSAGE>.*?</SYSTEM_MESSAGE>", "", text, flags=re.DOTALL)
        inner = extract_first_tool_call_block(cleaned)
        if inner is None:
            return None
        parsed = decode_tool_call_payload(inner)
        if parsed is None:
            return None
        # agy policy: both keys must be present; id is a fresh uuid.
        if "name" not in parsed or "arguments" not in parsed:
            return None
        call_id = f"agy_{uuid.uuid4().hex}"
        return [{
            "id": call_id,
            "name": parsed["name"],
            "arguments": parsed["arguments"]
        }]

    @staticmethod
    def _ensure_agy_supported() -> None:
        """agy is a TUI CLI that only emits its response to a TTY. DERPR captures
        stdout via a pipe — fine on POSIX, but on native Windows agy renders to
        the console and writes *nothing* to a non-TTY stdout/file, so the route
        silently returns empty. Fail loudly instead and point at the docs; run
        the engine on the POSIX host (Linux/macOS/WSL/Docker) to use agy.
        """
        if os.name != "posix":
            raise LLMCommunicationError(
                "The 'agy' provider is unsupported on native Windows: agy only "
                "writes its response to a TTY, but DERPR captures stdout via a "
                "pipe, so the response is always empty. Run the engine on the "
                "POSIX host (Linux/macOS/WSL/Docker) to use agy. "
                "See docs/user_guide.md (Antigravity / agy provider)."
            )

    @staticmethod
    def _sanitize_agy_workspace_name(persona_name: Optional[str]) -> Optional[str]:
        """Persona names come from config and may contain path separators or
        other filesystem-hostile characters; reduce to a safe slug. Returns
        None when nothing usable remains (caller falls back to global)."""
        if not persona_name:
            return None
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", persona_name).strip("._")
        return slug or None

    @staticmethod
    def _resolve_agy_workspace(persona_name: Optional[str]) -> Optional[str]:
        """Returns the persistent workspace dir for this call, or None when
        persistence is disabled (caller uses a throwaway temp dir). Does not
        create the directory."""
        if not global_config.AGY_PERSISTENT_WORKSPACES:
            return None
        workspaces_dir = global_config.AGY_WORKSPACES_DIR
        slug = TextEngine._sanitize_agy_workspace_name(persona_name)
        if global_config.AGY_WORKSPACE_MODE == "persona" and slug:
            return os.path.abspath(workspaces_dir / f"agy_{slug}")
        return os.path.abspath(workspaces_dir / "agy_global")

    async def _run_agy_cli(self, prompt: str, timeout: float = AGY_CALL_TIMEOUT_SECONDS, persona_name: Optional[str] = None) -> str:
        self._ensure_agy_supported()

        binary = os.environ.get("ANTIGRAVITY_HARNESS_PATH") or shutil.which("agy")
        if not binary:
            raise LLMCommunicationError("Antigravity harness/agy binary not found.")

        timeout_sec_str = f"{int(timeout) + 30}s"
        args = ["--print-timeout", timeout_sec_str, "-p", prompt]
        if global_config.AGY_SANDBOX:
            args = ["--sandbox", *args]

        workspace_dir = self._resolve_agy_workspace(persona_name)
        if workspace_dir is None:
            temp_dir = tempfile.mkdtemp()
            try:
                return await self._exec_agy(binary, args, temp_dir, timeout)
            finally:
                # The CLI leaves symlinks under .antigravitycli pointing at
                # files outside the temp dir; remove the targets so rmtree
                # doesn't strand them. Persistent workspaces keep this state
                # on purpose — that cache is the point of persistence.
                self._remove_agy_cli_link_targets(temp_dir)
                shutil.rmtree(temp_dir, ignore_errors=True)

        os.makedirs(workspace_dir, exist_ok=True)
        lock = self._agy_workspace_locks.setdefault(workspace_dir, asyncio.Lock())
        async with lock:
            return await self._exec_agy(binary, args, workspace_dir, timeout)

    @staticmethod
    def _remove_agy_cli_link_targets(workspace_dir: str) -> None:
        cli_dir = os.path.join(workspace_dir, ".antigravitycli")
        if not os.path.isdir(cli_dir):
            return
        for f in os.listdir(cli_dir):
            p = os.path.join(cli_dir, f)
            if os.path.islink(p):
                try:
                    target = os.readlink(p)
                    if os.path.exists(target):
                        os.remove(target)
                except Exception:
                    pass

    @staticmethod
    async def _exec_agy(binary: str, args: List[str], workspace_dir: str, timeout: float,
                        label: str = "agy") -> str:
        # `label` names the provider in error messages — this CLI runner is
        # shared by the agy and cc (Claude Code) routes, so a failure must
        # point at the route the caller actually invoked.
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                binary,
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace_dir,
                start_new_session=True
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
            except asyncio.TimeoutError as e:
                raise LLMCommunicationError(f"{label} CLI timed out after {timeout} seconds.") from e

            if proc.returncode != 0:
                stderr_excerpt = stderr.decode("utf-8", errors="replace").strip()
                excerpt = stderr_excerpt[-200:] if len(stderr_excerpt) > 200 else stderr_excerpt
                raise LLMCommunicationError(
                    f"{label} CLI failed with exit code {proc.returncode}. Stderr: {excerpt}"
                )

            return stdout.decode("utf-8", errors="replace")
        finally:
            if proc is not None:
                try:
                    import signal
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass

    async def _generate_agy_response(
        self, config: Dict[str, Any], history_object: Dict[str, Any], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """One-shot agy path. DP-206 decision: agy stays one-shot-only — it is
        a TUI CLI invoked as a subprocess (POSIX-only, see
        `_ensure_agy_supported`) whose entire response arrives at process exit;
        there is no token stream to make canonical. Streaming consumers get it
        via `stream_messages`' generate_response wrap (single text_delta)."""
        system_prompt, history = self._extract_system_prompt(history_object)

        prompt_parts = []
        if system_prompt:
            prompt_parts.append(system_prompt)

        if tools:
            rendered_tools = self._render_agy_tool_protocol(tools)
            if rendered_tools:
                prompt_parts.append(rendered_tools)

        rendered_history = self._render_agy_prompt(history)
        if rendered_history:
            prompt_parts.append(rendered_history)

        prompt = "\n\n".join(prompt_parts)

        tool_names = []
        if tools:
            tool_names = [t["function"]["name"] for t in tools if "function" in t and "name" in t["function"]]

        persona_name = config.get("persona_name")
        workspace_dir = self._resolve_agy_workspace(persona_name)
        api_payload = {
            "model": config.get("model_name"),
            "prompt_chars": len(prompt),
            "tools": tool_names,
            "isolation": {
                "stdin": "devnull",
                "skip_permissions": False,
                "workspace": workspace_dir if workspace_dir else "temp-dir-per-call",
            }
        }

        try:
            raw = await self._run_agy_cli(prompt, persona_name=persona_name)
        except LLMCommunicationError as e:
            if e.api_payload is None:
                e.api_payload = api_payload
            raise

        calls = self._parse_agy_tool_call(raw)
        if calls:
            return {"type": "tool_calls", "calls": calls}, api_payload
        else:
            cleaned_content = re.sub(r"<SYSTEM_MESSAGE>.*?</SYSTEM_MESSAGE>", "", raw, flags=re.DOTALL).strip()
            return {"type": "text", "content": cleaned_content}, api_payload

    async def _stream_agy_response(
        self, config: Dict[str, Any], history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """agy adapter into the unified event shape. agy stays one-shot by
        decision (subprocess TUI CLI — the entire response arrives at process
        exit, there is no token stream to make canonical); streaming consumers
        get the full text as a single text_delta."""
        result, api_payload = await self._generate_agy_response(
            config, history_object, tools
        )
        async for ev in self._events_from_one_shot(result, api_payload):
            yield ev

    # ------------------------------------------------------------------
    # Claude Code (cc-*) provider — DP-222
    #
    # Structural parity with the agy route (subprocess-per-call, one-shot,
    # POSIX-only, persistent per-persona workspace, dedicated rate limiter),
    # but with a deliberate behavioural divergence: the agy route CLAMPS tools
    # off and round-trips derpr's <tool_call> text protocol, whereas Claude
    # Code runs its OWN sandboxed tools autonomously (`--dangerously-skip-
    # permissions` bounded by the built-in OS sandbox). So the cc route ignores
    # the engine's `tools` argument and returns Claude Code's final text; derpr's
    # tool loop does not wrap it. Bridging derpr tools -> Claude Code is future
    # MCP work, where approval routing will also live.
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_cc_supported() -> None:
        """Claude Code's OS sandbox (Seatbelt/bubblewrap) only runs on
        macOS/Linux/WSL2 — never native Windows. Since this provider runs
        `--dangerously-skip-permissions` (yolo), the sandbox is the safety
        boundary, so refuse the route on non-POSIX hosts when the sandbox is
        enabled. Run the engine on the POSIX host (Linux/macOS/WSL/Docker).
        """
        if global_config.CC_SANDBOX and os.name != "posix":
            raise LLMCommunicationError(
                "The 'cc-*' (Claude Code) provider runs yolo bounded by Claude "
                "Code's OS sandbox, which is unavailable on native Windows. Run "
                "the engine on the POSIX host (Linux/macOS/WSL/Docker), or set "
                "CC_SANDBOX=False to run unsandboxed (no yolo; tools gated to "
                "CC_ALLOWED_TOOLS). "
                "See docs/user_guide.md (Claude Code / cc provider)."
            )

    @staticmethod
    def _cc_model_arg(model_name: str) -> str:
        """Map a `cc-<alias>` model name onto Claude Code's `--model` value
        (e.g. `cc-sonnet` -> `sonnet`). A bare `cc-` falls back to `sonnet`."""
        alias = model_name[len("cc-"):] if model_name.startswith("cc-") else model_name
        return alias or "sonnet"

    @staticmethod
    def _resolve_cc_workspace(persona_name: Optional[str]) -> Optional[str]:
        """Returns the working dir for this call, or None when persistence is
        disabled (caller uses a throwaway temp dir). Precedence: explicit
        CC_WORKSPACE_DIR override (e.g. the derpr checkout) > per-persona dir >
        global dir. Does not create the directory."""
        if global_config.CC_WORKSPACE_DIR:
            return os.path.abspath(global_config.CC_WORKSPACE_DIR)
        if not global_config.CC_PERSISTENT_WORKSPACES:
            return None
        workspaces_dir = global_config.CC_WORKSPACES_DIR
        slug = TextEngine._sanitize_agy_workspace_name(persona_name)
        if global_config.CC_WORKSPACE_MODE == "persona" and slug:
            return os.path.abspath(workspaces_dir / f"cc_{slug}")
        return os.path.abspath(workspaces_dir / "cc_global")

    @staticmethod
    def _build_cc_sandbox_settings() -> Optional[Dict[str, Any]]:
        """Build the `--settings` sandbox block, or None when CC_SANDBOX is off.
        Auto-allows sandboxed Bash so a headless run never blocks on a prompt;
        the OS sandbox confines it to the workspace + allowed domains."""
        if not global_config.CC_SANDBOX:
            return None
        sandbox: Dict[str, Any] = {
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
        }
        if global_config.CC_SANDBOX_WEAKER_NESTED:
            sandbox["enableWeakerNestedSandbox"] = True
        if global_config.CC_SANDBOX_ALLOWED_DOMAINS:
            sandbox["network"] = {"allowedDomains": list(global_config.CC_SANDBOX_ALLOWED_DOMAINS)}
        return {"sandbox": sandbox}

    def _build_cc_args(self, prompt: str, system_prompt: str, model_arg: str) -> List[str]:
        """Assemble the `claude -p` argv (without the binary)."""
        args = ["-p", prompt, "--output-format", "text", "--model", model_arg]
        if system_prompt:
            args += ["--system-prompt", system_prompt]
        if global_config.CC_SANDBOX:
            # yolo: skip per-tool approval prompts. The OS sandbox is the safety
            # boundary; root's skip-permissions check is waived inside it.
            args += ["--dangerously-skip-permissions"]
        elif global_config.CC_ALLOWED_TOOLS:
            # Unsandboxed (e.g. native Windows smoke): NEVER bare yolo. Use
            # Claude Code's OS-independent permission system — only the
            # explicitly allowlisted tools may run; everything else is refused
            # (headless cannot answer an approval prompt).
            args += ["--allowedTools", *global_config.CC_ALLOWED_TOOLS]
        if global_config.CC_MAX_TURNS > 0:
            args += ["--max-turns", str(global_config.CC_MAX_TURNS)]
        sandbox_settings = self._build_cc_sandbox_settings()
        if sandbox_settings is not None:
            args += ["--settings", json.dumps(sandbox_settings)]
        return args

    async def _run_cc_cli(
        self,
        prompt: str,
        system_prompt: str,
        model_arg: str,
        timeout: float = CC_CALL_TIMEOUT_SECONDS,
        persona_name: Optional[str] = None,
    ) -> str:
        self._ensure_cc_supported()

        binary = os.environ.get("CLAUDE_CLI_PATH") or shutil.which("claude")
        if not binary:
            raise LLMCommunicationError("Claude Code 'claude' binary not found on PATH.")

        args = self._build_cc_args(prompt, system_prompt, model_arg)

        workspace_dir = self._resolve_cc_workspace(persona_name)
        if workspace_dir is None:
            temp_dir = tempfile.mkdtemp()
            try:
                return await self._exec_agy(binary, args, temp_dir, timeout, label="Claude Code")
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        os.makedirs(workspace_dir, exist_ok=True)
        lock = self._cc_workspace_locks.setdefault(workspace_dir, asyncio.Lock())
        async with lock:
            return await self._exec_agy(binary, args, workspace_dir, timeout, label="Claude Code")

    async def _generate_cc_response(
        self, config: Dict[str, Any], history_object: Dict[str, Any], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """One-shot Claude Code path. The persona prompt is delivered via
        `--system-prompt` (replace); the rendered history transcript is the `-p`
        prompt. `tools` is intentionally ignored — Claude Code uses its own
        sandboxed tools and returns final text."""
        system_prompt, history = self._extract_system_prompt(history_object)
        prompt = self._render_agy_prompt(history)
        model_arg = self._cc_model_arg(config.get("model_name", ""))
        persona_name = config.get("persona_name")
        workspace_dir = self._resolve_cc_workspace(persona_name)

        if tools:
            logger.debug(
                "cc provider ignoring %d derpr tool(s) — Claude Code uses its own tools.",
                len(tools),
            )

        api_payload = {
            "model": config.get("model_name"),
            "cc_model": model_arg,
            "prompt_chars": len(prompt),
            "system_prompt_chars": len(system_prompt or ""),
            "tools_ignored": [
                t["function"]["name"] for t in tools or []
                if "function" in t and "name" in t["function"]
            ],
            "isolation": {
                "stdin": "devnull",
                "skip_permissions": True,
                "sandbox": global_config.CC_SANDBOX,
                "max_turns": global_config.CC_MAX_TURNS or None,
                "workspace": workspace_dir if workspace_dir else "temp-dir-per-call",
            },
        }

        try:
            raw = await self._run_cc_cli(
                prompt, system_prompt or "", model_arg, persona_name=persona_name
            )
        except LLMCommunicationError as e:
            if e.api_payload is None:
                e.api_payload = api_payload
            raise

        return {"type": "text", "content": raw.strip()}, api_payload

    async def _stream_cc_response(
        self, config: Dict[str, Any], history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Claude Code adapter into the unified event shape. One-shot by nature
        (the headless `claude -p` agentic run's full result arrives at process
        exit); streaming consumers get the final text as a single text_delta."""
        result, api_payload = await self._generate_cc_response(
            config, history_object, tools
        )
        async for ev in self._events_from_one_shot(result, api_payload):
            yield ev

    # ------------------------------------------------------------------
    # Provider streaming surface
    #
    # `stream_messages(persona, messages, params)` and
    # `stream_prompt(persona, prompt, params)` are the unified entries.
    #
    # DP-206b state: there is ONE driving layer. Each provider has a single
    # canonical streaming generator (`_stream_<provider>_response`; agy is a
    # one-shot subprocess adapted into the same event shape), the policy
    # driver `_stream_response` wraps it (routing, rate limiting, retries,
    # 429 fallback), `stream_messages` dispatches through the driver for
    # true token deltas, and `generate_response` is collect_stream over the
    # driver. Local (`model_name == "local"`) is the engine-owned
    # kobold-native StreamEngine for streaming AND one-shot (collect) —
    # the OpenAI-compat local transport is gone.
    # ------------------------------------------------------------------

    @staticmethod
    def _persona_config_with_params(
        persona_config: Dict[str, Any], params: GenerationParams
    ) -> Dict[str, Any]:
        """Overlay the structured params onto the legacy persona_config dict
        so the per-provider stream drivers keep working."""
        merged = dict(persona_config)
        if params.temperature is not None:
            merged["temperature"] = params.temperature
        if params.top_p is not None:
            merged["top_p"] = params.top_p
        if params.top_k is not None:
            merged["top_k"] = params.top_k
        if params.max_tokens is not None:
            merged["max_output_tokens"] = params.max_tokens
        return merged

    @staticmethod
    def _messages_to_history_object(
        messages: List[Dict[str, Any]],
        image_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Adapter for handlers that still take the legacy history_object
        shape. Splits a leading system message into `persona_prompt` so that
        `message_history` (and its legacy `history` alias) contain only
        non-system turns — matching the contract `_run_tool_loop` used to
        build before Phase C."""
        if messages and messages[0].get("role") == "system":
            persona_prompt = messages[0].get("content", "") or ""
            rest = list(messages[1:])
        else:
            persona_prompt = ""
            rest = list(messages)
        return {
            "persona_prompt": persona_prompt,
            "message_history": rest,
            "history": rest,  # legacy alias used by integration test mocks
            "current_message": {"text": "", "image_url": image_url},
        }

    async def stream_messages(
        self,
        persona_config: Dict[str, Any],
        messages: List[Dict[str, Any]],
        params: GenerationParams,
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        local_inference_config: Optional[Dict[str, Any]] = None,
        image_url: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Provider-agnostic streaming entry. Yields:
          - `{"type": "api_payload", "payload": ...}` first
          - one or more `{"type": "text_delta", "text": ...}` chunks
          - optional `{"type": "tool_calls", "calls": [...]}`
          - terminal `{"type": "done", "full_text": ...}`

        For `model_name == "local"` this routes straight to the kobold-native
        SSE stream (GenerationParams — including kobold provider_extras —
        pass through unchanged, and no retry policy applies, matching the
        pre-cutover portal path). All other models dispatch through the
        `_stream_response` policy driver: true token deltas from the
        canonical per-provider streams, with the same rate limiting / retry /
        fallback policy as `generate_response` (DP-206b cutover)."""
        merged_config = self._persona_config_with_params(persona_config, params)
        model_name: str = merged_config.get("model_name", "")

        if model_name == "local":
            async for ev in self.stream_engine.stream_messages(
                merged_config, messages, params, tools
            ):
                yield ev
            return

        history_object = self._messages_to_history_object(messages, image_url)
        async for ev in self._stream_response(
            merged_config, history_object, tools, local_inference_config,
        ):
            yield ev

    def stream_prompt(
        self,
        persona_config: Dict[str, Any],
        rendered_prompt: str,
        params: GenerationParams,
        *,
        stop_sequences: Optional[List[str]] = None,
        tools_advertised: Optional[List[str]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream from a caller-rendered prompt. Local-only — the portal
        path where kobold-lite owns templating. Non-local model raises."""
        model_name = persona_config.get("model_name", "")
        if model_name != "local":
            raise LLMCommunicationError(
                f"stream_prompt only supports local models, got '{model_name}'"
            )
        return self.stream_engine.stream_prompt(
            persona_config, rendered_prompt, params,
            stop_sequences=stop_sequences,
            tools_advertised=tools_advertised,
        )

    @staticmethod
    async def collect_stream(
        stream: AsyncIterator[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Drain a stream into `(result_dict, api_payload)` matching
        generate_response's return shape. Single source of truth for
        non-streaming consumers — they get the same tuple regardless of
        whether the underlying provider streams or not."""
        api_payload: Optional[Dict[str, Any]] = None
        text_parts: List[str] = []
        full_text: Optional[str] = None
        calls: Optional[List[Dict[str, Any]]] = None
        async for ev in stream:
            etype = ev.get("type")
            if etype == "api_payload":
                api_payload = ev.get("payload")
            elif etype == "text_delta":
                text_parts.append(ev.get("text", "") or "")
            elif etype == "tool_calls":
                calls = list(ev.get("calls", []))
            elif etype == "done":
                full_text = ev.get("full_text")
        if calls is not None:
            # A tool_calls event — even with an empty list (e.g. every call
            # failed to parse) — means the model chose the tool path; mirror
            # the pre-DP-206 one-shot handlers, which returned
            # {"type": "tool_calls", "calls": []} so the empty-response retry
            # logic fires rather than treating it as a text turn.
            return {"type": "tool_calls", "calls": calls}, api_payload
        text = full_text if full_text is not None else "".join(text_parts)
        return {"type": "text", "content": text}, api_payload
