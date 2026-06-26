# src/engine/driver.py

import json
import logging
import os
import asyncio
import re
import shutil
import tempfile
import uuid
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
import anthropic
from google import genai
from google.genai.types import Tool
from src.utils.claude_cli_env import build_claude_cli_env
from src.generation_params import GenerationParams
from src.llm_errors import LLMCommunicationError
from src.stream_engine import StreamEngine
from src.text_tool_protocol import (
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    decode_tool_call_payload,
    extract_first_tool_call_block,
    render_tool_descriptions,
)
# DP-244: Provider ABC family + ordered registry. OpenAI is fully extracted into
# `providers.openai`; the other five route through thin `_EngineProvider`
# adapters until their own slice. `_shared` holds the hoisted free helpers.
from src.engine.registry import build_registry
from src.engine.providers import _shared
from src.engine.providers.openai import stream_openai
from src.engine.providers.anthropic import stream_anthropic
from src.engine.providers import google

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
        # OpenAIProvider (providers.openai) lazily fills this cache slot.
        self.openai_client: Optional[Any] = None
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

        # DP-244: ordered Provider registry (replaces the _get_provider_route
        # string-prefix waterfall). Built after the limiters so providers can
        # reference them.
        self._registry = build_registry(self)

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


    @classmethod
    def _get_fallback_model(cls, model_name: str) -> Optional[str]:
        """Returns a fallback model for rate-limited requests, or None."""
        return cls._FALLBACK_MODELS.get(model_name)

    def _get_provider_route(self, model_name: str) -> Tuple[Callable, List[AsyncLimiter]]:
        """Back-compat shim over the registry (DP-244). Returns
        (stream_factory, [limiters]) — the bound `_stream_<provider>_response`
        method and its limiters — for callers/tests written against the
        pre-244 waterfall. The driver itself resolves providers directly via
        `self._registry`. Runs the provider host guard and raises
        LLMCommunicationError for unsupported models, exactly as before."""
        provider = self._registry.resolve(model_name)
        handler = getattr(self, provider.route_method_name)
        return handler, provider.limiters_for(model_name)

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

        provider = self._registry.resolve(model_name)

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
                async with self._rate_limited(provider.limiters_for(model_name)):
                    stream = provider.stream(
                        persona_config, history_object, tools,
                        local_inference_config=local_inference_config,
                    )
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
                        provider = self._registry.resolve(model_name)
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
        """Thin seam over `providers._shared.parse_openai_tool_calls` (DP-244);
        kept on the engine for callers/tests written against the pre-244 API."""
        return _shared.parse_openai_tool_calls(raw_calls)

    async def _stream_openai_response(
        self, config: Dict[str, Any], history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Engine seam for the OpenAI provider (DP-244). The logic lives in
        `providers.openai.stream_openai`; this delegator is the patch point the
        driver routes through (`OpenAIProvider.stream` calls back into it) so
        the driver-policy tests can inject fake streams as before."""
        async for ev in stream_openai(self, config, history_object, tools):
            yield ev

    async def _stream_anthropic_response(
        self, config: Dict[str, Any], history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Engine seam for the Anthropic provider (DP-244). The logic lives in
        `providers.anthropic.stream_anthropic`; this delegator is the patch point
        the driver routes through (`AnthropicProvider.stream` calls back into it)
        so the driver-policy tests can inject fake streams as before."""
        async for ev in stream_anthropic(self, config, history_object, tools):
            yield ev

    @staticmethod
    def _extract_system_prompt(history_object: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        """Thin seam over `providers._shared.extract_system_prompt` (DP-244)."""
        return _shared.extract_system_prompt(history_object)

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
        """Thin seam over `providers._shared.download_image` (DP-244)."""
        return await _shared.download_image(image_url)

    async def _build_google_history(
        self, system_prompt: str, history: List[Dict[str, Any]], image_url: Optional[str]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Engine seam for Google history-building (DP-244). Logic lives in
        `providers.google.build_google_history`; kept here because tests call it
        directly on the engine."""
        return await google.build_google_history(self, system_prompt, history, image_url)

    @staticmethod
    def _parse_google_response(
        response_obj: Any, api_params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Engine seam for Google response-parsing (DP-244). Logic lives in
        `providers.google.parse_google_response`; kept here because tests call it
        directly on the engine."""
        return google.parse_google_response(response_obj, api_params)

    async def _stream_google_response(
        self, config: Dict[str, Any], history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Engine seam for the Google provider (DP-244). The logic lives in
        `providers.google.stream_google`; this delegator is the patch point the
        driver routes through (`GoogleProvider.stream` calls back into it) so the
        driver-policy tests can inject fake streams as before."""
        async for ev in google.stream_google(self, config, history_object, tools):
            yield ev

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
                        label: str = "agy", env: Optional[Dict[str, str]] = None) -> str:
        # `label` names the provider in error messages — this CLI runner is
        # shared by the agy and cc (Claude Code) routes, so a failure must
        # point at the route the caller actually invoked. `env` overrides the
        # child environment (cc passes a subscription-scrubbed env; agy passes
        # None = inherit unchanged).
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                binary,
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace_dir,
                env=env,
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
    def _resolve_cc_workspace(
        persona_name: Optional[str], workspace_override: Optional[str] = None
    ) -> Optional[str]:
        """Returns the working dir for this call, or None when persistence is
        disabled (caller uses a throwaway temp dir). Precedence: per-call
        workspace_override (e.g. the DP-227 fixr clone, set by the orchestration
        layer) > explicit CC_WORKSPACE_DIR (the derpr checkout) > per-persona
        dir > global dir. Does not create the directory."""
        if workspace_override:
            return os.path.abspath(workspace_override)
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
        workspace_override: Optional[str] = None,
    ) -> str:
        self._ensure_cc_supported()

        binary = os.environ.get("CLAUDE_CLI_PATH") or shutil.which("claude")
        if not binary:
            raise LLMCommunicationError("Claude Code 'claude' binary not found on PATH.")

        args = self._build_cc_args(prompt, system_prompt, model_arg)
        # cc-* must use the Claude subscription, not the metered API: strip the
        # inherited ANTHROPIC_API_KEY so `-p` mode falls through to the OAuth token.
        cc_env = build_claude_cli_env()

        workspace_dir = self._resolve_cc_workspace(persona_name, workspace_override)
        if workspace_dir is None:
            temp_dir = tempfile.mkdtemp()
            try:
                return await self._exec_agy(binary, args, temp_dir, timeout, label="Claude Code", env=cc_env)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        os.makedirs(workspace_dir, exist_ok=True)
        lock = self._cc_workspace_locks.setdefault(workspace_dir, asyncio.Lock())
        async with lock:
            return await self._exec_agy(binary, args, workspace_dir, timeout, label="Claude Code", env=cc_env)

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
        # DP-227: the orchestration layer may inject a per-run workspace (the
        # fixr self-edit clone). It takes precedence over CC_WORKSPACE_DIR.
        workspace_override = config.get("cc_workspace_override")
        workspace_dir = self._resolve_cc_workspace(persona_name, workspace_override)

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
                prompt, system_prompt or "", model_arg,
                persona_name=persona_name, workspace_override=workspace_override,
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
