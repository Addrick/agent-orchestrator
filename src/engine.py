# src/engine.py

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
    RATE_LIMIT_AGY_RPM,
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
from src.text_tool_protocol import (
    TOOL_CALL_OPEN,
    TOOL_CALL_CLOSE,
    decode_tool_call_payload,
    extract_first_tool_call_block,
    render_tool_descriptions,
)

AGY_CALL_TIMEOUT_SECONDS = 120.0

logger = logging.getLogger(__name__)


class LLMCommunicationError(Exception):
    """Custom exception for when the TextEngine cannot communicate with an LLM provider."""

    def __init__(self, message: str, api_payload: Optional[Dict[str, Any]] = None,
                 rate_limited: bool = False):
        super().__init__(message)
        self.api_payload = api_payload
        self.rate_limited = rate_limited


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
        self.anthropic_client: Optional[anthropic.Anthropic] = None
        # Local-streaming delegate. Optional — non-local streaming wraps
        # generate_response so unit tests need not wire one up. Phase B.
        self.stream_engine: Optional[Any] = stream_engine

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
        logger.info(
            f"Rate limiters initialised — "
            f"Gemini 2.5: {RATE_LIMIT_GEMINI_25_RPM} RPM / {RATE_LIMIT_GEMINI_25_RPD} RPD | "
            f"Gemini 3.1: {RATE_LIMIT_GEMINI_3_RPM} RPM | "
            f"Gemma 3: {RATE_LIMIT_GEMMA_3_RPM} RPM | Gemma 4: {RATE_LIMIT_GEMMA_4_RPM} RPM | "
            f"OpenAI: {RATE_LIMIT_OPENAI_RPM} RPM | "
            f"Anthropic: {RATE_LIMIT_ANTHROPIC_RPM} RPM | "
            f"AGY: {RATE_LIMIT_AGY_RPM} RPM"
        )

        self._initialize_env()

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
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise LLMCommunicationError("OPENAI_API_KEY not set — skipping OpenAI provider.")
            self.openai_client = AsyncOpenAI(api_key=api_key)
        return self.openai_client

    def _get_anthropic_client(self) -> anthropic.Anthropic:
        """Initializes and returns the Anthropic client."""
        if self.anthropic_client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMCommunicationError("ANTHROPIC_API_KEY not set — skipping Anthropic provider.")
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        return self.anthropic_client

    def _initialize_google_client(self) -> None:
        """Initializes the Google client using the original project's method."""
        if self.google_client is not None:
            return

        api_key = os.environ.get("GOOGLE_GENERATIVEAI_API_KEY")
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
        """Returns (handler_method, [limiters]) for the model name.
        Raises LLMCommunicationError for unsupported models."""
        if model_name.startswith("gpt"):
            return self._generate_openai_response, [self._openai_limiter]
        if "claude" in model_name:
            return self._generate_anthropic_response, [self._anthropic_limiter]
        if "gemma-4" in model_name:
            return self._generate_google_response, [self._gemma_4_rpm_limiter]
        if "gemma" in model_name:
            return self._generate_google_response, [self._gemma_4_rpm_limiter]
        if "gemini-3.1" in model_name:
            return self._generate_google_response, [self._gemini_3_rpm_limiter]
        if "gemini" in model_name:
            return self._generate_google_response, [self._gemini_25_rpm_limiter, self._gemini_25_rpd_limiter]
        if model_name.startswith("agy"):
            self._ensure_agy_supported()
            return self._generate_agy_response, [self._agy_limiter]
        if model_name == 'local':
            return self._generate_local_response, []
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
        Routes the generation request and retries on empty responses.
        Returns: A tuple containing:
                 1. A structured dictionary:
                    - {'type': 'text', 'content': '...'} for a text response.
                    - {'type': 'tool_calls', 'calls': [{'id': '...', 'name': '...', 'arguments': {...}}]} for a tool call.
                 2. The API payload dictionary for debugging, or None.
        Raises: LLMCommunicationError if all retries fail or produce empty/invalid responses.
        """
        model_name: str = persona_config.get("model_name", "")

        if history_object["current_message"].get("image_url") and not self.model_supports_images(model_name):
            logger.info(f"Model {model_name} does not support images. Modifying prompt.")
            history_object["persona_prompt"] += (
                "\n\n[System note: The user has attached an image that you cannot see."
                " Please inform them of this fact in your response.]"
            )
            history_object["current_message"]["image_url"] = None

        handler, limiters = self._get_provider_route(model_name)

        for attempt in range(EMPTY_RESPONSE_RETRIES + 1):
            result: Dict[str, Any] = {}
            api_payload: Optional[Dict[str, Any]] = None

            try:
                async with self._rate_limited(limiters):
                    if model_name == 'local':
                        result, api_payload = await handler(persona_config, history_object, tools, local_inference_config)
                    else:
                        result, api_payload = await handler(persona_config, history_object, tools)

                # Validate the response structure and content
                if result.get('type') == 'text' and result.get('content', '').strip():
                    return result, api_payload
                if result.get('type') == 'tool_calls' and result.get('calls'):
                    return result, api_payload

            except LLMCommunicationError as e:
                if e.rate_limited:
                    fallback = self._get_fallback_model(model_name)
                    if fallback:
                        logger.warning(
                            f"Rate limit (429) hit for '{model_name}', "
                            f"falling back to '{fallback}'."
                        )
                        persona_config = {**persona_config, "model_name": fallback}
                        model_name = fallback
                        handler, limiters = self._get_provider_route(model_name)
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

    async def _generate_openai_response(self, config: Dict[str, Any], history_object: Dict[str, Any],
                                        tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[
        Dict[str, Any], Dict[str, Any]]:
        client = await self._get_openai_client()
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

        try:
            completion = await client.chat.completions.create(**api_params)
            response_message = completion.choices[0].message

            if "tools" in api_params:
                api_params["tools"] = [tool.get("function", {}).get("name", "unknown") for tool in
                                       api_params.get("tools", [])]

            if response_message.tool_calls:
                tool_calls = self._parse_openai_tool_calls(response_message.tool_calls)
                return {"type": "tool_calls", "calls": tool_calls}, api_params
            else:
                response_content: str = response_message.content or ""
                return {"type": "text", "content": response_content}, api_params

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

    async def _generate_anthropic_response(self, config: Dict[str, Any], history_object: Dict[str, Any],
                                           tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[
        Dict[str, Any], Dict[str, Any]]:
        client = self._get_anthropic_client()

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

        api_params = {k: v for k, v in api_params.items() if v is not None}

        try:
            response = client.messages.create(**api_params)

            if "tools" in api_params:
                api_params["tools"] = [tool.get("name", "unknown") for tool in api_params.get("tools", [])]

            if response.stop_reason == "tool_use":
                tool_calls: List[Dict[str, Any]] = []
                for content_block in response.content:
                    if content_block.type == 'tool_use':
                        tool_calls.append({
                            "id": content_block.id,
                            "name": content_block.name,
                            "arguments": content_block.input
                        })
                return {"type": "tool_calls", "calls": tool_calls}, api_params
            else:
                response_content: str = response.content[0].text or ""
                return {"type": "text", "content": response_content}, api_params

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

    async def _generate_google_response(self, config: Dict[str, Any], history_object: Dict[str, Any],
                                        tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[
        Dict[str, Any], Dict[str, Any]]:
        """Generates a response using the Google Gemini API."""
        try:
            self._initialize_google_client()
            assert self.google_client is not None and self.google_search_tool is not None
        except (ValueError, AssertionError) as e:
            raise LLMCommunicationError(f"Error: Google not configured: {e}") from e

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

        try:
            response_obj = await self.google_client.models.generate_content(
                model=config["model_name"],
                contents=history_for_api,
                config=GenerateContentConfig(**content_config_for_api)
            )
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

        return self._parse_google_response(response_obj, api_params_for_dumping)

    async def _get_local_client(self) -> AsyncOpenAI:
        """
        Creates a new AsyncOpenAI client configured to point to a local,
        OpenAI-compatible API endpoint (like KoboldCPP or Ollama).
        """
        # Use the configured URL from global_config or env
        local_api_url = os.environ.get("LOCAL_LLM_URL", global_config.LOCAL_LLM_URL)
        return AsyncOpenAI(base_url=local_api_url, api_key="not-required")

    async def _generate_local_response(self, config: Dict[str, Any], history_object: Dict[str, Any],
                                       tools: Optional[List[Dict[str, Any]]] = None,
                                       local_inference_config: Optional[Dict[str, Any]] = None) -> Tuple[
        Dict[str, Any], Dict[str, Any]]:
        """
        Generates a response from a local model by reusing the standard OpenAI
        API format. This standardizes local model integration.
        """
        local_client = await self._get_local_client()

        # Temporarily swap the main OpenAI client with our special local client
        original_openai_client = self.openai_client
        self.openai_client = local_client

        # Merge local_inference_config into config (Exact Parity)
        if local_inference_config:
            config = {**config, **{k: v for k, v in local_inference_config.items() if v is not None}}

        try:
            # Most local servers ignore the model name in the payload and use whatever is loaded.
            # We provide a placeholder for consistency.
            config['model_name'] = 'local-model'
            # We can now call our standard, well-tested OpenAI method!
            return await self._generate_openai_response(config, history_object, tools)
        except LLMCommunicationError as e:
            logger.error(f"Local OpenAI-compatible API error: {e}", exc_info=True)
            # Re-raise with a more specific "Local API" message, but preserving the original payload.
            raise LLMCommunicationError(f"Local API returned an error: {e}", api_payload=e.api_payload) from e
        except Exception as e:
            logger.error(f"An unexpected local API error occurred: {e}", exc_info=True)
            # This will catch errors before the API call is made (e.g., client init)
            raise LLMCommunicationError("An unexpected error occurred with the Local API.") from e
        finally:
            # CRITICAL: Always restore the original client to avoid breaking
            # subsequent calls to the actual OpenAI API.
            self.openai_client = original_openai_client

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

    async def _run_agy_cli(self, prompt: str, timeout: float = AGY_CALL_TIMEOUT_SECONDS) -> str:
        self._ensure_agy_supported()

        binary = os.environ.get("ANTIGRAVITY_HARNESS_PATH") or shutil.which("agy")
        if not binary:
            raise LLMCommunicationError("Antigravity harness/agy binary not found.")

        timeout_sec_str = f"{int(timeout) + 30}s"
        args = ["--print-timeout", timeout_sec_str, "-p", prompt]

        temp_dir = tempfile.mkdtemp()
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                binary,
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir,
                start_new_session=True
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
            except asyncio.TimeoutError as e:
                raise LLMCommunicationError(f"agy CLI timed out after {timeout} seconds.") from e

            if proc.returncode != 0:
                stderr_excerpt = stderr.decode("utf-8", errors="replace").strip()
                excerpt = stderr_excerpt[-200:] if len(stderr_excerpt) > 200 else stderr_excerpt
                raise LLMCommunicationError(
                    f"agy CLI failed with exit code {proc.returncode}. Stderr: {excerpt}"
                )

            return stdout.decode("utf-8", errors="replace")
        finally:
            if proc is not None:
                try:
                    import signal
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass

            cli_dir = os.path.join(temp_dir, ".antigravitycli")
            if os.path.isdir(cli_dir):
                for f in os.listdir(cli_dir):
                    p = os.path.join(cli_dir, f)
                    if os.path.islink(p):
                        try:
                            target = os.readlink(p)
                            if os.path.exists(target):
                                os.remove(target)
                        except Exception:
                            pass

            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _generate_agy_response(
        self, config: Dict[str, Any], history_object: Dict[str, Any], tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

        api_payload = {
            "model": config.get("model_name"),
            "prompt_chars": len(prompt),
            "tools": tool_names,
            "isolation": {
                "stdin": "devnull",
                "skip_permissions": False
            }
        }

        try:
            raw = await self._run_agy_cli(prompt)
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

    # ------------------------------------------------------------------
    # Phase B — provider streaming surface
    #
    # `stream_messages(persona, messages, params)` and
    # `stream_prompt(persona, prompt, params)` are the unified entries.
    # Local routes to the StreamEngine for real token-by-token SSE; the
    # SDK-backed providers (OpenAI/Anthropic/Google) currently wrap
    # generate_response and emit a single `text_delta` so the surface is
    # uniform without forcing per-provider streaming SDK work in this
    # phase. Phase C flips ownership: generate_response will become the
    # collect-stream wrapper around stream_messages.
    # See memory/project/plans/portal_engine_reintegration.md.
    # ------------------------------------------------------------------

    @staticmethod
    def _persona_config_with_params(
        persona_config: Dict[str, Any], params: GenerationParams
    ) -> Dict[str, Any]:
        """Overlay the structured params onto the legacy persona_config dict
        so existing _generate_*_response handlers keep working."""
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

        For `model_name == "local"` and a configured StreamEngine, this is a
        true SSE token stream. Other providers wrap `generate_response`."""
        merged_config = self._persona_config_with_params(persona_config, params)
        model_name: str = merged_config.get("model_name", "")

        if model_name == "local" and self.stream_engine is not None:
            async for ev in self.stream_engine.stream_messages(
                merged_config, messages, params, tools
            ):
                yield ev
            return

        history_object = self._messages_to_history_object(messages, image_url)
        try:
            result, api_payload = await self.generate_response(
                merged_config, history_object, tools, local_inference_config,
            )
        except LLMCommunicationError as e:
            if e.api_payload is not None:
                yield {"type": "api_payload", "payload": e.api_payload}
            raise

        yield {"type": "api_payload", "payload": api_payload or {}}
        if result.get("type") == "tool_calls":
            yield {"type": "tool_calls", "calls": list(result.get("calls", []))}
            yield {"type": "done", "full_text": ""}
        else:
            text = result.get("content", "") or ""
            if text:
                yield {"type": "text_delta", "text": text}
            yield {"type": "done", "full_text": text}

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
        if self.stream_engine is None:
            raise LLMCommunicationError(
                "StreamEngine not configured — cannot stream pre-rendered prompts."
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
        if calls:
            return {"type": "tool_calls", "calls": calls}, api_payload
        text = full_text if full_text is not None else "".join(text_parts)
        return {"type": "text", "content": text}, api_payload
