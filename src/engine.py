# src/engine.py

import json
import logging
import os
import time
from typing import Dict, Any, Optional, Tuple, List

from dotenv import load_dotenv

from config import global_config
from config.global_config import EMPTY_RESPONSE_RETRIES, EMPTY_RESPONSE_RETRY_DELAY
# --- Provider-specific imports ---
import aiohttp
import anthropic
from openai import AsyncOpenAI, APIStatusError, APITimeoutError
from google import genai
from google.genai.types import GenerateContentConfig, Tool, GoogleSearch, Candidate, GroundingMetadata, \
    FunctionDeclaration, Part
from src.utils.google_utils import process_grounding_metadata

logger = logging.getLogger(__name__)


class LLMCommunicationError(Exception):
    """Custom exception for when the TextEngine cannot communicate with an LLM provider."""
    pass


class TextEngine:
    """A centralized engine for handling requests to various LLM APIs."""

    def __init__(self) -> None:
        # --- Lazy-loaded clients ---
        self.openai_client: Optional[AsyncOpenAI] = None
        self.anthropic_client: Optional[anthropic.Anthropic] = None

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
        # Google: gemini models
        if 'gemini' in model_name:
            return True
        return False

    async def _get_openai_client(self) -> AsyncOpenAI:
        """Initializes and returns the OpenAI client."""
        if self.openai_client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key: raise ValueError("OPENAI_API_KEY not found in environment.")
            self.openai_client = AsyncOpenAI(api_key=api_key)
        return self.openai_client

    def _get_anthropic_client(self) -> anthropic.Anthropic:
        """Initializes and returns the Anthropic client."""
        if self.anthropic_client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key: raise ValueError("ANTHROPIC_API_KEY not found in environment.")
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

    async def generate_response(self, persona_config: Dict[str, Any], context_object: Dict[str, Any],
                                tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[
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

        for attempt in range(EMPTY_RESPONSE_RETRIES + 1):
            result: Dict[str, Any] = {}
            api_payload: Optional[Dict[str, Any]] = None

            try:
                if model_name.startswith("gpt"):
                    result, api_payload = await self._generate_openai_response(persona_config, context_object, tools)
                elif "claude" in model_name:
                    result, api_payload = await self._generate_anthropic_response(persona_config, context_object, tools)
                elif "gemini" in model_name:
                    result, api_payload = await self._generate_google_response(persona_config, context_object, tools)
                elif model_name == 'local':
                    result, api_payload = await self._generate_local_response(persona_config, context_object, tools)
                else:
                    raise LLMCommunicationError(f"Error: Model '{model_name}' is not supported.")

                # Validate the response structure and content
                if result.get('type') == 'text' and result.get('content', '').strip():
                    return result, api_payload
                if result.get('type') == 'tool_calls' and result.get('calls'):
                    return result, api_payload

            except LLMCommunicationError as e:
                # Re-raise critical errors immediately without retry
                if attempt >= EMPTY_RESPONSE_RETRIES:
                    raise
                logger.warning(f"LLM communication error (Attempt {attempt + 1}). Retrying... Error: {e}")

            if attempt < EMPTY_RESPONSE_RETRIES:
                logger.warning(f"LLM returned an empty or invalid response (Attempt {attempt + 1}). Retrying...")
                time.sleep(EMPTY_RESPONSE_RETRY_DELAY)

        logger.error(f"LLM returned an empty or invalid response after {EMPTY_RESPONSE_RETRIES + 1} attempts.")
        raise LLMCommunicationError(f"LLM provider returned an empty or invalid response after all retries.")

    async def _generate_openai_response(self, config: Dict[str, Any], context: Dict[str, Any],
                                        tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[
        Dict[str, Any], Dict[str, Any]]:
        client = await self._get_openai_client()
        messages: List[Dict[str, Any]] = []
        if context["history"] and context["history"][0]["role"] == "system":
            messages.append(context["history"][0])
            history_to_process = context["history"][1:]
        else:
            messages.append({"role": "system", "content": context["persona_prompt"]})
            history_to_process = context["history"]

        messages.extend(history_to_process)

        if context["current_message"].get("image_url"):
            last_message = messages[-1]
            if last_message['role'] == 'user':
                if isinstance(last_message['content'], str):
                    last_message['content'] = [{"type": "text", "text": last_message['content']}]
                last_message['content'].append(
                    {"type": "image_url", "image_url": {"url": context["current_message"]["image_url"]}})

        api_params: Dict[str, Any] = {
            "model": config["model_name"],
            "messages": messages,
            "max_tokens": config.get("max_output_tokens") or global_config.DEFAULT_TOKEN_LIMIT,
            "temperature": config.get("temperature"),
            "top_p": config.get("top_p")
        }
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        api_params = {k: v for k, v in api_params.items() if v is not None}

        try:
            completion = await client.chat.completions.create(**api_params)
            response_message = completion.choices[0].message

            if "tools" in api_params:
                api_params["tools"] = [tool.get("function", {}).get("name", "unknown") for tool in
                                       api_params.get("tools", [])]

            if response_message.tool_calls:
                tool_calls: List[Dict[str, Any]] = []
                for call in response_message.tool_calls:
                    try:
                        arguments = json.loads(call.function.arguments)
                        tool_calls.append(
                            {"id": call.id, "name": call.function.name, "arguments": arguments}
                        )
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse tool call arguments: {call.function.arguments}")
                        continue
                return {"type": "tool_calls", "calls": tool_calls}, api_params
            else:
                response_content: str = response_message.content or ""
                return {"type": "text", "content": response_content}, api_params

        except (APIStatusError, APITimeoutError) as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            raise LLMCommunicationError(f"OpenAI API returned an error: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected OpenAI error occurred: {e}", exc_info=True)
            raise LLMCommunicationError(f"An unexpected error occurred with the OpenAI API.") from e

    async def _generate_anthropic_response(self, config: Dict[str, Any], context: Dict[str, Any],
                                           tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[
        Dict[str, Any], Dict[str, Any]]:
        client = self._get_anthropic_client()

        system_prompt = context["persona_prompt"]
        history = context["history"]
        if history and history[0]["role"] == "system":
            system_prompt = f"{system_prompt}\n\n{history[0]['content']}"
            history = history[1:]

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
            api_params["tools"] = tools

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
            logger.error(f"Anthropic API error: {e}", exc_info=True)
            raise LLMCommunicationError(f"Anthropic API returned an error: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected Anthropic error occurred: {e}", exc_info=True)
            raise LLMCommunicationError(f"An unexpected error occurred with the Anthropic API.") from e

    async def _generate_google_response(self, config: Dict[str, Any], context: Dict[str, Any],
                                        tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[
        Dict[str, Any], Dict[str, Any]]:
        try:
            self._initialize_google_client()
            assert self.google_client is not None and self.google_search_tool is not None
        except (ValueError, AssertionError) as e:
            raise LLMCommunicationError(f"Error: Google not configured: {e}") from e

        history_for_api = []
        serializable_history = []

        system_prompt = context["persona_prompt"]
        history_to_process = context["history"]
        if history_to_process and history_to_process[0]["role"] == "system":
            system_prompt = f"{system_prompt}\n\n{history_to_process[0]['content']}"
            history_to_process = history_to_process[1:]

        first_turn = True
        for item in history_to_process:
            role = 'model' if item['role'] == 'assistant' else 'user'

            serializable_item = item.copy()

            if item['role'] == 'tool':
                part_dict = {'function_response': {'name': item['name'], 'response': json.loads(item['content'])}}
                history_for_api.append({'role': 'tool', 'parts': [Part(**part_dict)]})
                serializable_item['parts'] = [part_dict]
            elif item.get('tool_calls'):
                parts_list = [{'function_call': {'name': call['name'], 'args': call['arguments']}} for call in
                              item['tool_calls']]
                history_for_api.append({'role': 'model', 'parts': [Part(**p) for p in parts_list]})
                serializable_item['parts'] = parts_list
            else:
                content_text = item['content']
                if first_turn and role == 'user':
                    content_text = f"{system_prompt}\n\n### Conversation:\n{content_text}"
                    first_turn = False
                part_dict = {'text': content_text}
                history_for_api.append({'role': role, 'parts': [Part(**part_dict)]})
                serializable_item['parts'] = [part_dict]

            serializable_history.append(serializable_item)

        content_config_for_api: Dict[str, Any] = {"safety_settings": self.google_safety_settings}

        api_tools: List[Tool] = []
        if tools:
            converted_tools = [Tool(function_declarations=[FunctionDeclaration(**t['function'])])
                               for t in tools if t.get('type') == 'function' and t.get('function')]
            api_tools.extend(converted_tools)
            content_config_for_api['tool_config'] = {"function_calling_config": {"mode": "AUTO"}}
        else:
            api_tools.append(self.google_search_tool)
        content_config_for_api['tools'] = api_tools

        content_config_for_api['max_output_tokens'] = config.get(
            "max_output_tokens") or global_config.DEFAULT_TOKEN_LIMIT
        if isinstance(config.get("temperature"), (int, float)): content_config_for_api['temperature'] = config.get(
            "temperature")
        if isinstance(config.get("top_p"), (int, float)): content_config_for_api['top_p'] = config.get("top_p")
        if isinstance(config.get("top_k"), (int, float)): content_config_for_api['top_k'] = config.get("top_k")

        dump_config = content_config_for_api.copy()
        if 'tools' in dump_config:
            tool_names = []
            for t in dump_config['tools']:
                if hasattr(t, 'function_declarations') and t.function_declarations:
                    tool_names.extend([d.name for d in t.function_declarations])
                elif hasattr(t, 'google_search') and t.google_search is not None:
                    tool_names.append("google_search")
            dump_config['tools'] = tool_names

        api_params_for_dumping = {
            'model': config["model_name"], 'contents': serializable_history,
            'config': json.loads(json.dumps(dump_config, default=str))
        }

        try:
            response_obj = await self.google_client.models.generate_content(
                model=config["model_name"],
                contents=history_for_api,
                config=GenerateContentConfig(**content_config_for_api)
            )
        except Exception as e:
            logger.error(f"Google API error: {e}", exc_info=True)
            raise LLMCommunicationError(f"An error occurred with Google API: {e}") from e

        if response_obj.prompt_feedback and response_obj.prompt_feedback.block_reason:
            raise LLMCommunicationError(
                f"Response blocked by Google due to {response_obj.prompt_feedback.block_reason.name}.")

        candidate: Optional[Candidate] = response_obj.candidates[0] if response_obj.candidates else None
        if not candidate or not candidate.content or not candidate.content.parts:
            return {}, api_params_for_dumping

        tool_calls: List[Dict[str, Any]] = []
        for i, part in enumerate(candidate.content.parts):
            if part.function_call:
                arguments = {k: v for k, v in part.function_call.args.items()}
                tool_calls.append({"id": f"call_{part.function_call.name}_{i}", "name": part.function_call.name,
                                   "arguments": arguments})
        if tool_calls:
            return {"type": "tool_calls", "calls": tool_calls}, api_params_for_dumping

        base_text_from_response = "".join(
            part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text)
        final_text_content, search_query_display, citations_display = process_grounding_metadata(
            base_text_from_response, candidate.grounding_metadata, logger
        )
        if search_query_display: final_text_content += search_query_display
        if citations_display: final_text_content += citations_display

        return {"type": "text", "content": final_text_content.strip()}, api_params_for_dumping

    async def _generate_local_response(self, config: Dict[str, Any], context: Dict[str, Any],
                                       tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[
        Dict[str, Any], Dict[str, Any]]:
        url: str = 'http://localhost:5001/api/v1/generate'
        history_str: str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context['history']])
        full_prompt: str = f"{history_str}\nuser: {context['current_message']['text']}\nassistant:"

        payload: Dict[str, Any] = {
            "max_length": config.get("max_output_tokens") or global_config.DEFAULT_TOKEN_LIMIT,
            "temperature": config.get("temperature"),
            "top_p": config.get("top_p"),
            "top_k": config.get("top_k"),
            "memory": context["persona_prompt"],
            "prompt": full_prompt
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    response_text: str = data.get('results', [{}])[0].get('text', '')
                    return {"type": "text", "content": response_text.strip()}, payload
        except aiohttp.ClientError as e:
            logger.error(f"Local model connection error: {e}", exc_info=True)
            raise LLMCommunicationError("Could not connect to local model.") from e
