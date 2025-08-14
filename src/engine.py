# src/engine.py

import json
import logging
import os
import time
from typing import Dict, Any, Optional, Tuple, List

from dotenv import load_dotenv

from config.global_config import GEMINI_EMPTY_RESPONSE_RETRIES
# --- Provider-specific imports ---
import aiohttp
import anthropic
from openai import AsyncOpenAI, APIStatusError, APITimeoutError
from google import genai
from google.genai.types import GenerateContentConfig, Tool, GoogleSearch

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
        self.google_safety_settings: Optional[List[Dict[str, str]]] = [
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

        client = genai.client.BaseApiClient(api_key=api_key)
        self.google_client = genai.client.AsyncClient(client)
        self.google_search_tool = Tool(google_search=GoogleSearch())
        # self.google_tool_config is removed from here
        logger.info("Google AI Studio client initialized.")

    async def generate_response(self, persona_config: dict, context_object: Dict[str, Any]) -> Tuple[
        str, Optional[Dict]]:
        """
        Routes the generation request and returns the text response AND the API payload.
        Returns: (response_string, api_payload_dictionary_or_None)
        """
        model_name = persona_config.get("model_name", "")
        if model_name.startswith("gpt"):
            return await self._generate_openai_response(persona_config, context_object)
        elif "claude" in model_name:
            return await self._generate_anthropic_response(persona_config, context_object)
        elif "gemini" in model_name:
            return await self._generate_google_response(persona_config, context_object)
        elif model_name == 'local':
            return await self._generate_local_response(persona_config, context_object)
        else:
            error_msg = f"Error: Model '{model_name}' is not supported."
            logger.error(f"Unknown model provider for model_name: {model_name}")
            return error_msg, None

    async def _generate_openai_response(self, config: dict, context: Dict[str, Any]) -> Tuple[str, Dict]:
        client = await self._get_openai_client()
        messages = [{"role": "system", "content": context["persona_prompt"]}]
        messages.extend(context["history"])
        content_parts = [{"type": "text", "text": context["current_message"]["text"]}]
        if context["current_message"].get("image_url"):
            content_parts.append({"type": "image_url", "image_url": {"url": context["current_message"]["image_url"]}})
        messages.append({"role": "user", "content": content_parts})
        api_params = {"model": config["model_name"], "messages": messages,
                      "max_tokens": config.get("max_output_tokens"), "temperature": config.get("temperature"),
                      "top_p": config.get("top_p")}
        api_params = {k: v for k, v in api_params.items() if v is not None}

        try:
            completion = await client.chat.completions.create(**api_params)
            return completion.choices[0].message.content, api_params
        except (APIStatusError, APITimeoutError) as e:
            is_5xx_error = isinstance(e, APIStatusError) and 500 <= e.status_code < 600
            if is_5xx_error or isinstance(e, APITimeoutError):
                logger.warning(f"OpenAI API server error/timeout: {e}. Retrying once...")
                time.sleep(1)
                try:
                    completion = await client.chat.completions.create(**api_params)
                    return completion.choices[0].message.content, api_params
                except Exception as retry_e:
                    logger.error(f"OpenAI API failed on retry: {retry_e}")
                    raise LLMCommunicationError(f"OpenAI API is unavailable after retry.") from retry_e
            else:  # Not a 5xx error, so don't retry.
                logger.error(f"OpenAI API client error: {e}", exc_info=True)
                raise LLMCommunicationError(f"OpenAI API returned a client error: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected OpenAI error occurred: {e}", exc_info=True)
            raise LLMCommunicationError(f"An unexpected error occurred with the OpenAI API.") from e

    async def _generate_anthropic_response(self, config: dict, context: Dict[str, Any]) -> Tuple[str, Dict]:
        client = self._get_anthropic_client()
        history = [msg for msg in context["history"] if msg["role"] != "system"]
        history.append({"role": "user", "content": context["current_message"]["text"]})
        api_params = {"model": config["model_name"], "system": context["persona_prompt"], "messages": history,
                      "max_tokens": config["max_output_tokens"], "temperature": config.get("temperature"),
                      "top_p": config.get("top_p"), "top_k": config.get("top_k")}
        api_params = {k: v for k, v in api_params.items() if v is not None}
        try:
            response = client.messages.create(**api_params)
            return response.content[0].text, api_params
        except anthropic.APIStatusError as e:
            if 500 <= e.status_code < 600:
                logger.warning(f"Anthropic API server error: {e}. Retrying once...")
                time.sleep(1)
                try:
                    response = client.messages.create(**api_params)
                    return response.content[0].text, api_params
                except Exception as retry_e:
                    logger.error(f"Anthropic API failed on retry: {retry_e}")
                    raise LLMCommunicationError(f"Anthropic API is unavailable after retry.") from retry_e
            else:
                logger.error(f"Anthropic API client error: {e}", exc_info=True)
                raise LLMCommunicationError(f"Anthropic API returned a client error: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected Anthropic error occurred: {e}", exc_info=True)
            raise LLMCommunicationError(f"An unexpected error occurred with the Anthropic API.") from e

    async def _generate_google_response(self, config: dict, context: Dict[str, Any],
                                        tools: Optional[List[Dict]] = None) -> Tuple[str, Dict]:
        """Builds and sends a request to the Google API, with retries for empty responses."""
        try:
            self._initialize_google_client()
        except ValueError as e:
            return f"Error: Google not configured: {e}", {}

        history_str = "\n".join([f"{item['role']}: {item['content']}" for item in context['history']])
        prompt = context['persona_prompt']
        message = context['current_message']['text']
        request_content = (f"### Instructions: ###\n{prompt}\n\n"
                           f"### Conversation History: ###\n{history_str}\n\n"
                           f"### Current Message to Respond To: ###\n{message}")

        content_config_for_api = {
            'tools': [self.google_search_tool],
            'response_modalities': ["TEXT"],
            'safety_settings': self.google_safety_settings
        }
        if tools:
            content_config_for_api['tool_config'] = {"function_calling_config": {"mode": "AUTO"}}
        if isinstance(config.get("max_output_tokens"), int) and config.get("max_output_tokens") >= 100:
            content_config_for_api['max_output_tokens'] = config.get("max_output_tokens")
        if isinstance(config.get("temperature"), (int, float)):
            content_config_for_api['temperature'] = config.get("temperature")
        if isinstance(config.get("top_p"), (int, float)):
            content_config_for_api['top_p'] = config.get("top_p")
        if isinstance(config.get("top_k"), (int, float)):
            content_config_for_api['top_k'] = config.get("top_k")

        api_params_for_dumping = {
            'model': config["model_name"],
            'contents': request_content,
            'config': json.loads(json.dumps(content_config_for_api, default=str))
        }

        response_obj = None
        for attempt in range(GEMINI_EMPTY_RESPONSE_RETRIES + 1):
            try:
                response_obj = await self.google_client.models.generate_content(
                    model=config["model_name"],
                    contents=request_content,
                    config=GenerateContentConfig(**content_config_for_api)
                )
            except Exception as e:
                logger.warning(f"Google API error (Attempt {attempt + 1}): {e}. Retrying once...")
                time.sleep(1)
                if attempt == 0:  # Only one retry on connection error
                    try:
                        response_obj = await self.google_client.models.generate_content(
                            model=config["model_name"],
                            contents=request_content,
                            config=GenerateContentConfig(**content_config_for_api)
                        )
                    except Exception as retry_e:
                        logger.error(f"Google API failed on retry: {retry_e}", exc_info=True)
                        raise LLMCommunicationError(
                            f"An error occurred with Google API after retry: {retry_e}") from retry_e
                else:  # Final attempt failed
                    raise LLMCommunicationError(f"An error occurred with Google API after retry: {e}") from e

            if response_obj.prompt_feedback and response_obj.prompt_feedback.block_reason:
                raise LLMCommunicationError(
                    f"Response blocked by Google due to {response_obj.prompt_feedback.block_reason.name}.")

            base_text_from_response = "".join(
                part.text for part in response_obj.candidates[0].content.parts if hasattr(part, 'text') and part.text)

            if base_text_from_response.strip():
                # Success: we got content
                final_text_content, search_query_display, citations_display = _process_grounding_metadata(
                    base_text_from_response, response_obj.candidates[0].grounding_metadata, logger
                )
                if search_query_display: final_text_content += search_query_display
                if citations_display: final_text_content += citations_display
                return final_text_content.strip(), api_params_for_dumping

            if attempt < GEMINI_EMPTY_RESPONSE_RETRIES:
                logger.warning(f"Google returned an empty response (Attempt {attempt + 1}). Retrying...")
                time.sleep(0.5)

        # If the loop finishes, all retries were empty
        logger.error(f"Google returned an empty response after {GEMINI_EMPTY_RESPONSE_RETRIES + 1} attempts.")
        finish_reason_str = "Unknown"
        if response_obj.candidates and response_obj.candidates[0].finish_reason:
            finish_reason_str = str(response_obj.candidates[0].finish_reason.name)

        return (f"Google returned an empty response after all retries.\n"
                f"Final stop reason given: {finish_reason_str}"), api_params_for_dumping

    async def _generate_local_response(self, config: dict, context: Dict[str, Any]) -> Tuple[str, Dict]:
        url = 'http://localhost:5001/api/v1/generate'
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context['history']])
        full_prompt = f"{history_str}\nuser: {context['current_message']['text']}\nassistant:"
        payload = {"max_length": config["max_output_tokens"], "temperature": config.get("temperature", 0.7),
                   "top_p": config.get("top_p", 0.9), "top_k": config.get("top_k", 40),
                   "memory": context["persona_prompt"], "prompt": full_prompt}
        payload = {k: v for k, v in payload.items() if v is not None}

        async def do_request():
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data['results'][0]['text'].strip(), payload

        try:
            return await do_request()
        except aiohttp.ClientError as e:
            logger.warning(f"Local model connection error: {e}. Retrying once...")
            time.sleep(1)
            try:
                return await do_request()
            except Exception as retry_e:
                logger.error(f"Local model failed on retry: {retry_e}", exc_info=True)
                raise LLMCommunicationError("Could not connect to local model after retry.") from retry_e


def _process_grounding_metadata(base_text_from_response: str, metadata, logger):
    """Processes grounding metadata to insert citations and list sources."""
    if not (metadata and metadata.grounding_chunks and metadata.grounding_supports):
        return base_text_from_response, "", ""

    insertion_points_map = {}
    current_search_cursor = 0

    processed_sources = {}
    source_id_counter = 1
    chunk_idx_to_source_id_map = {}

    for chunk_index, chunk in enumerate(metadata.grounding_chunks):
        if chunk.web and chunk.web.uri:
            uri = chunk.web.uri
            title = chunk.web.title or uri
            if uri not in processed_sources:
                processed_sources[uri] = {'id': source_id_counter, 'title': title, 'url': uri}
                source_id_counter += 1
            chunk_idx_to_source_id_map[chunk_index] = processed_sources[uri]['id']

    segments_to_cite = []
    for support in metadata.grounding_supports:
        s_text = support.segment.text
        s_start_hint = support.segment.start_index if support.segment.start_index is not None else 0

        supporting_source_ids = set()
        if support.grounding_chunk_indices:
            for c_idx in support.grounding_chunk_indices:
                if c_idx in chunk_idx_to_source_id_map:
                    supporting_source_ids.add(chunk_idx_to_source_id_map[c_idx])

        if s_text and supporting_source_ids:
            segments_to_cite.append({
                "start_hint": s_start_hint,
                "text": s_text,
                "citations": sorted(list(supporting_source_ids))
            })

    segments_to_cite.sort(key=lambda s: s["start_hint"])

    for segment_data in segments_to_cite:
        segment_text = segment_data["text"]
        citation_ids_for_segment = segment_data["citations"]

        found_index = base_text_from_response.find(segment_text, current_search_cursor)
        if found_index == -1:
            alt_found_index = base_text_from_response.find(segment_text, 0)
            if alt_found_index != -1:
                found_index = alt_found_index
            else:
                continue

        insertion_location = found_index + len(segment_text)

        if citation_ids_for_segment:
            if insertion_location not in insertion_points_map:
                insertion_points_map[insertion_location] = set()
            insertion_points_map[insertion_location].update(citation_ids_for_segment)

        current_search_cursor = max(current_search_cursor, insertion_location)

    text_with_citations = base_text_from_response
    if insertion_points_map:
        modified_text_parts = []
        last_slice_end = 0
        for loc in sorted(insertion_points_map.keys()):
            modified_text_parts.append(base_text_from_response[last_slice_end:loc])

            hyperlinked_citation_parts = []
            sorted_citation_ids_at_loc = sorted(list(insertion_points_map[loc]))

            for src_id in sorted_citation_ids_at_loc:
                source_info = next((s_info for s_info in processed_sources.values() if s_info['id'] == src_id),
                                   None)
                if source_info:
                    hyperlinked_citation_parts.append(f"[{src_id}](<{source_info['url']}>)")

            citation_str_to_insert = ""
            if hyperlinked_citation_parts:
                citation_str_to_insert = f" [{', '.join(hyperlinked_citation_parts)}]"

            modified_text_parts.append(citation_str_to_insert)
            last_slice_end = loc
        modified_text_parts.append(base_text_from_response[last_slice_end:])
        text_with_citations = "".join(modified_text_parts)

    citations_text_list_str = ""
    if processed_sources:
        citations_text_list_str = "\n\nSources:\n"
        ordered_sources_for_display = sorted(processed_sources.values(), key=lambda s: s['id'])
        for src_info in ordered_sources_for_display:
            citations_text_list_str += f"{src_info['id']}. {src_info['title']}\n"

    search_query_text_str = ""
    if metadata.web_search_queries:
        search_query_text_str = f"\n\nSearch Query: {', '.join(metadata.web_search_queries)}"

    return text_with_citations, search_query_text_str, citations_text_list_str
