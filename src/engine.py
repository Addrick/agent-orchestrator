import json
import logging
import os
from typing import Dict, Any, Optional, Tuple, List

from dotenv import load_dotenv

# --- Provider-specific imports ---
import aiohttp
import anthropic
from openai import AsyncOpenAI
from google import genai
from google.genai.types import GenerateContentConfig, Tool, GoogleSearch

logger = logging.getLogger(__name__)


class TextEngine:
    """A centralized engine for handling requests to various LLM APIs."""

    def __init__(self) -> None:
        # --- Lazy-loaded clients ---
        self.openai_client: Optional[AsyncOpenAI] = None
        self.anthropic_client: Optional[anthropic.Anthropic] = None

        # --- Google Client (matching original implementation) ---
        self.google_client: Optional[genai.client.AsyncClient] = None
        self.google_search_tool: Optional[Tool] = None
        self.google_tool_config: Optional[Dict[str, Any]] = None
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
        self.google_tool_config = {"function_calling_config": {"mode": "AUTO"}}
        logger.info("Google AI Studio client initialized.")

    async def generate_response(self, persona_config: dict, context_object: Dict[str, Any]) -> Tuple[str, Optional[Dict]]:
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
        except Exception as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            return f"Error communicating with OpenAI: {e}", api_params

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
        except Exception as e:
            logger.error(f"Anthropic API error: {e}", exc_info=True)
            return f"Error communicating with Anthropic: {e}", api_params

    async def _generate_google_response(self, config: dict, context: Dict[str, Any]) -> Tuple[str, Dict]:
        """Builds and sends a request to the Google API using the original client structure."""
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

        # This dictionary contains the actual Tool OBJECTS and is used for the API call
        content_config_for_api = {
            'tools': [self.google_search_tool],
            'tool_config': self.google_tool_config,
            'response_modalities': ["TEXT"],
            'safety_settings': self.google_safety_settings
        }
        if isinstance(config.get("max_output_tokens"), int):
            if config.get("max_output_tokens") >= 100:
                content_config_for_api['max_output_tokens'] = config.get("max_output_tokens")
        if isinstance(config.get("temperature"), (int, float)):
            content_config_for_api['temperature'] = config.get("temperature")
        if isinstance(config.get("top_p"), (int, float)):
            content_config_for_api['top_p'] = config.get("top_p")
        if isinstance(config.get("top_k"), (int, float)):
            content_config_for_api['top_k'] = config.get("top_k")

        config_for_dumping = json.loads(json.dumps(content_config_for_api, default=str))

        # The final payload for dumping is now guaranteed to be serializable
        api_params_for_dumping = {
            'model': config["model_name"],
            'contents': request_content,
            'config': config_for_dumping
        }

        try:
            # The API call uses the original dictionary with the real objects
            response_obj = await self.google_client.models.generate_content(
                model=config["model_name"],
                contents=request_content,
                config=GenerateContentConfig(**content_config_for_api)
            )

            # Check for immediate blocking due to prompt or safety settings
            if response_obj.prompt_feedback and response_obj.prompt_feedback.block_reason:
                block_reason = response_obj.prompt_feedback.block_reason.name
                logger.error(f"Google AI Studio request blocked. Reason: {block_reason}")
                if response_obj.prompt_feedback.safety_ratings:
                    logger.error(f"Prompt Safety Ratings: {response_obj.prompt_feedback.safety_ratings}")
                return f"```Response blocked by Google due to {block_reason}. Try again, mix up your request a bit if it persists.```"

            base_text_from_response = ""
            candidate = None
            if response_obj.candidates:  # Check if candidates list is not empty
                candidate = response_obj.candidates[0]
                # Extract text from parts (more robust than response_obj.text)
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            base_text_from_response += part.text

            final_text_content = base_text_from_response
            search_query_display = ""
            citations_display = ""

            if candidate and candidate.grounding_metadata:
                metadata = candidate.grounding_metadata
                # Check if the necessary attributes for your processing exist
                if hasattr(metadata, 'grounding_chunks') and hasattr(metadata, 'grounding_supports'):
                    logger.debug("Processing grounding metadata for citations...")
                    final_text_content, search_query_display, citations_display = \
                        _process_grounding_metadata(base_text_from_response, metadata, logger)
                else:
                    logger.info(
                        "Grounding metadata present, but 'grounding_chunks' or 'grounding_supports' missing. Skipping citation processing.")

            # Handle cases where the response might be empty or only whitespace AFTER processing
            if not final_text_content.strip() and not search_query_display and not citations_display:
                logger.warning(
                    "Google AI Studio returned an effectively empty response (no text, no citations, no search query).")
                finish_reason_str = "Unknown"
                if candidate and candidate.finish_reason:
                    finish_reason_str = str(candidate.finish_reason.name)
                return (f"Google returned an empty response. Try request again.\n"
                        f"Stop reason given (if any): {finish_reason_str}")

            # Append search query and citations if they were generated
            if search_query_display:
                final_text_content += search_query_display
            if citations_display:
                final_text_content += citations_display

            return final_text_content.strip(), api_params_for_dumping
        except Exception as e:
            logger.error(f"Error during Google generation: {e}", exc_info=True)
            return f"An error occurred with Google API: {e}", api_params_for_dumping

    async def _generate_local_response(self, config: dict, context: Dict[str, Any]) -> Tuple[str, Dict]:
        url = 'http://localhost:5001/api/v1/generate'
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context['history']])
        full_prompt = f"{history_str}\nuser: {context['current_message']['text']}\nassistant:"
        payload = {"max_length": config["max_output_tokens"], "temperature": config.get("temperature", 0.7),
                   "top_p": config.get("top_p", 0.9), "top_k": config.get("top_k", 40),
                   "memory": context["persona_prompt"], "prompt": full_prompt}
        payload = {k: v for k, v in payload.items() if v is not None}
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['results'][0]['text'].strip(), payload
                    else:
                        error_text = await response.text()
                        logger.error(f"Local model error ({response.status}): {error_text}")
                        return f"Error with local model: Status {response.status}", payload
        except aiohttp.ClientError as e:
            logger.error(f"Local model connection error: {e}", exc_info=True)
            return "Error: Could not connect to local model at http://localhost:5001.", payload


def _process_grounding_metadata(base_text_from_response: str, metadata, logger):
    """Processes grounding metadata to insert citations and list sources."""
    if not (metadata and metadata.grounding_chunks and metadata.grounding_supports):
        return base_text_from_response, "", ""

    insertion_points_map = {}  # Maps insertion_location -> set of citation_ids
    current_search_cursor = 0

    # Store unique sources: uri -> {id: int (1-based), title: str, url: str}
    processed_sources = {}
    source_id_counter = 1
    # Maps API's internal grounding_chunk_index to our processed_sources ID
    chunk_idx_to_source_id_map = {}

    for chunk_index, chunk in enumerate(metadata.grounding_chunks):
        if chunk.web and chunk.web.uri:
            uri = chunk.web.uri
            title = chunk.web.title or uri  # Use URI if title is missing
            if uri not in processed_sources:
                processed_sources[uri] = {'id': source_id_counter, 'title': title, 'url': uri}
                source_id_counter += 1
            chunk_idx_to_source_id_map[chunk_index] = processed_sources[uri]['id']

    segments_to_cite = []
    for support in metadata.grounding_supports:
        s_text = support.segment.text
        # API provides start_index, but it might not always be accurate.
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
                "citations": sorted(list(supporting_source_ids))  # Store sorted citation IDs
            })

    # Sort segments by their start_hint to process them in order of appearance (ideally)
    segments_to_cite.sort(key=lambda s: s["start_hint"])

    for segment_data in segments_to_cite:
        segment_text = segment_data["text"]
        citation_ids_for_segment = segment_data["citations"]

        found_index = base_text_from_response.find(segment_text, current_search_cursor)
        if found_index == -1:  # Segment not found starting from current_search_cursor
            # Fallback: search from the beginning of the response
            alt_found_index = base_text_from_response.find(segment_text, 0)
            if alt_found_index != -1:
                logger.warning(
                    f"Segment (starts with: '{segment_text[:30].replace(chr(10), ' ')}...') "
                    f"not found from cursor {current_search_cursor}. Found at {alt_found_index} via global search."
                )
                found_index = alt_found_index
            else:
                logger.error(
                    f"Segment (starts with: '{segment_text[:30].replace(chr(10), ' ')}...') "
                    f"could not be found anywhere in the response. Skipping citation for this segment."
                )
                continue  # Skip this segment's citations

        # Position to insert the citation mark (after the segment)
        insertion_location = found_index + len(segment_text)

        if citation_ids_for_segment:  # Only proceed if there are citations for this segment
            if insertion_location not in insertion_points_map:
                insertion_points_map[insertion_location] = set()
            insertion_points_map[insertion_location].update(citation_ids_for_segment)

        # Update cursor to search for the *next* segment after the current one's *found_index*.
        # This helps avoid re-matching parts of already processed segments if hints are off.
        current_search_cursor = max(current_search_cursor, insertion_location)

    # Build the new text_content with citations inserted
    text_with_citations = base_text_from_response
    if insertion_points_map:
        modified_text_parts = []
        last_slice_end = 0
        for loc in sorted(insertion_points_map.keys()):  # Process in order of appearance
            modified_text_parts.append(base_text_from_response[last_slice_end:loc])

            hyperlinked_citation_parts = []
            # Get the unique, sorted citation IDs for this specific insertion location
            sorted_citation_ids_at_loc = sorted(list(insertion_points_map[loc]))

            for src_id in sorted_citation_ids_at_loc:
                # Find the source URL using our internal ID from processed_sources
                source_info = next((s_info for s_info in processed_sources.values() if s_info['id'] == src_id),
                                   None)
                if source_info:  # Should always find it if logic is correct
                    hyperlinked_citation_parts.append(f"[{src_id}](<{source_info['url']}>)")

            citation_str_to_insert = ""
            if hyperlinked_citation_parts:  # Only add if there are actual links
                # Example: " [\[1](<url1>), \[2](<url2>)]" - note the space for separation
                citation_str_to_insert = f" [{', '.join(hyperlinked_citation_parts)}]"

            modified_text_parts.append(citation_str_to_insert)
            last_slice_end = loc
        modified_text_parts.append(base_text_from_response[last_slice_end:])  # Append remaining text
        text_with_citations = "".join(modified_text_parts)

    # Build the list of sources for display
    citations_text_list_str = ""
    if processed_sources:
        citations_text_list_str = "\n\nSources:\n"
        # Sort sources by their assigned ID for consistent display order
        ordered_sources_for_display = sorted(processed_sources.values(), key=lambda s: s['id'])
        for src_info in ordered_sources_for_display:
            citations_text_list_str += f"{src_info['id']}. {src_info['title']}\n"

    # Build search query text
    search_query_text_str = ""
    if metadata.web_search_queries:  # Check if attribute exists and is not empty
        search_query_text_str = f"\n\nSearch Query: {', '.join(metadata.web_search_queries)}"

    return text_with_citations, search_query_text_str, citations_text_list_str

