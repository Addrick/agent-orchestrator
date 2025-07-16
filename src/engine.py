import json
import logging
import re
from dotenv import load_dotenv
from google.genai.types import GenerateContentConfig

from config import api_keys
from config.global_config import *

import aiohttp
import anthropic

import os

from vertexai.generative_models import HarmCategory, HarmBlockThreshold

from src.utils.model_utils import get_model_list


# Summary:
# This code defines a TextEngine class that handles text generation using various AI models.
# It supports OpenAI, Anthropic, Google, and local models. The class provides methods to
# set parameters, generate responses, and handle different API calls. It also includes
# a function to launch a local KoboldCPP instance.


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


class TextEngine:
    """Initialize the TextEngine with model settings."""

    def __init__(self, model_name='none',
                 token_limit=None,
                 temperature=None,
                 top_p=None,
                 top_k=None):
        self.logger = logging.getLogger(__name__)

        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = token_limit
        self.top_p = top_p
        self.top_k = top_k

        self.json_request = None
        self.json_response = None

        self.all_available_models = get_model_list()

        # OpenAI
        self.openai_models_available = self.all_available_models['From OpenAI']
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.openai_client = None

        # Google
        self.google_models_available = self.all_available_models['From Google']
        self.unsafe_settings_google_generativeai = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        self.google_aistudio_client = None
        self.google_aistudio_search_tool = None

        # Anthropic
        self.anthropic_models_available = self.all_available_models['From Anthropic']

    async def _initialize_openai(self):
        """Initializes OpenAI client if not already."""
        if self.openai_client is None:
            from openai import AsyncOpenAI  # Import here for lazy loading

            self.logger.info("Initializing OpenAI client...")
            # Load .env for API key - adjust path if .env is elsewhere
            current_dir = os.path.dirname(os.path.abspath(__file__))
            env_path = os.path.join(current_dir, '..', '.env')  # Assumes .env is in parent of current file's dir
            if os.path.exists(env_path):
                load_dotenv(env_path)
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                self.logger.error("OPENAI_API_KEY not found in environment variables.")
                raise ValueError("OPENAI_API_KEY not configured.")

            self.openai_client = AsyncOpenAI(api_key=openai_api_key)
            self.logger.info("OpenAI client initialized.")

    def _initialize_google_aistudio(self):
        """Initializes Google AI Studio model and search tool if not already."""
        if self.google_aistudio_client is not None:
            return  # Already initialized

        # Imports specific to Google AI Studio initialization
        from google import genai
        from google.genai.types import Tool, GoogleSearch
        self.logger.info("Initializing Google AI Studio client...")

        # Load .env for API key - adjust path if .env is elsewhere
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(current_dir, '..', '.env')  # Assumes .env is in parent of current file's dir
        if os.path.exists(env_path):
            load_dotenv(env_path)

        google_api_key = os.environ.get("GOOGLE_GENERATIVEAI_API_KEY")
        if not google_api_key:
            self.logger.error("GOOGLE_GENERATIVEAI_API_KEY not found in environment variables.")
            raise ValueError("GOOGLE_GENERATIVEAI_API_KEY not configured.")

        client = genai.client.BaseApiClient(api_key=google_api_key)
        self.google_aistudio_client = genai.client.AsyncClient(client)

        self.google_aistudio_search_tool = Tool(google_search=GoogleSearch())

    def _build_google_aistudio_gen_config(self):
        """Builds the GenerationConfig for Google AI Studio API call."""
        from google.genai.types import GenerateContentConfig  # Import here

        config_params = {}
        if isinstance(self.max_output_tokens, int) and self.max_output_tokens > 0:
            config_params['max_output_tokens'] = self.max_output_tokens
        if isinstance(self.temperature, (float, int)):  # Temperature can be 0
            config_params['temperature'] = float(self.temperature)
        if isinstance(self.top_p, (float, int)):
            config_params['top_p'] = float(self.top_p)
        if isinstance(self.top_k, int) and self.top_k > 0:
            config_params['top_k'] = self.top_k

        return GenerateContentConfig(**config_params) if config_params else None

    def get_raw_json_request(self):
        return self.json_request

    def get_max_tokens(self):
        return self.max_output_tokens

    def set_response_token_limit(self, new_response_token_limit):
        if isinstance(new_response_token_limit, int):
            self.max_output_tokens = new_response_token_limit
            return True
        else:
            logging.info("Error: Input is not an integer.")
            return False

    def set_temperature(self, new_temp):
        self.temperature = new_temp

    def set_top_p(self, top_p):
        self.top_p = top_p

    def set_top_k(self, top_k):
        self.top_k = top_k

    # Generates response based on model_name
    async def generate_response(self, prompt, message, context, image_url, token_limit):
        """Generate a response based on the selected model."""
        # route specific API and model to use based on model_name
        # if model_name matches models found in various APIs
        response = ''
        self.max_output_tokens = token_limit
        # OpenAI request
        if self.model_name in self.openai_models_available:
            # Search models
            if any(token in self.model_name for token in ["search"]):
                response = await self._generate_openai_search_response(prompt, message, context, image_url)
                return response
            # Chat models
            if any(token in self.model_name for token in ["gpt-4", "gpt-3.5-turbo", "chatgpt"]):
                response = await self._generate_openai_response(prompt, message, context, image_url)
                return response
            # Reasoning models
            if re.match(r'^o\d+(?:-|$)', self.model_name):
                response = await self._generate_openai_reasoning_response(prompt, message, context, image_url)
                return response

        # Anthropic request
        elif self.model_name in self.anthropic_models_available:
            if self.model_name in self.anthropic_models_available:
                response = await self._generate_anthropic_response(prompt, message, context)

        # Google request
        elif self.model_name in self.google_models_available:
            response = await self._generate_google_response_ai_studio_async(prompt, message, context)

        # Local koboldcpp request
        elif self.model_name == 'local':
            response = await self._generate_local_response(prompt, message, context)

        else:
            logging.info("Error: persona's model name not found.")
            response = "Error: persona's model name not found. Try: ```set model default``` or find a new model with: ```what models```"

        return response

    # OpenAI
    async def _generate_openai_response(self, prompt, message, context, image_url=None):
        """Prepare messages for async OpenAI API call."""
        if context is not None:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": context},
                # TODO: try iterating this for 1 msg/context block for better model processing?
                {"role": "user", "content": message}]
        else:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": message}]
        if image_url is not None:
            messages.append(
                {"role": "user", "content": [{
                    "type": "image_url",
                    "image_url":
                        {"url": image_url}}]})

        if self.openai_client is None:
            await self._initialize_openai()

        self.json_request = self.parse_request_json(messages)

        try:
            # Build API parameters with only non-None values
            api_params = {
                'messages': messages,
                'model': self.model_name,
            }

            if isinstance(self.max_output_tokens, int):
                api_params['max_tokens'] = self.max_output_tokens

            if isinstance(self.temperature, (int, float)):
                api_params['temperature'] = self.temperature

            if isinstance(self.temperature, (int, float)):
                api_params['top_p'] = self.top_p

            if isinstance(self.frequency_penalty, (int, float)):
                api_params['frequency_penalty'] = self.frequency_penalty

            if isinstance(self.presence_penalty, (int, float)):
                api_params['presence_penalty'] = self.presence_penalty

            completion = await self.openai_client.chat.completions.create(**api_params)
            self.json_response = completion
            token_count_and_model = f' ({str(completion.usage.total_tokens)} tokens using {self.model_name})'
            response = completion.choices[0].message.content
            logging.debug(response + token_count_and_model)
            return response + token_count_and_model

        except Exception as e:
            return str(e)

    async def _generate_openai_reasoning_response(self, prompt, message, context, image_url=None):

        """Prepare messages for async OpenAI API call."""
        if context is not None:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": context},
                # TODO: try iterating this for 1 msg/context block for better model processing?
                {"role": "user", "content": message}]
        else:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": message}]
        if image_url is not None:
            messages.append(
                {"role": "user", "content": [{
                    "type": "image_url",
                    "image_url":
                        {"url": image_url}}]})

        self.json_request = self.parse_request_json(messages)

        if self.openai_client is None:
            await self._initialize_openai()

        try:
            # Build API parameters with only non-None values
            api_params = {
                'messages': messages,
                'model': self.model_name,
            }

            if isinstance(self.max_output_tokens, int):
                api_params['max_tokens'] = self.max_output_tokens

            if isinstance(self.temperature, (int, float)):
                api_params['temperature'] = self.temperature

            if isinstance(self.temperature, (int, float)):
                api_params['top_p'] = self.top_p

            if isinstance(self.frequency_penalty, (int, float)):
                api_params['frequency_penalty'] = self.frequency_penalty

            if isinstance(self.presence_penalty, (int, float)):
                api_params['presence_penalty'] = self.presence_penalty

            completion = await self.openai_client.chat.completions.create(**api_params)
            self.json_response = completion
            token_count_and_model = f' ({str(completion.usage.total_tokens)} tokens using {self.model_name})'
            response = completion.choices[0].message.content
            logging.debug(response + token_count_and_model)
            return response + token_count_and_model

        except Exception as e:
            return str(e)

    async def _generate_openai_search_response(self, prompt, message, context, image_url=None):
        """Prepare messages for async OpenAI API call."""
        if context is not None:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": context},
                # TODO: try iterating this for 1 msg/context block for better model processing?
                {"role": "user", "content": message}]
        else:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": message}]
        if image_url is not None:
            messages.append(
                {"role": "user", "content": [{
                    "type": "image_url",
                    "image_url":
                        {"url": image_url}}]})

        self.json_request = self.parse_request_json(messages)

        if self.openai_client is None:
            await self._initialize_openai()

        try:
            # Build API parameters with only non-None values
            api_params = {
                'messages': messages,
                'model': self.model_name,
            }

            if isinstance(self.max_output_tokens, int):
                api_params['max_tokens'] = self.max_output_tokens

            if isinstance(self.temperature, (int, float)):
                api_params['temperature'] = self.temperature

            if isinstance(self.temperature, (int, float)):
                api_params['top_p'] = self.top_p

            if isinstance(self.frequency_penalty, (int, float)):
                api_params['frequency_penalty'] = self.frequency_penalty

            if isinstance(self.presence_penalty, (int, float)):
                api_params['presence_penalty'] = self.presence_penalty

            completion = await self.openai_client.chat.completions.create(**api_params)
            self.json_response = completion
            token_count_and_model = f' ({str(completion.usage.total_tokens)} tokens using {self.model_name})'
            response = completion.choices[0].message.content
            logging.debug(response + token_count_and_model)
            return response + token_count_and_model

        except Exception as e:
            return str(e)
        # except AttributeError as e:
        #     return str(e)

    # Google
    async def _generate_google_response_ai_studio_async(self, prompt: str, message: str, context: str):
        """Generate a response asynchronously using Google AI Studio (genai library)."""
        try:
            self._initialize_google_aistudio()  # Ensures model and tool are ready
        except ValueError as e:  # Catch API key or config errors from initializer
            self.logger.error(f"Initialization failed for Google AI Studio: {e}")
            return f"```Error: Google AI Studio not configured properly: {e}```"
        except Exception as e:  # Catch other init errors
            self.logger.error(f"Unexpected error during Google AI Studio initialization: {e}", exc_info=True)
            return f"```Error: Failed to initialize Google AI Studio: {e}```"

        request_content = f"### Instructions: ###\n{prompt}\n\n### Recent chat room history: ###\n{str(context)}\n\n### Now respond to the most recent message: ###\n{message}"
        self.json_request = self.parse_request_json(request_content)

        try:
            # Build API parameters with only valid values
            content_config = {
                'tools': [self.google_aistudio_search_tool],
                'response_modalities': ["TEXT"],
                'safety_settings': self.unsafe_settings_google_generativeai
            }
            if isinstance(self.max_output_tokens, int):
                content_config['max_output_tokens'] = self.max_output_tokens
            if isinstance(self.temperature, (int, float)):
                content_config['temperature'] = self.temperature
            if isinstance(self.temperature, (int, float)):
                content_config['top_p'] = self.top_p
            if isinstance(self.frequency_penalty, (int, float)):
                content_config['frequency_penalty'] = self.frequency_penalty
            if isinstance(self.presence_penalty, (int, float)):
                content_config['presence_penalty'] = self.presence_penalty
            api_params = {
                'model': self.model_name,
                'contents': request_content,
                'config': GenerateContentConfig(**content_config)
            }
            response_obj = await self.google_aistudio_client.models.generate_content(**api_params)

            # Check for immediate blocking due to prompt or safety settings
            if response_obj.prompt_feedback and response_obj.prompt_feedback.block_reason:
                block_reason = response_obj.prompt_feedback.block_reason.name
                self.logger.error(f"Google AI Studio request blocked. Reason: {block_reason}")
                if response_obj.prompt_feedback.safety_ratings:
                    self.logger.error(f"Prompt Safety Ratings: {response_obj.prompt_feedback.safety_ratings}")
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
                    self.logger.info("Processing grounding metadata for citations...")
                    final_text_content, search_query_display, citations_display = \
                        _process_grounding_metadata(base_text_from_response, metadata, self.logger)
                else:
                    self.logger.info(
                        "Grounding metadata present, but 'grounding_chunks' or 'grounding_supports' missing. Skipping citation processing.")

            # Handle cases where the response might be empty or only whitespace AFTER processing
            if not final_text_content.strip() and not search_query_display and not citations_display:
                self.logger.warning(
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

            return final_text_content.strip()

        except Exception as e:
            # Consider more specific error handling for google.api_core.exceptions if needed
            self.logger.error(f"Error during Google AI Studio generation: {e}", exc_info=True)
            return f"```An error occurred during Google AI Studio generation: {e}```"

    # Anthropic
    async def _generate_anthropic_response(self, prompt, message, context):
        """Generate a response using Anthropic API."""
        if context is not None:
            messages = [
                {"role": "user", "content": f'{context} \n {message}'}]
            # TODO: this needs distinct flagging of assistant message, so logic based on username
        else:
            messages = [
                {"role": "user", "content": message}]
        client = anthropic.Anthropic(api_key=api_keys.anthropic)
        self.json_request = self.parse_request_json(messages)

        # Build API parameters with only non-None values
        api_params = {
            'system': prompt,
            'model': self.model_name,
            'messages': messages,
            'stream': False
        }

        if self.temperature is not None:
            api_params['temperature'] = self.temperature

        if self.top_p is not None:
            api_params['top_p'] = self.top_p

        if self.top_k is not None:
            api_params['top_k'] = self.top_k

        if self.max_output_tokens is not None:
            api_params['max_tokens'] = self.max_output_tokens

        message = client.messages.create(**api_params)
        return message.content[0].text

    # KoboldCPP
    async def _generate_local_response(self, prompt, message, context=None):
        """Generate a response using a local model (KoboldCPP)."""
        # Message formatting must match model's expected syntax for best results (TODO: find a way to automate the formatting)
        if context is None:
            context = ''
        if isinstance(context, list):
            context = ' '.join(map(str, context))
        url = 'http://localhost:5001/api/v1/generate'

        payload = {
            "n": 1,
            "max_context_length": 2048,
            "max_length": self.max_output_tokens,
            "rep_pen": 1.1,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "top_a": 0.75,
            "typical": 0.19,
            "tfs": 0.97,
            "rep_pen_range": 1024,
            "rep_pen_slope": 0.7,
            "sampler_order": [6, 0, 1, 3, 4, 2, 5],
            "memory": prompt,
            "min_p": 0,
            "presence_penalty": 0,
            "genkey": "KCPP6857",  # TODO: bug here when context (probably) is a list
            "prompt": "" + context + ",\n now you respond: \n" + message + "\n",
            "quiet": False,
            "stop_sequence": ["You:"],
            "use_default_badwordsids": False
        }
        self.json_request = payload

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
                async with session.post(url, json=payload) as response:
                    response_data = await response.text()
                    response_data = response_data.replace('{"results": [{"text": "\nresponse:',
                                                          '{"results": [{"text": "\\nresponse:')
                    json_data = json.loads(response_data)
                    response_text = json_data['results'][0]['text'].split(': ')

                    logging.info(response_text)
                    return '\n'.join(response_text)
        except aiohttp.ClientError as e:
            err_response = f"An error occurred: {e}"
            logging.info(err_response)
            return err_response

    # JSON Utility
    def parse_request_json(self, messages):
        last_json = {
            "model": self.model_name,
            "messages": messages,
            "options": {}
        }

        # Only add parameters if they're not None
        options = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            options["max_completion_tokens"] = self.max_output_tokens
        if self.top_p is not None:
            options["top_p"] = self.top_p
        if self.frequency_penalty != 0:
            options["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty != 0:
            options["presence_penalty"] = self.presence_penalty

        if options:
            last_json["options"] = options

        last_json["id"] = self.model_name
        return last_json
