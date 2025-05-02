import json
import logging
import re
from dotenv import load_dotenv
from config import api_keys
from config.global_config import *
from src.utils.model_utils import get_model_list
from src.utils.message_utils import resolve_redirect_url

import aiohttp
import anthropic

from vertexai.generative_models import HarmCategory, HarmBlockThreshold


# Summary:
# This code defines a TextEngine class that handles text generation using various AI models.
# It supports OpenAI, Anthropic, Google, and local models. The class provides methods to
# set parameters, generate responses, and handle different API calls. It also includes
# a function to launch a local KoboldCPP instance.
# TODO: Implement a method to validate and sanitize input parameters
# TODO: Implement proper error handling and logging for all API calls


class TextEngine:
    """Initialize the TextEngine with model settings and API clients."""

    def __init__(self, model_name='none',
                 token_limit=None,
                 temperature=None,
                 top_p=None,
                 top_k=None):
        self.logger = logging.getLogger()

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = token_limit
        self.top_p = top_p
        self.json_request = None
        self.json_response = None
        self.all_available_models = get_model_list()

        # OpenAI models
        self.openai_models_available = self.all_available_models['From OpenAI']
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.openai_client = None

        # Google models
        self.google_models_available = self.all_available_models['From Google']
        self.top_k = top_k
        self.unsafe_settings_vertexai = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        self.unsafe_settings_google_generativeai = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]

        # Anthropic models
        self.anthropic_models_available = self.all_available_models['From Anthropic']

        # Local models
        # TODO: add me after finishing. Kobold will be launched with a particular model and takes ages to start up; model selection must be done on startup/shutdown

    async def initialize_openai_client(self):

        from openai import OpenAI, AsyncOpenAI, APIError

        self.openai_client = AsyncOpenAI(api_key=api_keys.openai)

    def get_raw_json_request(self):
        return self.json_request

    def get_max_tokens(self):
        return self.max_tokens

    def set_response_token_limit(self, new_response_token_limit):
        if isinstance(new_response_token_limit, int):
            self.max_tokens = new_response_token_limit
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
        self.max_tokens = token_limit
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
            # Image generation models
            # elif "dall-e" in self.model_name.lower():
            #     # return "image"
            #     # response = await self._generate_openai_image_response(prompt, message, context, image_url)
            # # Audio transcription/processing (e.g., Whisper)
            # elif "whisper" in self.model_name.lower():
            #     # return "audio"
            #     # response = await self._generate_openai_audio_response(prompt, message, context, image_url)
            # # Embedding models
            # elif "embedding" in self.model_name.lower():
            #     # return "embedding"
            #     # response = await self._generate_openai_embedding_response(prompt, message, context, image_url)
            # # Standard completions (older models)
            # elif any(token in self.model_name for token in ["davinci", "babbage", "curie", "ada"]):
            # return "completion"
            # response = await self._generate_openai_completion_response(prompt, message, context, image_url)


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
            response = "Error: persona's model name not found."

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
            await self.initialize_openai_client()

        self.json_request = self.parse_request_json(messages)

        try:
            # Build API parameters with only non-None values
            api_params = {
                'messages': messages,
                'model': self.model_name,
            }

            if isinstance(self.max_tokens, int):
                api_params['max_tokens'] = self.max_tokens

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
            await self.initialize_openai_client()

        try:
            # Build API parameters with only non-None values
            api_params = {
                'messages': messages,
                'model': self.model_name,
            }

            if isinstance(self.max_tokens, int):
                api_params['max_tokens'] = self.max_tokens

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
            await self.initialize_openai_client()

        try:
            # Build API parameters with only non-None values
            api_params = {
                'messages': messages,
                'model': self.model_name,
            }

            if isinstance(self.max_tokens, int):
                api_params['max_tokens'] = self.max_tokens

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

    # Google # TODO: async?
    def _generate_google_response_vertex(self, prompt, message, context):
        """Generate a response using Google's Vertex AI."""
        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=api_keys.google_project_id, location="us-east1")

        # TODO: further test and refine context and prompt structure
        # TODO: add model parameter customization

        model = GenerativeModel(self.model_name)

        request = '### Instructions: ###' + prompt + '\n### Recent chat history: ### \n' + str(
            context) + '\n### Now respond to the most recent message: ###\n' + message
        self.json_request = self.parse_request_json(request)

        response = model.generate_content(
            request,
            safety_settings=self.unsafe_settings_vertexai
        )
        try:
            text_content = response.text
        except ValueError:
            # Handle the absence of response.text
            text_content = f"```Response too spicy for Google, blocked at server. Try again, mix up your request a bit if it persists.```"  # TODO: do a retry instead of just reporting it failed
            logging.error(response.candidates[0].safety_ratings)
        return text_content

    async def _generate_google_response_ai_studio_async_old(self, prompt, message, context):
        """Generate a response asynchronously using Google AI Studio (genai library)."""

        import google.generativeai as genai
        from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
        import os

        # load .env for api key
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(current_dir, '..')
        load_dotenv(os.path.join(root_dir, '.env'))
        google_api_key = os.environ.get("GOOGLE_GENERATIVEAI_API_KEY")

        # Configure the genai library (ideally done once at application start)
        genai.configure(api_key=google_api_key)

        # Model instantiation using the genai library
        # You might want to cache this model instance if used frequently
        model = genai.GenerativeModel(self.model_name)
        request_content = '### Instructions: ###' + prompt + '\n### Recent chat room history: ### \n' + str(
            context) + '\n### Now respond to the most recent message: ###\n' + message

        # Log the structured request
        self.json_request = self.parse_request_json(request_content)

        try:
            # Build API parameters with only valid values
            api_params = {
                'messages': request_content,
                'model': self.model_name,
                'safety_settings' : self.unsafe_settings_google_generativeai,
                'tools' : GoogleSearch()
            }

            if isinstance(self.max_tokens, int):
                api_params['max_tokens'] = self.max_tokens

            if isinstance(self.temperature, (int, float)):
                api_params['temperature'] = self.temperature

            if isinstance(self.temperature, (int, float)):
                api_params['top_p'] = self.top_p

            if isinstance(self.frequency_penalty, (int, float)):
                api_params['frequency_penalty'] = self.frequency_penalty

            if isinstance(self.presence_penalty, (int, float)):
                api_params['presence_penalty'] = self.presence_penalty

            # Use the asynchronous generation method
            response = await model.generate_content_async(**api_params)


            # Check if the response was blocked due to safety settings
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason.name
                text_content = f"```Response blocked by Google due to {block_reason}. Try again, mix up your request a bit if it persists.```"
                self.logger.error(f"Google AI Studio response blocked. Reason: {block_reason}")
                if response.prompt_feedback.safety_ratings:
                    self.logger.error(f"Safety Ratings: {response.prompt_feedback.safety_ratings}")
                # TODO: Implement retry logic here instead of just reporting failure
            else:
                # Access the text content safely
                # The .text attribute should exist if not blocked, but good practice to check
                try:
                    text_content = response.text
                except ValueError:
                    # This path might be less common with genai if block_reason is checked first,
                    # but kept for robustness similar to original code.
                    text_content = "```Error retrieving text from response, even though not explicitly blocked.```"
                    self.logger.error("ValueError accessing response.text despite no block reason.")
                    if hasattr(response, 'candidates') and response.candidates:
                        self.logger.error(
                            f"Candidate safety ratings (if available): {response.candidates[0].safety_ratings}")


        except Exception as e:
            # Catch potential API errors or other issues
            self.logger.error(f"Error during Google AI Studio generation: {e}", exc_info=True)
            text_content = f"```An error occurred during generation: {e}```"

        return text_content

    async def _generate_google_response_ai_studio_async(self, prompt, message, context):
        """Generate a response asynchronously using Google AI Studio (genai library)."""

        from google import genai
        from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
        import os

        # load .env for api key
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(current_dir, '..')
        load_dotenv(os.path.join(root_dir, '.env'))
        google_api_key = os.environ.get("GOOGLE_GENERATIVEAI_API_KEY")
        if not google_api_key:
            self.logger.error("GOOGLE_GENERATIVEAI_API_KEY not found in environment variables.")
            return "```Error: API key not configured.```"

        client = genai.client.BaseApiClient(api_key=google_api_key)
        async_client = genai.client.AsyncClient(client)

        model_id = self.model_name

        google_search_tool = Tool(
            google_search=GoogleSearch()
        )

        request_content = '### Instructions: ###' + prompt + '\n### Recent chat room history: ### \n' + str(
            context) + '\n### Now respond to the most recent message: ###\n' + message

        # Log the structured request
        self.json_request = self.parse_request_json(request_content)

        try:
            # Build API parameters with only valid values
            content_config = {
                'tools' : [google_search_tool],
                'response_modalities' : ["TEXT"],
                'safety_settings' : self.unsafe_settings_google_generativeai
            }

            if isinstance(self.max_tokens, int):
                content_config['max_output_tokens'] = self.max_tokens

            if isinstance(self.temperature, (int, float)):
                content_config['temperature'] = self.temperature

            if isinstance(self.temperature, (int, float)):
                content_config['top_p'] = self.top_p

            if isinstance(self.frequency_penalty, (int, float)):
                content_config['frequency_penalty'] = self.frequency_penalty

            if isinstance(self.presence_penalty, (int, float)):
                content_config['presence_penalty'] = self.presence_penalty
            api_params = {
                'model':model_id,
                'contents':request_content,
                'config':GenerateContentConfig(**content_config)
            }
            # Use the asynchronous generation method
            response = await async_client.models.generate_content(**api_params)
            # response = await async_client.models.generate_content(
            #     model=model_id,
            #     contents=request_content,
            #     # safety_settings=self.unsafe_settings_google_generativeai, # Uncomment if you have this
            #     config=GenerateContentConfig(
            #         tools=[google_search_tool],
            #         response_modalities=["TEXT"],
            #     )
            # )

            # Check if the response was blocked due to safety settings
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason.name
                text_content = f"```Response blocked by Google due to {block_reason}. Try again, mix up your request a bit if it persists.```"
                self.logger.error(f"Google AI Studio response blocked. Reason: {block_reason}")
                if response.prompt_feedback.safety_ratings:
                    self.logger.error(f"Safety Ratings: {response.prompt_feedback.safety_ratings}")
            else:
                try:
                    text_content = response.text
                    # --- Citation Logic ---
                    citations_text = ""
                    search_query_text = ""

                    if response.candidates and response.candidates[0].grounding_metadata:
                        metadata = response.candidates[0].grounding_metadata

                        # 1. Get Search Query (optional, keep if desired)
                        if metadata.web_search_queries:
                            search_query_text = f"\n\nSearch Query: {', '.join(metadata.web_search_queries)}"

                        if metadata.grounding_chunks and metadata.grounding_supports:
                            # Map chunks to unique sources and assign citation numbers
                            source_map = {}  # Maps chunk index to source number (1-based)
                            source_list = []  # List of unique source URLs/info in citation order

                            for chunk_index, chunk in enumerate(metadata.grounding_chunks):
                                if chunk.web and chunk.web.uri:
                                    resolved_uri = resolve_redirect_url(chunk.web.uri)
                                    # Find if this URL is already in our source list
                                    try:
                                        source_number = source_list.index(resolved_uri) + 1
                                    except ValueError:
                                        # URL not found, add it
                                        source_list.append(resolved_uri)
                                        source_number = len(source_list)
                                    source_map[chunk_index] = source_number
                                # Handle other types of chunks if necessary (e.g., document chunks)
                                # Currently only handling web chunks as per original code's focus

                            # Sort supports by end index descending to insert citations from the end first
                            # This avoids shifting indices affecting subsequent insertions
                            sorted_supports = sorted(metadata.grounding_supports, key=lambda s: s.segment.end_index,
                                                     reverse=True)

                            modified_text = text_content
                            citation_insertions = {}  # Store insertion points and citation strings

                            # Collect all citation insertions
                            for support in sorted_supports:
                                # Get the unique source numbers for the chunks supporting this segment
                                supporting_source_numbers = sorted(list(set(
                                    source_map[c_idx] for c_idx in support.grounding_chunk_indices if c_idx in source_map
                                )))

                                if supporting_source_numbers:
                                    citation_string = f"[{', '.join(map(str, supporting_source_numbers))}]"
                                    end_index = support.segment.end_index

                                    # Append citation string at the end index.
                                    # If multiple supports end at the same index, combine their citations.
                                    if end_index in citation_insertions:
                                        # Append new citations, ensuring uniqueness and sorting
                                        existing_citations = set(citation_insertions[end_index].strip('[]').split(', '))
                                        new_citations = set(map(str, supporting_source_numbers))
                                        all_citations = sorted(list(existing_citations | new_citations),
                                                               key=int)  # Combine, sort, unique
                                        citation_insertions[end_index] = f"[{', '.join(all_citations)}]"
                                    else:
                                        citation_insertions[end_index] = citation_string

                            # Apply insertions from the end of the text
                            for end_index in sorted(citation_insertions.keys(), reverse=True):
                                citation_string = citation_insertions[end_index]
                                modified_text = modified_text[:end_index] + citation_string + modified_text[end_index:]

                            text_content = modified_text

                            # Build the list of sources
                            if source_list:
                                citations_text = "\n\nSources:\n"
                                for i, source_url in enumerate(source_list):
                                    citations_text += f"{i + 1}. <{source_url}>\n"  # Using angle brackets similar to some formats

                        text_content += search_query_text  # Add search query before sources
                        text_content += citations_text  # Add the list of sources at the end

                    # --- End Citation Logic ---

                except ValueError:
                    # This might happen if response.text is not available
                    text_content = "```Error retrieving text from response.```"
                    self.logger.error("ValueError accessing response.text.")
                    if hasattr(response, 'candidates') and response.candidates:
                        self.logger.error(
                            f"Candidate safety ratings (if available): {response.candidates[0].safety_ratings}")


        except Exception as e:
            # Catch potential API errors or other issues
            self.logger.error(f"Error during Google AI Studio generation: {e}", exc_info=True)
            text_content = f"```An error occurred during generation: {e}```"

        return text_content

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

        if self.max_tokens is not None:
            api_params['max_tokens'] = self.max_tokens

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
            "max_length": self.max_tokens,
            "rep_pen": 1.1,
            # "temperature": 0.44,
            "temperature": self.temperature,
            # "top_p": 0.5,
            "top_p": self.top_p,
            # "top_k": 0,
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
        if self.max_tokens is not None:
            options["max_completion_tokens"] = self.max_tokens
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

def launch_koboldcpp():
    """WIP: Launch a KoboldCPP instance with preconfigured settings."""
    # Currently will start the process successfully but can't be properly stopped or restart after (yet)
    import traceback
    import subprocess

    try:
        # Launches koboldcpp with preconfigured settings file
        command = [KOBOLDCPP_EXE, "--config", KOBOLDCPP_CONFIG]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        while True:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                break
            if output:
                output = output.strip().decode('utf-8')
                logging.info("koboldcpp: " + output)  # Process the output as needed
                if "Please connect to custom endpoint at http://localhost:5001" in output:
                    #  report startup status to chat
                    return True

        # Get the return code of the subprocess
        return_code = process.poll()
        logging.info('Subprocess returned with code: %s', return_code)

    except Exception:
        traceback.print_exc()
