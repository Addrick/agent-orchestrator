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
                 token_limit=DEFAULT_TOKEN_LIMIT,
                 temperature=DEFAULT_TEMPERATURE,
                 top_p=DEFAULT_TOP_P,
                 top_k=DEFAULT_TOP_K):
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
            completion = await self.openai_client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                # temperature=self.temperature,
                max_tokens=self.max_tokens,
                # top_p=self.top_p,
                # frequency_penalty=self.frequency_penalty,
                # presence_penalty=self.presence_penalty,
            )
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
            completion = await self.openai_client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
            )
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
            completion = await self.openai_client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                # temperature=self.temperature,
                max_tokens=self.max_tokens,
                # top_p=self.top_p,
                # frequency_penalty=self.frequency_penalty,
                # presence_penalty=self.presence_penalty,
            )
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
            # Use the asynchronous generation method
            response = await model.generate_content_async(
                request_content,
                safety_settings=self.unsafe_settings_google_generativeai,
                tools=GoogleSearch()
            )

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

        client = genai.Client(api_key=google_api_key)
        model_id = self.model_name

        google_search_tool = Tool(
            google_search=GoogleSearch()
        )

        request_content = '### Instructions: ###' + prompt + '\n### Recent chat room history: ### \n' + str(
            context) + '\n### Now respond to the most recent message: ###\n' + message

        # Log the structured request
        self.json_request = self.parse_request_json(request_content)

        try:
            # Use the asynchronous generation method
            response = client.models.generate_content(
                model=model_id,
                contents=request_content,
                # safety_settings=self.unsafe_settings_google_generativeai,
                config=GenerateContentConfig(
                    tools=[google_search_tool],
                    response_modalities=["TEXT"],
                )
            )
            # print(response.candidates[0].grounding_metadata.grounding_chunks)
            # print(response.candidates[0].grounding_metadata.grounding_supports)
            # Check if the response was blocked due to safety settings
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason.name
                text_content = f"```Response blocked by Google due to {block_reason}. Try again, mix up your request a bit if it persists.```"
                self.logger.error(f"Google AI Studio response blocked. Reason: {block_reason}")
                if response.prompt_feedback.safety_ratings:
                    self.logger.error(f"Safety Ratings: {response.prompt_feedback.safety_ratings}")
                # TODO: Implement retry logic here instead of just reporting failure
            else:
                try:
                    text_content = response.text
                    try:
                        # print the search details
                        text_content += f"\nSearch Query: {response.candidates[0].grounding_metadata.web_search_queries}"
                        # urls used for grounding
                        # text_content += f"\nSearch Pages: {', '.join([site.web.title for site in response.candidates[0].grounding_metadata.grounding_chunks])}"
                        text_content += f"\nSearch URLs: <{'>\n <'.join([resolve_redirect_url(site.web.uri) for site in response.candidates[0].grounding_metadata.grounding_chunks])}>"
                    except Exception as e:
                        self.logger.error(e)
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
            "options": {
                "temperature": self.temperature,
                "max_completion_tokens": self.max_tokens,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty
            },
            "id": self.model_name,
        }
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
