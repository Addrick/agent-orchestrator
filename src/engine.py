import json
import logging
import aiohttp
import anthropic

from openai import OpenAI, AsyncOpenAI, APIError
from vertexai.generative_models import HarmCategory, HarmBlockThreshold

from config.global_config import *
from src.utils import models
from config import api_keys


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

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = token_limit
        self.top_p = top_p
        self.json_request = None
        self.json_response = None
        self.all_available_models = models.get_model_list()

        # OpenAI models
        # self.openai_client = AsyncOpenAI(api_key=api_keys.openai) # TODO: this should be here instead of
        #  re-instantiated on every _generate_openai_response call but when moving declaration here 'await' throws an
        #  error and not using await blocks code during the request
        self.openai_models_available = self.all_available_models['From OpenAI']
        self.frequency_penalty = 0
        self.presence_penalty = 0

        # Google models
        self.google_models_available = self.all_available_models['From Google']
        self.top_k = top_k
        self.unsafe_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        # Anthropic models
        self.anthropic_models_available = self.all_available_models['From Anthropic']

        self.openai_client = OpenAI(api_key=api_keys.openai)

        # Local models
        # TODO: add me (Kobold cpp?)

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
            response = await self._generate_openai_response(prompt, message, context, image_url)

        # Anthropic request
        elif self.model_name in self.anthropic_models_available:
            if self.model_name in self.anthropic_models_available:
                response = await self._generate_anthropic_response(prompt, message, context)

        # Google request
        elif self.model_name in self.google_models_available:
            response = self._generate_google_response(prompt, message, context)

        # Local koboldcpp request
        elif self.model_name == 'local':
            response = await self._generate_local_response(prompt, message, context)

        else:
            logging.info("Error: persona's model name not found.")
            response = "Error: persona's model name not found."

        return response

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

        openai_client = AsyncOpenAI(api_key=api_keys.openai)  # TODO: put this somewhere better
        self.json_request = self.parse_request_json(messages)

        try:
            completion = await openai_client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
            )
            self.json_response = completion
            token_count_and_model = f' ({str(completion.usage.total_tokens)} tokens using {self.model_name})'
            response = completion.choices[0].message.content
            logging.debug(response + token_count_and_model)
            return response + token_count_and_model

        except APIError as e:
            return e.code + ": \n" + e.message
        except AttributeError as e:
            return str(e)

    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": "What's in this image?"},
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
    #                     }
    #                 },
    #             ],
    #         }
    #     ],
    #     max_tokens=300,
    # )
    #
    # print(response.choices[0])

    async def _generate_openai_response_stream(self, messages):
        # TODO: finish if I ever get a use case for streamed output
        # currently does not provide usage field for token usage reporting and provides no current benefit
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_keys.openai)  # TODO: put this somewhere better, may be a memory leak
        if self.model_name not in self.openai_models_available:
            logging.info("Error: model name not found in available OpenAI models.")
            return "Error: model name not found in available OpenAI models."
        else:
            completion = await client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stream=True
            )

            # Print the streamed output to the console
            for chunk in completion:
                print(chunk['choices'][0]['text'])

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
        message = client.messages.create(
            system=prompt,
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            messages=messages,
            stream=False
        )
        return message.content[0].text

    def _generate_google_response(self, prompt, message, context):
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
            safety_settings=self.unsafe_settings
        )
        return response.text

    def parse_request_json(self, messages):
        last_json = {
            "model": self.model_name,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
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
