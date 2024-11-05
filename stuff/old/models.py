from config import api_keys
from openai import OpenAI

import google.generativeai as palm
import inspect
import sys
from config.global_config import *


# TODO: make 2 superclasses, OpenAI and Google, oh and Local, and utilize the built-in model names from publishers
# TODO: delete this? I made engine.py I think this file is redundancy
# LanguageModel.__init__: Initializes a new LanguageModel instance with the given model_name, temperature,
# max_tokens, and top_p settings. The model_name defaults to 'basemodel', temperature to 0.8, max_tokens to
# a default limit defined in settings, and top_p to 1.0. Most of these should be overwritten by the persona config
class LanguageModel:
    def __init__(self, model_name='basemodel', temperature=0.8, max_tokens=DEFAULT_TOKEN_LIMIT, top_p=1.0):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.json_request = None
        self.json_response = None

    def get_raw_json_request(self):
        return self.json_request

    def get_max_tokens(self):
        return self.max_tokens

    def set_response_token_limit(self, new_response_token_limit):
        if isinstance(new_response_token_limit, int):
            self.max_tokens = new_response_token_limit
            return True
        else:
            print("Error: Input is not an integer.")
            return False


class Gpt3Turbo(LanguageModel):
    def __init__(self, model_name="gpt-3.5-turbo", temperature=.8, max_tokens=DEFAULT_TOKEN_LIMIT, top_p=1.0):
        super().__init__(model_name, temperature, max_tokens, top_p)
        self.api_key = api_keys.openai
        self.frequency_penalty = 0
        self.presence_penalty = 0
        self.client = OpenAI(api_key=api_keys.openai)

    def generate_response(self, prompt, message, context):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": context},
            {"role": "user", "content": message}]
        return self._create_completion(messages)

    def _create_completion(self, messages):
        completion = self.client.chat.completions.create(api_key=api_keys.openai,
                                                         messages=messages,
                                                         model=self.model_name,
                                                         temperature=self.temperature,
                                                         max_tokens=self.max_tokens,
                                                         top_p=self.top_p,
                                                         frequency_penalty=self.frequency_penalty,
                                                         presence_penalty=self.presence_penalty)
        self.json_request = {
            "model": self.model_name,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty
            },
            "object": "chat.completion",
            "id": self.model_name,
            "stream": False
        }
        self.json_response = completion
        # TODO: store/use token info: completion.usage.prompt_tokens/completion_tokens/total_tokens
        # could keep running tally of usage if I ever see shared usage
        return completion.choices[0].message.content

    def _create_completion_stream(self, messages):
        completion = self.client.chat.completions.create(api_key=api_keys.openai,
                                                         messages=messages,
                                                         model=self.model_name,
                                                         temperature=self.temperature,
                                                         max_tokens=self.max_tokens,
                                                         top_p=self.top_p,
                                                         frequency_penalty=self.frequency_penalty,
                                                         presence_penalty=self.presence_penalty,
                                                         stream=True)
        self.json_request = {
            "model": self.model_name,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty
            },
            "object": "chat.completion",
            "id": self.model_name,
            "stream": True
        }
        self.json_response = completion
        # TODO: test me (streaming text vs full completion)
        reply = ''
        for chunk in completion:
            print(chunk.choices[0].delta)
            reply.append(chunk.choices[0].delta)
        return reply


class Gpt4(Gpt3Turbo):
    def __init__(self, model_name="gpt-4", temperature=0.8, max_tokens=DEFAULT_TOKEN_LIMIT, top_p=1.0):
        super().__init__(model_name, temperature, max_tokens, top_p)


# https://developers.generativeai.google/guide/safety_setting
class PalmBison(LanguageModel):
    def __init__(self, model_name='palm-chat', temperature=0.8, max_tokens=2048,
                 top_p=1.0):  # palm chat is currently free so spam away
        super().__init__(model_name, temperature, max_tokens, top_p)
        self.api_key = api_keys.google

    def generate_response(self, prompt, message, context=[], examples=[]):
        palm.configure(api_key=self.api_key)
        context.append("NEXT REQUEST")
        # Build chat completion request for text completion model:
        persona_name = message.split()[0]  # name should be first word of latest message

        chat_request = f"you are in character as {persona_name}. {prompt} {persona_name} is chatting with others, here is the most recent conversation: \n{context}\n Now, respond to the chat request as {persona_name}: "
        completion = self._create_completion(prompt=chat_request)
        return completion

    def _create_completion(self, prompt):
        defaults = {
            'model': 'models/text-bison-001',
            'temperature': self.temperature,
            'candidate_count': 1,
            'top_k': 40,
            'top_p': 0.95,
        }
        completion = palm.generate_text(
            **defaults,
            model='models/text-bison-001',
            prompt=prompt,
            safety_settings=[
                {
                    "category": palm.safety_types.HarmCategory.HARM_CATEGORY_UNSPECIFIED,
                    "threshold": palm.safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": palm.safety_types.HarmCategory.HARM_CATEGORY_DEROGATORY,
                    "threshold": palm.safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": palm.safety_types.HarmCategory.HARM_CATEGORY_TOXICITY,
                    "threshold": palm.safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": palm.safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE,
                    "threshold": palm.safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": palm.safety_types.HarmCategory.HARM_CATEGORY_SEXUAL,
                    "threshold": palm.safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": palm.safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
                    "threshold": palm.safety_types.HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": palm.safety_types.HarmCategory.HARM_CATEGORY_DANGEROUS,
                    "threshold": palm.safety_types.HarmBlockThreshold.BLOCK_NONE,
                }])
        if completion.last is None:
            print('good job, no response')
            print('filter info:')
            print(completion.filters)
            print(completion.last)
        self.json_response = completion
        return completion.last[0]

    def get_available_chat_models(self):
        model_list = []
        module = sys.modules[__name__]
        classes = inspect.getmembers(module, inspect.isclass)
        for _, model_class in classes:
            if issubclass(model_class, LanguageModel):
                model_instance = model_class()
                if hasattr(model_instance, "model_name"):
                    if model_instance.model_name != 'basemodel':
                        model_list.append(model_instance.model_name.lower())

        # alternate method that utilizes teh model_name field and queries the API directly for all available models
        # OpenAI:
        openai_models = self.client.models.list(api_key=api_keys.openai)
        for model in openai_models['data']:
            # trim list down to just gpt models; syntax is likely poor/incompatible for completion or edits
            if 'gpt-3' in model['id'] or 'gpt-4' in model['id']:
                print(model['id'])

        return model_list


def get_model(model_name):
    module = sys.modules['models']
    classes = inspect.getmembers(module, inspect.isclass)
    for _, model_class in classes:
        if _ == model_name:
            return model_class()
    return False
