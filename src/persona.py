import logging

from src.message_handler import *
from config.global_config import *

# Summary:
# Data type to maintain discrete personas
# Accepts name, prompt and various other parameters for customizing model outputs


class Persona:
    def __init__(self, persona_name, model_name, prompt, context_limit=10, token_limit=100):
        self.persona_name = persona_name
        self.prompt = prompt
        self.context_length = int(context_limit)
        self.response_token_limit = token_limit

        self.model = None
        self.temperature = None
        self.top_p = None
        self.top_k = None

        self.last_json = 'none yet'
        self.conversation_mode = False
        # self.last_response = 'none yet'

        self.set_model(model_name)

    def get_context_length(self):
        return self.context_length

    def set_context_length(self, context_length):
        try:
            self.context_length = int(context_length)
        except Exception:
            self.context_length = context_length

    def get_response_token_limit(self):
        return self.response_token_limit

    def set_response_token_limit(self, response_token_limit):
        try:
            self.response_token_limit = int(response_token_limit)
        except Exception:
            self.response_token_limit = response_token_limit
        return True

    def set_temperature(self, new_temp):
        self.model.set_temperature(new_temp)

    def set_top_p(self, new_top_p):
        self.model.set_top_p(new_top_p)

    def set_top_k(self, new_top_k):
        self.model.set_top_k(new_top_k)

    def set_last_json(self, last_json):
        self.last_json = last_json

    def get_last_json(self):
        return self.last_json

    def set_prompt(self, prompt):
        self.prompt = prompt

    def update_prompt(self, message):
        self.prompt += message

    def get_prompt(self):
        return self.prompt

    def set_model(self, model_name):
        import src.engine
        model = src.engine.TextEngine(model_name,
                                  token_limit=self.response_token_limit,
                                  top_k=self.top_k,
                                  top_p=self.top_p)
        self.model = model
        return model

    def get_model_name(self):
        return self.model.model_name

    def set_conversation_mode(self, conversation_mode):
        self.conversation_mode = conversation_mode

    async def generate_response(self, message, context, image_url=None):
        logging.info('Querying response as ' + self.persona_name + '...')
        if self.context_length > 0:
            context = context[0:self.context_length+1]
            context = context[::-1]  # Reverse the history list
            context = " \n".join(context)
            context = 'recent chat history: \n' + context
        else:
            context = None

        token_limit = self.response_token_limit
        response = await self.model.generate_response(self.prompt, message, context, image_url, token_limit)
        self.last_json = self.model.get_raw_json_request()

        # conversation mode
        if self.conversation_mode and self.context_length <= (GLOBAL_CONTEXT_LIMIT-2):
            self.context_length += 2

        return response

