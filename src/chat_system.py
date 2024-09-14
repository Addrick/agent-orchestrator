import os
import logging

import src.utils.messages
from src.message_handler import *
from src.persona import *
from src.utils.config import save_personas_to_file, load_personas_from_file
from src.utils.models import get_model_list


# ChatSystem
# Maintains personas, queries the engine system for responses, executes dev commands
# It handles loading and saving personas from/to a file, adding and deleting personas,
# and generating responses using these personas.
class ChatSystem:
    def __init__(self):
        self.persona_save_file = PERSONA_SAVE_FILE
        self.personas = load_personas_from_file()
        self.models_available = get_model_list()
        self.bot_logic = BotLogic(self)  # Pass the instance of ChatSystem to BotLogic

    def get_persona_list(self):
        """Return the dictionary of personas."""
        return self.personas

    def add_persona(self, name, model_name, prompt, context_limit, token_limit, save_new=True):
        """Add a new persona to the system."""
        persona = Persona(name, model_name, prompt, context_limit, token_limit)
        self.personas[name] = persona
        # add to persona bank file
        if save_new:
            save_personas_to_file(self.personas)

    def delete_persona(self, name, save=False):
        """Delete a persona from the system."""
        del self.personas[name]
        if save:
            save_personas_to_file(self.personas)

    def add_to_prompt(self, persona_name, text_to_add):
        """Add text to a persona's prompt."""
        if persona_name in self.personas:
            self.personas[persona_name].add_to_prompt(text_to_add)
        else:
            logging.info(f"Failed to add to prompt, persona '{persona_name}' does not exist.")

    async def generate_response(self, persona_name, message, context='', image_url=None):
        """Generate a response using the specified persona and message channel."""
        # skip any images if the model can't handle it (currently only gpt-4o)
        # todo: determine if other apis support this and if I can tell which do via some request
        if self.personas[persona_name].model.model_name != 'gpt-4o':
            image_url = None
            logging.warning('Image URL discarded because the selected model is not gpt-4o and it cannot handle images (yet).')
        if persona_name in self.personas:
            persona = self.personas[persona_name]
            reply = await persona.generate_response(message, context, image_url)
            if persona_name != 'derpr':
                reply = f"{persona_name}: {reply}"
            return reply

        else:
            logging.warning(f"persona '{persona_name}' does not exist.")
            return "Error in chat_system generate_response"

