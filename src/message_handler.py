# src/message_handler.py

from config.global_config import DEFAULT_MODEL_NAME, DEFAULT_CONTEXT_LIMIT, DEFAULT_PERSONA, DEFAULT_WELCOME_REQUEST
from src.persona import Persona
from src.utils import model_utils
from src.utils.model_utils import get_model_list

import logging
import re
import json
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class BotLogic:
    def __init__(self, chat_system):
        self.chat_system = chat_system
        self.command_handlers = {
            'help': self._handle_help,
            'update_models': self._handle_update_models,
            'remember': self._handle_remember,
            'save': self._handle_save,
            'add': self._handle_add,
            'delete': self._handle_delete,
            'detail': self._handle_detail,
            'what': self._handle_what,
            'set': self._handle_set,
            'hello': self._handle_start_conversation,
            'goodbye': self._handle_stop_conversation,
            'dump_last': self._handle_dump_last,
        }
        self.what_handlers = {
            'prompt': self._what_prompt,
            'model': self._what_model,
            'models': self._what_models,
            'personas': self._what_personas,
            'context': self._what_context,
            'tokens': self._what_tokens,
            'temp': self._what_temp,
        }
        self.set_handlers = {
            'prompt': self._set_prompt,
            'default_prompt': self._set_default_prompt,
            'model': self._set_model,
            'tokens': self._set_tokens,
            'context': self._set_context,
            'temp': self._set_temp,
            'top_p': self._set_top_p,
            'top_k': self._set_top_k,
        }

    async def preprocess_message(self, persona_name: str, user_identifier: str, message: str) -> Optional[
        Dict[str, Any]]:
        split_args = re.split(r'[ ]', message.lower())
        try:
            command, args = split_args[0], split_args[1:]
        except IndexError:
            return None

        handler = self.command_handlers.get(command)
        if not handler:
            return None

        current_persona = self.chat_system.personas.get(persona_name)
        if not current_persona:
            return {"response": "Error: Current persona not found.", "mutated": False}

        response, mutated = handler(args, current_persona, user_identifier)
        if response is None:
            return None

        return {"response": response, "mutated": mutated}

    def _handle_help(self, args: list, persona: Persona, user_identifier: str) -> tuple[str, bool] | tuple[None, bool]:
        if args:
            return None, False
        help_msg = ("Talk to a specific persona by starting your message with their name. \n \n"
                    "Currently active personas: \n" +
                    ', '.join(self.chat_system.personas.keys()) + "\n\n"
                                                                  "Bot commands: \n"
                                                                  "hello (start new conversation), \n"
                                                                  "goodbye (end conversation), \n"
                                                                  "remember <+prompt>, \n"
                                                                  "what prompt/model/models (google/openai/anthropic)/personas/context/tokens/temp, \n"
                                                                  "set prompt/model/context/tokens/temp, \n"
                                                                  "add <persona>, \n"
                                                                  "delete <persona>, \n"
                                                                  "detail, \n"
                                                                  "save, \n"
                                                                  "update_models, \n"
                                                                  "dump_last")
        return help_msg, False

    def _handle_remember(self, args: list, persona: Persona, user_identifier: str) -> tuple[str, bool] | tuple[None, bool]:
        if not args:
            return None, False
        text_to_add = ' '.join(args)
        persona.append_to_prompt(' ' + text_to_add)
        return f'Prompt for {persona.get_name()} updated.', True

    def _handle_add(self, args: list, persona: Persona, user_identifier: str) -> tuple[str, bool] | tuple[None, bool]:
        if not args:
            return None, False # Invalid syntax, fall through to LLM
        new_persona_name = args[0]

        if new_persona_name in self.chat_system.personas:
            return f"Error: Persona '{new_persona_name}' already exists.", False

        prompt_args = args[1:]
        prompt = ' '.join(prompt_args) if prompt_args else 'you are in character as ' + new_persona_name

        new_persona = Persona(
            persona_name=new_persona_name,
            model_name=DEFAULT_MODEL_NAME,
            prompt=prompt
        )
        self.chat_system.personas[new_persona_name] = new_persona
        return f"Added '{new_persona_name}' with prompt: '{prompt}'", True

    def _handle_delete(self, args: list, persona: Persona, user_identifier: str) -> tuple[str, bool] | tuple[None, bool]:
        if not args:
            return None, False # Invalid syntax, fall through to LLM
        persona_to_delete = args[0]

        if persona_to_delete not in self.chat_system.personas:
            return f"Error: Persona '{persona_to_delete}' not found.", False

        del self.chat_system.personas[persona_to_delete]
        return f"Deleted persona '{persona_to_delete}'.", True

    def _handle_detail(self, args: list, persona: Persona, user_identifier: str) -> tuple[str, bool] | tuple[None, bool]:
        if args:
            return None, False
        details = (
            f"Details for Persona: {persona.get_name()}\n"
            f"----------------------------------------\n"
            f"Model: {persona.get_model_name() or 'default'}\n"
            f"Context Length: {persona.get_context_length()}\n"
            f"Response Token Limit: {persona.get_response_token_limit() or 'default'}\n"
            f"Generation Parameters:\n"
            f"  - Temperature: {persona.get_temperature() or 'default'}\n"
            f"  - Top P: {persona.get_top_p() or 'default'}\n"
            f"  - Top K: {persona.get_top_k() or 'default'}\n"
            f"----------------------------------------\n"
            f"Prompt:\n{persona.get_prompt()}"
        )
        return details, False

    def _handle_what(self, args: list, persona: Persona, user_identifier: str) -> Tuple[Optional[str], bool]:
        if not args:
            return None, False
        sub_command = args[0]
        handler = self.what_handlers.get(sub_command)
        if handler:
            return handler(args, persona)
        return None, False

    def _what_prompt(self, args: list, persona: Persona) -> Tuple[str, bool]:
        return f"Prompt for '{persona.get_name()}': {persona.get_prompt()}", False

    def _what_model(self, args: list, persona: Persona) -> Tuple[str, bool]:
        return f"{persona.get_name()} is using {persona.get_model_name()}", False

    def _what_models(self, args: list, persona: Persona) -> Tuple[Optional[str], bool]:
        all_models = self.chat_system.models_available
        # Case 1: No vendor specified, show all models.
        if len(args) == 1:
            return f"Available model options: {json.dumps(all_models, indent=2)}", False

        # Case 2: Vendor specified, try to find and filter.
        if len(args) == 2:
            vendor_arg = args[1].lower()
            for key, models in all_models.items():
                if vendor_arg in key.lower():
                    return f"Available models from {key}: {json.dumps({key: models}, indent=2)}", False

        # Case 3: Invalid vendor or too many args, fall through to LLM.
        return None, False

    def _what_personas(self, args: list, persona: Persona) -> Tuple[str, bool]:
        return f"Available personas are: {list(self.chat_system.personas.keys())}", False

    def _what_context(self, args: list, persona: Persona) -> Tuple[str, bool]:
        return f"{persona.get_name()} currently looks back {persona.get_context_length()} previous messages.", False

    def _what_tokens(self, args: list, persona: Persona) -> Tuple[str, bool]:
        return f"{persona.get_name()} is limited to {persona.get_response_token_limit()} response tokens.", False

    def _what_temp(self, args: list, persona: Persona) -> Tuple[str, bool]:
        return f"Temperature for {persona.get_name()} is set to {persona.get_temperature() or 'default'}.", False

    def _handle_set(self, args: list, persona: Persona, user_identifier: str) -> tuple[str, bool] | tuple[None, bool]:
        if not args:
            return None, False
        sub_command = args[0]
        handler = self.set_handlers.get(sub_command)
        if handler:
            return handler(args, persona)
        return f"Error: Unknown 'set' command: {sub_command}", False

    def _set_prompt(self, args: list, persona: Persona) -> tuple[str, bool] | tuple[None, bool]:
        prompt = ' '.join(args[1:])
        if not prompt:
            return None, False
        persona.set_prompt(prompt)
        return 'Prompt saved.', True

    def _set_default_prompt(self, args: list, persona: Persona) -> tuple[str, bool] | tuple[None, bool]:
        persona.set_prompt(DEFAULT_PERSONA)
        return f"Prompt for {persona.get_name()} reset to default.", True

    def _set_model(self, args: list, persona: Persona) -> tuple[str, bool] | tuple[None, bool]:
        try:
            model_name = args[1]
        except IndexError:
            return None, False
        if model_name == 'default':
            model_name = DEFAULT_MODEL_NAME
        if model_utils.check_model_available(model_name):
            persona.set_model_name(model_name)
            return f"Model for {persona.get_name()} set to '{model_name}'.", True
        else:
            return f"Error: Model '{model_name}' does not exist.", False

    def _set_tokens(self, args: list, persona: Persona) -> tuple[str, bool] | tuple[None, bool]:
        try:
            limit_str = args[1]
            token_limit = int(limit_str)
            persona.set_response_token_limit(token_limit)
            return f"Set token limit to '{token_limit}' for {persona.get_name()}.", True
        except IndexError:
            return None, False
        except ValueError:
            persona.set_response_token_limit(None)
            return f"Non-numeric token limit '{limit_str}' provided. The default token limit will be used for {persona.get_name()}.", True

    def _set_context(self, args: list, persona: Persona) -> tuple[str, bool] | tuple[None, bool]:
        try:
            limit_str = args[1]
            context_limit = int(limit_str)
            persona.set_context_length(context_limit)
            return f"Set context limit for {persona.get_name()} to '{context_limit}'.", True
        except IndexError:
            return None, False
        except ValueError:
            persona.set_context_length(None)
            return f"Non-numeric context limit '{limit_str}' provided. The default context length will be used for {persona.get_name()}.", True

    def _set_temp(self, args: list, persona: Persona) -> tuple[str, bool] | tuple[None, bool]:
        try:
            temp_str = args[1]
            new_temp = float(temp_str)
            if not 0 <= new_temp <= 2:
                return "Error: Temperature must be between 0 and 2.", False
            persona.set_temperature(new_temp)
            return f"Set temperature to {new_temp} for {persona.get_name()}.", True
        except IndexError:
            return None, False
        except ValueError:
            persona.set_temperature(None)
            return f"Non-numeric temperature '{temp_str}' provided. The default temperature will be used for {persona.get_name()}.", True

    def _set_top_p(self, args: list, persona: Persona) -> tuple[str, bool] | tuple[None, bool]:
        try:
            top_p_str = args[1]
            new_top_p = float(top_p_str)
            if not 0 <= new_top_p <= 1:
                return "Error: Top P must be between 0 and 1.", False
            persona.set_top_p(new_top_p)
            return f"Set top_p to {new_top_p} for {persona.get_name()}.", True
        except IndexError:
            return None, False
        except ValueError:
            persona.set_top_p(None)
            return f"Non-numeric Top P '{top_p_str}' provided. The default Top P will be used for {persona.get_name()}.", True

    def _set_top_k(self, args: list, persona: Persona) -> tuple[str, bool] | tuple[None, bool]:
        try:
            top_k_str = args[1]
            new_top_k = int(top_k_str)
            persona.set_top_k(new_top_k)
            return f"Set top_k to {new_top_k} for {persona.get_name()}.", True
        except IndexError:
            return None, False
        except ValueError:
            persona.set_top_k(None)
            return f"Non-numeric Top K '{top_k_str}' provided. The default Top K will be used for {persona.get_name()}.", True

    def _handle_start_conversation(self, args: list, persona: Persona, user_identifier: str) -> tuple[str, bool] | tuple[None, bool]:
        if args:
            return None, False
        persona.set_context_length(0)
        return f"{persona.get_name()}: Hello! Starting new conversation...", True

    def _handle_stop_conversation(self, args: list, persona: Persona, user_identifier: str) -> tuple[str, bool] | tuple[None, bool]:
        if args:
            return None, False
        persona.set_context_length(DEFAULT_CONTEXT_LIMIT)
        return f"{persona.get_name()}: Goodbye! Resetting context.", True

    def _handle_dump_last(self, args: list, persona: Persona, user_identifier: str) -> tuple[str, bool] | tuple[None, bool]:
        if args:
            return None, False

        persona_name = persona.get_name()
        last_request = self.chat_system.last_api_requests.get(user_identifier, {}).get(persona_name)

        if not last_request:
            return f"{persona_name}: No previous request to dump for your session with this persona.", False

        pretty_json = json.dumps(last_request, indent=2, sort_keys=True)
        display_json = pretty_json.replace('\\n', '\n')

        return f"{persona_name}: Last API Request Payload\n{display_json}", False

    def _handle_save(self, args: list, persona: Persona, user_identifier: str) -> tuple[str, bool] | tuple[None, bool]:
        if args:
            return None, False
        return 'Personas saved.', True

    def _handle_update_models(self, args: list, persona: Persona, user_identifier: str) -> tuple[str, bool] | tuple[None, bool]:
        if args:
            return None, False
        self.chat_system.models_available = get_model_list(update=True)
        return f"Model list updated. Currently available: {json.dumps(self.chat_system.models_available, indent=2)}", False
