from src import kobold_api
from src.engine import *
from src.persona import *
from src.app_manager import *
from src.utils import save_utils, model_utils
from src.utils.model_utils import get_model_list

import logging
import re
import json

logger = logging.getLogger(__name__)


# --- Decorator Definition ---
def save_after_change(func):
    """Decorator to save persona data after a method executes successfully and returns a non-error string."""

    def wrapper(self, *args, **kwargs):
        response = func(self, *args, **kwargs)
        if isinstance(response, str) and not response.lower().startswith('error'):
            save_utils.save_personas_to_file(self.chat_system.personas)
            logging.debug(f"Personas saved automatically after function '{func.__name__}' execution.")
        return response

    return wrapper


class BotLogic:
    def __init__(self, chat_system):
        self.message = None
        self.chat_system = chat_system
        self.koboldcpp_thread = None
        self.local_model = kobold_api.LocalModel()
        self.command_handlers = {
            'help': self._handle_help,
            'update_models': self._handle_update_models,
            'remember': self._handle_remember,
            'save': self._handle_save,
            'add': self._handle_add,
            'delete': self._handle_delete,
            'what': self._handle_what,
            'set': self._handle_set,
            'hello': self._handle_start_conversation,
            'goodbye': self._handle_stop_conversation,
            'dump_last': self._handle_dump_last,
            'start_koboldcpp': self._handle_start_koboldcpp,
            'stop_koboldcpp': self._handle_stop_koboldcpp,
            'check_koboldcpp': self._handle_check_koboldcpp,
            'query_generation': self._handle_koboldcpp_query,
            'update_app': self._handle_update_app,
            'restart_app': self._handle_restart_app,
            'stop_app': self._handle_stop_app,
        }
        self.what_handlers = {
            'prompt': self._what_prompt,
            'model': self._what_model,
            'models': self._what_models,
            'personas': self._what_personas,
            'context': self._what_context,
            'tokens': self._what_tokens,
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

    def preprocess_message(self, message, check_only=False):
        logging.debug('Checking for dev commands...')
        self.message = message
        split_args = re.split(r'[ ]', message.content.lower())
        try:
            persona_name, command, args = split_args[0].lower(), split_args[1].lower(), split_args[2:]
        except IndexError:
            return None

        current_persona = self.chat_system.personas.get(persona_name)
        if not current_persona:
            return None

        handler = self.command_handlers.get(command)
        if handler:
            if check_only:
                return True
            else:
                return handler(args, current_persona)

        logging.debug("No dev commands found.")
        return None

    def _handle_help(self, args, persona):
        if args:
            return None
        help_msg = "" \
                   "Talk to a specific persona by starting your message with their name. \n \n" \
                   "Currently active personas: \n" + \
                   ', '.join(self.chat_system.personas.keys()) + "\n\n" \
                                                                 "Bot commands: \n" \
                                                                 "hello (start new conversation), \n" \
                                                                 "goodbye (end conversation), \n" \
                                                                 "remember <+prompt>, \n" \
                                                                 "what prompt/model/models(+openai/google/anthropic)/personas/context/tokens, \n" \
                                                                 "set prompt/model/context/tokens, \n" \
                                                                 "add <persona>, \n" \
                                                                 "delete <persona>, \n" \
                                                                 "save, \n" \
                                                                 "update_models, \n" \
                                                                 "dump_last"
        return help_msg

    def _handle_stop_app(self, args, persona):
        return f'App stopping...'

    def _handle_restart_app(self, args, persona):
        return f'App restarting...'

    def _handle_update_app(self, args, persona):
        return update_app()

    @save_after_change
    def _handle_remember(self, args, persona):
        if not args:
            return None
        text_to_add = ' '.join(args)
        new_prompt = persona.get_prompt() + ' ' + text_to_add
        persona.set_prompt(new_prompt)
        return f'New prompt for {persona.name}: {persona.get_prompt()}'

    @save_after_change
    def _handle_add(self, args, persona):
        if not args:
            return None
        new_persona_name = args[0]
        prompt_args = args[1:]
        prompt = ' '.join(prompt_args) if prompt_args else 'you are in character as ' + new_persona_name
        self.chat_system.add_persona(new_persona_name, DEFAULT_MODEL_NAME, prompt,
                                     context_limit=DEFAULT_CONTEXT_LIMIT, token_limit=1024, save_new=False)
        return f"added '{new_persona_name}' with prompt: '{prompt}'"

    @save_after_change
    def _handle_delete(self, args, persona):
        if not args:
            return None
        persona_to_delete = args[0]
        self.chat_system.delete_persona(persona_to_delete, save=False)
        return persona_to_delete + " has been deleted."

    def _handle_what(self, args, persona):
        try:
            sub_command = args[0]
            handler = self.what_handlers.get(sub_command)
            if handler:
                return handler(args, persona)
        except IndexError:
            return None
        return None

    def _what_prompt(self, args, persona):
        return f"Prompt for '{persona.name}': {persona.get_prompt()}"

    def _what_model(self, args, persona):
        return f"{persona.name} is using {persona.get_model_name()}"

    def _what_models(self, args, persona):
        model_names = self.chat_system.models_available
        try:
            provider = args[1]
            provider_key_map = {'openai': 'From OpenAI', 'google': 'From Google', 'anthropic': 'From Anthropic'}
            if provider in provider_key_map:
                provider_key = provider_key_map[provider]
                if provider_key in model_names:
                    return json.dumps(model_names[provider_key], indent=2, ensure_ascii=False,
                                      separators=(',', ':')).replace('\"', '')
        except IndexError:
            pass
        formatted_models = json.dumps(model_names, indent=2, ensure_ascii=False, separators=(',', ':')).replace('\"',
                                                                                                                '')
        return f"Available model options: {formatted_models}"

    def _what_personas(self, args, persona):
        return f"Available personas are: {self.chat_system.get_persona_list()}"

    def _what_context(self, args, persona):
        return f"{persona.name} currently looks back {persona.get_context_length()} previous messages for context."

    def _what_tokens(self, args, persona):
        return f"{persona.name} is limited to {persona.get_response_token_limit()} response tokens."

    def _handle_set(self, args, persona):
        try:
            sub_command = args[0]
            handler = self.set_handlers.get(sub_command)
            if handler:
                return handler(args, persona)
        except IndexError:
            return None
        return None

    @save_after_change
    def _set_prompt(self, args, persona):
        prompt = ' '.join(args[1:])
        if not prompt:
            return "Error: 'set prompt' requires text for the new prompt."
        persona.set_prompt(prompt)
        logging.debug(f"Prompt set for '{persona.name}'.")
        return 'Prompt saved.'

    @save_after_change
    def _set_default_prompt(self, args, persona):
        prompt = DEFAULT_PERSONA
        persona.set_prompt(prompt)
        logging.debug(f"Prompt set for '{persona.name}'.")
        message = DEFAULT_WELCOME_REQUEST
        return persona.generate_response(persona.name, message)

    @save_after_change
    def _set_model(self, args, persona):
        try:
            model_name = args[1]
        except IndexError:
            return "Error: 'set model' requires a model name."
        if model_name == 'default':
            model_name = DEFAULT_MODEL_NAME
        if model_utils.check_model_available(model_name):
            persona.set_model(model_name)
            return f"Model set to '{model_name}'."
        else:
            return f"Model '{model_name}' does not exist. Try 'set model default' or 'what models'."

    @save_after_change
    def _set_tokens(self, args, persona):
        try:
            token_limit = args[1]
            return f"Set token limit: '{token_limit}' response tokens." if persona.set_response_token_limit(
                token_limit) else "Error setting response token limit."
        except IndexError:
            return "Error: 'set tokens' requires a number."

    @save_after_change
    def _set_context(self, args, persona):
        try:
            context_limit = args[1]
            persona.set_context_length(context_limit)
            return f"Set context_limit for {persona.name}, now reading '{context_limit}' previous messages."
        except IndexError:
            return "Error: 'set context' requires a number."

    @save_after_change
    def _set_temp(self, args, persona):
        try:
            new_temp = float(args[1])
            if not 0 <= new_temp <= 2:
                return f"Error: temperature value must be between 0 and 2, received {new_temp}"
            persona.set_temperature(new_temp)
            return f"Set temperature to {new_temp} for {persona.name}."
        except (IndexError, ValueError):
            return "Error: 'set temp' requires a numeric value between 0 and 2."

    @save_after_change
    def _set_top_p(self, args, persona):
        try:
            new_top_p = int(args[1])
            persona.set_top_p(new_top_p)
            return f"Set top_p to {new_top_p} for {persona.name}."
        except (IndexError, ValueError):
            return "Error: 'set top_p' requires an integer value."

    @save_after_change
    def _set_top_k(self, args, persona):
        try:
            new_top_k = int(args[1])
            persona.set_top_k(new_top_k)
            return f"Set top_k to {new_top_k} for {persona.name}."
        except (IndexError, ValueError):
            return "Error: 'set top_k' requires an integer value."

    def _handle_start_conversation(self, args, persona):
        if args:
            return None
        persona.set_context_length(0)
        persona.set_conversation_mode(True)
        return f"{persona.name}: Hello! Starting new conversation..."

    def _handle_check_koboldcpp(self, args, persona):
        logger.info('checking if koboldcpp is running...')
        if self.koboldcpp_thread is not None:
            return self.koboldcpp_thread.isAlive()

    def _handle_stop_koboldcpp(self, args, persona):
        logger.info('attempting to stop koboldcpp...')
        if self.koboldcpp_thread is not None:
            self.koboldcpp_thread.do_run = False
            self.koboldcpp_thread.join()
            return "koboldcpp process stopped"

    def _handle_start_koboldcpp(self, args, persona):
        import threading
        self.koboldcpp_thread = threading.Thread(target=launch_koboldcpp)
        self.koboldcpp_thread.start()
        return "Starting koboldcpp..."

    def _handle_koboldcpp_query(self, args, persona):
        return self.local_model.poll_generation_results()

    def _handle_stop_conversation(self, args, persona):
        if args:
            return None
        persona.set_context_length(DEFAULT_CONTEXT_LIMIT)
        persona.set_conversation_mode(False)
        return f"{persona.name}: Goodbye! Resetting context length to {DEFAULT_CONTEXT_LIMIT} previous messages..."

    def _handle_dump_last(self, args, persona):
        if args:
            return None
        raw_json_response = persona.get_last_json()
        last_request = json.dumps(raw_json_response, indent=2, ensure_ascii=False, separators=(',', ':')).replace('\\n',
                                                                                                                  '\n').replace(
            '\\"', '\"')
        return f"{persona.name}: {last_request}"

    @save_after_change
    def _handle_save(self, args, persona):
        if args:
            return None
        return 'Personas saved.'

    def _handle_update_models(self, args, persona):
        if args:
            return None
        self.chat_system.models_available = get_model_list(update=True)
        return f"Model names currently available: {json.dumps(self.chat_system.models_available, indent=4)}"