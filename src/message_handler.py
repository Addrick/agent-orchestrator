# src/message_handler.py

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from config.global_config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PERSONA,
    MODEL_SELECTOR_PERSONA_NAME,
    TOOL_SELECTOR_PERSONA_NAME
)

from src.persona import Persona, ExecutionMode, MemoryMode
from src.utils import model_utils
from src.utils.model_utils import get_model_list

if TYPE_CHECKING:
    from src.chat_system import ChatSystem

logger = logging.getLogger(__name__)


class BotLogic:
    def __init__(self, chat_system: "ChatSystem") -> None:
        self.chat_system: "ChatSystem" = chat_system
        self.command_handlers = {
            'help': self._handle_help,
            'update_models': self._handle_update_models,
            'remember': self._handle_remember,
            'add': self._handle_add,
            'delete': self._handle_delete,
            'detail': self._handle_detail,
            'what': self._handle_what,
            'set': self._handle_set,
            'hello': self._handle_start_conversation,
            'goodbye': self._handle_stop_conversation,
            'dump_last': self._handle_dump_last,
            'dump_context': self._handle_dump_context,
        }
        self.what_handlers = {
            'prompt': self._what_prompt,
            'model': self._what_model,
            'models': self._what_models,
            'personas': self._what_personas,
            'context': self._what_context,
            'tokens': self._what_tokens,
            'temp': self._what_temp,
            'execution_mode': self._what_execution_mode,
            'tools': self._what_tools,
            'memory_mode': self._what_memory_mode,
            'service_bindings': self._what_service_bindings,
            'top_p': self._what_top_p,
            'top_k': self._what_top_k,
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
            'display_name': self._set_display_name,
            'execution_mode': self._set_execution_mode,
            'tools': self._set_tools,
            'memory_mode': self._set_memory_mode,
            'service_bindings': self._set_service_bindings,
        }

    async def preprocess_message(
            self,
            persona_name: str,
            user_identifier: str,
            message: str
    ) -> Optional[Dict[str, Any]]:
        split_args: List[str] = re.split(r'[ ]', message.lower())
        command: str
        args: List[str]
        try:
            command, args = split_args[0], split_args[1:]
        except IndexError:
            return None

        handler: Any = self.command_handlers.get(command)
        if not handler:
            return None

        current_persona: Optional[Persona] = self.chat_system.personas.get(persona_name)
        if not current_persona:
            return {"response": "Error: Current persona not found.", "mutated": False}

        response: Optional[str]
        mutated: bool

        # Only 'set' is async currently
        if command == 'set':
            response, mutated = await handler(args, current_persona, user_identifier)
        else:
            response, mutated = handler(args, current_persona, user_identifier)

        if response is None:
            return None

        return {"response": response, "mutated": mutated}

    def _handle_help(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[Optional[str], bool]:
        if args:
            return None, False
        help_msg: str = ("Talk to a specific persona by starting your message with their name. \n \n"
                         "Currently active personas: \n" +
                         ', '.join(self.chat_system.personas.keys()) + "\n\n"
                                                                       "Bot commands: \n"
                                                                       "hello (start new conversation), \n"
                                                                       "goodbye (end conversation), \n"
                                                                       "remember <+prompt>, \n"
                                                                       "what prompt/model/models/personas/context/tokens/temp/top_p/top_k/execution_mode/tools/memory_mode/service_bindings, \n"
                                                                       "set prompt/model/context/tokens/temp/top_p/top_k/display_name/execution_mode/tools/memory_mode/service_bindings, \n"
                                                                       "add <persona>, \n"
                                                                       "delete <persona>, \n"
                                                                       "detail, \n"
                                                                       "update_models, \n"
                                                                       "dump_last, \n"
                                                                       "dump_context")
        return help_msg, False

    async def _query_llm_for_model_selection(self, user_query: str) -> Optional[str]:
        """
        Query model selector persona to find best matching model.
        Returns model name if successful, None if persona unavailable or parsing fails.
        """
        try:
            # # Get the model selector persona
            # selector_persona = self.chat_system.personas.get(MODEL_SELECTOR_PERSONA_NAME)
            #
            # if not selector_persona:
            #     logger.warning(f"Model selector persona '{MODEL_SELECTOR_PERSONA_NAME}' not found")
            #     return None

            # Build available models list
            models_str = json.dumps(self.chat_system.models_available, indent=2)

            # Construct query message
            user_message = f"Available models:\n{models_str}\n\nUser query: {user_query}\n\nSelected model:"
            #
            # # Build context for text_engine
            # context = {
            #     "persona_prompt": selector_persona.get_prompt(),
            #     "history": [],
            #     "current_message": {"text": user_message, "image_url": None}
            # }

            response_text, response_type, ticket_id = await self.chat_system.generate_response(
                persona_name=MODEL_SELECTOR_PERSONA_NAME,
                user_identifier="n/a",
                channel="model_selector_query",
                message=user_message,
                server_id="model_selector_query",
                image_url=None,
                history_limit=0,
                user_display_name="n/a"
            )

            if response_text:
                model_name = response_text.strip()

                # Validate response
                if model_name == "DEFAULT":
                    return None

                # Check if returned model exists
                if model_utils.check_model_available(model_name):
                    return model_name

            return None

        except Exception as e:
            logger.error(f"Error during LLM model selection: {e}", exc_info=True)
            return None

    async def _query_llm_for_tool_selection(self, user_query: str, available_tools: List[str]) -> Optional[str]:
        """
        Query tool selector persona to find best matching tool name.
        Returns exact tool name if successful, None if persona unavailable or no match.
        """
        try:
            tools_str = "\n".join(available_tools)
            user_message = f"Available tools:\n{tools_str}\n\nUser query: {user_query}\n\nMatched tool:"

            response_text, _, _ = await self.chat_system.generate_response(
                persona_name=TOOL_SELECTOR_PERSONA_NAME,
                user_identifier="n/a",
                channel="tool_selector_query",
                message=user_message,
                server_id="tool_selector_query",
                image_url=None,
                history_limit=0,
                user_display_name="n/a"
            )

            if response_text:
                tool_name = response_text.strip()
                if tool_name == "NONE":
                    return None
                if tool_name in available_tools:
                    return tool_name

            return None

        except Exception as e:
            logger.error(f"Error during LLM tool selection: {e}", exc_info=True)
            return None

    def _handle_remember(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[Optional[str], bool]:
        if not args:
            return None, False
        text_to_add: str = ' '.join(args)
        persona.append_to_prompt(' ' + text_to_add)
        return f'Prompt for {persona.get_name()} updated.', True

    def _handle_add(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[Optional[str], bool]:
        if not args:
            return None, False
        new_persona_name: str = args[0]

        if new_persona_name in self.chat_system.personas:
            return f"Error: Persona '{new_persona_name}' already exists.", False

        prompt_args: List[str] = args[1:]
        prompt: str = ' '.join(prompt_args) if prompt_args else 'you are in character as ' + new_persona_name

        new_persona = Persona(
            persona_name=new_persona_name,
            model_name=DEFAULT_MODEL_NAME,
            prompt=prompt
        )
        self.chat_system.personas[new_persona_name] = new_persona
        return f"Added '{new_persona_name}' with prompt: '{prompt}'", True

    def _handle_delete(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[Optional[str], bool]:
        if not args:
            return None, False
        persona_to_delete: str = args[0]

        if persona_to_delete not in self.chat_system.personas:
            return f"Error: Persona '{persona_to_delete}' not found.", False

        del self.chat_system.personas[persona_to_delete]
        return f"Deleted persona '{persona_to_delete}'.", True

    def _handle_detail(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[Optional[str], bool]:
        if args:
            return None, False

        enabled_tools = persona.get_enabled_tools()
        if enabled_tools == ['*']:
            tools_display = "All"
        elif not enabled_tools:
            tools_display = "None"
        else:
            tools_display = ", ".join(enabled_tools)

        context_display: str
        if persona.is_in_dynamic_context():
            next_limit = persona.get_current_effective_context_length()
            context_display = f"{next_limit} (Dynamic, will grow on next message)"
        else:
            context_display = str(persona.get_current_effective_context_length())

        details: str = (
            f"Details for Persona: {persona.get_name()}\n"
            f"----------------------------------------\n"
            f"Model: {persona.get_model_name() or 'default'}\n"
            f"Memory Mode: {persona.get_memory_mode().name}\n"
            f"Execution Mode: {persona.get_execution_mode().name}\n"
            f"Service Bindings: {', '.join(persona.get_service_bindings()) or 'none'}\n"
            f"Enabled Tools: {tools_display}\n"
            f"Context Length: {context_display}\n"
            f"Display Name in Chat: {persona.should_display_name_in_chat()}\n"
            f"Response Token Limit: {persona.get_response_token_limit() or 'default'}\n"
            f"Generation Parameters:\n"
            f"  - Temperature: {persona.get_temperature() or 'default'}\n"
            f"  - Top P: {persona.get_top_p() or 'default'}\n"
            f"  - Top K: {persona.get_top_k() or 'default'}\n"
            f"----------------------------------------\n"
            f"Prompt:\n{persona.get_prompt()}"
        )
        return details, False

    def _handle_what(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[Optional[str], bool]:
        if not args:
            return None, False
        sub_command: str = args[0]
        handler = self.what_handlers.get(sub_command)
        if handler:
            return handler(args, persona)
        return None, False

    def _what_prompt(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        return f"Prompt for '{persona.get_name()}': {persona.get_prompt()}", False

    def _what_model(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        return f"{persona.get_name()} is using {persona.get_model_name()}", False

    def _what_models(self, args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
        all_models: Dict[str, Any] = self.chat_system.models_available
        if len(args) == 1:
            return f"Available model options: {json.dumps(all_models, indent=2)}", False

        if len(args) == 2:
            vendor_arg: str = args[1].lower()
            for key, models in all_models.items():
                if vendor_arg in key.lower():
                    return f"Available models from {key}: {json.dumps({key: models}, indent=2)}", False

        return None, False

    def _what_personas(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        return f"Available personas are: {list(self.chat_system.personas.keys())}", False

    def _what_context(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        return f"{persona.get_name()} default context length is {persona.get_base_context_length()}.", False

    def _what_tokens(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        return f"{persona.get_name()} is limited to {persona.get_response_token_limit()} response tokens.", False

    def _what_temp(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        return f"Temperature for {persona.get_name()} is set to {persona.get_temperature() or 'default'}.", False

    def _what_execution_mode(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        return f"Execution mode for '{persona.get_name()}' is set to {persona.get_execution_mode().name}.", False

    def _what_tools(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        all_tool_defs = self.chat_system.tool_manager.get_tool_definitions()
        all_tool_names = {tool['function']['name'] for tool in all_tool_defs}
        enabled_tools = persona.get_enabled_tools()

        response_lines = ["Available Tools & Status for " + persona.get_name() + ":"]
        if not all_tool_names:
            return "No tools are currently available in the system.", False

        for tool_name in sorted(list(all_tool_names)):
            status = "[ENABLED]" if enabled_tools == ['*'] or tool_name in enabled_tools else "[DISABLED]"
            response_lines.append(f"- {tool_name} {status}")

        return "\n".join(response_lines), False

    def _what_memory_mode(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        valid_modes = ", ".join([e.name.lower() for e in MemoryMode])
        return f"Memory mode for '{persona.get_name()}' is {persona.get_memory_mode().name.lower()}.\nValid modes are: {valid_modes}.", False

    def _what_service_bindings(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        bindings = persona.get_service_bindings()
        display = ', '.join(bindings) if bindings else 'none'
        return f"Service bindings for '{persona.get_name()}': {display}.", False

    def _what_top_p(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        return f"Top P for {persona.get_name()} is set to {persona.get_top_p() or 'default'}.", False

    def _what_top_k(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        return f"Top K for {persona.get_name()} is set to {persona.get_top_k() or 'default'}.", False

    async def _handle_set(
            self,
            args: List[str],
            persona: Persona,
            user_identifier: str
    ) -> Tuple[Optional[str], bool]:
        if not args:
            return None, False

        sub_command: str = args[0]
        set_handler: Any = self.set_handlers.get(sub_command)

        if set_handler:
            result: Tuple[Optional[str], bool]
            if sub_command in ('model', 'tools'):
                result = await set_handler(args, persona)
            else:
                result = set_handler(args, persona)
            return result

        return f"Error: Unknown 'set' command: {sub_command}", False

    def _set_prompt(self, args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
        prompt: str = ' '.join(args[1:])
        if not prompt:
            return None, False
        persona.set_prompt(prompt)
        return 'Prompt saved.', True

    def _set_default_prompt(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        persona.set_prompt(DEFAULT_PERSONA)
        return f"Prompt for {persona.get_name()} reset to default.", True

    async def _set_model(self, args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
        model_name: str
        try:
            model_name = args[1]
        except IndexError:
            return None, False

        # Handle 'default' keyword
        if model_name == 'default':
            model_name = DEFAULT_MODEL_NAME

        # Try exact match first (fast path)
        if model_utils.check_model_available(model_name):
            persona.set_model_name(model_name)
            return f"Model for {persona.get_name()} set to '{model_name}'.", True

        # Try LLM-assisted selection
        model = " ".join(args)
        selected_model = await self._query_llm_for_model_selection(model)

        if selected_model:
            persona.set_model_name(selected_model)
            return f"Model for {persona.get_name()} set to '{selected_model}' (matched from '{model}').", True

        # Fallback to default
        persona.set_model_name(DEFAULT_MODEL_NAME)
        return f"Could not find '{model}'. Model for {persona.get_name()} set to default: '{DEFAULT_MODEL_NAME}'.", True

    def _set_tokens(self, args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
        limit_str: str
        try:
            limit_str = args[1]
            token_limit: int = int(limit_str)
            persona.set_response_token_limit(token_limit)
            return f"Set token limit to '{token_limit}' for {persona.get_name()}.", True
        except IndexError:
            return None, False
        except ValueError:
            limit_str = args[1]
            persona.set_response_token_limit(None)
            return f"Non-numeric token limit '{limit_str}' provided. The default token limit will be used for {persona.get_name()}.", True

    def _set_context(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        if len(args) < 2:
            return "Usage: set context <number|dynamic> [start_value]", False

        mode = args[1].lower()
        if mode == 'dynamic':
            start_value: int
            if len(args) > 2:
                try:
                    start_value = int(args[2])
                except ValueError:
                    return f"Error: Invalid start value '{args[2]}'. Must be an integer.", False
            else:
                start_value = persona.get_current_effective_context_length()

            persona.start_new_conversation(start_value)
            return f"Dynamic context mode enabled for {persona.get_name()}, starting at size {start_value}.", True
        else:
            try:
                context_limit = int(mode)
                persona.set_context_length(context_limit)
                return f"Set static context limit for {persona.get_name()} to '{context_limit}'.", True
            except ValueError:
                return f"Error: Invalid context command '{mode}'. Use a number or 'dynamic'.", False

    def _set_temp(self, args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
        temp_str: str
        try:
            temp_str = args[1]
            new_temp: float = float(temp_str)
            if not 0 <= new_temp <= 2:
                return "Error: Temperature must be between 0 and 2.", False
            persona.set_temperature(new_temp)
            return f"Set temperature to {new_temp} for {persona.get_name()}.", True
        except IndexError:
            return None, False
        except ValueError:
            temp_str = args[1]
            persona.set_temperature(None)
            return f"Non-numeric temperature '{temp_str}' provided. The default temperature will be used for {persona.get_name()}.", True

    def _set_top_p(self, args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
        top_p_str: str
        try:
            top_p_str = args[1]
            new_top_p: float = float(top_p_str)
            if not 0 <= new_top_p <= 1:
                return "Error: Top P must be between 0 and 1.", False
            persona.set_top_p(new_top_p)
            return f"Set top_p to {new_top_p} for {persona.get_name()}.", True
        except IndexError:
            return None, False
        except ValueError:
            top_p_str = args[1]
            persona.set_top_p(None)
            return f"Non-numeric Top P '{top_p_str}' provided. The default Top P will be used for {persona.get_name()}.", True

    def _set_top_k(self, args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
        top_k_str: str
        try:
            top_k_str = args[1]
            new_top_k: int = int(top_k_str)
            persona.set_top_k(new_top_k)
            return f"Set top_k to {new_top_k} for {persona.get_name()}.", True
        except IndexError:
            return None, False
        except ValueError:
            top_k_str = args[1]
            persona.set_top_k(None)
            return f"Non-numeric Top K '{top_k_str}' provided. The default Top K will be used for {persona.get_name()}.", True

    def _set_display_name(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        value_str: str
        try:
            value_str = args[1].lower()
        except IndexError:
            return "Error: Please specify 'on' or 'off' for the display name.", False

        new_value: bool
        if value_str in ['true', 'on', 'yes', '1']:
            new_value = True
        elif value_str in ['false', 'off', 'no', '0']:
            new_value = False
        else:
            return f"Error: Invalid value '{value_str}'. Please use 'on' or 'off'.", False

        persona.set_display_name_in_chat(new_value)
        status: str = "enabled" if new_value else "disabled"
        return f"Displaying name in chat for {persona.get_name()} is now {status}.", True

    def _set_execution_mode(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        try:
            mode_str = args[1].upper()
        except IndexError:
            valid_modes = ", ".join([e.name.lower() for e in ExecutionMode])
            return f"Error: Please specify an execution mode. Valid modes are: {valid_modes}.", False

        try:
            ExecutionMode[mode_str]
            persona.set_execution_mode(mode_str)
            return f"Execution mode for {persona.get_name()} set to '{mode_str}'.", True
        except KeyError:
            valid_modes = ", ".join([e.name.lower() for e in ExecutionMode])
            return f"Error: Invalid execution mode '{args[1]}'. Valid modes are: {valid_modes}.", False

    async def _set_tools(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        if len(args) < 2:
            return "Usage: set tools <all|none|tool_name_1> [tool_name_2]... (prefix with - to exclude)", False

        all_tool_defs = self.chat_system.tool_manager.get_tool_definitions()
        available_tool_names = sorted(tool['function']['name'] for tool in all_tool_defs)
        available_set = set(available_tool_names)

        raw_args = args[1:]
        # Check for 'all' or 'none' keywords
        use_all = raw_args[0].lower() == 'all'
        use_none = raw_args[0].lower() == 'none'

        if use_none:
            persona.set_enabled_tools([])
            return f"All tools have been disabled for {persona.get_name()}.", True

        # Split into includes and excludes
        include_names = []
        exclude_names = []
        names_to_process = raw_args[1:] if use_all else raw_args

        for name in names_to_process:
            if name.startswith('-'):
                exclude_names.append(name[1:])
            else:
                if use_all:
                    # bare names after 'all' without '-' are errors
                    return f"Error: Use '-' prefix to exclude tools when using 'all'. Example: set tools all -{name}", False
                include_names.append(name)

        # Excludes without 'all' and without includes is an error
        if exclude_names and not use_all and not include_names:
            return "Error: Exclude syntax requires 'all' as a base. Usage: set tools all -tool_name", False

        # Resolve each name: exact match first, then fuzzy LLM
        resolved_includes = []
        resolved_excludes = []
        fuzzy_matches = []

        async def resolve_tool_name(name: str) -> Optional[str]:
            if name in available_set:
                return name
            resolved = await self._query_llm_for_tool_selection(name, available_tool_names)
            if resolved:
                fuzzy_matches.append(f"'{name}' -> '{resolved}'")
            return resolved

        for name in include_names:
            resolved = await resolve_tool_name(name)
            if not resolved:
                return f"Error: Could not match tool '{name}'. Available: {', '.join(available_tool_names)}", False
            resolved_includes.append(resolved)

        for name in exclude_names:
            resolved = await resolve_tool_name(name)
            if not resolved:
                return f"Error: Could not match excluded tool '{name}'. Available: {', '.join(available_tool_names)}", False
            resolved_excludes.append(resolved)

        # Build final tool set
        if use_all:
            if resolved_excludes:
                final_tools = [t for t in available_tool_names if t not in set(resolved_excludes)]
                persona.set_enabled_tools(final_tools)
                fuzzy_note = f" (fuzzy: {', '.join(fuzzy_matches)})" if fuzzy_matches else ""
                return (f"All tools enabled for {persona.get_name()} except: "
                        f"{', '.join(resolved_excludes)}.{fuzzy_note}"), True
            else:
                persona.set_enabled_tools(['*'])
                return f"All tools have been enabled for {persona.get_name()}.", True
        else:
            persona.set_enabled_tools(resolved_includes)
            fuzzy_note = f" (fuzzy: {', '.join(fuzzy_matches)})" if fuzzy_matches else ""
            return (f"Enabled tools for {persona.get_name()} set to: "
                    f"{', '.join(resolved_includes)}.{fuzzy_note}"), True

    def _set_memory_mode(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        try:
            mode_str = args[1].upper()
        except IndexError:
            valid_modes = ", ".join([e.name.lower() for e in MemoryMode])
            return f"Error: Please specify a memory mode. Valid modes are: {valid_modes}.", False

        try:
            MemoryMode[mode_str]
            persona.set_memory_mode(mode_str)
            return f"Memory mode for {persona.get_name()} set to '{mode_str}'.", True
        except KeyError:
            valid_modes = ", ".join([e.name.lower() for e in MemoryMode])
            return f"Error: Invalid memory mode '{args[1]}'. Valid modes are: {valid_modes}.", False

    def _set_service_bindings(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        if len(args) < 2:
            return "Error: Please specify service bindings (comma-separated, or 'none' to clear).", False
        value_str = args[1].lower().strip()
        if value_str in ['none', 'clear', '[]']:
            persona.set_service_bindings([])
            return f"Service bindings for {persona.get_name()} cleared.", True
        bindings = [b.strip() for b in value_str.split(',') if b.strip()]
        persona.set_service_bindings(bindings)
        return f"Service bindings for {persona.get_name()} set to: {', '.join(bindings)}.", True

    def _handle_start_conversation(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[
        Optional[str], bool]:
        if args:
            return None, False
        persona.start_new_conversation()
        return f"{persona.get_name()}: Hello! Starting new conversation...", True

    def _handle_stop_conversation(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[
        Optional[str], bool]:
        if args:
            return None, False
        persona.end_new_conversation()
        return f"{persona.get_name()}: Goodbye! Resetting context.", True

    def _handle_dump_last(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[str, bool]:
        if args:
            return "Usage: dump_last", False

        persona_name: str = persona.get_name()
        last_request: Optional[Dict[str, Any]] = self.chat_system.last_api_requests.get(user_identifier, {}).get(
            persona_name)

        if not last_request:
            return f"{persona_name}: No previous request to dump for your session with this persona.", False

        # --- Finalized Summary Logic for Standardized Payloads ---
        model_name = last_request.get('model', last_request.get('model_name', 'N/A'))

        # Check for a system prompt to correctly count conversational turns
        has_system_prompt = False
        history_list = last_request.get('contents', last_request.get('messages', []))
        if last_request.get('system') or (history_list and history_list[0].get('role') == 'system'):
            has_system_prompt = True

        # Count conversational turns (excluding any system prompt)
        conversational_turns = 0
        if history_list:
            # Subtract 1 from the total length if a system prompt was the first item in the list
            conversational_turns = len(history_list) - 1 if has_system_prompt else len(history_list)

        # Extract generation parameters from potentially different locations
        config = last_request.get('config', {})
        temp = config.get('temperature', last_request.get('temperature', 'default'))
        max_tokens = config.get('max_output_tokens', last_request.get('max_length', 'default'))

        # Safely get and format tool names
        tools_list = config.get('tools', [])
        if tools_list and isinstance(tools_list[0], dict):  # Handle OpenAI format
            tools_list = [t.get('function', {}).get('name', 'unknown') for t in tools_list]
        tools = ", ".join(tools_list) if tools_list else "None"

        summary = (
            f"{persona_name}: Summary of Last API Request\n"
            f"----------------------------------------\n"
            f"Model Used: {model_name}\n"
            f"Context Sent:\n"
            f"  - Context Messages: {conversational_turns}\n"
            f"  - Memory Mode Used: {persona.get_memory_mode().name.lower()}\n"
            f"Generation Params:\n"
            f"  - Temperature: {temp}\n"
            f"  - Max Output Tokens: {max_tokens}\n"
            f"  - Tools Available: {tools}\n"
            f"----------------------------------------\n"
            f"Tip: Use `dump context` to see the exact history file sent to the model."
        )
        return summary, False

        # In src/message_handler.py

    def _handle_dump_context(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[
        Optional[str], bool]:
        """
        Generates a detailed text file containing the full context of the last API call.

        Includes everything the LLM received: persona config, service bindings,
        tool definitions (with full schemas), system prompt, and conversation history.

        Returns:
            A specially formatted string that signals to the Discord interface
            to create and upload a file, or an error message.
            Format: "FILE_RESPONSE::filename.txt::file_content"
        """
        if args:
            return "Usage: dump_context", False

        persona_name = persona.get_name()
        last_request = self.chat_system.last_api_requests.get(user_identifier, {}).get(persona_name)

        if not last_request:
            return f"{persona_name}: No previous request to analyze.", False

        output_lines = [f"=== Context Dump for {persona_name} ==="]
        self._dump_persona_config(output_lines, persona)
        self._dump_tools(output_lines, last_request.get('_tools_for_llm', []))
        self._dump_api_config(output_lines, last_request.get('config', {}))
        self._dump_conversation(output_lines, last_request)

        file_content = "\n".join(output_lines)
        return f"FILE_RESPONSE::context_dump.txt::{file_content}", False

    @staticmethod
    def _dump_persona_config(lines: List[str], persona: Persona) -> None:
        """Append persona configuration section to dump output."""
        lines.append("\n--- Persona Configuration ---")
        lines.append(f"Model: {persona.get_model_name()}")
        lines.append(f"Memory Mode: {persona.get_memory_mode().name}")
        lines.append(f"Execution Mode: {persona.get_execution_mode().name}")
        bindings = persona.get_service_bindings()
        lines.append(f"Service Bindings: {', '.join(bindings) if bindings else 'none'}")
        lines.append(f"Context Length Setting: {persona.get_base_context_length()}")
        lines.append(f"Response Token Limit: {persona.get_response_token_limit() or 'default'}")
        lines.append(
            f"Temp: {persona.get_temperature()}, Top P: {persona.get_top_p()}, Top K: {persona.get_top_k()}")

    @staticmethod
    def _dump_tools(lines: List[str], tools_for_llm: List[Dict[str, Any]]) -> None:
        """Append tool definitions section to dump output."""
        lines.append(f"\n--- Tools Sent to LLM ({len(tools_for_llm)} total) ---")
        if not tools_for_llm:
            lines.append("No tools were sent to the LLM.")
            return
        for tool in tools_for_llm:
            func = tool.get('function', {})
            name = func.get('name', 'unknown')
            params = func.get('parameters', {})
            props = params.get('properties', {})
            required = params.get('required', [])

            lines.append(f"\n  [{name}]")
            lines.append(f"    Description: {func.get('description', 'No description')}")
            lines.append(f"    Service Binding: {tool.get('service_binding', 'none')}")
            lines.append(f"    Write Operation: {tool.get('is_write', False)}")
            if props:
                lines.append("    Parameters:")
                for pname, pdef in props.items():
                    req = " (required)" if pname in required else ""
                    lines.append(f"      - {pname}: {pdef.get('type', 'any')}{req}"
                                 f" — {pdef.get('description', '')}")
            else:
                lines.append("    Parameters: none")

    @staticmethod
    def _dump_api_config(lines: List[str], config_data: Dict[str, Any]) -> None:
        """Append API request config section to dump output."""
        if not config_data:
            return
        lines.append("\n--- API Request Config ---")
        for key, value in config_data.items():
            if key not in ('tools', 'tool_config', 'safety_settings'):
                lines.append(f"  {key}: {value}")

    @staticmethod
    def _dump_conversation(lines: List[str], last_request: Dict[str, Any]) -> None:
        """Append conversation history section to dump output."""
        lines.append("\n--- Context Sent to Model ---")
        contents = last_request.get('contents', last_request.get('messages', []))

        if not contents:
            lines.append("No content was sent to the model.")
            return

        conversation_history = contents
        if contents[0].get('role') == 'system':
            sys_content = contents[0].get('content', '')
            if not sys_content:
                parts = contents[0].get('parts', [{}])
                sys_content = parts[0].get('text', '[NO TEXT CONTENT]') if parts else '[NO TEXT CONTENT]'
            lines.append("\n[System Prompt]")
            lines.append(sys_content)
            lines.append("-" * 40)
            conversation_history = contents[1:]

        if not conversation_history:
            lines.append("\nNo conversational messages were sent (only a system prompt).")
            return

        for i, item in enumerate(conversation_history):
            role = item.get('role', 'unknown').upper()
            lines.append(f"\n[Message {i + 1} - ROLE: {role}]")
            lines.append(BotLogic._extract_message_content(item))
            lines.append("-" * 40)

    @staticmethod
    def _extract_message_content(message: Dict[str, Any]) -> str:
        """Extract displayable text from a message dict across all provider formats."""
        # Tool call messages (assistant requesting tool use)
        if 'tool_calls' in message:
            calls = message['tool_calls']
            call_strs = [f"  {c.get('name', 'unknown')}({json.dumps(c.get('arguments', {}), indent=2)})"
                         for c in calls]
            return "[TOOL CALLS]\n" + "\n".join(call_strs)

        # Tool result messages (results returned to assistant)
        if message.get('role') == 'tool':
            return f"[TOOL RESULT: {message.get('name', 'unknown')}]\n{message.get('content', '')}"

        content = message.get('content')

        # Direct content string (OpenAI/Anthropic)
        if isinstance(content, str) and content:
            return content

        # OpenAI/Anthropic multimodal: content is a list of typed parts
        if isinstance(content, list):
            return BotLogic._extract_multimodal_parts(content)

        # Google format: parts list
        parts = message.get('parts', [])
        if parts and isinstance(parts, list):
            texts = [p['text'] for p in parts if isinstance(p, dict) and 'text' in p]
            return '\n'.join(texts) if texts else '[NO TEXT CONTENT]'

        return '[NO TEXT CONTENT]'

    @staticmethod
    def _extract_multimodal_parts(content: List[Any]) -> str:
        """Extract text from OpenAI/Anthropic multimodal content arrays."""
        texts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get('type', '')
            if ptype == 'text':
                texts.append(part.get('text', ''))
            elif ptype in ('image_url', 'image'):
                texts.append('[IMAGE]')
        return '\n'.join(texts) if texts else '[MULTIMODAL CONTENT]'

    def _handle_update_models(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[str, bool]:
        if args:
            return "Usage: update_models", False
        self.chat_system.models_available = get_model_list(update=True) or {}
        return f"Model list updated. Currently available: {json.dumps(self.chat_system.models_available, indent=2)}", False
