# src/message_handler.py

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from config.global_config import (DEFAULT_CONTEXT_LIMIT, DEFAULT_MODEL_NAME,
                                  DEFAULT_PERSONA)
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
        }

    async def preprocess_message(self, persona_name: str, user_identifier: str, message: str) -> Optional[
        Dict[str, Any]]:
        split_args: List[str] = re.split(r'[ ]', message.lower())
        command: str
        args: List[str]
        try:
            command, args = split_args[0], split_args[1:]
        except IndexError:
            return None

        handler = self.command_handlers.get(command)
        if not handler:
            return None

        current_persona: Optional[Persona] = self.chat_system.personas.get(persona_name)
        if not current_persona:
            return {"response": "Error: Current persona not found.", "mutated": False}

        response: Optional[str]
        mutated: bool
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
                                                                       "what prompt/model/models/personas/context/tokens/temp/execution_mode/tools/memory_mode, \n"
                                                                       "set prompt/model/context/tokens/temp/display_name/execution_mode/tools/memory_mode, \n"
                                                                       "add <persona>, \n"
                                                                       "delete <persona>, \n"
                                                                       "detail, \n"
                                                                       "update_models, \n"
                                                                       "dump_last, \n"
                                                                       "dump_context")
        return help_msg, False

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
        return f"{persona.get_name()} default context length is {persona._context_length}.", False

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

    def _handle_set(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[Optional[str], bool]:
        if not args:
            return None, False
        sub_command: str = args[0]
        handler = self.set_handlers.get(sub_command)
        if handler:
            return handler(args, persona)
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

    def _set_model(self, args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
        model_name: str
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

    def _set_tools(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        if len(args) < 2:
            return "Usage: set tools <all|none|tool_name_1> [tool_name_2]...", False

        all_tool_defs = self.chat_system.tool_manager.get_tool_definitions()
        available_tool_names = {tool['function']['name'] for tool in all_tool_defs}

        mode = args[1].lower()
        if mode == 'all':
            persona.set_enabled_tools(['*'])
            return f"All tools have been enabled for {persona.get_name()}.", True
        elif mode == 'none':
            persona.set_enabled_tools([])
            return f"All tools have been disabled for {persona.get_name()}.", True
        else:
            tools_to_set = args[1:]
            invalid_tools = [name for name in tools_to_set if name not in available_tool_names]
            if invalid_tools:
                return f"Error: The following tools are not valid: {', '.join(invalid_tools)}", False

            persona.set_enabled_tools(tools_to_set)
            return f"Enabled tools for {persona.get_name()} set to: {', '.join(tools_to_set)}", True

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

        # Extract key information from the payload for a concise summary
        model_name = last_request.get('model', 'N/A')
        config = last_request.get('config', {})
        contents = last_request.get('contents', [])
        history_count = len(contents) - 1 if contents else 0
        current_message_count = 1 if contents else 0
        total_messages = history_count + current_message_count

        # Check if a system prompt was included based on the formatting logic for Google's API
        system_prompt_included = "No"
        if contents and "### Conversation:" in contents[0].get('parts', [{}])[0].get('text', ''):
            system_prompt_included = "Yes"

        # Extract generation parameters
        temp = config.get('temperature', 'default')
        max_tokens = config.get('max_output_tokens', 'default')
        # Format tool names for readability
        tools_list = config.get('tools', [])
        tools = ", ".join(tools_list) if tools_list else "None"

        # Assemble the formatted summary string
        summary = (
            f"{persona_name}: Summary of Last API Request\n"
            f"----------------------------------------\n"
            f"Model Used: {model_name}\n"
            f"Context Sent:\n"
            f"  - Total Messages: {total_messages} ({history_count} from history + {current_message_count} current)\n"
            f"  - Memory Mode Used: {persona.get_memory_mode().name.lower()}\n"
            f"  - System Prompt Included: {system_prompt_included}\n"
            f"Generation Params:\n"
            f"  - Temperature: {temp}\n"
            f"  - Max Output Tokens: {max_tokens}\n"
            f"  - Tools Available: {tools}\n"
            f"----------------------------------------\n"
            f"Tip: Use `dump context` to see the exact history file sent to the model."
        )
        return summary, False

    def _handle_dump_context(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[Optional[str], bool]:
        """
        Generates a detailed text file containing the full context of the last API call,
        including persona configuration and the exact conversation history sent to the model.
        """
        if args:
            return "Usage: dump_context", False

        persona_name = persona.get_name()
        last_request = self.chat_system.last_api_requests.get(user_identifier, {}).get(persona_name)

        if not last_request:
            return f"{persona_name}: No previous request to analyze.", False

        # Start building the detailed context string
        output_lines = [f"--- Context Dump for {persona_name} ---"]

        # Add persona details for a comprehensive debugging view
        output_lines.append("\n--- Persona Configuration ---")
        output_lines.append(f"Model: {persona.get_model_name()}")
        output_lines.append(f"Memory Mode: {persona.get_memory_mode().name}")
        output_lines.append(f"Execution Mode: {persona.get_execution_mode().name}")
        output_lines.append(f"Context Length Setting: {persona.get_base_context_length()}")
        output_lines.append(f"Temp: {persona.get_temperature()}, Top P: {persona.get_top_p()}, Top K: {persona.get_top_k()}")

        # Add the conversation history from the API payload
        output_lines.append("\n--- Conversation History Sent to Model ---")
        contents = last_request.get('contents', [])
        if not contents:
            output_lines.append("No conversation history was sent.")
        else:
            for i, item in enumerate(contents):
                role = item.get('role', 'unknown').upper()
                # Safely access potentially nested content
                content_text = '[NO TEXT CONTENT]'
                parts = item.get('parts', [])
                if parts and isinstance(parts, list) and isinstance(parts[0], dict):
                    content_text = parts[0].get('text', content_text)

                output_lines.append(f"\n[Message {i + 1} - ROLE: {role}]")
                output_lines.append(content_text)
                output_lines.append("-" * 20)

        # Use a special prefix to signal to the Discord interface that this is a file response.
        # Format: "FILE_RESPONSE::filename.txt::file_content"
        return f"FILE_RESPONSE::{'context_dump.txt'}::{'\n'.join(output_lines)}", False

    def _handle_update_models(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[str, bool]:
        if args:
            return "Usage: update_models", False
        self.chat_system.models_available = get_model_list(update=True) or {}
        return f"Model list updated. Currently available: {json.dumps(self.chat_system.models_available, indent=2)}", False
