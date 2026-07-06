# src/message_handler.py

import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from config.global_config import (
    DEFAULT_MODEL_NAME,
    MODEL_SELECTOR_PERSONA_NAME,
    TOOL_SELECTOR_PERSONA_NAME
)

from src.persona import Persona
from src.persona_fields import cli_set_handlers, cli_what_handlers
from src.tools.composition import revalidate_persona_security
from src.utils import model_utils
from src.utils.model_utils import get_model_list

if TYPE_CHECKING:
    from src.engine import TextEngine
    from src.memory.memory_manager import MemoryManager
    from src.tools.tool_manager import ToolManager
    from src.turn_persistence import TurnPersistence

logger = logging.getLogger(__name__)


class BotLogic:
    """Dev-command layer. Takes its dependencies explicitly (DP-202) — it
    must never import or receive the ChatSystem orchestrator.

    Rebindable collaborators (`personas`, `visible_personas`, `text_engine`,
    `tool_manager`, the model catalog) are injected as zero-arg providers
    (the RequestBuilder `persona_lookup` / ConfirmationManager
    `tool_manager_lookup` pattern) so post-init rebinds on the owner stay
    visible. `turn_persistence` and `memory_manager` are stable instances.
    """

    def __init__(
            self,
            *,
            personas: Callable[[], Dict[str, Persona]],
            visible_personas: Callable[[], Dict[str, Persona]],
            text_engine: Callable[[], "TextEngine"],
            tool_manager: Callable[[], "ToolManager"],
            turn_persistence: "TurnPersistence",
            memory_manager: "MemoryManager",
            get_models_available: Callable[[], Dict[str, Any]],
            set_models_available: Callable[[Dict[str, Any]], None],
    ) -> None:
        # Providers returning the LIVE objects (call to dereference).
        self.personas = personas
        self.visible_personas = visible_personas
        self.text_engine = text_engine
        self.tool_manager = tool_manager
        # Stable instances.
        self.turn_persistence = turn_persistence
        self.memory_manager = memory_manager
        # Model catalog accessors — the catalog lives on (and is rebound by)
        # the owner; `update_models` writes back through the setter.
        self.get_models_available = get_models_available
        self.set_models_available = set_models_available
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
            'dump_history': self._handle_dump_history,
            'trust': self._handle_trust,
            'untrust': self._handle_untrust,
        }
        # Persona-field handlers come from the declarative registry
        # (src/persona_fields.py — DP-200 slice D); only handlers that need
        # ChatSystem state or an LLM call stay as bespoke methods here.
        self.what_handlers = {
            **cli_what_handlers(),
            'models': self._what_models,
            'personas': self._what_personas,
            'tools': self._what_tools,
            'security': self._what_security,
        }
        self.set_handlers = {
            **cli_set_handlers(),
            'model': self._set_model,
            'tools': self._set_tools,
        }

    async def preprocess_message(
            self,
            persona_name: str,
            user_identifier: str,
            message: str
    ) -> Optional[Dict[str, Any]]:
        # Preserve the original case of VALUE args — only the dispatch keys are
        # matched case-insensitively (lowercased at each lookup site below).
        # Blanket-lowercasing the whole message (an early-project shortcut for
        # persona-name/command matching) silently corrupted case-sensitive
        # values like `set prompt …`, `set model …`, and `set tool_policy <json>`.
        split_args: List[str] = message.split(' ')
        command: str
        args: List[str]
        try:
            command, args = split_args[0].lower(), split_args[1:]
        except IndexError:
            return None

        handler: Any = self.command_handlers.get(command)
        if not handler:
            return None

        current_persona: Optional[Persona] = self.personas().get(persona_name)
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
        # Field lists are generated from the dispatch tables (registry +
        # bespoke) so help can never go stale against what's actually wired.
        what_fields = '/'.join(self.what_handlers.keys())
        set_fields = '/'.join(self.set_handlers.keys())
        help_msg: str = ("Talk to a specific persona by starting your message with their name. \n \n"
                         "Currently active personas: \n" +
                         ', '.join(self.visible_personas().keys()) + "\n\n"
                         "Bot commands: \n"
                         "hello (start new conversation), \n"
                         "goodbye (end conversation), \n"
                         "remember <+prompt>, \n"
                         f"what {what_fields}, \n"
                         f"set {set_fields}, \n"
                         "set <provider>.<key> <value> (e.g. 'set kobold.mirostat 2', 'set kobold.rep_pen none' to clear), \n"
                         "add <persona>, \n"
                         "delete <persona>, \n"
                         "detail, \n"
                         "update_models, \n"
                         "dump_last, \n"
                         "dump_history, \n"
                         "trust <id> <reason>, \n"
                         "untrust <id> <reason>")
        return help_msg, False

    async def _query_llm_with_selection_tool(
        self,
        persona_name: str,
        tool_name: str,
        arg_name: str,
        choices: List[str],
        none_sentinel: str,
        user_query: str,
        tool_description: str,
    ) -> Optional[str]:
        """
        Run a single-shot structured selection via TextEngine + inline tool schema.
        Bypasses ChatSystem (no history, memory, or channel side-effects).
        Returns the chosen string (verbatim from `choices`) or None for no-match.
        """
        persona = self.personas().get(persona_name)
        if not persona:
            logger.warning(f"Selector persona '{persona_name}' not found")
            return None

        enum_values = list(choices) + [none_sentinel]
        selection_tool = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        arg_name: {
                            "type": "string",
                            "enum": enum_values,
                            "description": (
                                f"Exact value from the provided list, or '{none_sentinel}' if no reasonable match."
                            ),
                        }
                    },
                    "required": [arg_name],
                },
            },
        }

        prompt = (
            f"User query: {user_query}\n\n"
            f"Available options:\n" + "\n".join(f"- {c}" for c in choices) + "\n\n"
            f"The user's query is a fuzzy/abbreviated name. Match it to the closest option "
            f"even if the query is a partial name, is missing a suffix like '-it' or '-preview', "
            f"uses spaces instead of hyphens, or drops version details. "
            f"Only use '{none_sentinel}' if the query is completely unrelated to every option.\n\n"
            f"Call {tool_name} with your selection."
        )

        try:
            response, _ = await self.text_engine().generate_response(
                persona_config=persona.get_config_for_engine(),
                history_object={
                    "persona_prompt": persona.get_prompt(),
                    "message_history": [{"role": "user", "content": prompt}],
                    "current_message": {"text": prompt, "image_url": None},
                },
                tools=[selection_tool],
            )
        except Exception as e:
            logger.error(f"Error during LLM selection ({tool_name}): {e}", exc_info=True)
            return None

        return self._parse_selection_response(
            response, tool_name, arg_name, choices, none_sentinel
        )

    @staticmethod
    def _parse_selection_response(
        response: Dict[str, Any],
        tool_name: str,
        arg_name: str,
        choices: List[str],
        none_sentinel: str,
    ) -> Optional[str]:
        if response.get("type") != "tool_calls":
            logger.warning(f"{tool_name}: model returned text, not a tool call: {response.get('content', '')[:200]}")
            return None

        for call in response.get("calls", []):
            if call.get("name") != tool_name:
                continue
            args = call.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    logger.warning(f"{tool_name}: bad JSON args: {args!r}")
                    return None
            choice = args.get(arg_name)
            logger.debug(f"{tool_name}: model selected '{choice}'")
            if not isinstance(choice, str) or choice == none_sentinel:
                logger.info(f"{tool_name}: model returned sentinel/null (choice={choice!r})")
                return None
            for c in choices:
                if c.lower() == choice.lower():
                    return c
            logger.warning(f"{tool_name}: model returned off-list value '{choice}'")
            return None

        return None

    async def _query_llm_for_model_selection(self, user_query: str) -> Optional[str]:
        """
        Query model selector persona to find best matching model.
        Returns model name if successful, None if persona unavailable or no match.
        """
        models_list: List[str] = []
        for models in self.get_models_available().values():
            if isinstance(models, list):
                models_list.extend(models)
        if not models_list:
            return None

        return await self._query_llm_with_selection_tool(
            persona_name=MODEL_SELECTOR_PERSONA_NAME,
            tool_name="select_model",
            arg_name="model",
            choices=sorted(models_list),
            none_sentinel="DEFAULT",
            user_query=user_query,
            tool_description="Pick the model ID from the available list that best matches the user query.",
        )

    async def _query_llm_for_tool_selection(self, user_query: str, available_tools: List[str]) -> Optional[str]:
        """
        Query tool selector persona to find best matching tool name.
        Returns exact tool name if successful, None if no match.
        """
        if not available_tools:
            return None
        return await self._query_llm_with_selection_tool(
            persona_name=TOOL_SELECTOR_PERSONA_NAME,
            tool_name="select_tool",
            arg_name="tool",
            choices=available_tools,
            none_sentinel="NONE",
            user_query=user_query,
            tool_description="Pick the tool name from the available list that best matches the user query.",
        )

    def _handle_remember(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[Optional[str], bool]:
        if not args:
            return None, False
        text_to_add: str = ' '.join(args)
        persona.append_to_prompt(' ' + text_to_add)
        return f'Prompt for {persona.get_name()} updated.', True

    def _handle_add(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[Optional[str], bool]:
        if not args:
            return None, False
        # persona names are lowercase-keyed by convention (prompt text keeps case)
        new_persona_name: str = args[0].lower()

        if new_persona_name in self.personas():
            return f"Error: Persona '{new_persona_name}' already exists.", False

        prompt_args: List[str] = args[1:]
        prompt: str = ' '.join(prompt_args) if prompt_args else 'you are in character as ' + new_persona_name

        new_persona = Persona(
            persona_name=new_persona_name,
            model_name=DEFAULT_MODEL_NAME,
            prompt=prompt
        )
        self.personas()[new_persona_name] = new_persona
        return f"Added '{new_persona_name}' with prompt: '{prompt}'", True

    def _handle_delete(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[Optional[str], bool]:
        if not args:
            return None, False
        persona_to_delete: str = args[0].lower()

        if persona_to_delete not in self.personas():
            return f"Error: Persona '{persona_to_delete}' not found.", False

        del self.personas()[persona_to_delete]
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

        history_display: str
        if persona.is_in_dynamic_history():
            next_limit = persona.get_current_effective_history_messages()
            history_display = f"{next_limit} (Dynamic, will grow on next message)"
        else:
            history_display = str(persona.get_current_effective_history_messages())

        details: str = (
            f"Details for Persona: {persona.get_name()}\n"
            f"----------------------------------------\n"
            f"Model: {persona.get_model_name() or 'default'}\n"
            f"Memory Mode: {persona.get_memory_mode().name}\n"
            f"Long-term Memory: {'on' if persona.get_long_term_memory() else 'off'}\n"
            f"Include Ambient Memory: {'on' if persona.get_include_ambient_memory() else 'off'}\n"
            f"Execution Mode: {persona.get_execution_mode().name}\n"
            f"Service Bindings: {', '.join(persona.get_service_bindings()) or 'none'}\n"
            f"Enabled Tools: {tools_display}\n"
            f"History Length (Messages): {history_display}\n"
            f"Display Name in Chat: {persona.should_display_name_in_chat()}\n"
            f"Response Token Limit: {persona.get_response_token_limit() or 'default'}\n"
            f"Generation Parameters:\n"
            f"  - Temperature: {persona.get_temperature() or 'default'}\n"
            f"  - Top P: {persona.get_top_p() or 'default'}\n"
            f"  - Top K: {persona.get_top_k() or 'default'}\n"
            f"  - Thinking Level: {persona.get_thinking_level() or 'default'}\n"
            f"  - Chat Template: {persona.get_chat_template() or 'default'}\n"
            f"----------------------------------------\n"
            f"Prompt:\n{persona.get_prompt()}"
        )
        return details, False

    def _handle_what(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[Optional[str], bool]:
        if not args:
            return None, False
        sub_command: str = args[0].lower()
        handler = self.what_handlers.get(sub_command)
        if handler:
            return handler(args, persona)
        if '.' in sub_command:
            return self._what_provider_extra(sub_command, persona)
        return None, False

    @staticmethod
    def _what_provider_extra(dotted: str, persona: Persona) -> Tuple[Optional[str], bool]:
        provider, _, key = dotted.partition('.')
        if not provider or not key:
            return None, False
        value = persona.get_provider_extra(provider, key)
        if value is None:
            return f"{provider}.{key} for '{persona.get_name()}' is not set.", False
        return f"{provider}.{key} for '{persona.get_name()}' is {value!r}.", False

    def _what_models(self, args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
        all_models: Dict[str, Any] = self.get_models_available()
        if len(args) == 1:
            return f"Available model options: {json.dumps(all_models, indent=2)}", False

        if len(args) == 2:
            vendor_arg: str = args[1].lower()
            for key, models in all_models.items():
                if vendor_arg in key.lower():
                    return f"Available models from {key}: {json.dumps({key: models}, indent=2)}", False

        return None, False

    def _what_personas(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        return f"Available personas are: {list(self.visible_personas().keys())}", False

    def _what_tools(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        all_tool_defs = self.tool_manager().get_tool_definitions()
        all_tool_names = {tool['function']['name'] for tool in all_tool_defs}
        enabled_tools = persona.get_enabled_tools()

        response_lines = ["Available Tools & Status for " + persona.get_name() + ":"]
        if not all_tool_names:
            return "No tools are currently available in the system.", False

        for tool_name in sorted(list(all_tool_names)):
            status = "[ENABLED]" if enabled_tools == ['*'] or tool_name in enabled_tools else "[DISABLED]"
            response_lines.append(f"- {tool_name} {status}")

        return "\n".join(response_lines), False

    def _what_security(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        """Report whether the persona is quarantined for an insecure tool
        composition, and why. See DP-128."""
        reasons = persona.get_security_block_reasons()
        if not reasons:
            return f"Persona '{persona.get_name()}' is not security-blocked.", False
        detail = "\n".join(f" - {r}" for r in reasons)
        return (
            f"Persona '{persona.get_name()}' is QUARANTINED (insecure tool "
            f"composition); generation is blocked until fixed:\n{detail}\n"
            "Fix with `set tools <safe list>` or `set tool_policy <json>`.",
            False,
        )

    async def _handle_set(
            self,
            args: List[str],
            persona: Persona,
            user_identifier: str
    ) -> Tuple[Optional[str], bool]:
        if not args:
            return None, False

        sub_command: str = args[0].lower()
        set_handler: Any = self.set_handlers.get(sub_command)

        if set_handler:
            # DP-277: the explicit_overrides mutation is audited (operator +
            # prior/new state) — capture the prior value at the edit boundary.
            prior_overrides: Optional[List[str]] = (
                persona.get_explicit_overrides()
                if sub_command == 'explicit_overrides' else None
            )

            result: Tuple[Optional[str], bool]
            if sub_command in ('model', 'tools'):
                result = await set_handler(args, persona)
            else:
                result = set_handler(args, persona)

            # DP-128: a live tool/policy edit re-runs security validation so the
            # operator can clear (or trip) the quarantine without a restart. Done
            # here — the operator-edit boundary — rather than in the pure setters,
            # which internal callers use for many non-policy reasons.
            message, mutated = result
            if mutated and sub_command == 'explicit_overrides':
                self.memory_manager.log_audit_event(
                    event_type="explicit_overrides_change",
                    operator_id=user_identifier,
                    prior_state=json.dumps(prior_overrides),
                    new_state=json.dumps(persona.get_explicit_overrides()),
                    reason="dev command: set explicit_overrides",
                    metadata={"persona": persona.get_name()},
                )
            if mutated and sub_command in ('tools', 'tool_policy', 'explicit_overrides'):
                if revalidate_persona_security(persona):
                    reasons = "; ".join(persona.get_security_block_reasons())
                    message = (
                        f"{message}\n⚠️ Persona '{persona.get_name()}' is now QUARANTINED "
                        f"(insecure tool composition): {reasons}. Generation is blocked "
                        "until you scope its tools to a safe set."
                    )
                result = (message, mutated)
            return result

        if '.' in sub_command:
            return self._set_provider_extra(args, persona)

        return f"Error: Unknown 'set' command: {sub_command}", False

    @staticmethod
    def _set_provider_extra(args: List[str], persona: Persona) -> Tuple[Optional[str], bool]:
        """Fallback dotted-path setter: `set <provider>.<key> <value>`.
        Stores in persona.params.provider_extras[provider][key]. Phase E
        of plans/portal_engine_reintegration.md."""
        dotted = args[0].lower()  # provider/key are matched case-insensitively
        provider, _, key = dotted.partition('.')
        if not provider or not key:
            return f"Error: Invalid dotted path '{dotted}'. Use '<provider>.<key>'.", False
        if len(args) < 2:
            return f"Usage: set {dotted} <value> (or 'none' to clear).", False

        raw = args[1]
        if raw.lower() in ('none', 'null', 'clear'):
            cleared = persona.clear_provider_extra(provider, key)
            if cleared:
                return f"{provider}.{key} cleared for {persona.get_name()}.", True
            return f"{provider}.{key} was not set for {persona.get_name()}.", False

        value = BotLogic._coerce_extra_value(raw)
        persona.set_provider_extra(provider, key, value)
        return f"{provider}.{key} for {persona.get_name()} set to {value!r}.", True

    @staticmethod
    def _coerce_extra_value(raw: str) -> Any:
        """Best-effort coerce an arg string into int/float/bool, fall back to str."""
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            pass
        low = raw.lower()
        if low in ('true', 'on', 'yes'):
            return True
        if low in ('false', 'off', 'no'):
            return False
        return raw

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
        # Strip command words to avoid confusing the selector
        query_parts = [a for i, a in enumerate(args) if not (i == 0 and a.lower() == 'model')]
        model_query = " ".join(query_parts)

        selected_model = await self._query_llm_for_model_selection(model_query)

        if selected_model:
            persona.set_model_name(selected_model)
            return f"Model for {persona.get_name()} set to '{selected_model}' (fuzzy matched '{model_query}').", True

        # Fallback to default
        persona.set_model_name(DEFAULT_MODEL_NAME)
        return f"Could not find a match for '{model_query}'. Falling back to default: '{DEFAULT_MODEL_NAME}'.", True

    async def _set_tools(self, args: List[str], persona: Persona) -> Tuple[str, bool]:
        if len(args) < 2:
            return "Usage: set tools <all|none|tool_name_1> [tool_name_2]... (prefix with - to exclude)", False

        all_tool_defs = self.tool_manager().get_tool_definitions()
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
        return f"{persona.get_name()}: Goodbye! Resetting history.", True

    def _handle_dump_last(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[str, bool]:
        if args:
            return "Usage: dump_last", False

        persona_name: str = persona.get_name()
        last_request: Optional[Dict[str, Any]] = self.turn_persistence.last_api_requests.get(user_identifier, {}).get(
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

        # Safely get and format tool names — prefer full definitions, fall back to stripped names
        tools_for_llm = last_request.get('_tools_for_llm', [])
        if tools_for_llm:
            tools = ", ".join(t.get('function', {}).get('name', '?') for t in tools_for_llm)
        else:
            tools_list = config.get('tools', last_request.get('tools', []))
            if tools_list and isinstance(tools_list[0], dict):
                tools_list = [t.get('function', {}).get('name', 'unknown') for t in tools_list]
            tools = ", ".join(tools_list) if tools_list else "None"

        summary = (
            f"{persona_name}: Summary of Last API Request\n"
            f"----------------------------------------\n"
            f"Model Used: {model_name}\n"
            f"History Sent:\n"
            f"  - History Messages: {conversational_turns}\n"
            f"  - Memory Mode Used: {persona.get_memory_mode().name.lower()}\n"
            f"Generation Params:\n"
            f"  - Temperature: {temp}\n"
            f"  - Max Output Tokens: {max_tokens}\n"
            f"  - Tools Available: {tools}\n"
            f"----------------------------------------\n"
            f"Tip: Use `dump_history` to see the exact history file sent to the model."
        )
        return summary, False

        # In src/message_handler.py

    def _handle_dump_history(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[
        Optional[str], bool]:
        """
        Generates a detailed text file containing the full history/context of the last API call.

        Includes everything the LLM received: persona config, service bindings,
        tool definitions (with full schemas), system prompt, and message history.

        Returns:
            A specially formatted string that signals to the Discord interface
            to create and upload a file, or an error message.
            Format: "FILE_RESPONSE::filename.txt::file_content"
        """
        if args:
            return "Usage: dump_history", False

        persona_name = persona.get_name()
        last_request = self.turn_persistence.last_api_requests.get(user_identifier, {}).get(persona_name)

        if not last_request:
            return f"{persona_name}: No previous request to analyze.", False

        # Each entry is one LLM call in the last turn's tool loop (the messages
        # grow as tool results accumulate). Falls back to the single last
        # payload if the per-turn list isn't populated.
        iterations = (
            self.turn_persistence.last_api_iterations.get(user_identifier, {}).get(persona_name)
            or [last_request]
        )

        output_lines = [f"=== History Dump for {persona_name} ==="]
        self._dump_persona_config(output_lines, persona)
        self._dump_tools(output_lines, last_request.get('_tools_for_llm', []))

        output_lines.append(f"\n--- Tool Loop: {len(iterations)} LLM call(s) this turn ---")
        for idx, payload in enumerate(iterations):
            output_lines.append(f"\n========== LLM Call {idx + 1} of {len(iterations)} ==========")
            self._dump_api_config(output_lines, payload)
            self._dump_conversation(output_lines, payload)

        file_content = "\n".join(output_lines)
        return f"FILE_RESPONSE::history_dump.txt::{file_content}", False

    @staticmethod
    def _dump_persona_config(lines: List[str], persona: Persona) -> None:
        """Append persona configuration section to dump output."""
        lines.append("\n--- Persona Configuration ---")
        lines.append(f"Model: {persona.get_model_name()}")
        lines.append(f"Memory Mode: {persona.get_memory_mode().name}")
        lines.append(f"Execution Mode: {persona.get_execution_mode().name}")
        enabled = persona.get_enabled_tools()
        lines.append(f"Enabled Tools: {', '.join(enabled) if enabled else 'none'}")
        bindings = persona.get_service_bindings()
        lines.append(f"Service Bindings: {', '.join(bindings) if bindings else 'none'}")
        lines.append(f"History Messages Setting: {persona.get_base_history_messages()}")
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
    def _dump_api_config(lines: List[str], last_request: Dict[str, Any]) -> None:
        """Append API request config section to dump output.

        Handles both Google (nested 'config' dict) and OpenAI/Anthropic (flat top-level keys).
        """
        EXCLUDED_KEYS = {
            'tools', 'tool_config', 'tool_choice', 'safety_settings',
            'messages', 'contents', 'system', 'model', '_tools_for_llm',
        }
        config_data = last_request.get('config')
        if config_data and isinstance(config_data, dict):
            params = {k: v for k, v in config_data.items() if k not in EXCLUDED_KEYS}
        else:
            params = {k: v for k, v in last_request.items() if k not in EXCLUDED_KEYS}

        if not params:
            return
        lines.append("\n--- API Request Config ---")
        for key, value in params.items():
            lines.append(f"  {key}: {value}")

    @staticmethod
    def _dump_conversation(lines: List[str], last_request: Dict[str, Any]) -> None:
        """Append message history section to dump output."""
        lines.append("\n--- History (Messages) Sent to Model ---")
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

        # Tool result messages (OpenAI/Anthropic format: has 'content' key)
        if message.get('role') == 'tool' and 'content' in message:
            return f"[TOOL RESULT: {message.get('name', 'unknown')}]\n{message.get('content', '')}"

        content = message.get('content')

        # Direct content string (OpenAI/Anthropic)
        if isinstance(content, str) and content:
            return content

        # OpenAI/Anthropic multimodal: content is a list of typed parts
        if isinstance(content, list):
            return BotLogic._extract_multimodal_parts(content)

        # Google format: parts list (may contain text, function_call, or function_response)
        parts = message.get('parts', [])
        if parts and isinstance(parts, list):
            return BotLogic._extract_google_parts(parts)

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

    @staticmethod
    def _extract_google_parts(parts: List[Any]) -> str:
        """Extract text, tool calls, and tool results from Google-format parts."""
        segments = []
        for p in parts:
            if not isinstance(p, dict):
                continue
            if 'text' in p:
                segments.append(p['text'])
            elif 'function_call' in p:
                fc = p['function_call']
                segments.append(f"[TOOL CALL] {fc.get('name', 'unknown')}"
                                f"({json.dumps(fc.get('args', {}), indent=2)})")
            elif 'function_response' in p:
                fr = p['function_response']
                segments.append(f"[TOOL RESULT: {fr.get('name', 'unknown')}]\n"
                                f"{json.dumps(fr.get('response', {}), indent=2)}")
        return '\n'.join(segments) if segments else '[NO TEXT CONTENT]'

    def _handle_update_models(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[str, bool]:
        if args:
            return "Usage: update_models", False
        self.set_models_available(get_model_list(update=True) or {})
        return f"Model list updated. Currently available: {json.dumps(self.get_models_available(), indent=2)}", False

    def _handle_trust(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[str, bool]:
        if not args:
            return "Usage: trust <summary_id> <reason>", False
        try:
            summary_id = int(args[0])
        except ValueError:
            return f"Error: Invalid summary_id '{args[0]}'. Must be an integer.", False
        
        reason = " ".join(args[1:]) if len(args) > 1 else "No reason provided"
        
        success = self.memory_manager.mark_trusted(
            summary_id=summary_id,
            operator_id=user_identifier,
            reason=reason
        )
        
        if success:
            return f"Memory {summary_id} marked as TRUSTED. Audit log updated.", True
        else:
            return f"Error: Could not find or update memory {summary_id}.", False

    def _handle_untrust(self, args: List[str], persona: Persona, user_identifier: str) -> Tuple[str, bool]:
        if not args:
            return "Usage: untrust <summary_id> <reason>", False
        try:
            summary_id = int(args[0])
        except ValueError:
            return f"Error: Invalid summary_id '{args[0]}'. Must be an integer.", False
        
        reason = " ".join(args[1:]) if len(args) > 1 else "No reason provided"
        
        success = self.memory_manager.mark_untrusted(
            summary_id=summary_id,
            operator_id=user_identifier,
            reason=reason
        )
        
        if success:
            return f"Memory {summary_id} marked as UNTRUSTED. Audit log updated.", True
        else:
            return f"Error: Could not find or update memory {summary_id}.", False
