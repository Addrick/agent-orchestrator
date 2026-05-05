# src/persona.py

import logging
from enum import Enum, auto
from typing import Optional, Dict, Any, List, Type, TypeVar, Union

from config import global_config
from src.generation_params import GenerationParams
from src.tools.policy import ToolPolicy

logger = logging.getLogger(__name__)

E = TypeVar('E', bound=Enum)


class ExecutionMode(Enum):
    """Defines the autonomy level for a persona's tool-use capabilities."""
    AUTONOMOUS = auto()       # Execute tools immediately
    CONFIRM = auto()          # Present write-tools for user approval before executing


class MemoryMode(Enum):
    """Defines the strategy for retrieving conversation history."""
    CHANNEL_ISOLATED = auto()
    SERVER_WIDE = auto()
    PERSONAL = auto()
    GLOBAL = auto()
    TICKET_ISOLATED = auto()


class Persona:
    """
    A data class to hold settings and state for a specific LLM persona.
    Attributes are managed via getter and setter methods for robust control.
    """

    def __init__(
            self,
            persona_name: str,
            model_name: str,
            prompt: str,
            token_limit: Optional[int] = None,
            history_messages: Optional[int] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            top_k: Optional[int] = None,
            display_name_in_chat: bool = False,
            execution_mode: Any = ExecutionMode.AUTONOMOUS,
            enabled_tools: Optional[List[str]] = None,
            memory_mode: Any = MemoryMode.CHANNEL_ISOLATED,
            service_bindings: Optional[List[str]] = None,
            include_ambient_memory: bool = True,
            thinking_level: Optional[str] = None,
            long_term_memory: bool = True,
            max_context_tokens: Optional[int] = None,
            params: Any = None,
            chat_template: Optional[str] = None,
            tool_policy: Optional[Union[Dict[str, Any], ToolPolicy]] = None,
            **kwargs: Any,
    ) -> None:
        self._name: str = persona_name
        self._model_name: str = model_name
        self._prompt: str = prompt

        # Generation params: prefer the structured `params` dict/object when
        # present (new save shape), otherwise start fresh from defaults.
        # Flat kwargs (temperature/top_p/top_k/token_limit) override on top so
        # legacy callers and per-field overrides keep working. Phase A facade
        # — see src/generation_params.py.
        if isinstance(params, GenerationParams):
            self._params: GenerationParams = params
        elif isinstance(params, dict):
            self._params = GenerationParams.from_dict(params)
        else:
            self._params = GenerationParams()
        if temperature is not None:
            self._params.temperature = temperature
        if top_p is not None:
            self._params.top_p = top_p
        if top_k is not None:
            self._params.top_k = top_k

        self._set_and_sanitize_token_limit(
            token_limit if token_limit is not None else self._params.max_tokens
        )

        # Handle legacy context_length from tests or old configs
        effective_history = history_messages
        if effective_history is None:
            effective_history = kwargs.get("context_length")
        if effective_history is None:
            effective_history = global_config.DEFAULT_HISTORY_MESSAGES

        self._history_messages: int = int(effective_history)
        self._execution_mode: ExecutionMode = self._resolve_enum(
            ExecutionMode, execution_mode, ExecutionMode.AUTONOMOUS)
        self._enabled_tools: List[str] = enabled_tools if enabled_tools is not None else []
        self._memory_mode: MemoryMode = self._resolve_enum(
            MemoryMode, memory_mode, MemoryMode.CHANNEL_ISOLATED)
        self._temp_history_override: Optional[int] = None

        self._display_name_in_chat: bool = display_name_in_chat
        self._service_bindings: List[str] = service_bindings if service_bindings is not None else []
        self._include_ambient_memory: bool = include_ambient_memory
        self._thinking_level: Optional[str] = thinking_level
        self._long_term_memory: bool = long_term_memory
        self._chat_template: Optional[str] = chat_template if chat_template else None

        try:
            self._max_context_tokens: int = int(max_context_tokens) if max_context_tokens is not None else global_config.DEFAULT_MAX_CONTEXT_TOKENS
        except (ValueError, TypeError):
            self._max_context_tokens = global_config.DEFAULT_MAX_CONTEXT_TOKENS

        if isinstance(tool_policy, ToolPolicy):
            self._tool_policy = tool_policy
        elif isinstance(tool_policy, dict):
            self._tool_policy = ToolPolicy.from_dict(tool_policy)
        else:
            self._tool_policy = ToolPolicy.from_legacy_list(self._enabled_tools)

    # --- Getters ---

    def get_name(self) -> str:
        return self._name

    def get_model_name(self) -> str:
        return self._model_name

    def get_prompt(self) -> str:
        return self._prompt

    def get_response_token_limit(self) -> int:
        # _set_and_sanitize_token_limit guarantees max_tokens is always int.
        assert self._params.max_tokens is not None
        return self._params.max_tokens

    def get_generation_params(self) -> GenerationParams:
        """Returns the underlying structured GenerationParams. Phase A seam
        for Section B providers (stream_messages / stream_prompt)."""
        return self._params

    def get_history_messages(self) -> int:
        """
        Returns the effective history message count.
        If a temporary override is active (from a 'hello' command), it returns
        the override value and increments it for the next turn.
        """
        if self._temp_history_override is not None:
            current_limit = self._temp_history_override
            # Increment by 2 for the user message and the assistant's reply.
            self._temp_history_override += 2
            return current_limit

        return self._history_messages

    def get_context_length(self) -> int:
        """Legacy alias for get_history_messages."""
        return self.get_history_messages()

    def get_base_history_messages(self) -> int:
        """Returns the persona's static, default history message count."""
        return self._history_messages

    def get_base_context_length(self) -> int:
        """Legacy alias for get_base_history_messages."""
        return self.get_base_history_messages()

    def get_temperature(self) -> Optional[float]:
        return self._params.temperature

    def get_top_p(self) -> Optional[float]:
        return self._params.top_p

    def get_top_k(self) -> Optional[int]:
        return self._params.top_k

    def should_display_name_in_chat(self) -> bool:
        return self._display_name_in_chat

    def get_execution_mode(self) -> ExecutionMode:
        return self._execution_mode

    def get_enabled_tools(self) -> List[str]:
        """Returns the list of tool names this persona is allowed to use."""
        if self._tool_policy.default == "allow" and "*" in self._tool_policy.allow:
            return ["*"]
        # Combine allowed and ask tools for the engine to consider both
        return sorted(list(set(self._tool_policy.allow + self._tool_policy.ask)))

    def get_tool_policy(self) -> ToolPolicy:
        """Returns the persona's structured tool security policy."""
        return self._tool_policy

    def get_service_bindings(self) -> List[str]:
        """Returns the list of service integrations this persona is bound to."""
        return self._service_bindings

    def get_include_ambient_memory(self) -> bool:
        """Whether to include ambient channel memories in long-term memory retrieval."""
        return self._include_ambient_memory

    def get_long_term_memory(self) -> bool:
        """Whether long-term memory retrieval is enabled for this persona."""
        return self._long_term_memory

    def get_thinking_level(self) -> Optional[str]:
        """Returns the thinking level override for extended thinking models (e.g. 'minimal')."""
        return self._thinking_level

    def get_chat_template(self) -> Optional[str]:
        """Returns the instruct template name used when rendering prompts for local inference.

        Maps to StreamEngine.CHAT_TEMPLATES keys: 'chatml', 'gemma', 'llama3', 'alpaca'.
        None means fall back to KOBOLD_CHAT_TEMPLATE env/config or 'chatml'.
        """
        return self._chat_template

    def get_memory_mode(self) -> MemoryMode:
        """Returns the persona's current memory retrieval strategy."""
        return self._memory_mode

    def get_max_context_tokens(self) -> int:
        """Total ctx budget (prompt + reserved response). Matches kobold-lite's
        localsettings.max_context_length semantic — see context_budget.py."""
        return self._max_context_tokens

    def get_provider_extra(self, provider: str, key: str) -> Any:
        """Read a single provider-specific knob from `provider_extras[provider][key]`."""
        return self._params.provider_extras.get(provider, {}).get(key)

    def set_provider_extra(self, provider: str, key: str, value: Any) -> None:
        """Write a single provider-specific knob into `provider_extras[provider][key]`.
        Phase E dotted-path setter (see plans/portal_engine_reintegration.md)."""
        block = self._params.provider_extras.setdefault(provider, {})
        block[key] = value
        logger.info(f"Persona '{self._name}' provider_extras[{provider}][{key}] set to {value!r}.")

    def clear_provider_extra(self, provider: str, key: str) -> bool:
        """Remove `provider_extras[provider][key]`. Returns True if it existed."""
        block = self._params.provider_extras.get(provider)
        if not block or key not in block:
            return False
        del block[key]
        if not block:
            del self._params.provider_extras[provider]
        logger.info(f"Persona '{self._name}' provider_extras[{provider}][{key}] cleared.")
        return True

    # --- Private Helpers ---

    @staticmethod
    def _resolve_enum(enum_class: Type[E], value: Any, default: E) -> E:
        """Accepts a string or enum member, returns a valid enum member or the default."""
        if isinstance(value, enum_class):
            return value
        if isinstance(value, str):
            try:
                return enum_class[value.upper()]
            except KeyError:
                logger.warning(f"Invalid {enum_class.__name__} '{value}'. Defaulting to {default.name}.")
        return default

    # --- Setters ---

    def _set_and_sanitize_token_limit(self, new_limit: Any) -> None:
        """
        Private method to handle the core logic of setting the token limit. No logging.
        """
        try:
            parsed_limit = int(new_limit)
            if parsed_limit < 100:
                self._params.max_tokens = 100
                logger.debug(f"Warning: low token limit {parsed_limit} provided, clamping to 100.")
            else:
                self._params.max_tokens = parsed_limit
        except (ValueError, TypeError):
            self._params.max_tokens = global_config.DEFAULT_TOKEN_LIMIT

    def set_response_token_limit(self, new_limit: Any) -> int:
        """
        Public setter for token limit. Logs the change and returns the final value.
        """
        original_value = self._params.max_tokens
        self._set_and_sanitize_token_limit(new_limit)
        if self._params.max_tokens != original_value:
            logger.info(f"Persona '{self._name}' response token limit set to {self._params.max_tokens}.")
        else:
            logger.info(
                f"Invalid or no token limit provided: '{new_limit}'. Using value: {self._params.max_tokens}.")
        assert self._params.max_tokens is not None
        return self._params.max_tokens

    def set_model_name(self, new_model_name: str) -> None:
        """Sets the model name for the persona."""
        self._model_name = str(new_model_name)
        logger.info(f"Persona '{self._name}' model set to {self._model_name}.")

    def set_prompt(self, new_prompt: str) -> None:
        """Sets the persona's base prompt."""
        self._prompt = str(new_prompt)
        logger.info(f"Persona '{self._name}' prompt has been updated.")

    def set_history_messages(self, new_length: Any) -> int:
        """
        Sets the static default history message count and disables any active dynamic override.
        """
        self.end_new_conversation()  # Ensure dynamic mode is off when setting a static length.
        try:
            self._history_messages = int(new_length)
            logger.info(f"Persona '{self._name}' history messages set to {self._history_messages}.")
        except (ValueError, TypeError):
            self._history_messages = global_config.DEFAULT_HISTORY_MESSAGES
            logger.info(
                f"Invalid history length provided: '{new_length}'. Setting to default value: {self._history_messages}.")
        return self._history_messages

    def set_context_length(self, new_length: Any) -> int:
        """Legacy alias for set_history_messages."""
        return self.set_history_messages(new_length)

    def set_temperature(self, new_temp: Any) -> Optional[float]:
        """
        Sets the temperature. Returns the float value if successful,
        or None if the input is invalid (in which case the temperature is also set to None).
        """
        try:
            self._params.temperature = float(new_temp)
            logger.info(f"Persona '{self._name}' temperature set to {self._params.temperature}.")
        except (ValueError, TypeError):
            self._params.temperature = None
            logger.info(f"Invalid temperature value provided: '{new_temp}'. Must be a number. Setting to None.")
        return self._params.temperature

    def set_top_p(self, new_top_p: Any) -> Optional[float]:
        """
        Sets top_p. Returns the float value if successful,
        or None if the input is invalid (in which case top_p is also set to None).
        """
        try:
            self._params.top_p = float(new_top_p)
            logger.info(f"Persona '{self._name}' top_p set to {self._params.top_p}.")
        except (ValueError, TypeError):
            self._params.top_p = None
            logger.info(f"Invalid top_p value provided: '{new_top_p}'. Must be a number. Setting to None.")
        return self._params.top_p

    def set_top_k(self, new_top_k: Any) -> Optional[int]:
        """
        Sets top_k. Returns the integer value if successful,
        or None if the input is invalid (in which case top_k is also set to None).
        """
        try:
            self._params.top_k = int(new_top_k)
            logger.info(f"Persona '{self._name}' top_k set to {self._params.top_k}.")
        except (ValueError, TypeError):
            self._params.top_k = None
            logger.info(f"Invalid top_k value provided: '{new_top_k}'. Must be an integer. Setting to None.")
        return self._params.top_k

    def set_display_name_in_chat(self, new_value: bool) -> None:
        """Sets whether the persona's name should be displayed in chat replies."""
        self._display_name_in_chat = new_value
        logger.info(f"Persona '{self._name}' display_name_in_chat set to {new_value}.")

    def set_execution_mode(self, new_mode: Any) -> None:
        """Sets the execution mode from a string or an ExecutionMode member."""
        if isinstance(new_mode, ExecutionMode):
            self._execution_mode = new_mode
        elif isinstance(new_mode, str):
            try:
                self._execution_mode = ExecutionMode[new_mode.upper()]
            except KeyError:
                logger.warning(f"Invalid execution mode string: '{new_mode}'. No change made.")
                return
        else:
            logger.warning(f"Invalid type for execution mode: {type(new_mode)}. No change made.")
            return
        logger.info(f"Persona '{self._name}' execution mode set to {self._execution_mode.name}.")

    def set_include_ambient_memory(self, value: bool) -> None:
        """Sets whether ambient channel memories are included in long-term retrieval."""
        self._include_ambient_memory = value
        logger.info(f"Persona '{self._name}' include_ambient_memory set to {value}.")

    def set_long_term_memory(self, value: bool) -> None:
        """Enables or disables long-term memory retrieval for this persona."""
        self._long_term_memory = value
        logger.info(f"Persona '{self._name}' long_term_memory set to {value}.")

    def set_thinking_level(self, value: Optional[str]) -> None:
        """Sets the thinking level for extended thinking models (e.g. 'minimal', None to clear)."""
        self._thinking_level = value
        logger.info(f"Persona '{self._name}' thinking_level set to {value}.")

    def set_chat_template(self, value: Optional[str]) -> None:
        """Sets the instruct template name for local inference prompt rendering.

        Accepted values: 'chatml', 'gemma', 'llama3', 'alpaca', or None to clear.
        Invalid/unknown values are accepted and will fall back at render time.
        """
        self._chat_template = value if value else None
        logger.info(f"Persona '{self._name}' chat_template set to {value!r}.")

    def set_service_bindings(self, bindings: List[str]) -> None:
        """Sets the list of service integrations this persona is bound to."""
        self._service_bindings = bindings
        logger.info(f"Persona '{self._name}' service_bindings set to {self._service_bindings}.")

    def set_enabled_tools(self, new_tools: List[str]) -> None:
        """Sets the list of tools the persona is allowed to use, updating the policy."""
        self._enabled_tools = new_tools
        self._tool_policy = ToolPolicy.from_legacy_list(new_tools)
        logger.info(f"Persona '{self._name}' enabled tools set to: {self._enabled_tools}")

    def set_tool_policy(self, policy: Union[Dict[str, Any], ToolPolicy]) -> None:
        """Sets the structured tool security policy."""
        if isinstance(policy, dict):
            self._tool_policy = ToolPolicy.from_dict(policy)
        else:
            self._tool_policy = policy
        # Update legacy list for compatibility
        self._enabled_tools = self._tool_policy.allow
        logger.info(f"Persona '{self._name}' tool policy updated.")

    def set_max_context_tokens(self, new_value: Any) -> int:
        """Sets the total context budget. Falls back to default on invalid input."""
        try:
            parsed = int(new_value)
            if parsed < 100:
                logger.warning(f"max_context_tokens {parsed} too low; clamping to 100.")
                parsed = 100
            self._max_context_tokens = parsed
            logger.info(f"Persona '{self._name}' max_context_tokens set to {self._max_context_tokens}.")
        except (ValueError, TypeError):
            self._max_context_tokens = global_config.DEFAULT_MAX_CONTEXT_TOKENS
            logger.info(
                f"Invalid max_context_tokens '{new_value}'. Using default: {self._max_context_tokens}.")
        return self._max_context_tokens

    def set_memory_mode(self, new_mode: Any) -> None:
        """Sets the memory retrieval strategy from a string or a MemoryMode member."""
        if isinstance(new_mode, MemoryMode):
            self._memory_mode = new_mode
        elif isinstance(new_mode, str):
            try:
                self._memory_mode = MemoryMode[new_mode.upper()]
            except KeyError:
                logger.warning(f"Invalid memory mode string: '{new_mode}'. No change made.")
                return
        else:
            logger.warning(f"Invalid type for memory mode: {type(new_mode)}. No change made.")
            return
        logger.info(f"Persona '{self._name}' memory mode set to {self._memory_mode.name}.")

    # --- Conversation State Methods ---

    def start_new_conversation(self, start_value: int = 0) -> None:
        """Initiates a 'fresh start' mode by setting a temporary history override."""
        self._temp_history_override = start_value
        logger.info(f"Persona '{self._name}' starting new conversation with temporary history at size {start_value}.")

    def end_new_conversation(self) -> None:
        """Ends the 'fresh start' mode and reverts to the default history length."""
        if self.is_in_dynamic_history():
            self._temp_history_override = None
            logger.info(f"Persona '{self._name}' ending temporary history, reverting to default.")

    def is_in_dynamic_history(self) -> bool:
        """Returns True if the persona is in a temporary, dynamic history conversation."""
        return self._temp_history_override is not None

    def is_in_dynamic_context(self) -> bool:
        """Legacy alias for is_in_dynamic_history."""
        return self.is_in_dynamic_history()

    def get_current_effective_history_messages(self) -> int:
        """
        Returns the next history value that will be used, without incrementing the counter.
        Useful for inspecting state with the 'detail' command.
        """
        if self._temp_history_override is not None:
            return self._temp_history_override
        return self._history_messages

    def get_current_effective_context_length(self) -> int:
        """Legacy alias for get_current_effective_history_messages."""
        return self.get_current_effective_history_messages()

    # --- Utility Methods ---

    def append_to_prompt(self, message: str) -> None:
        """Appends text to the persona's base prompt."""
        self._prompt += message

    def get_config_for_engine(self) -> Dict[str, Any]:
        """Returns a dictionary of the current generation parameters for the TextEngine."""
        config: Dict[str, Any] = {
            "model_name": self._model_name,
            "max_output_tokens": self._params.max_tokens,
            "temperature": self._params.temperature,
            "top_p": self._params.top_p,
            "top_k": self._params.top_k,
        }
        if self._thinking_level is not None:
            config["thinking_level"] = self._thinking_level
        if self._chat_template is not None:
            config["chat_template"] = self._chat_template
        config["max_context_tokens"] = self._max_context_tokens
        if self._params.provider_extras:
            config["provider_extras"] = {
                k: dict(v) for k, v in self._params.provider_extras.items()
            }
        return config
