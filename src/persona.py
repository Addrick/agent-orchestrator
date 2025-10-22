# src/persona.py

import logging
from enum import Enum, auto
from typing import Optional, Dict, Any, List

from config import global_config

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Defines the autonomy level for a persona's tool-use capabilities."""
    SILENT_ANALYSIS = auto()
    ASSISTED_DISPATCH = auto()


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
            token_limit: Optional[int] = global_config.DEFAULT_TOKEN_LIMIT,
            context_length: Optional[int] = global_config.DEFAULT_CONTEXT_LIMIT,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            top_k: Optional[int] = None,
            display_name_in_chat: bool = False,
            execution_mode: ExecutionMode = ExecutionMode.SILENT_ANALYSIS,
            enabled_tools: Optional[List[str]] = None
    ) -> None:
        self._name: str = persona_name
        self._model_name: str = model_name
        self._prompt: str = prompt
        self._response_token_limit: Optional[int] = token_limit
        self._context_length: int = context_length if context_length is not None else global_config.DEFAULT_CONTEXT_LIMIT
        self._execution_mode: ExecutionMode = execution_mode
        self._enabled_tools: List[str] = enabled_tools if enabled_tools is not None else []
        self._temp_context_override: Optional[int] = None

        # Model-specific generation parameters
        self._temperature: Optional[float] = temperature
        self._top_p: Optional[float] = top_p
        self._top_k: Optional[int] = top_k
        self._display_name_in_chat: bool = display_name_in_chat

    # --- Getters ---

    def get_name(self) -> str:
        return self._name

    def get_model_name(self) -> str:
        return self._model_name

    def get_prompt(self) -> str:
        return self._prompt

    def get_response_token_limit(self) -> Optional[int]:
        return self._response_token_limit

    def get_context_length(self) -> int:
        """
        Returns the effective context length. If a temporary override is active
        (from a 'hello' command), it returns the override value and increments it
        for the next turn. Otherwise, it returns the default context length.
        """
        if self._temp_context_override is not None:
            current_limit = self._temp_context_override
            # Increment by 2 for the user message and the assistant's reply.
            self._temp_context_override += 2
            return current_limit

        return self._context_length

    def get_temperature(self) -> Optional[float]:
        return self._temperature

    def get_top_p(self) -> Optional[float]:
        return self._top_p

    def get_top_k(self) -> Optional[int]:
        return self._top_k

    def should_display_name_in_chat(self) -> bool:
        return self._display_name_in_chat

    def get_execution_mode(self) -> ExecutionMode:
        return self._execution_mode

    def get_enabled_tools(self) -> List[str]:
        """Returns the list of tool names this persona is allowed to use."""
        return self._enabled_tools

    # --- Setters ---

    def set_model_name(self, new_model_name: str) -> None:
        """Sets the model name for the persona."""
        self._model_name = str(new_model_name)
        logger.info(f"Persona '{self._name}' model set to {self._model_name}.")

    def set_prompt(self, new_prompt: str) -> None:
        """Sets the persona's base prompt."""
        self._prompt = str(new_prompt)
        logger.info(f"Persona '{self._name}' prompt has been updated.")

    def set_response_token_limit(self, new_limit: Any) -> Optional[int]:
        """
        Sets the response token limit. Returns the integer value if successful,
        or None if the input is invalid (in which case the limit is also set to None).
        """
        try:
            self._response_token_limit = int(new_limit)
            if self._response_token_limit < 100:
                self._response_token_limit = 100
                logger.warning(f"Warning: very low token response limit received, setting value to 100.")
            logger.info(f"Persona '{self._name}' response token limit set to {self._response_token_limit}.")
        except (ValueError, TypeError):
            self._response_token_limit = None
            logger.info(
                f"Non-integer token limit provided: '{new_limit}'. No limit set (this will use provider default).")
        return self._response_token_limit

    def set_context_length(self, new_length: Any) -> int:
        """
        Sets the static default context length and disables any active dynamic context override.
        """
        self.end_new_conversation()  # Ensure dynamic mode is off when setting a static length.
        try:
            self._context_length = int(new_length)
            logger.info(f"Persona '{self._name}' context length set to {self._context_length}.")
        except (ValueError, TypeError):
            self._context_length = global_config.DEFAULT_CONTEXT_LIMIT
            logger.info(
                f"Invalid context length provided: '{new_length}'. Setting to default value: {self._context_length}.")
        return self._context_length

    def set_temperature(self, new_temp: Any) -> Optional[float]:
        """
        Sets the temperature. Returns the float value if successful,
        or None if the input is invalid (in which case the temperature is also set to None).
        """
        try:
            self._temperature = float(new_temp)
            logger.info(f"Persona '{self._name}' temperature set to {self._temperature}.")
        except (ValueError, TypeError):
            self._temperature = None
            logger.info(f"Invalid temperature value provided: '{new_temp}'. Must be a number. Setting to None.")
        return self._temperature

    def set_top_p(self, new_top_p: Any) -> Optional[float]:
        """
        Sets top_p. Returns the float value if successful,
        or None if the input is invalid (in which case top_p is also set to None).
        """
        try:
            self._top_p = float(new_top_p)
            logger.info(f"Persona '{self._name}' top_p set to {self._top_p}.")
        except (ValueError, TypeError):
            self._top_p = None
            logger.info(f"Invalid top_p value provided: '{new_top_p}'. Must be a number. Setting to None.")
        return self._top_p

    def set_top_k(self, new_top_k: Any) -> Optional[int]:
        """
        Sets top_k. Returns the integer value if successful,
        or None if the input is invalid (in which case top_k is also set to None).
        """
        try:
            self._top_k = int(new_top_k)
            logger.info(f"Persona '{self._name}' top_k set to {self._top_k}.")
        except (ValueError, TypeError):
            self._top_k = None
            logger.info(f"Invalid top_k value provided: '{new_top_k}'. Must be an integer. Setting to None.")
        return self._top_k

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

    def set_enabled_tools(self, new_tools: List[str]) -> None:
        """Sets the list of tools the persona is allowed to use."""
        if not isinstance(new_tools, list):
            logger.warning(f"Invalid type for enabled tools: {type(new_tools)}. Must be a list. No change made.")
            return
        self._enabled_tools = new_tools
        logger.info(f"Persona '{self._name}' enabled tools set to: {self._enabled_tools}")

    # --- Conversation State Methods ---

    def start_new_conversation(self, start_value: int = 0) -> None:
        """Initiates a 'fresh start' mode by setting a temporary context override."""
        self._temp_context_override = start_value
        logger.info(f"Persona '{self._name}' starting new conversation with temporary context at size {start_value}.")

    def end_new_conversation(self) -> None:
        """Ends the 'fresh start' mode and reverts to the default context length."""
        if self.is_in_dynamic_context():
            self._temp_context_override = None
            logger.info(f"Persona '{self._name}' ending temporary context, reverting to default.")

    def is_in_dynamic_context(self) -> bool:
        """Returns True if the persona is in a temporary, dynamic context conversation."""
        return self._temp_context_override is not None

    def get_current_effective_context_length(self) -> int:
        """
        Returns the next context value that will be used, without incrementing the counter.
        Useful for inspecting state with the 'detail' command.
        """
        if self._temp_context_override is not None:
            return self._temp_context_override
        return self._context_length

    # --- Utility Methods ---

    def append_to_prompt(self, message: str) -> None:
        """Appends text to the persona's base prompt."""
        self._prompt += message

    def get_config_for_engine(self) -> Dict[str, Any]:
        """Returns a dictionary of the current generation parameters for the TextEngine."""
        return {
            "model_name": self._model_name,
            "max_output_tokens": self._response_token_limit,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
        }
