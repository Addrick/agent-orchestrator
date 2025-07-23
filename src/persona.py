import logging
from typing import Optional, Dict, Any

from config import global_config

logger = logging.getLogger(__name__)


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
            top_k: Optional[int] = None
    ) -> None:
        self._name: str = persona_name
        self._model_name: str = model_name
        self._prompt: str = prompt
        self._response_token_limit: Optional[int] = token_limit
        self._context_length: Optional[int] = context_length

        # Model-specific generation parameters
        self._temperature: Optional[float] = temperature
        self._top_p: Optional[float] = top_p
        self._top_k: Optional[int] = top_k

    # --- Getters ---

    def get_name(self) -> str:
        return self._name

    def get_model_name(self) -> str:
        return self._model_name

    def get_prompt(self) -> str:
        return self._prompt

    def get_response_token_limit(self) -> Optional[int]:
        return self._response_token_limit

    def get_context_length(self) -> Optional[int]:
        return self._context_length

    def get_temperature(self) -> Optional[float]:
        return self._temperature

    def get_top_p(self) -> Optional[float]:
        return self._top_p

    def get_top_k(self) -> Optional[int]:
        return self._top_k

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
            logger.info(f"Invalid token limit provided: '{new_limit}'. Must be an integer. Setting to None.")
        return self._response_token_limit

    def set_context_length(self, new_length: Any) -> Optional[int]:
        """
        Sets the context length. Returns the integer value if successful,
        or None if the input is invalid (in which case the length is also set to None).
        """
        try:
            self._context_length = int(new_length)
            logger.info(f"Persona '{self._name}' context length set to {self._context_length}.")
        except (ValueError, TypeError):
            self._context_length = None
            logger.info(f"Invalid context length provided: '{new_length}'. Must be an integer. Setting to None.")
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