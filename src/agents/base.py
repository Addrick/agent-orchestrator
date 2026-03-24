# src/agents/base.py

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from src.chat_system import ChatSystem
from src.persona import Persona
from src.utils.save_utils import load_system_personas_from_file

logger = logging.getLogger(__name__)


class Agent(ABC):
    """
    Base class for scheduled agents.

    Provides:
    - System persona injection into ChatSystem
    - Common LLM context builder
    - Shortcut references to text_engine and memory_manager
    - Cooperative shutdown flag (`stopping`)

    Subclasses must implement `poll()` and may override `on_start()`
    for one-time setup (e.g. verifying external service identity).

    Scheduling is owned by AppManager, not the agent itself.
    """

    poll_interval: float = 60

    def __init__(self, chat_system: ChatSystem, inject_personas: bool = True) -> None:
        self.chat_system = chat_system
        self.text_engine = chat_system.text_engine
        self.memory_manager = chat_system.memory_manager
        self._stopping = False

        if inject_personas:
            self._inject_system_personas()

    @property
    def stopping(self) -> bool:
        """True when the agent has been asked to stop. Check in long-running loops."""
        return self._stopping

    def request_stop(self) -> None:
        """Signal the agent to finish its current work and stop."""
        self._stopping = True

    def _inject_system_personas(self) -> None:
        system_personas = load_system_personas_from_file()
        if system_personas:
            self.chat_system.personas.update(system_personas)
            logger.info(f"Injected {len(system_personas)} system personas into ChatSystem.")
        else:
            logger.warning("No system personas loaded. Agent may fail if personas are missing.")

    async def on_start(self) -> None:
        """Hook for subclass-specific startup work. Called once before the first poll."""
        pass

    @abstractmethod
    async def poll(self) -> None:
        """Called each scheduling cycle. Subclasses implement their work here."""
        ...

    @staticmethod
    def _build_llm_context(persona: Persona, prompt: str) -> Dict[str, Any]:
        """Build a minimal context object for a single-shot LLM call."""
        return {
            "persona_prompt": persona.get_prompt(),
            "history": [{"role": "user", "content": prompt}],
            "current_message": {"text": prompt, "image_url": None}
        }
