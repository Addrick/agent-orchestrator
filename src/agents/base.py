# src/agents/base.py

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from src.chat_system import ChatSystem
from src.persona import Persona
from src.utils.save_utils import load_system_personas_from_file

logger = logging.getLogger(__name__)


class AgentLoop(ABC):
    """
    Base class for polling-based agent loops.

    Provides:
    - Async polling loop with graceful shutdown
    - System persona injection into ChatSystem
    - Common LLM context builder
    - Shortcut references to text_engine and memory_manager

    Subclasses must implement `_poll()` and may override `_on_start()`
    for one-time setup (e.g. verifying external service identity).
    """

    poll_interval: float = 60

    def __init__(self, chat_system: ChatSystem, inject_personas: bool = True) -> None:
        self.chat_system = chat_system
        self.text_engine = chat_system.text_engine
        self.memory_manager = chat_system.memory_manager
        self._shutdown_event = asyncio.Event()

        if inject_personas:
            self._inject_system_personas()

    def _inject_system_personas(self) -> None:
        system_personas = load_system_personas_from_file()
        if system_personas:
            self.chat_system.personas.update(system_personas)
            logger.info(f"Injected {len(system_personas)} system personas into ChatSystem.")
        else:
            logger.warning("No system personas loaded. Agent may fail if personas are missing.")

    async def start(self) -> None:
        """Run the polling loop until stop() is called."""
        logger.info(f"{self.__class__.__name__} started.")
        await self._on_start()

        while not self._shutdown_event.is_set():
            try:
                await self._poll()
            except Exception as e:
                logger.error(f"Error in {self.__class__.__name__} polling loop: {e}", exc_info=True)

            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=self.poll_interval)
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        """Signal the agent to stop after the current poll cycle."""
        self._shutdown_event.set()

    async def _on_start(self) -> None:
        """Hook for subclass-specific startup work. Called once before the first poll."""
        pass

    @abstractmethod
    async def _poll(self) -> None:
        """Called each polling cycle. Subclasses implement their work here."""
        ...

    @staticmethod
    def _build_llm_context(persona: Persona, prompt: str) -> Dict[str, Any]:
        """Build a minimal context object for a single-shot LLM call."""
        return {
            "persona_prompt": persona.get_prompt(),
            "history": [{"role": "user", "content": prompt}],
            "current_message": {"text": prompt, "image_url": None}
        }
