# src/agents/base.py

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

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
    - Common LLM context builder with action history injection
    - Step-level action logging
    - Shortcut references to text_engine and memory_manager

    Subclasses must implement `_poll()` and may override `_on_start()`
    for one-time setup (e.g. verifying external service identity).

    Subclasses should set `agent_name` and `action_history_limit` to
    enable action history injection into LLM prompts.
    """

    poll_interval: float = 60
    agent_name: str = ""
    action_history_limit: int = 0  # 0 = disabled

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

    # --- Step Logging ---

    def _log_step(
        self, parent_id: int, action_type: str,
        action_payload: Optional[str] = None,
        outcome: str = "success",
        outcome_payload: Optional[str] = None,
    ) -> int:
        """Log a child step under a parent task action."""
        return self.memory_manager.log_agent_action(
            agent_name=self.agent_name,
            action_type=action_type,
            action_payload=action_payload,
            outcome=outcome,
            outcome_payload=outcome_payload,
            parent_id=parent_id,
        )

    # --- LLM Context Building ---

    def _build_llm_context(
        self, persona: Persona, prompt: str,
        task_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a context object for a single-shot LLM call.

        If action_history_limit > 0 and agent_name is set, recent action
        history is prepended as a system message. Pass task_data with
        'match_contexts' (list of (type, value) tuples) to scope retrieval
        to relevant entities.
        """
        history: List[Dict[str, str]] = [{"role": "user", "content": prompt}]

        if self.action_history_limit > 0 and self.agent_name:
            history_text = self._get_action_history_message(task_data)
            if history_text:
                history.insert(0, {"role": "system", "content": history_text})

        return {
            "persona_prompt": persona.get_prompt(),
            "history": history,
            "current_message": {"text": prompt, "image_url": None}
        }

    def _get_action_history_message(
        self, task_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Retrieve and format recent action history for LLM injection."""
        match_contexts: Optional[List[Tuple[str, str]]] = None
        match_types: Optional[List[str]] = None

        if task_data:
            match_contexts = task_data.get("match_contexts")
            match_types = task_data.get("match_types")

        actions = self.memory_manager.get_relevant_agent_actions(
            agent_name=self.agent_name,
            match_contexts=match_contexts,
            match_types=match_types,
            limit=self.action_history_limit,
        )

        if not actions:
            return ""

        return self._format_action_history(actions)

    def _format_action_history(self, actions: List[Dict[str, Any]]) -> str:
        """Format action history for LLM consumption. Override for custom formatting."""
        if not actions:
            return ""

        # Reverse for chronological display (DB returns newest-first per bucket)
        actions = list(reversed(actions))

        lines = [f"--- RECENT ACTIONS ({self.agent_name}) ---"]
        for i, action in enumerate(actions, 1):
            ts = action.get("timestamp", "?")
            if hasattr(ts, "strftime"):
                ts = ts.strftime("%Y-%m-%d %H:%M")
            else:
                ts = str(ts)[:16]

            action_type = action.get("action_type", "?").upper()
            trigger = action.get("trigger_context", "")
            outcome = action.get("outcome", "?")

            line = f"{i}. [{ts}] {action_type} {trigger} -> {outcome}"

            # Add compact payload summary
            payload_summary = self._summarize_payload(action.get("outcome_payload"))
            if payload_summary:
                line += f" ({payload_summary})"

            # For failed actions, show which step failed
            if outcome in ("failed", "error", "notification_failed"):
                failed_step = self._get_failed_step_name(action.get("id"))
                if failed_step:
                    line += f" [failed at: {failed_step}]"

            lines.append(line)

        lines.append("---")
        return "\n".join(lines)

    def _summarize_payload(self, payload: Optional[str], max_len: int = 120) -> str:
        """Extract a compact summary from an outcome_payload string."""
        if not payload:
            return ""
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                # Extract key fields for a compact summary
                parts = []
                for key in ("priority", "channel", "sent", "reason"):
                    if key in data:
                        parts.append(f"{key}={data[key]}")
                if parts:
                    return ", ".join(parts)
                # Fallback: show first few keys
                summary = json.dumps(data, default=str)
                if len(summary) > max_len:
                    return summary[:max_len] + "..."
                return summary
        except (json.JSONDecodeError, TypeError):
            pass
        # Plain string fallback
        if len(payload) > max_len:
            return payload[:max_len] + "..."
        return payload

    def _get_failed_step_name(self, action_id: Optional[int]) -> str:
        """Look up child steps to find which one failed."""
        if action_id is None:
            return ""
        steps = self.memory_manager.get_action_steps(action_id)
        for step in steps:
            if step.get("outcome") in ("failed", "error"):
                result: str = step.get("action_type", "unknown")
                return result
        return ""
