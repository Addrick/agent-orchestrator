# src/agents/base.py

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, cast

from src.chat_system import ChatSystem
from src.persona import Persona
from src.personas.store import load_system_personas_from_file
from src.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class Agent(ABC):
    """
    Base class for all agents.

    Provides:
    - Schedule-driven async execution loop with graceful shutdown
    - System persona injection into ChatSystem
    - Common LLM context builder with action history injection
    - Step-level action logging
    - Observable status properties for external monitoring

    Subclasses must implement `deploy()` and may override `_on_start()`
    for one-time setup (e.g. verifying external service identity).

    Subclasses should set `agent_name` and `action_history_limit` to
    enable action history injection into LLM prompts.
    """

    agent_name: str = ""
    action_history_limit: int = 0  # 0 = disabled
    schedule: Dict[str, Any] = {}  # e.g. {"interval": 30}

    # Hindsight bridging (DP-116b). Both default to agent_name when unset.
    # Subclasses override `experience_bank` to mingle agent series into a
    # human-facing persona bank (e.g. dispatch -> "dispatch_analyst").
    experience_bank: Optional[str] = None
    experience_persona: Optional[str] = None

    def __init__(self, chat_system: ChatSystem, inject_personas: bool = True) -> None:
        self.chat_system = chat_system
        self.text_engine = chat_system.text_engine
        self.memory_manager: MemoryManager = chat_system.memory_manager
        self._shutdown_event = asyncio.Event()

        # Observable status properties — read by AgentManager or any monitor
        self.started_at: Optional[datetime] = None
        self.last_deploy_time: Optional[datetime] = None
        self.deploy_count: int = 0
        self.error_count: int = 0
        self.consecutive_errors: int = 0
        self.last_error: Optional[str] = None

        if inject_personas:
            self._inject_system_personas()

    def _inject_system_personas(self) -> None:
        system_personas = load_system_personas_from_file()
        if system_personas:
            self.chat_system.personas.update(system_personas)
            self.chat_system.system_persona_names.update(system_personas.keys())
            logger.info(f"Injected {len(system_personas)} system personas into ChatSystem.")
        else:
            logger.warning("No system personas loaded. Agent may fail if personas are missing.")

    @property
    def is_running(self) -> bool:
        """True if the agent has started and has not been signalled to stop."""
        return self.started_at is not None and not self._shutdown_event.is_set()

    async def start(self) -> None:
        """Run the agent loop until stop() is called."""
        logger.info(f"{self.__class__.__name__} started.")
        self.started_at = datetime.now(timezone.utc)
        await self._on_start()

        while not self._shutdown_event.is_set():
            try:
                await self.deploy()
                self.deploy_count += 1
                self.last_deploy_time = datetime.now(timezone.utc)
                self.consecutive_errors = 0
            except Exception as e:
                self.error_count += 1
                self.consecutive_errors += 1
                self.last_error = str(e)
                self.last_deploy_time = datetime.now(timezone.utc)
                logger.error(f"Error in {self.__class__.__name__} deploy: {e}", exc_info=True)

            await self._wait_for_next_run()

    def stop(self) -> None:
        """Signal the agent to stop after the current deploy cycle."""
        self._shutdown_event.set()

    async def _on_start(self) -> None:
        """Hook for subclass-specific startup work. Called once before the first deploy."""
        pass

    @abstractmethod
    async def deploy(self) -> None:
        """Execute one cycle of the agent's work. Subclasses implement this."""
        ...

    async def _wait_for_next_run(self) -> None:
        """Sleep until the next scheduled run, respecting shutdown.

        Supports:
        - {"interval": <seconds>} — fixed interval between runs
        - {"daily_at": "HH:MM"}   — run once a day at specific local time
        """
        if "daily_at" in self.schedule:
            try:
                target_time_str = self.schedule["daily_at"]
                target_hour, target_minute = map(int, target_time_str.split(':'))
                
                while not self._shutdown_event.is_set():
                    now = datetime.now() # Schedule relative to local time
                    target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
                    
                    if target <= now:
                        target += timedelta(days=1)
                    
                    wait_seconds = (target - now).total_seconds()
                    # Sleep in chunks to handle system clock changes and DST gracefully
                    sleep_time = min(wait_seconds, 3600.0)
                    
                    try:
                        await asyncio.wait_for(self._shutdown_event.wait(), timeout=sleep_time)
                        return
                    except asyncio.TimeoutError:
                        if datetime.now() >= target:
                            break
            except Exception as e:
                logger.error(f"Failed to parse daily_at schedule '{self.schedule.get('daily_at')}': {e}")
                wait_seconds = 60
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=wait_seconds)
                except asyncio.TimeoutError:
                    pass
        else:
            wait_seconds = float(self.schedule.get("interval", 60))
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=wait_seconds)
            except asyncio.TimeoutError:
                pass

    # --- Step Logging ---

    MAX_PAYLOAD_CHARS: int = 4000
    _TRUNC_MARKER: str = "...[truncated]"

    @classmethod
    def _truncate_ascii(cls, text: Optional[str], max_len: Optional[int] = None) -> Optional[str]:
        """ASCII-safe + length-cap. Downstream Hindsight has utf-8 mangling on
        some endpoints, so action payloads are stored ASCII-only by default."""
        if text is None:
            return None
        text = text.encode("ascii", "replace").decode("ascii")
        cap = cls.MAX_PAYLOAD_CHARS if max_len is None else max_len
        if len(text) > cap:
            return text[: cap - len(cls._TRUNC_MARKER)] + cls._TRUNC_MARKER
        return text

    @classmethod
    def _serialize_payload(cls, data: Any) -> Optional[str]:
        """Serialize an arbitrary payload to an ASCII-safe, capped string.
        dict/list → JSON; str → passed through; None → None."""
        if data is None:
            return None
        if isinstance(data, str):
            return cls._truncate_ascii(data)
        try:
            return cls._truncate_ascii(json.dumps(data, default=str, ensure_ascii=True))
        except (TypeError, ValueError):
            return cls._truncate_ascii(str(data))

    def _log_task_root(
        self, action_type: str,
        trigger_context: Optional[str] = None,
        action_payload: Any = None,
        contexts: Optional[List[Tuple[str, Any]]] = None,
        outcome: str = "pending",
    ) -> int:
        """Log the root of an agent task and attach context tags atomically.

        Use _log_step under the returned id for child trajectory steps, and
        _finalize_action to set the outcome/outcome_payload when done.
        """
        action_id = int(self.memory_manager.log_agent_action(
            agent_name=self.agent_name,
            action_type=action_type,
            trigger_context=self._truncate_ascii(trigger_context),
            action_payload=self._serialize_payload(action_payload),
            outcome=outcome,
        ))
        if contexts:
            self._add_contexts(action_id, contexts)
        return action_id

    def _add_contexts(self, action_id: int, contexts: List[Tuple[str, Any]]) -> None:
        cleaned = [
            (str(t), str(v))
            for t, v in contexts
            if t is not None and v is not None and str(v) != ""
        ]
        if cleaned:
            self.memory_manager.add_action_contexts(action_id, cleaned)

    def _finalize_action(
        self, action_id: int, outcome: str,
        outcome_payload: Any = None,
    ) -> None:
        """Set the terminal outcome + outcome_payload on a root action."""
        self.memory_manager.update_agent_action_outcome(
            action_id, outcome, self._serialize_payload(outcome_payload),
        )

    # --- Hindsight bridging (DP-116b) ---

    AGENT_HISTORY_DOC_PREFIX: str = "agent_action"

    @classmethod
    def _action_document_id(cls, action_id: int) -> str:
        return f"{cls.AGENT_HISTORY_DOC_PREFIX}:{action_id}"

    def _format_action_series_prose(
        self,
        parent: Dict[str, Any],
        steps: List[Dict[str, Any]],
        contexts: List[Tuple[str, str]],
    ) -> str:
        """Dense ASCII prose for a parent + children + context-tag series.

        Hindsight's extractor performs best on chat-like prose. Plain k:v
        lines, no JSON braces / nulls, repeated keys collapsed.
        """
        def _kv_lines(payload: Optional[str], indent: str = "  ") -> List[str]:
            if not payload:
                return []
            try:
                data = json.loads(payload)
            except (TypeError, ValueError):
                return [f"{indent}{payload}"]
            if not isinstance(data, dict):
                return [f"{indent}{data}"]
            lines = []
            for k, v in data.items():
                if v is None or v == "" or v == [] or v == {}:
                    continue
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, ensure_ascii=True, default=str)
                lines.append(f"{indent}{k}: {v}")
            return lines

        action_id = parent.get("id")
        out: List[str] = []
        out.append(
            f"agent={self.agent_name} action_id={action_id} "
            f"type={parent.get('action_type')} outcome={parent.get('outcome')}"
        )
        trigger = parent.get("trigger_context")
        if trigger:
            out.append(f"trigger: {trigger}")
        if contexts:
            ctx_str = ", ".join(f"{t}={v}" for t, v in contexts)
            out.append(f"contexts: {ctx_str}")
        ap_lines = _kv_lines(parent.get("action_payload"))
        if ap_lines:
            out.append("inputs:")
            out.extend(ap_lines)
        if steps:
            out.append("steps:")
            for i, step in enumerate(steps, 1):
                out.append(
                    f"  {i}. {step.get('action_type')} -> {step.get('outcome')}"
                )
                for line in _kv_lines(step.get("action_payload"), indent="     in: "):
                    out.append(line)
                for line in _kv_lines(step.get("outcome_payload"), indent="     out: "):
                    out.append(line)
        op_lines = _kv_lines(parent.get("outcome_payload"))
        if op_lines:
            out.append("result:")
            out.extend(op_lines)
        prose = "\n".join(out)
        # 16k safety cap. Hindsight chunks at 10k internally, so this is a
        # belt on pathological loops — not a semantic boundary. Series are
        # kept small at log time via ref-substitution for blob payloads.
        return cast("str", self._truncate_ascii(prose, max_len=16000)) or ""

    async def _retain_action_series(self, action_id: int) -> None:
        """Bridge a finalized agent-action series into Hindsight.

        Fire-and-forget through the backend's per-bank queue. Stable
        document_id = `agent_action:<id>` makes the retain idempotent
        (re-running on the same id replaces; no double-extract).
        """
        if not self.agent_name:
            return
        parent = self.memory_manager.get_agent_action(action_id)
        if parent is None:
            logger.warning("retain_action_series: action %s not found", action_id)
            return
        steps = self.memory_manager.get_action_steps(action_id)
        contexts = self.memory_manager.get_action_contexts(action_id)
        prose = self._format_action_series_prose(parent, steps, contexts)
        bank = self.experience_bank or self.agent_name
        persona = self.experience_persona or self.experience_bank or self.agent_name
        scope_tags = [f"agent:{self.agent_name}", f"action_id:{action_id}"]
        ts = parent.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = None
        if ts is None:
            ts = datetime.now(timezone.utc)
        try:
            await self.memory_manager.retain_experience(
                bank_id=bank,
                action_type=str(parent.get("action_type") or "agent_action"),
                context={"action_id": action_id},
                outcome=str(parent.get("outcome") or ""),
                scope_tags=scope_tags,
                source_persona=persona,
                timestamp=ts,
                metadata={"action_id": str(action_id),
                          "agent": self.agent_name},
                document_id=self._action_document_id(action_id),
                content_override=prose,
            )
        except NotImplementedError:
            # SQLite-shape backend (semantic backend = sqlite). Bridging is a
            # Hindsight-only feature; silently skip when the backend isn't wired.
            pass
        except Exception as e:  # noqa: BLE001 — never break the agent loop
            logger.warning(
                "retain_action_series enqueue failed for action %s: %s",
                action_id, e,
            )

    def _log_step(
        self, parent_id: int, action_type: str,
        action_payload: Any = None,
        outcome: str = "success",
        outcome_payload: Any = None,
        contexts: Optional[List[Tuple[str, Any]]] = None,
    ) -> int:
        """Log a child step under a parent task action.

        action_payload / outcome_payload accept dicts (JSON-serialised) or strings.
        Both are ASCII-safe and capped at MAX_PAYLOAD_CHARS.
        """
        action_id = int(self.memory_manager.log_agent_action(
            agent_name=self.agent_name,
            action_type=action_type,
            action_payload=self._serialize_payload(action_payload),
            outcome=outcome,
            outcome_payload=self._serialize_payload(outcome_payload),
            parent_id=parent_id,
        ))
        if contexts:
            self._add_contexts(action_id, contexts)
        return action_id

    # --- LLM Context Building ---

    def _build_history_object(
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
            "message_history": history,
            "history": history,  # Legacy key for tests
            "current_message": {"text": prompt, "image_url": None}
        }

    def _build_llm_context(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Legacy alias for _build_history_object."""
        return self._build_history_object(*args, **kwargs)

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
