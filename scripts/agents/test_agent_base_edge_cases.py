# tests/agents/test_agent_base_edge_cases.py
"""DP-199 Batch 8 — Agent base class edge cases (slice 9 prereq)."""

import asyncio
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agents.base import Agent


class ConcreteAgent(Agent):
    agent_name = "test_agent"

    def __init__(self, chat_system: Any, inject_personas: bool = False) -> None:
        super().__init__(chat_system, inject_personas)
        self.deploy_calls = 0

    async def deploy(self) -> None:
        self.deploy_calls += 1


@pytest.fixture
def mock_chat_system() -> Any:
    cs = MagicMock()
    cs.text_engine = MagicMock()
    cs.memory_manager = MagicMock()
    cs.personas = {}
    return cs


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestAgentLifecycle:
    async def test_async_context_manager_not_supported(self, mock_chat_system: Any) -> None:
        """Agent does NOT implement async context manager — verify that calling
        `async with` raises AttributeError, so tests / callers know to use
        explicit start/stop instead."""
        agent = ConcreteAgent(mock_chat_system)
        with pytest.raises((AttributeError, TypeError)):
            async with agent:  # type: ignore[misc]
                pass

    async def test_stop_idempotent(self, mock_chat_system: Any) -> None:
        agent = ConcreteAgent(mock_chat_system)
        agent.stop()
        agent.stop()  # second call must not raise
        assert agent._shutdown_event.is_set()

    async def test_double_start_finishes_when_stopped(self, mock_chat_system: Any) -> None:
        """Calling start() twice on an already-stopped agent shouldn't hang
        and shouldn't crash. Start, stop, then a second start should also exit."""
        agent = ConcreteAgent(mock_chat_system)
        agent.schedule = {"interval": 0.01}
        agent.stop()  # pre-stop so the loop exits immediately
        await asyncio.wait_for(agent.start(), timeout=1.0)
        # started_at gets set even though loop body didn't run
        assert agent.started_at is not None
        # Second start: still no hang
        await asyncio.wait_for(agent.start(), timeout=1.0)

    async def test_on_start_exception_propagates(self, mock_chat_system: Any) -> None:
        """The base class does NOT wrap _on_start; an exception raised there
        propagates out of start(). Confirms current contract for slice 9."""
        class BrokenAgent(ConcreteAgent):
            async def _on_start(self) -> None:
                raise RuntimeError("startup boom")

        agent = BrokenAgent(mock_chat_system)
        agent.schedule = {"interval": 0.01}
        agent.stop()
        with pytest.raises(RuntimeError, match="startup boom"):
            await agent.start()


# ---------------------------------------------------------------------------
# Wait-for-next-run
# ---------------------------------------------------------------------------

class TestWaitForNextRun:
    async def test_daily_at_past_time_schedules_next_day(
        self, mock_chat_system: Any,
    ) -> None:
        """When the daily_at time is earlier than now, the agent should sleep
        until that time tomorrow (positive wait, large enough to be 'tomorrow')."""
        agent = ConcreteAgent(mock_chat_system)
        # Pick a time 5 minutes in the past
        past = (datetime.now() - timedelta(minutes=5))
        agent.schedule = {"daily_at": past.strftime("%H:%M")}

        slept: dict[str, float] = {}

        async def fake_wait_for(coro: Any, timeout: float) -> None:
            slept["timeout"] = timeout
            # cancel the underlying coroutine to avoid warnings
            if hasattr(coro, 'close'):
                coro.close()
            raise asyncio.TimeoutError

        with patch('src.agents.base.asyncio.wait_for', new=fake_wait_for):
            await agent._wait_for_next_run()

        # Should wait nearly a full day, not negative
        assert "timeout" in slept
        assert slept["timeout"] > 23 * 3600  # at least 23 hours
        assert slept["timeout"] < 25 * 3600

    async def test_daily_at_malformed_falls_back_to_60s(
        self, mock_chat_system: Any,
    ) -> None:
        agent = ConcreteAgent(mock_chat_system)
        agent.schedule = {"daily_at": "not a time"}

        slept: dict[str, float] = {}

        async def fake_wait_for(coro: Any, timeout: float) -> None:
            slept["timeout"] = timeout
            if hasattr(coro, 'close'):
                coro.close()
            raise asyncio.TimeoutError

        with patch('src.agents.base.asyncio.wait_for', new=fake_wait_for):
            await agent._wait_for_next_run()
        assert slept["timeout"] == 60


# ---------------------------------------------------------------------------
# Build history with task_data contexts
# ---------------------------------------------------------------------------

class TestBuildHistoryTaskData:
    def test_build_history_with_task_data_contexts(
        self, mock_chat_system: Any,
    ) -> None:
        """When task_data carries match_contexts / match_types, they should be
        forwarded to memory_manager.get_relevant_agent_actions."""
        mm = mock_chat_system.memory_manager
        mm.get_relevant_agent_actions.return_value = []
        agent = ConcreteAgent(mock_chat_system)
        agent.action_history_limit = 3

        persona = MagicMock()
        persona.get_prompt.return_value = "sys"

        task_data = {
            "match_contexts": [("ticket_id", "42")],
            "match_types": ["dispatch"],
        }
        agent._build_history_object(persona, "do thing", task_data=task_data)

        mm.get_relevant_agent_actions.assert_called_once()
        kw = mm.get_relevant_agent_actions.call_args.kwargs
        assert kw["agent_name"] == "test_agent"
        assert kw["match_contexts"] == [("ticket_id", "42")]
        assert kw["match_types"] == ["dispatch"]
        assert kw["limit"] == 3


# ---------------------------------------------------------------------------
# Format action history — failed step name
# ---------------------------------------------------------------------------

class TestFormatActionHistory:
    def test_format_action_history_failed_step_name(
        self, mock_chat_system: Any,
    ) -> None:
        """When an action's outcome is 'failed', the formatter should look up
        the failed child step and append its name."""
        mm = mock_chat_system.memory_manager
        mm.get_action_steps.return_value = [
            {"action_type": "tool:zammad.get_ticket", "outcome": "success"},
            {"action_type": "llm_step", "outcome": "failed"},
            {"action_type": "tool:notification.send", "outcome": "success"},
        ]
        agent = ConcreteAgent(mock_chat_system)

        actions = [{
            "id": 7, "action_type": "dispatch",
            "trigger_context": "ticket:1",
            "outcome": "failed",
            "timestamp": "2026-05-21 10:00",
            "outcome_payload": None,
        }]

        prose = agent._format_action_history(actions)
        assert "[failed at: llm_step]" in prose
        assert "DISPATCH" in prose

    def test_format_action_history_no_failed_step(
        self, mock_chat_system: Any,
    ) -> None:
        """Failed outcome but no failed child -> no '[failed at: ...]' suffix."""
        mm = mock_chat_system.memory_manager
        mm.get_action_steps.return_value = [
            {"action_type": "tool:x", "outcome": "success"},
        ]
        agent = ConcreteAgent(mock_chat_system)
        actions = [{
            "id": 7, "action_type": "x",
            "outcome": "failed",
            "timestamp": "2026-05-21",
            "outcome_payload": None,
            "trigger_context": "",
        }]
        prose = agent._format_action_history(actions)
        assert "[failed at:" not in prose
