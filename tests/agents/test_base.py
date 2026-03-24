# tests/agents/test_base.py

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.base import AgentLoop
from src.persona import Persona


class ConcreteAgent(AgentLoop):
    """Minimal concrete subclass for testing the abstract AgentLoop."""

    def __init__(self, chat_system, inject_personas=True):
        super().__init__(chat_system, inject_personas)
        self.poll_count = 0

    async def _poll(self):
        self.poll_count += 1


@pytest.fixture
def mock_chat_system():
    cs = MagicMock()
    cs.text_engine = MagicMock()
    cs.memory_manager = MagicMock()
    cs.personas = {}
    return cs


class TestAgentLoopInit:
    @patch('src.agents.base.load_system_personas_from_file')
    def test_init_injects_personas(self, mock_load, mock_chat_system):
        mock_load.return_value = {"triage_analyst": MagicMock()}
        agent = ConcreteAgent(mock_chat_system)
        assert "triage_analyst" in mock_chat_system.personas
        assert agent.text_engine is mock_chat_system.text_engine
        assert agent.memory_manager is mock_chat_system.memory_manager

    @patch('src.agents.base.load_system_personas_from_file')
    def test_init_skip_persona_injection(self, mock_load, mock_chat_system):
        agent = ConcreteAgent(mock_chat_system, inject_personas=False)
        mock_load.assert_not_called()
        assert agent.chat_system is mock_chat_system

    @patch('src.agents.base.load_system_personas_from_file')
    def test_init_warns_on_empty_personas(self, mock_load, mock_chat_system):
        mock_load.return_value = {}
        with patch('src.agents.base.logger') as mock_logger:
            ConcreteAgent(mock_chat_system)
            mock_logger.warning.assert_called_once()


class TestAgentLoopPolling:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_start_and_stop(self, mock_load, mock_chat_system):
        agent = ConcreteAgent(mock_chat_system)
        agent.poll_interval = 0.05

        async def stop_after_polls():
            while agent.poll_count < 2:
                await asyncio.sleep(0.01)
            agent.stop()

        await asyncio.gather(agent.start(), stop_after_polls())
        assert agent.poll_count >= 2

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_poll_error_does_not_crash_loop(self, mock_load, mock_chat_system):
        """A single poll failure should not stop the agent."""
        agent = ConcreteAgent(mock_chat_system)
        agent.poll_interval = 0.05
        call_count = 0

        async def failing_poll():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated poll failure")

        agent._poll = failing_poll  # type: ignore[assignment]

        async def stop_after_recovery():
            while call_count < 3:
                await asyncio.sleep(0.01)
            agent.stop()

        await asyncio.gather(agent.start(), stop_after_recovery())
        assert call_count >= 3  # Continued polling after the error

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_on_start_hook_called(self, mock_load, mock_chat_system):
        agent = ConcreteAgent(mock_chat_system)
        agent.poll_interval = 0.05
        on_start_called = False

        original_on_start = agent._on_start

        async def track_on_start():
            nonlocal on_start_called
            on_start_called = True
            await original_on_start()

        agent._on_start = track_on_start  # type: ignore[assignment]

        async def stop_quickly():
            await asyncio.sleep(0.02)
            agent.stop()

        await asyncio.gather(agent.start(), stop_quickly())
        assert on_start_called


class TestBuildLlmContext:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_build_llm_context_structure(self, mock_load, mock_chat_system):
        """With action_history_limit=0 (default), no history injection occurs."""
        agent = ConcreteAgent(mock_chat_system)
        persona = MagicMock(spec=Persona)
        persona.get_prompt.return_value = "You are a test bot."

        result = agent._build_llm_context(persona, "test prompt")

        assert result["persona_prompt"] == "You are a test bot."
        assert len(result["history"]) == 1
        assert result["history"][0] == {"role": "user", "content": "test prompt"}
        assert result["current_message"]["text"] == "test prompt"
        assert result["current_message"]["image_url"] is None

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_build_llm_context_with_action_history(self, mock_load, mock_chat_system):
        """When action_history_limit > 0, a system message with history is prepended."""
        agent = ConcreteAgent(mock_chat_system)
        agent.agent_name = "test_agent"
        agent.action_history_limit = 5

        mock_chat_system.memory_manager.get_relevant_agent_actions.return_value = [
            {
                "id": 1, "action_type": "dispatch", "trigger_context": "ticket:42",
                "outcome": "success", "outcome_payload": '{"priority": "high"}',
                "timestamp": "2025-01-15 09:23:00",
            }
        ]
        mock_chat_system.memory_manager.get_action_steps.return_value = []

        persona = MagicMock(spec=Persona)
        persona.get_prompt.return_value = "System prompt"

        result = agent._build_llm_context(persona, "test prompt")

        assert len(result["history"]) == 2
        assert result["history"][0]["role"] == "system"
        assert "RECENT ACTIONS" in result["history"][0]["content"]
        assert "ticket:42" in result["history"][0]["content"]
        assert result["history"][1] == {"role": "user", "content": "test prompt"}

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_build_llm_context_passes_task_data(self, mock_load, mock_chat_system):
        """match_contexts from task_data should be forwarded to get_relevant_agent_actions."""
        agent = ConcreteAgent(mock_chat_system)
        agent.agent_name = "test_agent"
        agent.action_history_limit = 5

        mock_chat_system.memory_manager.get_relevant_agent_actions.return_value = []

        persona = MagicMock(spec=Persona)
        persona.get_prompt.return_value = "System prompt"

        task_data = {"match_contexts": [("ticket", "42"), ("customer", "jane")]}
        agent._build_llm_context(persona, "prompt", task_data=task_data)

        mock_chat_system.memory_manager.get_relevant_agent_actions.assert_called_once_with(
            agent_name="test_agent",
            match_contexts=[("ticket", "42"), ("customer", "jane")],
            match_types=None,
            limit=5,
        )


class TestLogStep:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_log_step_sets_parent_and_agent(self, mock_load, mock_chat_system):
        """_log_step should delegate to memory_manager with correct parent_id and agent_name."""
        agent = ConcreteAgent(mock_chat_system)
        agent.agent_name = "test_agent"
        mock_chat_system.memory_manager.log_agent_action.return_value = 99

        result = agent._log_step(
            parent_id=1,
            action_type="fetch_ticket",
            action_payload='{"ticket_id": 42}',
            outcome="success",
            outcome_payload='{"title": "test"}',
        )

        assert result == 99
        mock_chat_system.memory_manager.log_agent_action.assert_called_once_with(
            agent_name="test_agent",
            action_type="fetch_ticket",
            action_payload='{"ticket_id": 42}',
            outcome="success",
            outcome_payload='{"title": "test"}',
            parent_id=1,
        )


class TestFormatActionHistory:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_empty_returns_empty(self, mock_load, mock_chat_system):
        agent = ConcreteAgent(mock_chat_system)
        agent.agent_name = "test"
        assert agent._format_action_history([]) == ""

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_chronological_order(self, mock_load, mock_chat_system):
        """Actions should be displayed oldest-first (input is newest-first from DB)."""
        agent = ConcreteAgent(mock_chat_system)
        agent.agent_name = "dispatch"
        mock_chat_system.memory_manager.get_action_steps.return_value = []

        actions = [
            {"id": 2, "action_type": "dispatch", "trigger_context": "ticket:2",
             "outcome": "success", "outcome_payload": None, "timestamp": "2025-01-15 09:25:00"},
            {"id": 1, "action_type": "dispatch", "trigger_context": "ticket:1",
             "outcome": "success", "outcome_payload": None, "timestamp": "2025-01-15 09:20:00"},
        ]
        result = agent._format_action_history(actions)
        lines = result.split("\n")
        # First action line (after header) should be the older one
        assert "ticket:1" in lines[1]
        assert "ticket:2" in lines[2]

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_failed_action_shows_failing_step(self, mock_load, mock_chat_system):
        """Failed actions should include the name of the failed step."""
        agent = ConcreteAgent(mock_chat_system)
        agent.agent_name = "dispatch"
        mock_chat_system.memory_manager.get_action_steps.return_value = [
            {"action_type": "fetch_ticket", "outcome": "success"},
            {"action_type": "llm_decision", "outcome": "failed"},
        ]

        actions = [
            {"id": 1, "action_type": "dispatch", "trigger_context": "ticket:42",
             "outcome": "failed", "outcome_payload": None, "timestamp": "2025-01-15 09:20:00"},
        ]
        result = agent._format_action_history(actions)
        assert "failed at: llm_decision" in result

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_truncates_long_payload(self, mock_load, mock_chat_system):
        """Long outcome_payload strings should be truncated."""
        agent = ConcreteAgent(mock_chat_system)
        agent.agent_name = "test"
        mock_chat_system.memory_manager.get_action_steps.return_value = []

        long_payload = "x" * 500
        actions = [
            {"id": 1, "action_type": "test", "trigger_context": "ctx",
             "outcome": "success", "outcome_payload": long_payload,
             "timestamp": "2025-01-15 09:20:00"},
        ]
        result = agent._format_action_history(actions)
        # Should contain truncation indicator
        assert "..." in result

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_json_payload_extraction(self, mock_load, mock_chat_system):
        """JSON outcome_payload should extract key fields."""
        agent = ConcreteAgent(mock_chat_system)
        agent.agent_name = "dispatch"
        mock_chat_system.memory_manager.get_action_steps.return_value = []

        actions = [
            {"id": 1, "action_type": "dispatch", "trigger_context": "ticket:42",
             "outcome": "success",
             "outcome_payload": '{"priority": "high", "channel": "discord"}',
             "timestamp": "2025-01-15 09:20:00"},
        ]
        result = agent._format_action_history(actions)
        assert "priority=high" in result
        assert "channel=discord" in result

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_header_and_footer(self, mock_load, mock_chat_system):
        """Output should have RECENT ACTIONS header and footer markers."""
        agent = ConcreteAgent(mock_chat_system)
        agent.agent_name = "test"
        mock_chat_system.memory_manager.get_action_steps.return_value = []

        actions = [
            {"id": 1, "action_type": "test", "trigger_context": "ctx",
             "outcome": "success", "outcome_payload": None,
             "timestamp": "2025-01-15 09:20:00"},
        ]
        result = agent._format_action_history(actions)
        assert result.startswith("--- RECENT ACTIONS (test) ---")
        assert result.endswith("---")


class TestIsRunning:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_is_running_false_before_start(self, mock_load, mock_chat_system):
        """is_running returns False before start."""
        agent = ConcreteAgent(mock_chat_system)
        assert agent.is_running is False

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_is_running_true_after_start(self, mock_load, mock_chat_system):
        """is_running returns True after start."""
        agent = ConcreteAgent(mock_chat_system)
        agent.poll_interval = 0.05

        async def check_and_stop():
            while agent.started_at is None:
                await asyncio.sleep(0.01)
            assert agent.is_running is True
            agent.stop()

        await asyncio.gather(agent.start(), check_and_stop())

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_is_running_false_after_stop(self, mock_load, mock_chat_system):
        """is_running returns False after stop."""
        agent = ConcreteAgent(mock_chat_system)
        agent.poll_interval = 0.05

        async def stop_quickly():
            while agent.started_at is None:
                await asyncio.sleep(0.01)
            agent.stop()

        await asyncio.gather(agent.start(), stop_quickly())
        assert agent.is_running is False


class TestSummarizePayloadFallbacks:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_json_dict_no_known_keys_falls_back_to_json_dumps(self, mock_load, mock_chat_system):
        """JSON dict with no known keys falls back to json.dumps."""
        agent = ConcreteAgent(mock_chat_system)
        result = agent._summarize_payload('{"unknown_key": "value"}')
        assert "unknown_key" in result
        assert "value" in result

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_json_dict_no_known_keys_long_gets_truncated(self, mock_load, mock_chat_system):
        """JSON dict with no known keys that's very long gets truncated."""
        agent = ConcreteAgent(mock_chat_system)
        import json
        long_data = {f"key_{i}": "x" * 20 for i in range(20)}
        payload = json.dumps(long_data)
        result = agent._summarize_payload(payload)
        assert result.endswith("...")
        assert len(result) <= 124  # max_len (120) + "..."

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_short_plain_string_returned_as_is(self, mock_load, mock_chat_system):
        """Short non-JSON string is returned unchanged (line 236)."""
        agent = ConcreteAgent(mock_chat_system)
        result = agent._summarize_payload("simple status text")
        assert result == "simple status text"


class TestGetFailedStepNameNoFailures:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_returns_empty_string_when_all_steps_succeed(self, mock_load, mock_chat_system):
        """Returns empty string when no steps have failed outcome."""
        agent = ConcreteAgent(mock_chat_system)
        mock_chat_system.memory_manager.get_action_steps.return_value = [
            {"action_type": "fetch_ticket", "outcome": "success"},
            {"action_type": "llm_decision", "outcome": "success"},
            {"action_type": "send_notification", "outcome": "success"},
        ]
        result = agent._get_failed_step_name(1)
        assert result == ""

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_returns_empty_string_for_none_action_id(self, mock_load, mock_chat_system):
        """action_id=None returns empty string without querying DB (line 241)."""
        agent = ConcreteAgent(mock_chat_system)
        result = agent._get_failed_step_name(None)
        assert result == ""
        mock_chat_system.memory_manager.get_action_steps.assert_not_called()


class TestFormatActionHistoryDatetime:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_datetime_timestamp_formatted(self, mock_load, mock_chat_system):
        """Real datetime objects use strftime formatting (line 186)."""
        from datetime import datetime
        agent = ConcreteAgent(mock_chat_system)
        agent.agent_name = "test"
        mock_chat_system.memory_manager.get_action_steps.return_value = []

        ts = datetime(2025, 1, 15, 9, 23, 0)
        actions = [
            {"id": 1, "action_type": "dispatch", "trigger_context": "ticket:42",
             "outcome": "success", "outcome_payload": None, "timestamp": ts},
        ]
        result = agent._format_action_history(actions)
        assert "2025-01-15 09:23" in result
