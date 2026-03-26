# tests/agents/test_base.py

import asyncio
import pytest
from unittest.mock import MagicMock, patch

from src.agents.base import AgentLoop
from src.persona import Persona


class ConcreteAgent(AgentLoop):
    """Minimal concrete subclass for testing the abstract AgentLoop."""
    agent_name = "test_agent"

    def __init__(self, chat_system, inject_personas=True):
        super().__init__(chat_system, inject_personas)

    async def _poll(self):
        pass


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


class TestAgentLoopLifecycle:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_is_running_default_false(self, mock_load, mock_chat_system):
        agent = ConcreteAgent(mock_chat_system)
        assert agent.is_running is False

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_stop_sets_shutdown_event(self, mock_load, mock_chat_system):
        agent = ConcreteAgent(mock_chat_system)
        agent.stop()
        assert agent._shutdown_event.is_set()

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_on_start_is_noop_by_default(self, mock_load, mock_chat_system):
        agent = ConcreteAgent(mock_chat_system)
        await agent._on_start()  # Should not raise

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_initial_status_properties(self, mock_load, mock_chat_system):
        agent = ConcreteAgent(mock_chat_system)
        assert agent.started_at is None
        assert agent.last_poll_time is None
        assert agent.poll_count == 0
        assert agent.error_count == 0
        assert agent.consecutive_errors == 0
        assert agent.last_error is None


class TestBuildLlmContext:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_build_llm_context_structure(self, mock_load, mock_chat_system):
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
        """When action_history_limit > 0, history should be injected."""
        mock_chat_system.memory_manager.get_relevant_agent_actions.return_value = [
            {
                "id": 1, "action_type": "dispatch", "trigger_context": "ticket:123",
                "outcome": "success", "timestamp": "2026-01-01 12:00",
                "outcome_payload": '{"priority": "high"}',
            }
        ]
        agent = ConcreteAgent(mock_chat_system)
        agent.action_history_limit = 5

        persona = MagicMock(spec=Persona)
        persona.get_prompt.return_value = "You are a test bot."

        result = agent._build_llm_context(persona, "test prompt")
        # Should have system message (action history) + user message
        assert len(result["history"]) == 2
        assert result["history"][0]["role"] == "system"
        assert "RECENT ACTIONS" in result["history"][0]["content"]

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_build_llm_context_no_history_when_disabled(self, mock_load, mock_chat_system):
        """When action_history_limit is 0, no history should be injected."""
        agent = ConcreteAgent(mock_chat_system)
        agent.action_history_limit = 0

        persona = MagicMock(spec=Persona)
        persona.get_prompt.return_value = "You are a test bot."

        result = agent._build_llm_context(persona, "test prompt")
        assert len(result["history"]) == 1
        mock_chat_system.memory_manager.get_relevant_agent_actions.assert_not_called()
