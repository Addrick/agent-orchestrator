# tests/agents/test_base.py

import pytest
from unittest.mock import MagicMock, patch

from src.agents.base import Agent
from src.persona import Persona


class ConcreteAgent(Agent):
    """Minimal concrete subclass for testing the abstract Agent."""

    def __init__(self, chat_system, inject_personas=True):
        super().__init__(chat_system, inject_personas)
        self.poll_count = 0

    async def poll(self):
        self.poll_count += 1


@pytest.fixture
def mock_chat_system():
    cs = MagicMock()
    cs.text_engine = MagicMock()
    cs.memory_manager = MagicMock()
    cs.personas = {}
    return cs


class TestAgentInit:
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


class TestAgentStopping:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_stopping_default_false(self, mock_load, mock_chat_system):
        agent = ConcreteAgent(mock_chat_system)
        assert agent.stopping is False

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_request_stop_sets_flag(self, mock_load, mock_chat_system):
        agent = ConcreteAgent(mock_chat_system)
        agent.request_stop()
        assert agent.stopping is True


class TestAgentPoll:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_poll_increments_count(self, mock_load, mock_chat_system):
        agent = ConcreteAgent(mock_chat_system)
        await agent.poll()
        assert agent.poll_count == 1

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_on_start_is_noop_by_default(self, mock_load, mock_chat_system):
        agent = ConcreteAgent(mock_chat_system)
        await agent.on_start()  # Should not raise


class TestBuildLlmContext:
    def test_build_llm_context_structure(self):
        persona = MagicMock(spec=Persona)
        persona.get_prompt.return_value = "You are a test bot."

        result = Agent._build_llm_context(persona, "test prompt")

        assert result["persona_prompt"] == "You are a test bot."
        assert len(result["history"]) == 1
        assert result["history"][0] == {"role": "user", "content": "test prompt"}
        assert result["current_message"]["text"] == "test prompt"
        assert result["current_message"]["image_url"] is None
