# tests/agents/test_base.py

import asyncio
import pytest
from unittest.mock import MagicMock, patch

from src.agents.base import Agent
from src.persona import Persona


class ConcreteAgent(Agent):
    """Minimal concrete subclass for testing the abstract Agent."""
    agent_name = "test_agent"

    def __init__(self, chat_system, inject_personas=True):
        super().__init__(chat_system, inject_personas)

    async def deploy(self):
        pass


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


class TestAgentLifecycle:
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
        assert agent.last_deploy_time is None
        assert agent.deploy_count == 0
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


class TestRetainActionSeries:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_format_action_series_prose_dense_kv(self, mock_load, mock_chat_system):
        agent = ConcreteAgent(mock_chat_system)
        parent = {
            "id": 7,
            "agent_name": "test_agent",
            "action_type": "dispatch",
            "trigger_context": "ticket:1234",
            "outcome": "success",
            "action_payload": '{"ticket_id": 1234, "empty": "", "none": null}',
            "outcome_payload": '{"priority": "high", "sent": true}',
        }
        steps = [
            {"action_type": "tool:zammad.get_ticket",
             "outcome": "success",
             "action_payload": '{"ticket_id": 1234}',
             "outcome_payload": '{"number": "10042", "title": "Billing 500"}'},
            {"action_type": "llm_step", "outcome": "success",
             "action_payload": '{"persona": "dispatch_analyst"}',
             "outcome_payload": '{"priority": "high"}'},
        ]
        contexts = [("ticket_id", "1234"), ("priority", "high")]

        prose = agent._format_action_series_prose(parent, steps, contexts)

        assert "action_id=7" in prose
        assert "type=dispatch" in prose
        assert "outcome=success" in prose
        assert "trigger: ticket:1234" in prose
        assert "ticket_id=1234" in prose
        assert "priority=high" in prose
        # Empty + null collapsed
        assert "empty:" not in prose
        assert "none:" not in prose
        # No JSON braces from inner payloads
        assert '{"ticket_id"' not in prose
        # Steps numbered
        assert "1. tool:zammad.get_ticket" in prose
        assert "2. llm_step" in prose

    @pytest.mark.asyncio
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    async def test_retain_action_series_calls_memory_manager(
        self, mock_load, mock_chat_system,
    ):
        from unittest.mock import AsyncMock
        agent = ConcreteAgent(mock_chat_system)
        agent.experience_bank = "test_bank"
        agent.experience_persona = "test_persona"

        mm = mock_chat_system.memory_manager
        mm.get_agent_action.return_value = {
            "id": 9, "action_type": "dispatch", "outcome": "success",
            "trigger_context": "ticket:1", "action_payload": None,
            "outcome_payload": None, "timestamp": "2026-05-16T00:00:00+00:00",
        }
        mm.get_action_steps.return_value = []
        mm.get_action_contexts.return_value = [("ticket_id", "1")]
        mm.retain_experience = AsyncMock(return_value="")

        await agent._retain_action_series(9)

        mm.retain_experience.assert_awaited_once()
        kwargs = mm.retain_experience.call_args.kwargs
        assert kwargs["bank_id"] == "test_bank"
        assert kwargs["source_persona"] == "test_persona"
        assert kwargs["document_id"] == "agent_action:9"
        assert "agent=test_agent" in kwargs["content_override"]
        assert kwargs["action_type"] == "dispatch"

    @pytest.mark.asyncio
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    async def test_retain_action_series_swallows_not_implemented(
        self, mock_load, mock_chat_system,
    ):
        """SQLite backend raises NotImplementedError on retain_experience —
        bridging is opportunistic, must not break the agent loop."""
        from unittest.mock import AsyncMock
        agent = ConcreteAgent(mock_chat_system)
        mm = mock_chat_system.memory_manager
        mm.get_agent_action.return_value = {
            "id": 1, "action_type": "x", "outcome": "success",
            "trigger_context": None, "action_payload": None,
            "outcome_payload": None, "timestamp": None,
        }
        mm.get_action_steps.return_value = []
        mm.get_action_contexts.return_value = []
        mm.retain_experience = AsyncMock(side_effect=NotImplementedError())

        # Should not raise
        await agent._retain_action_series(1)
