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
    def test_build_llm_context_structure(self):
        persona = MagicMock(spec=Persona)
        persona.get_prompt.return_value = "You are a test bot."

        result = AgentLoop._build_llm_context(persona, "test prompt")

        assert result["persona_prompt"] == "You are a test bot."
        assert len(result["history"]) == 1
        assert result["history"][0] == {"role": "user", "content": "test prompt"}
        assert result["current_message"]["text"] == "test prompt"
        assert result["current_message"]["image_url"] is None
