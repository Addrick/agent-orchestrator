# tests/agents/test_dispatch_agent.py

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.dispatch_agent import DispatchAgent
from src.clients.notification import NotificationRouter, LogNotifier


@pytest.fixture
def mock_chat_system():
    cs = MagicMock()
    cs.text_engine = MagicMock()
    cs.memory_manager = MagicMock()
    # Each call returns a fresh id so root vs. child rows can be told apart.
    cs._next_action_id = 0

    def _next_id(*_args, **_kwargs):
        cs._next_action_id += 1
        return cs._next_action_id

    cs.memory_manager.log_agent_action = MagicMock(side_effect=_next_id)
    cs.memory_manager.update_agent_action_outcome = MagicMock()
    cs.memory_manager.add_action_contexts = MagicMock()
    cs.personas = {}
    return cs


@pytest.fixture
def mock_zammad_client():
    return MagicMock()


@pytest.fixture
def notification_router():
    router = NotificationRouter()
    return router


@pytest.fixture
@patch('src.agents.base.load_system_personas_from_file', return_value={})
def dispatch_agent(mock_load, mock_chat_system, mock_zammad_client, notification_router):
    return DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)


class TestDispatchAgentInit:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_init_stores_references(self, mock_load, mock_chat_system, mock_zammad_client, notification_router):
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)
        assert agent.zammad_client is mock_zammad_client
        assert agent.notification_router is notification_router


class TestExtractTriageNote:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_finds_triage_note(self, mock_load, mock_chat_system, mock_zammad_client, notification_router):
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)
        articles = [
            {"body": "User reported issue", "internal": False},
            {"body": "AI analysis\n[ AI TRIAGE CONTEXT DUMP ]\nKeywords: test", "internal": True},
        ]
        result = agent._extract_triage_note(articles)
        assert "AI TRIAGE CONTEXT DUMP" in result

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_finds_recommended_action_note(self, mock_load, mock_chat_system, mock_zammad_client, notification_router):
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)
        articles = [
            {"body": "Recommended Action: escalate to admin", "internal": True},
        ]
        result = agent._extract_triage_note(articles)
        assert "Recommended Action" in result

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_fallback_to_last_article(self, mock_load, mock_chat_system, mock_zammad_client, notification_router):
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)
        articles = [
            {"body": "Just a regular message", "internal": False},
        ]
        result = agent._extract_triage_note(articles)
        assert result == "Just a regular message"

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_empty_articles(self, mock_load, mock_chat_system, mock_zammad_client, notification_router):
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)
        result = agent._extract_triage_note([])
        assert result == "No content"


class TestParseJsonResponse:
    def test_bare_json(self):
        result = DispatchAgent._parse_json_response('{"priority": "high"}')
        assert result["priority"] == "high"

    def test_markdown_fenced_json(self):
        content = '```json\n{"priority": "high", "summary": "test"}\n```'
        result = DispatchAgent._parse_json_response(content)
        assert result["priority"] == "high"

    def test_unfenced_markdown_block(self):
        content = '```\n{"priority": "low"}\n```'
        result = DispatchAgent._parse_json_response(content)
        assert result["priority"] == "low"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            DispatchAgent._parse_json_response("not json at all")


class TestGetDispatchDecision:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_missing_persona_returns_none(self, mock_load, mock_chat_system, mock_zammad_client,
                                                notification_router):
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)
        # personas dict is empty, so dispatch_analyst won't be found
        result = await agent._get_dispatch_decision("Test Ticket", "Triage note")
        assert result is None

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_valid_llm_response(self, mock_load, mock_chat_system, mock_zammad_client, notification_router):
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)
        mock_persona = MagicMock()
        mock_persona.get_prompt.return_value = "You are a dispatch agent."
        mock_persona.get_config_for_engine.return_value = {}
        mock_chat_system.personas["dispatch_analyst"] = mock_persona

        decision_json = json.dumps({
            "priority": "high",
            "summary": "Server is down",
            "reasoning": "Critical infrastructure"
        })
        mock_chat_system.text_engine.generate_response = AsyncMock(
            return_value=({"type": "text", "content": decision_json}, None)
        )

        result = await agent._get_dispatch_decision("Server Down", "Triage note content")
        assert result is not None
        assert result["priority"] == "high"
        assert result["summary"] == "Server is down"

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(self, mock_load, mock_chat_system, mock_zammad_client,
                                             notification_router):
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)
        mock_persona = MagicMock()
        mock_persona.get_prompt.return_value = "prompt"
        mock_persona.get_config_for_engine.return_value = {}
        mock_chat_system.personas["dispatch_analyst"] = mock_persona

        mock_chat_system.text_engine.generate_response = AsyncMock(
            return_value=({"type": "text", "content": "not valid json"}, None)
        )

        result = await agent._get_dispatch_decision("Title", "Note")
        assert result is None

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_non_text_response_returns_none(self, mock_load, mock_chat_system, mock_zammad_client,
                                                  notification_router):
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)
        mock_persona = MagicMock()
        mock_persona.get_config_for_engine.return_value = {}
        mock_chat_system.personas["dispatch_analyst"] = mock_persona

        mock_chat_system.text_engine.generate_response = AsyncMock(
            return_value=({"type": "tool_calls", "calls": []}, None)
        )

        result = await agent._get_dispatch_decision("Title", "Note")
        assert result is None


class TestDispatchTicket:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_successful_dispatch(self, mock_load, mock_chat_system, mock_zammad_client, notification_router):
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)

        # Set up mocks on the injected zammad_client
        mock_zammad_client.get_ticket = MagicMock(
            return_value={"id": 42, "title": "Test Issue", "number": 10042}
        )
        mock_zammad_client.get_ticket_articles = MagicMock(
            return_value=[{"body": "AI triage\n[ AI TRIAGE CONTEXT DUMP ]", "internal": True}]
        )
        mock_zammad_client.add_tag = MagicMock()

        decision = {
            "priority": "medium",
            "summary": "Test issue summary",
            "reasoning": "Routine ticket"
        }
        agent._get_dispatch_decision = AsyncMock(return_value=decision)

        await agent._dispatch_ticket(42)

        # Root + at least one child action logged
        log_calls = mock_chat_system.memory_manager.log_agent_action.call_args_list
        assert len(log_calls) >= 2
        root_kwargs = log_calls[0].kwargs
        assert root_kwargs["action_type"] == "dispatch"
        assert root_kwargs.get("parent_id") is None
        # Child rows carry parent_id linking back to the root
        child_calls = [c for c in log_calls[1:] if c.kwargs.get("parent_id") == 1]
        assert child_calls, "expected child step rows under the root action"
        child_types = {c.kwargs["action_type"] for c in child_calls}
        assert "llm_step" in child_types
        assert any(t.startswith("tool:") for t in child_types)
        # Ticket tagged
        mock_zammad_client.add_tag.assert_called_once()
        # Outcome on the root row is success with structured payload
        final_call = mock_chat_system.memory_manager.update_agent_action_outcome.call_args
        assert final_call[0][0] == 1  # root action_id
        assert final_call[0][1] == "success"
        payload = json.loads(final_call[0][2])
        assert payload["priority"] == "medium"
        assert payload["sent"] is True
        # Contexts attached to the root row
        ctx_calls = mock_chat_system.memory_manager.add_action_contexts.call_args_list
        root_ctx_calls = [c for c in ctx_calls if c[0][0] == 1]
        assert root_ctx_calls
        flat_contexts = {(t, v) for call in root_ctx_calls for (t, v) in call[0][1]}
        ctx_types = {t for t, _ in flat_contexts}
        assert {"ticket_id", "priority", "channel", "recipient"} <= ctx_types

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_channel_from_config_not_decision(self, mock_load, mock_chat_system, mock_zammad_client, notification_router):
        """Channel is determined by agent_config, not by the LLM decision."""
        agent = DispatchAgent(
            mock_chat_system, mock_zammad_client, notification_router,
            agent_config={"notification_defaults": {"channel": "discord_dm", "recipient": "someone"}},
        )

        mock_zammad_client.get_ticket = MagicMock(
            return_value={"id": 42, "title": "Test", "number": 10042}
        )
        mock_zammad_client.get_ticket_articles = MagicMock(
            return_value=[{"body": "[ AI TRIAGE CONTEXT DUMP ]", "internal": True}]
        )
        mock_zammad_client.add_tag = MagicMock()

        decision = {"priority": "high", "summary": "Outage", "reasoning": "Critical"}
        agent._get_dispatch_decision = AsyncMock(return_value=decision)

        await agent._dispatch_ticket(42)

        update_call = mock_chat_system.memory_manager.update_agent_action_outcome.call_args
        payload = json.loads(update_call[0][2])
        assert payload["channel"] == "discord_dm"

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_dispatch_with_no_decision(self, mock_load, mock_chat_system, mock_zammad_client,
                                             notification_router):
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)

        mock_zammad_client.get_ticket = MagicMock(
            return_value={"id": 42, "title": "Test", "number": 10042}
        )
        mock_zammad_client.get_ticket_articles = MagicMock(
            return_value=[{"body": "content", "internal": False}]
        )

        agent._get_dispatch_decision = AsyncMock(return_value=None)

        await agent._dispatch_ticket(42)

        # Should log failure, not tag
        update_call = mock_chat_system.memory_manager.update_agent_action_outcome.call_args
        assert update_call[0][1] == "failed"
        mock_zammad_client.add_tag.assert_not_called()

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_dispatch_error_logged(self, mock_load, mock_chat_system, mock_zammad_client, notification_router):
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)

        mock_zammad_client.get_ticket = MagicMock(
            side_effect=RuntimeError("API down")
        )

        await agent._dispatch_ticket(42)

        update_call = mock_chat_system.memory_manager.update_agent_action_outcome.call_args
        assert update_call[0][1] == "error"
        assert "API down" in update_call[0][2]
