# tests/agents/test_dispatch_agent.py

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch

from src.agents.dispatch_agent import DispatchAgent
from src.clients.notification import NotificationRouter, LogNotifier


@pytest.fixture
def mock_chat_system():
    cs = MagicMock()
    cs.text_engine = MagicMock()
    cs.memory_manager = MagicMock()
    cs.memory_manager.log_agent_action = MagicMock(return_value=1)
    cs.memory_manager.update_agent_action_outcome = MagicMock()
    cs.memory_manager.add_action_contexts = MagicMock()
    cs.memory_manager.get_relevant_agent_actions = MagicMock(return_value=[])
    cs.memory_manager.get_action_steps = MagicMock(return_value=[])
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

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_class_attributes(self, mock_load, mock_chat_system, mock_zammad_client, notification_router):
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)
        assert agent.agent_name == "dispatch"
        assert agent.action_history_limit == 10


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


class TestGetDispatchDecision:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_missing_persona_returns_none(self, mock_load, mock_chat_system, mock_zammad_client,
                                                notification_router):
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)
        # personas dict is empty, so dispatch_analyst won't be found
        result = await agent._get_dispatch_decision("Test Ticket", "Triage note", 42)
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
            "notify_channel": "zammad",
            "summary": "Server is down",
            "reasoning": "Critical infrastructure"
        })
        mock_chat_system.text_engine.generate_response = AsyncMock(
            return_value=({"type": "text", "content": decision_json}, None)
        )

        result = await agent._get_dispatch_decision("Server Down", "Triage note content", 42)
        assert result is not None
        assert result["priority"] == "high"
        assert result["notify_channel"] == "zammad"

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

        result = await agent._get_dispatch_decision("Title", "Note", 42)
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

        result = await agent._get_dispatch_decision("Title", "Note", 42)
        assert result is None


class TestDispatchTicket:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_successful_dispatch(self, mock_load, mock_chat_system, mock_zammad_client, notification_router):
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)

        # Set up mocks on the injected zammad_client
        mock_zammad_client.get_ticket = MagicMock(
            return_value={"id": 42, "title": "Test Issue", "number": 10042, "customer": "jane@acme.com"}
        )
        mock_zammad_client.get_ticket_articles = MagicMock(
            return_value=[{"body": "AI triage\n[ AI TRIAGE CONTEXT DUMP ]", "internal": True}]
        )
        mock_zammad_client.add_tag = MagicMock()

        decision = {
            "priority": "medium",
            "notify_channel": "zammad",
            "summary": "Test issue summary",
            "reasoning": "Routine ticket"
        }
        agent._get_dispatch_decision = AsyncMock(return_value=decision)

        await agent._dispatch_ticket(42)

        # Verify parent action logged (first call) + step calls
        log_calls = mock_chat_system.memory_manager.log_agent_action.call_args_list
        assert len(log_calls) >= 1  # At least the parent action
        # Verify ticket tagged
        mock_zammad_client.add_tag.assert_called_once()
        # Verify outcome updated to success
        update_call = mock_chat_system.memory_manager.update_agent_action_outcome.call_args
        assert update_call[0][1] == "success"

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


class TestDispatchStepLogging:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_dispatch_logs_all_steps(self, mock_load, mock_chat_system, mock_zammad_client, notification_router):
        """Each operation in the dispatch pipeline should create a child step."""
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)

        mock_zammad_client.get_ticket = MagicMock(
            return_value={"id": 42, "title": "Test", "number": 10042, "customer": "jane@acme.com"}
        )
        mock_zammad_client.get_ticket_articles = MagicMock(
            return_value=[{"body": "AI triage\n[ AI TRIAGE CONTEXT DUMP ]", "internal": True}]
        )
        mock_zammad_client.add_tag = MagicMock()

        decision = {"priority": "high", "notify_channel": "zammad", "summary": "Issue", "reasoning": "Test"}
        agent._get_dispatch_decision = AsyncMock(return_value=decision)

        await agent._dispatch_ticket(42)

        # Count log_agent_action calls: 1 parent + 5 steps
        # (fetch_ticket, fetch_articles, llm_decision, send_notification, tag_ticket)
        log_calls = mock_chat_system.memory_manager.log_agent_action.call_args_list
        assert len(log_calls) == 6  # 1 parent + 5 steps

        # First call is the parent (no parent_id)
        parent_call_kwargs = log_calls[0][1]
        assert parent_call_kwargs.get("parent_id") is None or "parent_id" not in parent_call_kwargs

        # Remaining calls are steps (with parent_id=1)
        step_types = []
        for step_call in log_calls[1:]:
            kwargs = step_call[1]
            assert kwargs["parent_id"] == 1
            step_types.append(kwargs["action_type"])

        assert step_types == [
            "fetch_ticket", "fetch_articles", "llm_decision",
            "send_notification", "tag_ticket",
        ]

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_dispatch_adds_context_tags(self, mock_load, mock_chat_system, mock_zammad_client,
                                              notification_router):
        """Dispatch should tag the action with ticket and customer contexts."""
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)

        mock_zammad_client.get_ticket = MagicMock(
            return_value={"id": 42, "title": "Test", "number": 10042, "customer": "jane@acme.com"}
        )
        mock_zammad_client.get_ticket_articles = MagicMock(
            return_value=[{"body": "content", "internal": False}]
        )
        mock_zammad_client.add_tag = MagicMock()

        decision = {"priority": "low", "notify_channel": "zammad", "summary": "Test", "reasoning": "Test"}
        agent._get_dispatch_decision = AsyncMock(return_value=decision)

        await agent._dispatch_ticket(42)

        mock_chat_system.memory_manager.add_action_contexts.assert_called_once()
        contexts = mock_chat_system.memory_manager.add_action_contexts.call_args[0][1]
        context_types = {ct for ct, _ in contexts}
        assert "ticket" in context_types
        assert "customer" in context_types

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_failed_step_updates_parent_outcome(self, mock_load, mock_chat_system, mock_zammad_client,
                                                      notification_router):
        """When the LLM decision fails, parent should be marked failed."""
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)

        mock_zammad_client.get_ticket = MagicMock(
            return_value={"id": 42, "title": "Test", "number": 10042}
        )
        mock_zammad_client.get_ticket_articles = MagicMock(
            return_value=[{"body": "content", "internal": False}]
        )

        agent._get_dispatch_decision = AsyncMock(return_value=None)

        await agent._dispatch_ticket(42)

        update_call = mock_chat_system.memory_manager.update_agent_action_outcome.call_args
        assert update_call[0][1] == "failed"
        assert "llm_decision" in update_call[0][2]


class TestResolveRecipient:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_resolves_discord_dm(self, mock_load, mock_chat_system, mock_zammad_client, notification_router):
        """Should resolve logical name to Discord user ID from config."""
        notification_router.register("discord_dm", MagicMock())
        agent_config = {
            "notification_defaults": {"channel": "discord_dm", "recipient": "adrich"},
            "_recipients": {"adrich": {"discord_user_id": "321783731146850305"}},
        }
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router, agent_config)

        result = agent._resolve_recipient("discord_dm")
        assert result == "321783731146850305"

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_returns_none_for_unavailable_channel(self, mock_load, mock_chat_system, mock_zammad_client,
                                                   notification_router):
        """Channel not registered with router should return None."""
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)
        result = agent._resolve_recipient("discord_dm")
        assert result is None

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_returns_none_when_no_default_recipient(self, mock_load, mock_chat_system, mock_zammad_client,
                                                     notification_router):
        """No notification_defaults.recipient configured should return None."""
        notification_router.register("discord_dm", MagicMock())
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router, agent_config={})
        result = agent._resolve_recipient("discord_dm")
        assert result is None

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_returns_none_when_user_id_not_configured(self, mock_load, mock_chat_system, mock_zammad_client,
                                                       notification_router):
        """Recipient exists but has no discord_user_id should return None."""
        notification_router.register("discord_dm", MagicMock())
        agent_config = {
            "notification_defaults": {"recipient": "adrich"},
            "_recipients": {"adrich": {"discord_user_id": None}},
        }
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router, agent_config)
        result = agent._resolve_recipient("discord_dm")
        assert result is None

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    def test_resolves_email_channel(self, mock_load, mock_chat_system, mock_zammad_client, notification_router):
        """Email channel should resolve to the email field."""
        notification_router.register("email", MagicMock())
        agent_config = {
            "notification_defaults": {"recipient": "adrich"},
            "_recipients": {"adrich": {"email": "adam@tech-ops.it"}},
        }
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router, agent_config)
        result = agent._resolve_recipient("email")
        assert result == "adam@tech-ops.it"

    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_falls_back_to_zammad_in_dispatch_pipeline(self, mock_load, mock_chat_system,
                                                              mock_zammad_client, notification_router):
        """When discord_dm resolution fails, _dispatch_ticket should fall back to zammad."""
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)

        mock_zammad_client.get_ticket = MagicMock(
            return_value={"id": 42, "title": "Test", "number": 10042, "customer": "jane"}
        )
        mock_zammad_client.get_ticket_articles = MagicMock(
            return_value=[{"body": "content", "internal": False}]
        )
        mock_zammad_client.add_tag = MagicMock()

        # LLM decides discord_dm but no recipient mapping exists
        decision = {"priority": "high", "notify_channel": "discord_dm", "summary": "Test", "reasoning": "Test"}
        agent._get_dispatch_decision = AsyncMock(return_value=decision)

        await agent._dispatch_ticket(42)

        # Notification should have been sent to zammad (fallback)
        log_calls = mock_chat_system.memory_manager.log_agent_action.call_args_list
        notification_step = [c for c in log_calls if c[1].get("action_type") == "send_notification"]
        assert len(notification_step) == 1
        payload = json.loads(notification_step[0][1]["action_payload"])
        assert payload["channel"] == "zammad"


class TestDispatchActionHistoryInContext:
    @patch('src.agents.base.load_system_personas_from_file', return_value={})
    @pytest.mark.asyncio
    async def test_dispatch_decision_includes_action_history(self, mock_load, mock_chat_system,
                                                             mock_zammad_client, notification_router):
        """The LLM context should contain action history as a system message."""
        agent = DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)

        mock_persona = MagicMock()
        mock_persona.get_prompt.return_value = "You are a dispatch agent."
        mock_persona.get_config_for_engine.return_value = {}
        mock_chat_system.personas["dispatch_analyst"] = mock_persona

        # Set up past actions to be returned
        mock_chat_system.memory_manager.get_relevant_agent_actions.return_value = [
            {
                "id": 1, "action_type": "dispatch", "trigger_context": "ticket:40",
                "outcome": "success", "outcome_payload": '{"priority": "medium"}',
                "timestamp": "2025-01-15 09:20:00",
            }
        ]
        mock_chat_system.memory_manager.get_action_steps.return_value = []

        decision_json = json.dumps({
            "priority": "high", "notify_channel": "zammad",
            "summary": "Test", "reasoning": "Test"
        })
        mock_chat_system.text_engine.generate_response = AsyncMock(
            return_value=({"type": "text", "content": decision_json}, None)
        )

        result = await agent._get_dispatch_decision("Test", "Triage note", 42, "jane@acme.com")

        # Verify generate_response was called with a context that includes action history
        gen_call = mock_chat_system.text_engine.generate_response.call_args
        context_object = gen_call[1]["context_object"] if "context_object" in gen_call[1] else gen_call[0][1]
        history = context_object["history"]

        # Should have system message (action history) + user message (prompt)
        assert len(history) == 2
        assert history[0]["role"] == "system"
        assert "RECENT ACTIONS" in history[0]["content"]
        assert "ticket:40" in history[0]["content"]
        assert history[1]["role"] == "user"
