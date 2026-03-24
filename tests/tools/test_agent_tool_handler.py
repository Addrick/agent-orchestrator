# tests/tools/test_agent_tool_handler.py

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.tools.agent_tool_handler import AgentToolHandler


@pytest.fixture
def mock_agent_manager():
    mgr = MagicMock()
    mgr.get_status.return_value = {
        "agents": {
            "dispatch": {
                "registered": True,
                "class": "DispatchAgent",
                "running": True,
                "started_at": "2025-01-15T09:00:00+00:00",
                "last_poll_time": "2025-01-15T09:05:00+00:00",
                "poll_count": 10,
                "poll_interval": 300,
                "error_count": 0,
                "consecutive_errors": 0,
                "last_error": None,
            }
        }
    }
    mgr.start_agent = AsyncMock(return_value="Agent 'dispatch' started successfully.")
    mgr.stop_agent = AsyncMock(return_value="Agent 'dispatch' stopped successfully.")
    mgr.restart_agent = AsyncMock(return_value="Agent 'dispatch' started successfully.")
    return mgr


@pytest.fixture
def mock_memory_manager():
    mm = MagicMock()
    mm.get_relevant_agent_actions.return_value = []
    mm.get_action_steps.return_value = []
    return mm


@pytest.fixture
def handler(mock_agent_manager, mock_memory_manager):
    return AgentToolHandler(mock_agent_manager, mock_memory_manager)


class TestGetAgentStatus:
    @pytest.mark.asyncio
    async def test_returns_status_for_named_agent(self, handler, mock_agent_manager):
        result = await handler._get_agent_status(agent_name="dispatch")
        mock_agent_manager.get_status.assert_called_once_with("dispatch")
        assert "agents" in result
        assert "dispatch" in result["agents"]

    @pytest.mark.asyncio
    async def test_returns_all_agents_when_no_name(self, handler, mock_agent_manager):
        result = await handler._get_agent_status()
        mock_agent_manager.get_status.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_registers_with_tool_manager(self, handler):
        tm = MagicMock()
        handler.register(tm)
        registered_names = [call.args[0] for call in tm.register.call_args_list]
        assert "get_agent_status" in registered_names
        assert "get_agent_history" in registered_names
        assert "manage_agent" in registered_names


class TestGetAgentHistory:
    @pytest.mark.asyncio
    async def test_returns_empty_history(self, handler, mock_memory_manager):
        result = await handler._get_agent_history(agent_name="dispatch")
        assert result["agent_name"] == "dispatch"
        assert result["action_count"] == 0
        assert result["actions"] == []

    @pytest.mark.asyncio
    async def test_returns_formatted_actions(self, handler, mock_memory_manager):
        mock_memory_manager.get_relevant_agent_actions.return_value = [
            {
                "id": 1,
                "action_type": "dispatch",
                "trigger_context": "ticket:42",
                "outcome": "success",
                "outcome_payload": '{"priority": "high", "channel": "discord_dm"}',
                "timestamp": "2025-01-15 09:23:00",
            },
        ]
        result = await handler._get_agent_history(agent_name="dispatch")

        assert result["action_count"] == 1
        action = result["actions"][0]
        assert action["action_type"] == "dispatch"
        assert action["outcome"] == "success"
        assert action["outcome_details"]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_includes_steps_for_failed_actions(self, handler, mock_memory_manager):
        mock_memory_manager.get_relevant_agent_actions.return_value = [
            {
                "id": 5,
                "action_type": "dispatch",
                "trigger_context": "ticket:99",
                "outcome": "failed",
                "outcome_payload": "llm_decision step failed",
                "timestamp": "2025-01-15 10:00:00",
            },
        ]
        mock_memory_manager.get_action_steps.return_value = [
            {"action_type": "fetch_ticket", "outcome": "success"},
            {"action_type": "llm_decision", "outcome": "failed"},
        ]

        result = await handler._get_agent_history(agent_name="dispatch")
        action = result["actions"][0]
        assert "steps" in action
        assert len(action["steps"]) == 2
        assert action["steps"][1]["outcome"] == "failed"

    @pytest.mark.asyncio
    async def test_passes_ticket_filter(self, handler, mock_memory_manager):
        await handler._get_agent_history(agent_name="dispatch", ticket_id="42")
        call_kwargs = mock_memory_manager.get_relevant_agent_actions.call_args.kwargs
        assert ("ticket", "42") in call_kwargs["match_contexts"]

    @pytest.mark.asyncio
    async def test_passes_customer_filter(self, handler, mock_memory_manager):
        await handler._get_agent_history(agent_name="dispatch", customer="jane@acme.com")
        call_kwargs = mock_memory_manager.get_relevant_agent_actions.call_args.kwargs
        assert ("customer", "jane@acme.com") in call_kwargs["match_contexts"]

    @pytest.mark.asyncio
    async def test_passes_both_filters(self, handler, mock_memory_manager):
        await handler._get_agent_history(
            agent_name="dispatch", ticket_id="42", customer="jane@acme.com"
        )
        call_kwargs = mock_memory_manager.get_relevant_agent_actions.call_args.kwargs
        assert ("ticket", "42") in call_kwargs["match_contexts"]
        assert ("customer", "jane@acme.com") in call_kwargs["match_contexts"]

    @pytest.mark.asyncio
    async def test_no_filters_passes_none(self, handler, mock_memory_manager):
        await handler._get_agent_history(agent_name="dispatch")
        call_kwargs = mock_memory_manager.get_relevant_agent_actions.call_args.kwargs
        assert call_kwargs["match_contexts"] is None

    @pytest.mark.asyncio
    async def test_respects_limit(self, handler, mock_memory_manager):
        await handler._get_agent_history(agent_name="dispatch", limit=5)
        call_kwargs = mock_memory_manager.get_relevant_agent_actions.call_args.kwargs
        assert call_kwargs["limit"] == 5

    @pytest.mark.asyncio
    async def test_plain_string_payload(self, handler, mock_memory_manager):
        """Non-JSON outcome_payload should be returned as-is."""
        mock_memory_manager.get_relevant_agent_actions.return_value = [
            {
                "id": 1,
                "action_type": "dispatch",
                "trigger_context": "ticket:42",
                "outcome": "error",
                "outcome_payload": "Connection timeout",
                "timestamp": "2025-01-15 09:23:00",
            },
        ]
        result = await handler._get_agent_history(agent_name="dispatch")
        assert result["actions"][0]["outcome_details"] == "Connection timeout"


class TestManageAgent:
    @pytest.mark.asyncio
    async def test_start_action(self, handler, mock_agent_manager):
        result = await handler._manage_agent(agent_name="dispatch", action="start")
        mock_agent_manager.start_agent.assert_awaited_once_with("dispatch")
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_stop_action(self, handler, mock_agent_manager):
        result = await handler._manage_agent(agent_name="dispatch", action="stop")
        mock_agent_manager.stop_agent.assert_awaited_once_with("dispatch")
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_restart_action(self, handler, mock_agent_manager):
        result = await handler._manage_agent(agent_name="dispatch", action="restart")
        mock_agent_manager.restart_agent.assert_awaited_once_with("dispatch")
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_unknown_action_raises(self, handler):
        with pytest.raises(ValueError, match="Unknown action"):
            await handler._manage_agent(agent_name="dispatch", action="explode")
