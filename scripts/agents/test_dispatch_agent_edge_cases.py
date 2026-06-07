# tests/agents/test_dispatch_agent_edge_cases.py
"""DP-199 Batch 8 — DispatchAgent edge cases."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.dispatch_agent import DispatchAgent
from src.clients.notification import NotificationRouter


@pytest.fixture
def mock_chat_system() -> Any:
    cs = MagicMock()
    cs.text_engine = MagicMock()
    cs.memory_manager = MagicMock()
    cs._next_action_id = 0

    def _next_id(*_args: Any, **_kwargs: Any) -> int:
        cs._next_action_id += 1
        return cs._next_action_id

    cs.memory_manager.log_agent_action = MagicMock(side_effect=_next_id)
    cs.memory_manager.update_agent_action_outcome = MagicMock()
    cs.memory_manager.add_action_contexts = MagicMock()
    cs.personas = {}
    return cs


@pytest.fixture
def mock_zammad_client() -> Any:
    return MagicMock()


@pytest.fixture
def notification_router() -> NotificationRouter:
    return NotificationRouter()


@pytest.fixture
@patch('src.agents.base.load_system_personas_from_file', return_value={})
def dispatch_agent(
    _mock_load: Any, mock_chat_system: Any,
    mock_zammad_client: Any, notification_router: NotificationRouter,
) -> DispatchAgent:
    return DispatchAgent(mock_chat_system, mock_zammad_client, notification_router)


# ---------------------------------------------------------------------------
# Recipient resolution — email channels
# ---------------------------------------------------------------------------

class TestResolveRecipientEmail:
    def test_resolves_email_channel(self, dispatch_agent: DispatchAgent) -> None:
        dispatch_agent.agent_config = {
            "notification_defaults": {"recipient": "alice"},
            "_recipients": {"alice": {"email": "alice@example.com"}},
        }
        result = dispatch_agent._resolve_recipient("email", 42)
        assert result == "alice@example.com"

    def test_resolves_smtp_email_channel(self, dispatch_agent: DispatchAgent) -> None:
        """Any channel containing 'email' picks the email field."""
        dispatch_agent.agent_config = {
            "notification_defaults": {"recipient": "alice"},
            "_recipients": {"alice": {"email": "alice@example.com"}},
        }
        result = dispatch_agent._resolve_recipient("smtp_email", 42)
        assert result == "alice@example.com"

    def test_falls_back_to_ticket_id_when_unmapped(
        self, dispatch_agent: DispatchAgent,
    ) -> None:
        dispatch_agent.agent_config = {
            "notification_defaults": {"recipient": "alice"},
            "_recipients": {"alice": {"discord_channel_id": "999"}},
        }
        # email field missing on the recipient; fallback to ticket id
        result = dispatch_agent._resolve_recipient("email", 7)
        assert result == "7"


# ---------------------------------------------------------------------------
# Deploy shutdown
# ---------------------------------------------------------------------------

class TestDeployShutdown:
    async def test_shutdown_event_stops_loop(
        self, dispatch_agent: DispatchAgent, mock_zammad_client: Any,
    ) -> None:
        mock_zammad_client.search_tickets = MagicMock(
            return_value=[{"id": 1}, {"id": 2}, {"id": 3}]
        )

        async def fake_dispatch(ticket_id: int) -> None:
            if ticket_id == 1:
                dispatch_agent._shutdown_event.set()

        dispatch_agent._dispatch_ticket = AsyncMock(side_effect=fake_dispatch)

        await dispatch_agent.deploy()
        # Only the first ticket processed before shutdown trip
        assert dispatch_agent._dispatch_ticket.await_count == 1


# ---------------------------------------------------------------------------
# Ticket fetch exception logged
# ---------------------------------------------------------------------------

class TestDispatchTicketExceptionLogged:
    async def test_ticket_fetch_exception_logged_and_recorded(
        self, dispatch_agent: DispatchAgent, mock_chat_system: Any,
        mock_zammad_client: Any,
    ) -> None:
        """If get_ticket raises, the agent should:
        1. Log the error (not propagate)
        2. Finalize the root action with outcome='error'
        """
        mock_zammad_client.get_ticket = MagicMock(
            side_effect=RuntimeError("ticket fetch boom")
        )

        with patch('src.agents.dispatch_agent.logger') as mock_logger:
            await dispatch_agent._dispatch_ticket(42)
            mock_logger.error.assert_called()

        # Root action_id=1; outcome update should mark 'error'
        update_call = mock_chat_system.memory_manager.update_agent_action_outcome.call_args
        assert update_call[0][1] == "error"
        assert "ticket fetch boom" in update_call[0][2]

    async def test_deploy_search_exception_returns_silently(
        self, dispatch_agent: DispatchAgent, mock_zammad_client: Any,
    ) -> None:
        mock_zammad_client.search_tickets = MagicMock(
            side_effect=RuntimeError("zammad down")
        )
        dispatch_agent._dispatch_ticket = AsyncMock()
        await dispatch_agent.deploy()  # must not raise
        dispatch_agent._dispatch_ticket.assert_not_awaited()
