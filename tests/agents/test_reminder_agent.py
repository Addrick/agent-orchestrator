# tests/agents/test_reminder_agent.py

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta

from src.agents.reminder_agent import ReminderAgent
from src.clients.notification import NotificationRouter


@pytest.fixture
def mock_chat_system():
    cs = MagicMock()
    cs.text_engine = MagicMock()
    cs.memory_manager = MagicMock()
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
    client = MagicMock()
    client.api_url = "https://zammad.example.com"
    return client


@pytest.fixture
def notification_router():
    return NotificationRouter()


@pytest.fixture
def reminder_agent(mock_chat_system, mock_zammad_client, notification_router):
    agent = ReminderAgent(mock_chat_system, mock_zammad_client, notification_router)
    # Default multi-target config
    agent.agent_config = {
        "notification_targets": [
            {"channel": "discord_dm", "recipient": "adrich"},
            {"channel": "discord_channel", "recipient": "99999"}
        ],
        "_recipients": {
            "adrich": {"discord_user_id": "12345"}
        }
    }
    return agent


class TestReminderAgentInit:
    def test_init_stores_references(self, mock_chat_system, mock_zammad_client, notification_router):
        agent = ReminderAgent(mock_chat_system, mock_zammad_client, notification_router)
        assert agent.zammad_client is mock_zammad_client
        assert agent.notification_router is notification_router


class TestReminderAgentDeploy:
    @pytest.mark.asyncio
    async def test_no_tickets(self, reminder_agent, mock_zammad_client):
        # Set deploy_count > 0 to test regular run
        reminder_agent.deploy_count = 1
        mock_zammad_client.search_tickets = MagicMock(return_value=[])
        
        await reminder_agent.deploy()
        
        mock_zammad_client.search_tickets.assert_called_once()
        reminder_agent.notification_router.send = AsyncMock()
        reminder_agent.notification_router.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_startup_sends_to_adrich_only(self, reminder_agent, mock_zammad_client):
        # deploy_count is 0 by default
        now_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        mock_zammad_client.search_tickets = MagicMock(return_value=[
            {"id": 101, "number": "10101", "title": "New Ticket", "customer_id": 1, "updated_at": now_iso},
        ])
        mock_zammad_client.get_user = MagicMock(return_value={"firstname": "Alice", "lastname": "Doe"})
        reminder_agent.notification_router.send = AsyncMock(return_value=True)
        
        await reminder_agent.deploy()
        
        # Should only send ONE notification (to adrich)
        assert reminder_agent.notification_router.send.call_count == 1
        call_args = reminder_agent.notification_router.send.call_args
        assert call_args[1]["channel"] == "discord_dm"
        assert call_args[1]["recipient"] == "12345" # Resolved adrich ID

    @pytest.mark.asyncio
    async def test_sends_daily_summary_to_multiple_targets(self, reminder_agent, mock_zammad_client, mock_chat_system):
        # Set deploy_count > 0 to simulate scheduled run
        reminder_agent.deploy_count = 1
        
        # Mock tickets
        now_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        mock_zammad_client.search_tickets = MagicMock(return_value=[
            {"id": 101, "number": "10101", "title": "New Ticket", "customer_id": 1, "updated_at": now_iso},
        ])
        
        mock_zammad_client.get_user = MagicMock(return_value={"firstname": "Alice", "lastname": "Doe"})
        reminder_agent.notification_router.send = AsyncMock(return_value=True)
        
        await reminder_agent.deploy()
        
        # Verify summary sent to BOTH targets
        assert reminder_agent.notification_router.send.call_count == 2
        
        calls = reminder_agent.notification_router.send.call_args_list
        channels = [c[1]["channel"] for c in calls]
        recipients = [c[1]["recipient"] for c in calls]
        
        assert "discord_dm" in channels
        assert "discord_channel" in channels
        assert "12345" in recipients # adrich ID
        assert "99999" in recipients # direct channel ID


class TestReminderAgentActionLogging:
    @pytest.mark.asyncio
    async def test_action_payload_and_contexts_populated(
        self, reminder_agent, mock_zammad_client, mock_chat_system,
    ):
        reminder_agent.deploy_count = 1
        now_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        mock_zammad_client.search_tickets = MagicMock(return_value=[
            {"id": 101, "number": "10101", "title": "T1", "customer_id": 1, "updated_at": now_iso},
            {"id": 102, "number": "10102", "title": "T2", "customer_id": 2, "updated_at": now_iso},
        ])
        mock_zammad_client.get_user = MagicMock(return_value={"firstname": "A", "lastname": "B"})
        reminder_agent.notification_router.send = AsyncMock(return_value=True)

        await reminder_agent.deploy()

        log_calls = mock_chat_system.memory_manager.log_agent_action.call_args_list
        # One root per target (2 targets configured)
        assert len(log_calls) == 2
        for call in log_calls:
            kw = call.kwargs
            assert kw["action_type"] == "daily_summary"
            assert kw.get("parent_id") is None
            payload = json.loads(kw["action_payload"])
            assert payload["subject"] == "Daily Ticket Summary"
            assert payload["ticket_count"] == 2
            assert "T1" in payload["body_excerpt"]
            assert payload["channel"] in {"discord_dm", "discord_channel"}

        ctx_calls = mock_chat_system.memory_manager.add_action_contexts.call_args_list
        assert len(ctx_calls) == 2
        for call in ctx_calls:
            ctx_types = {t for t, _ in call[0][1]}
            assert "channel" in ctx_types
            assert "recipient" in ctx_types

        # Outcome serialised with sent flag
        for fcall in mock_chat_system.memory_manager.update_agent_action_outcome.call_args_list:
            assert fcall[0][1] == "success"
            assert json.loads(fcall[0][2])["sent"] is True

    @pytest.mark.asyncio
    async def test_hindsight_bridge_fires_per_target(
        self, reminder_agent, mock_zammad_client, mock_chat_system,
    ):
        """DP-116b: at series completion, reminder enqueues one
        retain_experience call per target. bank/persona default to agent_name
        ('reminder') because ReminderAgent does not override experience_bank."""
        reminder_agent.deploy_count = 1
        now_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        mock_zammad_client.search_tickets = MagicMock(return_value=[
            {"id": 101, "number": "10101", "title": "T1", "customer_id": 1, "updated_at": now_iso},
        ])
        mock_zammad_client.get_user = MagicMock(return_value={"firstname": "A", "lastname": "B"})
        reminder_agent.notification_router.send = AsyncMock(return_value=True)

        mm = mock_chat_system.memory_manager
        # _retain_action_series re-reads the row it just finalized.
        mm.get_agent_action.side_effect = lambda aid: {
            "id": aid, "agent_name": "reminder", "action_type": "daily_summary",
            "trigger_context": "target:discord_dm:adrich",
            "action_payload": '{"channel": "discord_dm"}',
            "outcome": "success",
            "outcome_payload": '{"sent": true}',
            "timestamp": "2026-05-16T00:00:00+00:00",
        }
        mm.get_action_steps.return_value = []
        mm.get_action_contexts.return_value = [("channel", "discord_dm")]
        mm.retain_experience = AsyncMock(return_value="")

        await reminder_agent.deploy()

        # One bridge call per configured target (2)
        assert mm.retain_experience.await_count == 2
        for call in mm.retain_experience.await_args_list:
            kw = call.kwargs
            assert kw["bank_id"] == "reminder"
            assert kw["source_persona"] == "reminder"
            assert kw["action_type"] == "daily_summary"
            assert kw["document_id"].startswith("agent_action:")
            assert "agent=reminder" in kw["content_override"]


class TestReminderAgentRecipientResolution:
    def test_resolves_mapped_recipient(self, reminder_agent):
        recipient = reminder_agent._resolve_recipient("discord_dm", "adrich", 101)
        assert recipient == "12345"

    def test_resolves_direct_id(self, reminder_agent):
        recipient = reminder_agent._resolve_recipient("discord_channel", "1498777752197796003", 101)
        assert recipient == "1498777752197796003"
