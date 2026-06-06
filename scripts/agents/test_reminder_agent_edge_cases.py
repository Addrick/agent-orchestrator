# tests/agents/test_reminder_agent_edge_cases.py
"""DP-199 Batch 8 — ReminderAgent edge cases."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.reminder_agent import ReminderAgent
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
    client = MagicMock()
    client.api_url = "https://zammad.example.com"
    return client


@pytest.fixture
def notification_router() -> NotificationRouter:
    return NotificationRouter()


@pytest.fixture
def reminder_agent(
    mock_chat_system: Any, mock_zammad_client: Any,
    notification_router: NotificationRouter,
) -> ReminderAgent:
    agent = ReminderAgent(mock_chat_system, mock_zammad_client, notification_router)
    agent.agent_config = {
        "notification_targets": [
            {"channel": "discord_dm", "recipient": "adrich"},
        ],
        "_recipients": {
            "adrich": {"discord_user_id": "12345"},
        },
    }
    return agent


# ---------------------------------------------------------------------------
# Startup vs. scheduled deploy
# ---------------------------------------------------------------------------

class TestDeployStartupVsScheduled:
    async def test_startup_run_dispatches_to_override_target(
        self, reminder_agent: ReminderAgent, mock_zammad_client: Any,
    ) -> None:
        """deploy_count == 0 + send_startup_dm=True -> uses override target,
        not configured notification_targets."""
        reminder_agent.deploy_count = 0
        reminder_agent.agent_config["send_startup_dm"] = True
        # Add a second target that should NOT be used during startup
        reminder_agent.agent_config["notification_targets"].append(
            {"channel": "discord_channel", "recipient": "99999"},
        )
        mock_zammad_client.search_tickets = MagicMock(return_value=[
            {"id": 1, "number": "1", "title": "T", "customer_id": 1,
             "updated_at": datetime.now(timezone.utc).isoformat()},
        ])
        mock_zammad_client.get_user = MagicMock(
            return_value={"firstname": "A", "lastname": "B"}
        )
        reminder_agent.notification_router.send = AsyncMock(return_value=True)

        await reminder_agent.deploy()

        # Only ONE send (the adrich override), not two
        assert reminder_agent.notification_router.send.call_count == 1
        call_kw = reminder_agent.notification_router.send.call_args.kwargs
        assert call_kw["channel"] == "discord_dm"
        assert call_kw["recipient"] == "12345"

    async def test_scheduled_run_dispatches_to_all_targets(
        self, reminder_agent: ReminderAgent, mock_zammad_client: Any,
    ) -> None:
        reminder_agent.deploy_count = 5  # any > 0
        reminder_agent.agent_config["notification_targets"].append(
            {"channel": "discord_channel", "recipient": "99999"},
        )
        mock_zammad_client.search_tickets = MagicMock(return_value=[
            {"id": 1, "number": "1", "title": "T", "customer_id": 1,
             "updated_at": datetime.now(timezone.utc).isoformat()},
        ])
        mock_zammad_client.get_user = MagicMock(
            return_value={"firstname": "A", "lastname": "B"}
        )
        reminder_agent.notification_router.send = AsyncMock(return_value=True)

        await reminder_agent.deploy()

        # Both configured targets dispatched
        assert reminder_agent.notification_router.send.call_count == 2


# ---------------------------------------------------------------------------
# Clock injection via monkeypatch
# ---------------------------------------------------------------------------

class TestClockInjection:
    async def test_send_batch_summary_with_mocked_clock(
        self, reminder_agent: ReminderAgent, mock_zammad_client: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pin datetime.now to a known UTC instant so the relative time
        formatting in the summary body is deterministic."""
        reminder_agent.deploy_count = 1
        fixed_now = datetime(2026, 5, 21, 12, 0, 0, tzinfo=timezone.utc)
        ticket_ts = datetime(2026, 5, 21, 9, 0, 0, tzinfo=timezone.utc)  # 3h ago

        class FakeDatetime(datetime):
            @classmethod
            def now(cls, tz: Any = None) -> "datetime":  # type: ignore[override]
                if tz is not None:
                    return fixed_now.astimezone(tz)
                return fixed_now.replace(tzinfo=None)

        monkeypatch.setattr("src.agents.reminder_agent.datetime", FakeDatetime)

        mock_zammad_client.search_tickets = MagicMock(return_value=[
            {"id": 1, "number": "1", "title": "T", "customer_id": 1,
             "updated_at": ticket_ts.isoformat().replace("+00:00", "Z")},
        ])
        mock_zammad_client.get_user = MagicMock(
            return_value={"firstname": "A", "lastname": "B"}
        )
        reminder_agent.notification_router.send = AsyncMock(return_value=True)

        await reminder_agent._send_batch_summary()

        # Body should say "3h ago" given the mocked clock
        sent_body = reminder_agent.notification_router.send.call_args.kwargs["body"]
        assert "3h ago" in sent_body

    async def test_missed_run_dispatches_idempotently_once_per_target(
        self, reminder_agent: ReminderAgent, mock_zammad_client: Any,
    ) -> None:
        """Two back-to-back deploy() calls (simulating a missed schedule that
        catches up) should each dispatch once per configured target — the
        agent does not skip on its own. Confirms current contract."""
        reminder_agent.deploy_count = 1
        mock_zammad_client.search_tickets = MagicMock(return_value=[
            {"id": 1, "number": "1", "title": "T", "customer_id": 1,
             "updated_at": datetime.now(timezone.utc).isoformat()},
        ])
        mock_zammad_client.get_user = MagicMock(
            return_value={"firstname": "A", "lastname": "B"}
        )
        reminder_agent.notification_router.send = AsyncMock(return_value=True)

        await reminder_agent.deploy()
        await reminder_agent.deploy()
        # 1 target * 2 deploys = 2 sends
        assert reminder_agent.notification_router.send.call_count == 2

    async def test_send_batch_summary_respects_shutdown_via_router_failure(
        self, reminder_agent: ReminderAgent, mock_zammad_client: Any,
    ) -> None:
        """If the notification router raises (e.g. due to mid-run cancellation
        cleanup), the agent should record 'error' on the action and continue
        without re-raising."""
        reminder_agent.deploy_count = 1
        mock_zammad_client.search_tickets = MagicMock(return_value=[
            {"id": 1, "number": "1", "title": "T", "customer_id": 1,
             "updated_at": datetime.now(timezone.utc).isoformat()},
        ])
        mock_zammad_client.get_user = MagicMock(
            return_value={"firstname": "A", "lastname": "B"}
        )
        reminder_agent.notification_router.send = AsyncMock(
            side_effect=RuntimeError("router down"),
        )

        await reminder_agent.deploy()  # must not raise

        mm = reminder_agent.memory_manager
        # Action should be finalized with 'error'
        update_calls = mm.update_agent_action_outcome.call_args_list
        outcomes = [c[0][1] for c in update_calls]
        assert "error" in outcomes


# ---------------------------------------------------------------------------
# _get_user_info fallback
# ---------------------------------------------------------------------------

class TestGetUserInfoFallback:
    async def test_fallback_on_exception(
        self, reminder_agent: ReminderAgent, mock_zammad_client: Any,
    ) -> None:
        mock_zammad_client.get_user = MagicMock(side_effect=RuntimeError("404"))
        name, link = await reminder_agent._get_user_info(123)
        # Falls back to "Unknown" + "#" defaults
        assert name == "Unknown"
        assert link == "#"

    async def test_fallback_on_missing_customer_id(
        self, reminder_agent: ReminderAgent,
    ) -> None:
        name, link = await reminder_agent._get_user_info(None)
        assert name == "Unknown"
        assert link == "#"

    async def test_uses_login_when_name_blank(
        self, reminder_agent: ReminderAgent, mock_zammad_client: Any,
    ) -> None:
        mock_zammad_client.get_user = MagicMock(return_value={
            "firstname": "", "lastname": "", "login": "alice",
        })
        name, _ = await reminder_agent._get_user_info(7)
        assert name == "alice"


# ---------------------------------------------------------------------------
# No api_url configured
# ---------------------------------------------------------------------------

class TestNoApiUrl:
    async def test_send_batch_summary_no_api_url_aborts(
        self, reminder_agent: ReminderAgent, mock_zammad_client: Any,
    ) -> None:
        mock_zammad_client.api_url = None
        mock_zammad_client.search_tickets = MagicMock(return_value=[
            {"id": 1, "number": "1", "title": "T", "customer_id": 1,
             "updated_at": "2026-05-21T00:00:00Z"},
        ])
        reminder_agent.notification_router.send = AsyncMock()

        with patch('src.agents.reminder_agent.logger') as mock_logger:
            await reminder_agent._send_batch_summary()
            mock_logger.error.assert_called()
            assert "API URL is not configured" in mock_logger.error.call_args[0][0]
        # No search, no send
        mock_zammad_client.search_tickets.assert_not_called()
        reminder_agent.notification_router.send.assert_not_called()
