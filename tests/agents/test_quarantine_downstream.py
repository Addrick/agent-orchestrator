# tests/agents/test_quarantine_downstream.py
"""DP-288 review fixes: quarantine must hold in EVERY downstream consumer,
not just managr. DispatchAgent (LLM prompt exposure) and ReminderAgent
(outbound Discord exposure) each respect the quarantine tags, and unknown tag
state fails closed everywhere.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from config.global_config import (
    DISPATCH_DISPATCHED_TAG,
    QUARANTINE_TAGS,
    SECURITY_REPORT_TAG,
)
from src.agents.dispatch_agent import DispatchAgent
from src.agents.reminder_agent import ReminderAgent

BAIT_TITLE = "FW: Invoice overdue - URGENT wire transfer required"


def _make_dispatch():
    chat_system = MagicMock()
    chat_system.text_engine.generate_response = AsyncMock()
    chat_system.memory_manager.log_agent_action = MagicMock(return_value=1)
    chat_system.personas = {}
    zammad = MagicMock()
    zammad.get_ticket = MagicMock(return_value={
        "id": 42, "number": "10042", "title": BAIT_TITLE})
    zammad.get_ticket_articles = MagicMock(return_value=[{"body": "bait body"}])
    zammad.get_tags = MagicMock(return_value=[])
    router = MagicMock()
    router.send = AsyncMock(return_value=True)
    with patch("src.agents.base.load_system_personas_from_file", return_value={}):
        agent = DispatchAgent(chat_system, zammad, router)
    agent._retain_action_series = AsyncMock()
    return agent, chat_system, zammad, router


@pytest.mark.asyncio
async def test_dispatch_selector_query_excludes_quarantine_tags():
    agent, chat_system, zammad, router = _make_dispatch()
    zammad.search_tickets = MagicMock(return_value=[])

    await agent.deploy()

    query = zammad.search_tickets.call_args.kwargs["query"]
    for tag in QUARANTINE_TAGS:
        assert f"NOT tags:{tag}" in query
    assert f"NOT tags:{DISPATCH_DISPATCHED_TAG}" in query


@pytest.mark.asyncio
async def test_dispatch_belt_skips_quarantined_ticket_before_any_content_fetch():
    """Even if the selector query returns a quarantined ticket (query
    semantics are Zammad's), the per-ticket belt stops it before the ticket
    body or title is fetched into a prompt."""
    agent, chat_system, zammad, router = _make_dispatch()
    zammad.get_tags = MagicMock(return_value=[SECURITY_REPORT_TAG])

    await agent._dispatch_ticket(42)

    zammad.get_ticket.assert_not_called()
    zammad.get_ticket_articles.assert_not_called()
    chat_system.text_engine.generate_response.assert_not_awaited()
    router.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_unknown_tag_state_fails_closed():
    agent, chat_system, zammad, router = _make_dispatch()
    zammad.get_tags = MagicMock(side_effect=RuntimeError("api down"))

    await agent._dispatch_ticket(42)

    zammad.get_ticket.assert_not_called()
    chat_system.text_engine.generate_response.assert_not_awaited()
    router.send.assert_not_awaited()


def _make_reminder(tags):
    chat_system = MagicMock()
    chat_system.memory_manager.log_agent_action = MagicMock(return_value=1)
    zammad = MagicMock()
    zammad.api_url = "http://zammad.local"
    zammad.search_tickets = MagicMock(return_value=[
        {"id": 42, "number": "10042", "title": BAIT_TITLE,
         "customer_id": 7, "updated_at": "2026-07-09T10:00:00Z"},
    ])
    zammad.get_user = MagicMock(return_value={"firstname": "A", "lastname": "B"})
    if isinstance(tags, Exception):
        zammad.get_tags = MagicMock(side_effect=tags)
    else:
        zammad.get_tags = MagicMock(return_value=tags)
    router = MagicMock()
    router.send = AsyncMock(return_value=True)
    with patch("src.agents.base.load_system_personas_from_file", return_value={}):
        agent = ReminderAgent(chat_system, zammad, router, agent_config={
            "notification_targets": [
                {"channel": "discord_channel", "recipient": "123"}],
        })
    agent._retain_action_series = AsyncMock()
    return agent, router


@pytest.mark.asyncio
async def test_reminder_masks_quarantined_title_in_summary():
    agent, router = _make_reminder(tags=[SECURITY_REPORT_TAG])
    agent.deploy_count = 1  # past the startup special case

    await agent.deploy()

    body = router.send.await_args.kwargs["body"]
    assert BAIT_TITLE not in body
    assert "CONTENT QUARANTINED" in body
    assert SECURITY_REPORT_TAG in body
    # Ticket is still listed — it needs human attention
    assert "#10042" in body


@pytest.mark.asyncio
async def test_reminder_shows_clean_titles_unchanged():
    agent, router = _make_reminder(tags=["vip"])
    agent.deploy_count = 1

    await agent.deploy()

    body = router.send.await_args.kwargs["body"]
    assert BAIT_TITLE in body


@pytest.mark.asyncio
async def test_reminder_unknown_tag_state_withholds_title():
    agent, router = _make_reminder(tags=RuntimeError("api down"))
    agent.deploy_count = 1

    await agent.deploy()

    body = router.send.await_args.kwargs["body"]
    assert BAIT_TITLE not in body
    assert "title withheld" in body
    assert "#10042" in body
