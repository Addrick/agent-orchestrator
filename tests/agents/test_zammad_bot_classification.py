# tests/agents/test_zammad_bot_classification.py
"""Unit tests for ZammadBot's classification gate (DP-288 Phase 1).

Contract under test:
- quarantined content NEVER reaches a triage persona prompt (the gate runs
  before the scout/analyst and stops the pipeline)
- quarantine = internal note + quarantine tag + triage tag (so deploy() stops
  re-selecting the ticket)
- a pre-existing quarantine tag (operator-applied) is respected without
  re-judging
- classification failure fails open: the pipeline proceeds unclassified
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from config.global_config import (
    ZAMMAD_TRIAGE_TAG,
    TRIAGE_ANALYST_NAME,
    SECURITY_REPORT_TAG,
    PHISHING_SUSPECT_TAG,
)
from src.agents.content_classifier import Classification
from src.agents.zammad_bot import ZammadBot
from src.persona import Persona


def _make_bot():
    chat_system = MagicMock()
    chat_system.text_engine.generate_response = AsyncMock(
        return_value=({"type": "text", "content": "analysis"}, {}))
    chat_system.personas = {
        TRIAGE_ANALYST_NAME: Persona(
            persona_name=TRIAGE_ANALYST_NAME, model_name="mock",
            prompt="analyst prompt"),
    }
    zammad = MagicMock()
    zammad.get_ticket = MagicMock(return_value={
        "id": 42, "customer_id": 7, "title": "FW: Invoice overdue - URGENT"})
    zammad.get_ticket_articles = MagicMock(return_value=[
        {"body": "phishing — see forwarded mail below\n\nPay $9000 now."}])
    zammad.get_tags = MagicMock(return_value=[])
    zammad.add_tag = MagicMock()
    zammad.add_article_to_ticket = MagicMock()
    zammad.search_tickets = MagicMock(return_value=[])
    with patch("src.agents.base.load_system_personas_from_file", return_value={}):
        bot = ZammadBot(chat_system, zammad)
    return bot, chat_system, zammad


def _tagged(zammad, tag):
    return any(c.kwargs.get("tag") == tag or (len(c.args) > 1 and c.args[1] == tag)
               for c in zammad.add_tag.call_args_list)


@pytest.mark.asyncio
async def test_reported_phishing_is_quarantined_without_any_llm_call():
    bot, chat_system, zammad = _make_bot()

    await bot._process_ticket(42)

    # The reporter marker short-circuits: zero LLM calls, so the bait never
    # reached any persona prompt.
    chat_system.text_engine.generate_response.assert_not_awaited()
    assert _tagged(zammad, SECURITY_REPORT_TAG)
    assert _tagged(zammad, ZAMMAD_TRIAGE_TAG)
    note = zammad.add_article_to_ticket.call_args.kwargs["body"]
    assert "QUARANTINED" in note
    assert "phishing_report" in note
    assert zammad.add_article_to_ticket.call_args.kwargs["internal"] is True


@pytest.mark.asyncio
async def test_existing_quarantine_tag_skips_triage_without_rejudging():
    bot, chat_system, zammad = _make_bot()
    zammad.get_tags = MagicMock(return_value=[PHISHING_SUSPECT_TAG])
    bot.classifier.classify = AsyncMock()

    await bot._process_ticket(42)

    bot.classifier.classify.assert_not_awaited()
    chat_system.text_engine.generate_response.assert_not_awaited()
    assert _tagged(zammad, ZAMMAD_TRIAGE_TAG)
    # No new note: the verdict was already recorded when the tag was applied
    zammad.add_article_to_ticket.assert_not_called()


@pytest.mark.asyncio
async def test_high_confidence_suspect_verdict_quarantines():
    bot, chat_system, zammad = _make_bot()
    zammad.get_ticket = MagicMock(return_value={
        "id": 42, "customer_id": 7, "title": "Payment request"})
    zammad.get_ticket_articles = MagicMock(return_value=[
        {"body": "Please wire funds to the new account immediately."}])
    bot.classifier.classify = AsyncMock(return_value=Classification(
        label="phishing_suspect", confidence=0.9, indicators=["urgency"]))

    await bot._process_ticket(42)

    assert _tagged(zammad, PHISHING_SUSPECT_TAG)
    assert _tagged(zammad, ZAMMAD_TRIAGE_TAG)
    # Pipeline stopped: no scout/analyst calls
    chat_system.text_engine.generate_response.assert_not_awaited()
    note = zammad.add_article_to_ticket.call_args.kwargs["body"]
    assert "QUARANTINED" in note
    assert "urgency" in note


@pytest.mark.asyncio
async def test_low_confidence_suspect_notes_but_does_not_quarantine():
    bot, chat_system, zammad = _make_bot()
    zammad.get_ticket = MagicMock(return_value={
        "id": 42, "customer_id": 7, "title": "Question about invoice"})
    zammad.get_ticket_articles = MagicMock(return_value=[{"body": "hello"}])
    bot.classifier.classify = AsyncMock(return_value=Classification(
        label="phishing_suspect", confidence=0.3, indicators=[]))
    bot._get_search_keywords = AsyncMock(return_value=None)

    await bot._process_ticket(42)

    assert not _tagged(zammad, PHISHING_SUSPECT_TAG)
    # First note is the low-confidence advisory; triage then proceeded to
    # post its own analysis note
    first_note = zammad.add_article_to_ticket.call_args_list[0].kwargs["body"]
    assert "LOW CONFIDENCE" in first_note
    bot._get_search_keywords.assert_awaited()
    assert _tagged(zammad, ZAMMAD_TRIAGE_TAG)


@pytest.mark.asyncio
async def test_clean_verdict_proceeds_with_pipeline():
    bot, chat_system, zammad = _make_bot()
    zammad.get_ticket = MagicMock(return_value={
        "id": 42, "customer_id": 7, "title": "Printer broken"})
    zammad.get_ticket_articles = MagicMock(return_value=[{"body": "it is broken"}])
    bot.classifier.classify = AsyncMock(return_value=Classification(
        label="clean", confidence=0.95))
    bot._get_search_keywords = AsyncMock(return_value=None)

    await bot._process_ticket(42)

    assert not _tagged(zammad, SECURITY_REPORT_TAG)
    assert not _tagged(zammad, PHISHING_SUSPECT_TAG)
    bot._get_search_keywords.assert_awaited()
    # Normal triage completed: analyst note + triage tag
    assert _tagged(zammad, ZAMMAD_TRIAGE_TAG)
    assert "analysis" in zammad.add_article_to_ticket.call_args.kwargs["body"]


@pytest.mark.asyncio
async def test_classifier_failure_fails_open():
    bot, chat_system, zammad = _make_bot()
    zammad.get_ticket = MagicMock(return_value={
        "id": 42, "customer_id": 7, "title": "Printer broken"})
    zammad.get_ticket_articles = MagicMock(return_value=[{"body": "it is broken"}])
    bot.classifier.classify = AsyncMock(return_value=None)
    bot._get_search_keywords = AsyncMock(return_value=None)

    await bot._process_ticket(42)

    # No quarantine artifacts; pipeline ran to completion as before DP-288
    assert not _tagged(zammad, SECURITY_REPORT_TAG)
    assert not _tagged(zammad, PHISHING_SUSPECT_TAG)
    bot._get_search_keywords.assert_awaited()
    assert _tagged(zammad, ZAMMAD_TRIAGE_TAG)


@pytest.mark.asyncio
async def test_tag_fetch_failure_still_classifies():
    bot, chat_system, zammad = _make_bot()
    zammad.get_tags = MagicMock(side_effect=RuntimeError("api down"))

    await bot._process_ticket(42)

    # Pre-signal path still quarantined the reported phishing
    assert _tagged(zammad, SECURITY_REPORT_TAG)
