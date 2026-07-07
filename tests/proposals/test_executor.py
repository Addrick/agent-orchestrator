# tests/proposals/test_executor.py
"""Unit tests for ProposalExecutor (DP-282): whitelist dispatch + execution-time
re-validation. Zammad fully mocked."""

import pytest
from unittest.mock import MagicMock

from src.proposals.executor import ProposalExecutor


def _make_executor(ticket=None):
    client = MagicMock()
    client.search_tickets = MagicMock(
        return_value=[ticket] if ticket else [{"id": 7, "number": "10001", "title": "T"}]
    )
    return ProposalExecutor(client), client


def _proposal(action_type, args):
    return {"proposal_id": 1, "action_type": action_type, "action_args": args}


@pytest.mark.asyncio
async def test_add_note_forces_internal():
    executor, client = _make_executor()
    ok, result = await executor.execute(
        _proposal("add_note", {"ticket_number": 10001, "body": "call them back"}))
    assert ok
    client.add_article_to_ticket.assert_called_once_with(
        ticket_id=7, body="call them back", internal=True)
    assert "#10001" in result


@pytest.mark.asyncio
async def test_set_priority_dispatch():
    executor, client = _make_executor()
    ok, result = await executor.execute(
        _proposal("set_priority", {"ticket_number": 10001, "priority": "3 high"}))
    assert ok
    client.update_ticket.assert_called_once_with(
        ticket_id=7, payload={"priority": "3 high"})


@pytest.mark.asyncio
async def test_remind_dispatch():
    executor, client = _make_executor()
    ok, result = await executor.execute(
        _proposal("remind", {"ticket_number": 10001, "pending_until": "2026-08-01"}))
    assert ok
    client.update_ticket.assert_called_once_with(
        ticket_id=7,
        payload={"state": "pending reminder", "pending_time": "2026-08-01T09:00:00Z"})


@pytest.mark.asyncio
async def test_tampered_row_fails_revalidation_without_any_call():
    """A row edited between review and execution still can't smuggle args."""
    executor, client = _make_executor()
    ok, result = await executor.execute(
        _proposal("add_note", {"ticket_number": 10001, "body": "x", "internal": False}))
    assert not ok
    assert "re-validation" in result
    client.search_tickets.assert_not_called()
    client.add_article_to_ticket.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_action_fails_revalidation():
    executor, client = _make_executor()
    ok, result = await executor.execute(_proposal("close_ticket", {"ticket_number": 10001}))
    assert not ok
    client.update_ticket.assert_not_called()


@pytest.mark.asyncio
async def test_missing_ticket_fails_cleanly():
    executor, client = _make_executor()
    client.search_tickets.return_value = []
    ok, result = await executor.execute(
        _proposal("set_priority", {"ticket_number": 999, "priority": "1 low"}))
    assert not ok
    assert "not found" in result


@pytest.mark.asyncio
async def test_resolution_error_reported_not_raised():
    """A network/HTTP failure during ticket resolution must return a failure
    result, not raise — a raising executor strands the row in 'approved'."""
    executor, client = _make_executor()
    client.search_tickets.side_effect = RuntimeError("connection refused")
    ok, result = await executor.execute(
        _proposal("set_priority", {"ticket_number": 10001, "priority": "1 low"}))
    assert not ok
    assert "ticket resolution failed" in result
    client.update_ticket.assert_not_called()


@pytest.mark.asyncio
async def test_zammad_error_reported_not_raised():
    executor, client = _make_executor()
    client.update_ticket.side_effect = RuntimeError("boom")
    ok, result = await executor.execute(
        _proposal("set_priority", {"ticket_number": 10001, "priority": "1 low"}))
    assert not ok
    assert "zammad call failed" in result
