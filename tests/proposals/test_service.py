# tests/proposals/test_service.py
"""Unit tests for the proposal review tool handlers (DP-282): real in-memory
proposal store, mocked executor. Covers the list/approve/deny lifecycle and
the audit trail."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from memory.memory_manager import MemoryManager
from src.proposals.executor import ProposalExecutor
from src.proposals.service import ProposalIntegration, ProposalToolHandler


@pytest.fixture
def mem_manager():
    manager = MemoryManager(db_path=":memory:")
    manager.create_schema()
    yield manager
    manager.close()


@pytest.fixture
def handler(mem_manager):
    executor = MagicMock(spec=ProposalExecutor)
    executor.execute = AsyncMock(return_value=(True, "internal note added to ticket #10001"))
    return ProposalToolHandler(mem_manager, executor)


def _queue(mem_manager, **overrides):
    defaults = dict(
        agent_name="managr",
        action_type="add_note",
        action_args={"ticket_number": 10001, "body": "follow up"},
        rationale="stale for 14d",
        taint={"source": "zammad_board_snapshot", "cycle_action_id": 1},
        source_action_id=1,
    )
    defaults.update(overrides)
    return mem_manager.create_proposal(**defaults)


def _audit_events(mem_manager):
    cursor = mem_manager._get_connection().cursor()
    cursor.execute("SELECT event_type, target_id FROM Audit_Log ORDER BY audit_id")
    return [(row["event_type"], row["target_id"]) for row in cursor.fetchall()]


def test_integration_registers_all_proposal_tools(mem_manager):
    integration = ProposalIntegration(mem_manager, MagicMock(spec=ProposalExecutor))
    assert integration.name == "proposals"
    tool_manager = MagicMock()
    integration.register_tools(tool_manager)
    registered = {call.args[0] for call in tool_manager.register.call_args_list}
    assert registered == {"list_proposals", "approve_proposal", "deny_proposal",
                          "add_standing_order", "list_standing_orders",
                          "retire_standing_order"}


@pytest.mark.asyncio
async def test_list_proposals_default_pending(handler, mem_manager):
    pid = _queue(mem_manager)
    denied = _queue(mem_manager, action_type="set_priority",
                    action_args={"ticket_number": 10002, "priority": "1 low"})
    mem_manager.review_proposal(denied, "denied", "operator", "no")

    result = await handler._list_proposals()
    assert result["count"] == 1
    entry = result["proposals"][0]
    assert entry["proposal_id"] == pid
    assert entry["action_type"] == "add_note"
    assert entry["args"] == {"ticket_number": 10001, "body": "follow up"}
    assert entry["rationale"] == "stale for 14d"

    everything = await handler._list_proposals(status="all")
    assert everything["count"] == 2


@pytest.mark.asyncio
async def test_approve_executes_and_audits(handler, mem_manager):
    pid = _queue(mem_manager)
    result = await handler._approve_proposal(pid, note="looks right")

    assert result["executed"] is True
    handler.executor.execute.assert_awaited_once()
    row = mem_manager.get_proposal(pid)
    assert row["status"] == "executed"
    assert row["reviewer"] == "operator"
    assert row["review_note"] == "looks right"
    assert _audit_events(mem_manager) == [
        ("proposal_approved", pid), ("proposal_executed", pid)]


@pytest.mark.asyncio
async def test_approve_execution_failure_recorded(handler, mem_manager):
    handler.executor.execute = AsyncMock(return_value=(False, "ticket #10001 not found"))
    pid = _queue(mem_manager)
    result = await handler._approve_proposal(pid)

    assert result["executed"] is False
    row = mem_manager.get_proposal(pid)
    assert row["status"] == "execution_failed"
    assert row["execution_result"] == "ticket #10001 not found"
    assert ("proposal_execution_failed", pid) in _audit_events(mem_manager)


@pytest.mark.asyncio
async def test_approve_survives_raising_executor(handler, mem_manager):
    """A raising executor must not strand the row in 'approved': the failure
    is recorded on the proposal and audited like any execution failure."""
    handler.executor.execute = AsyncMock(side_effect=RuntimeError("boom"))
    pid = _queue(mem_manager)
    result = await handler._approve_proposal(pid)

    assert result["executed"] is False
    assert "executor error" in result["result"]
    row = mem_manager.get_proposal(pid)
    assert row["status"] == "execution_failed"
    assert "boom" in row["execution_result"]
    assert ("proposal_execution_failed", pid) in _audit_events(mem_manager)


@pytest.mark.asyncio
async def test_approve_rejects_non_pending_and_missing(handler, mem_manager):
    pid = _queue(mem_manager)
    mem_manager.review_proposal(pid, "denied", "operator", "no")

    with pytest.raises(ValueError, match="not pending"):
        await handler._approve_proposal(pid)
    handler.executor.execute.assert_not_awaited()

    with pytest.raises(ValueError, match="No proposal"):
        await handler._approve_proposal(9999)


@pytest.mark.asyncio
async def test_deny_records_reason_without_executing(handler, mem_manager):
    pid = _queue(mem_manager)
    result = await handler._deny_proposal(pid, reason="priority is fine as-is")

    assert result["status"] == "denied"
    handler.executor.execute.assert_not_awaited()
    row = mem_manager.get_proposal(pid)
    assert row["status"] == "denied"
    assert row["review_note"] == "priority is fine as-is"
    assert _audit_events(mem_manager) == [("proposal_denied", pid)]


@pytest.mark.asyncio
async def test_expired_proposal_cannot_be_approved(handler, mem_manager):
    from datetime import datetime, timedelta, timezone
    pid = _queue(mem_manager,
                 expires_at=datetime.now(timezone.utc) - timedelta(days=1))

    listing = await handler._list_proposals()
    assert listing["expired_now"] == 1
    assert listing["count"] == 0

    with pytest.raises(ValueError, match="not pending"):
        await handler._approve_proposal(pid)
    handler.executor.execute.assert_not_awaited()


# --- Standing orders (DP-281) ---


@pytest.mark.asyncio
async def test_add_standing_order_records_and_audits(handler, mem_manager):
    result = await handler._add_standing_order("client Y tickets are always low priority")

    assert result["status"] == "active"
    order_id = result["order_id"]
    rows = mem_manager.list_standing_orders()
    assert len(rows) == 1
    assert rows[0]["order_text"] == "client Y tickets are always low priority"
    assert rows[0]["source"] == "operator"
    assert _audit_events(mem_manager) == [("standing_order_added", order_id)]


@pytest.mark.asyncio
async def test_add_standing_order_rejects_empty_text(handler, mem_manager):
    with pytest.raises(ValueError, match="empty"):
        await handler._add_standing_order("   ")
    assert mem_manager.list_standing_orders() == []


@pytest.mark.asyncio
async def test_list_standing_orders_filters_and_all(handler, mem_manager):
    active = (await handler._add_standing_order("keep flagging stale tickets"))["order_id"]
    retired = (await handler._add_standing_order("old rule"))["order_id"]
    await handler._retire_standing_order(retired)

    default = await handler._list_standing_orders()
    assert [o["order_id"] for o in default["orders"]] == [active]

    everything = await handler._list_standing_orders(status="all")
    assert everything["count"] == 2

    retired_only = await handler._list_standing_orders(status="retired")
    assert [o["order_id"] for o in retired_only["orders"]] == [retired]


@pytest.mark.asyncio
async def test_retire_standing_order_audits_and_rejects_double(handler, mem_manager):
    order_id = (await handler._add_standing_order("stop flagging ticket #123"))["order_id"]

    result = await handler._retire_standing_order(order_id, note="ticket closed")
    assert result["status"] == "retired"
    rows = mem_manager.list_standing_orders(status="retired")
    assert rows[0]["retire_note"] == "ticket closed"
    assert _audit_events(mem_manager) == [
        ("standing_order_added", order_id),
        ("standing_order_retired", order_id),
    ]

    with pytest.raises(ValueError, match="No active standing order"):
        await handler._retire_standing_order(order_id)
    with pytest.raises(ValueError, match="No active standing order"):
        await handler._retire_standing_order(9999)
