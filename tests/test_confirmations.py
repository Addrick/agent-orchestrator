# tests/test_confirmations.py
"""DP-297 — the token-keyed gated-write store.

Unit-level coverage of ConfirmationManager itself: the token index, the
double-resolve guard, in-place history patching, and lazy expiry. The
end-to-end park → approve → summarize flow lives in
tests/integration/test_resume_kernel_convergence.py.
"""

import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from config.global_config import PENDING_ACTION_TTL
from src.confirmations import (
    DENIAL_INSTRUCTION, ConfirmationManager, Decision, ParkedWrite,
)
from src.memory.memory_manager import MemoryManager


@pytest.fixture
def mem_manager():
    manager = MemoryManager(db_path=":memory:")
    manager.create_schema()
    yield manager
    manager.close()


@pytest.fixture
def manager(mem_manager):
    tool_manager = MagicMock()
    tool_manager.execute_tool = AsyncMock(return_value={"ok": True})
    mgr = ConfirmationManager(lambda: tool_manager, mem_manager)
    mgr._tool_manager = tool_manager  # handle for assertions
    return mgr


def _park(token="t1", user="u", persona="p", tool="update_ticket",
          call_id="c1", row_id=None, created_at=None):
    return ParkedWrite(
        token=token,
        write_call={"id": call_id, "name": tool, "arguments": {"x": 1}},
        audit_info={"actions": [{"tool": tool}]},
        confirmation_text=f"Run {tool}?",
        user_identifier=user,
        persona_name=persona,
        parked_assistant_id=row_id,
        created_at=created_at if created_at is not None else time.time(),
    )


# ---- store ---------------------------------------------------------------

def test_parks_accumulate_in_order(manager):
    manager.park(_park(token="a"))
    manager.park(_park(token="b"))
    manager.park(_park(token="c"))

    assert [p.token for p in manager.list_for("u", "p")] == ["a", "b", "c"]


def test_parks_are_isolated_per_conversation(manager):
    manager.park(_park(token="a", user="alice"))
    manager.park(_park(token="b", user="bob"))

    assert [p.token for p in manager.list_for("alice", "p")] == ["a"]
    assert [p.token for p in manager.list_for("bob", "p")] == ["b"]


def test_take_is_single_shot(manager):
    manager.park(_park(token="a"))

    assert manager.take("a") is not None
    # The second caller loses — this is the double-click guard, and it works
    # because `take` never awaits, so no other task can interleave inside it.
    assert manager.take("a") is None
    assert manager.list_for("u", "p") == []


def test_restore_puts_a_claimed_park_back(manager):
    manager.park(_park(token="a"))
    parked = manager.take("a")
    manager.restore(parked)

    assert [p.token for p in manager.list_for("u", "p")] == ["a"]


def test_taking_one_park_leaves_its_siblings(manager):
    manager.park(_park(token="a"))
    manager.park(_park(token="b"))

    manager.take("a")

    assert [p.token for p in manager.list_for("u", "p")] == ["b"]


# ---- history patching ----------------------------------------------------

def _row_with_awaiting_entries(mem_manager, call_ids):
    """Persist an assistant row whose tool_context gates `call_ids`."""
    msgs = [{
        "role": "assistant",
        "tool_calls": [{"id": cid, "name": "update_ticket", "arguments": {}}
                       for cid in call_ids],
    }]
    for cid in call_ids:
        msgs.append({
            "role": "tool", "tool_call_id": cid, "name": "update_ticket",
            "content": json.dumps({"status": "awaiting_human_approval",
                                   "token": f"tok-{cid}"}),
        })
    row_id = mem_manager.log_message(
        user_identifier="u", persona_name="p", channel="c",
        author_role="assistant", author_name="p", content="Proposed.",
        timestamp=datetime.now(), tool_context=json.dumps(msgs),
    )
    return row_id


def test_patch_rewrites_only_the_matching_entry(manager, mem_manager):
    """Sibling proposals in the same row must be left alone.

    A turn seals ALL its gated writes into one tool_context blob, so patching
    is a read-modify-write of a shared structure — resolving one proposal must
    not disturb the others.
    """
    row_id = _row_with_awaiting_entries(mem_manager, ["c1", "c2", "c3"])
    parked = _park(token="tok-c2", call_id="c2", row_id=row_id)

    assert manager.patch_parked_entry(parked, "approved", {"ticket_id": 9})

    msgs = json.loads(mem_manager.get_tool_context(row_id))
    by_id = {m["tool_call_id"]: json.loads(m["content"])
             for m in msgs if m.get("role") == "tool"}
    assert by_id["c2"] == {"status": "approved", "token": "tok-c2",
                           "result": {"ticket_id": 9}}
    assert by_id["c1"]["status"] == "awaiting_human_approval"
    assert by_id["c3"]["status"] == "awaiting_human_approval"


def test_patch_scrubs_the_result(manager, mem_manager):
    """The patched result reaches replayed history and the portal transcript,
    so it is scrubbed exactly like a live tool result."""
    from src.security.scrubber import get_scrubber, reset_scrubber

    reset_scrubber()
    get_scrubber().register("hunter2supersecret", "TEST_KEY")
    try:
        row_id = _row_with_awaiting_entries(mem_manager, ["c1"])
        parked = _park(token="tok-c1", call_id="c1", row_id=row_id)

        manager.patch_parked_entry(
            parked, "approved", {"echo": "hunter2supersecret"})

        blob = mem_manager.get_tool_context(row_id)
        assert "hunter2supersecret" not in blob
        assert "[REDACTED:TEST_KEY]" in blob
    finally:
        reset_scrubber()


def test_patch_is_a_noop_without_a_row(manager):
    """A park whose turn never committed a row cannot be patched — and must
    not raise, since the write itself may already have executed."""
    assert manager.patch_parked_entry(_park(row_id=None), "approved", {}) is False


def test_patch_reports_a_missing_entry(manager, mem_manager):
    """A call_id absent from the blob returns False rather than corrupting it."""
    row_id = _row_with_awaiting_entries(mem_manager, ["c1"])
    parked = _park(token="tok-zz", call_id="does-not-exist", row_id=row_id)

    assert manager.patch_parked_entry(parked, "approved", {}) is False
    msgs = json.loads(mem_manager.get_tool_context(row_id))
    assert json.loads(msgs[1]["content"])["status"] == "awaiting_human_approval"


# ---- resolution ----------------------------------------------------------

@pytest.mark.asyncio
async def test_apply_patches_history_after_executing(manager, mem_manager):
    row_id = _row_with_awaiting_entries(mem_manager, ["c1"])
    parked = _park(token="tok-c1", call_id="c1", row_id=row_id)
    manager.park(parked)

    decision = Decision(park=parked, approved=True)
    await manager.apply(decision)

    entry = json.loads(mem_manager.get_tool_context(row_id))[1]
    assert json.loads(entry["content"])["status"] == "approved"


@pytest.mark.asyncio
async def test_apply_records_a_denial_in_history(manager, mem_manager):
    row_id = _row_with_awaiting_entries(mem_manager, ["c1"])
    parked = _park(token="tok-c1", call_id="c1", row_id=row_id)

    await manager.apply(Decision(park=parked, approved=False, note="nope"))

    payload = json.loads(json.loads(
        mem_manager.get_tool_context(row_id))[1]["content"])
    assert payload["status"] == "denied"
    assert payload["result"]["note"] == "nope"
    manager._tool_manager.execute_tool.assert_not_called()


# ---- queue / drain -------------------------------------------------------

def test_drain_takes_everything_queued(manager):
    a, b = _park(token="a"), _park(token="b")
    manager.enqueue(Decision(park=a, approved=True))
    manager.enqueue(Decision(park=b, approved=False))

    batch = manager.drain(("u", "p"))
    assert [d.park.token for d in batch] == ["a", "b"]
    # Draining is destructive — a second holder of the lock must not re-run
    # decisions the first one already applied.
    assert manager.drain(("u", "p")) == []


def test_drain_is_per_conversation(manager):
    manager.enqueue(Decision(park=_park(token="a", user="alice"), approved=True))
    manager.enqueue(Decision(park=_park(token="b", user="bob"), approved=True))

    assert [d.park.token for d in manager.drain(("alice", "p"))] == ["a"]
    assert [d.park.token for d in manager.drain(("bob", "p"))] == ["b"]


# ---- expiry --------------------------------------------------------------

def test_sweep_expires_only_stale_parks(manager, mem_manager):
    manager.park(_park(token="fresh"))
    manager.park(_park(token="stale",
                       created_at=time.time() - PENDING_ACTION_TTL - 10))

    assert manager.sweep_expired() == 1
    assert [p.token for p in manager.list_for("u", "p")] == ["fresh"]


def test_sweep_patches_the_expired_entry(manager, mem_manager):
    """An expired proposal must stop reading as pending in history, or the
    model keeps believing a write is still awaiting review."""
    row_id = _row_with_awaiting_entries(mem_manager, ["c1"])
    manager.park(_park(token="tok-c1", call_id="c1", row_id=row_id,
                       created_at=time.time() - PENDING_ACTION_TTL - 10))

    manager.sweep_expired()

    payload = json.loads(json.loads(
        mem_manager.get_tool_context(row_id))[1]["content"])
    assert payload["status"] == "expired"


def test_sweep_logs_an_audit_row(manager, mem_manager):
    manager.park(_park(token="stale",
                       created_at=time.time() - PENDING_ACTION_TTL - 10))
    manager.sweep_expired()

    conn = mem_manager._get_connection()
    row = conn.execute(
        "SELECT * FROM Audit_Log WHERE event_type='audit_park_expired'"
    ).fetchone()
    assert row is not None
    assert row["new_state"] == "expired"


def test_listing_sweeps_lazily(manager):
    """Expiry has no background task (a daemon here would reintroduce the
    DP-304 shutdown-contract problem), so reads must sweep."""
    manager.pending["stale"] = _park(
        token="stale", created_at=time.time() - PENDING_ACTION_TTL - 10)
    manager._by_key[("u", "p")] = ["stale"]

    assert manager.list_for("u", "p") == []


def test_is_expired_boundary(manager):
    just_inside = _park(created_at=time.time() - PENDING_ACTION_TTL + 5)
    just_outside = _park(created_at=time.time() - PENDING_ACTION_TTL - 5)

    assert manager.is_expired(just_inside) is False
    assert manager.is_expired(just_outside) is True


# ---- denial carries its own standing instruction -------------------------


@pytest.mark.asyncio
async def test_denial_instruction_persists_in_the_patched_entry(
        manager, mem_manager):
    """The "wait" instruction must live in the entry, not in the nudge.

    The continuation nudge is ephemeral by design, so a denial framed only
    there decays after one turn into a bare `error` — which is the shape this
    codebase uses everywhere else to mean "the tool failed, adapt and retry".
    The verdict and what to do about it have the same lifetime because they are
    the same fact.
    """
    row_id = _row_with_awaiting_entries(mem_manager, ["c1"])
    parked = _park(token="tok-c1", call_id="c1", row_id=row_id)

    await manager.apply(Decision(park=parked, approved=False))

    entry = json.loads(next(
        m for m in json.loads(mem_manager.get_tool_context(row_id))
        if m.get("tool_call_id") == "c1"
    )["content"])
    assert entry["status"] == "denied"
    assert entry["result"]["error"] == DENIAL_INSTRUCTION
    # Not merely a verdict: it names the state the model should now be in.
    assert "Wait for corrections" in entry["result"]["error"]


@pytest.mark.asyncio
async def test_denial_instruction_survives_without_an_operator_note(
        manager, mem_manager):
    """Discord's reaction path sends no note, so the note must not be what
    carries the instruction."""
    row_id = _row_with_awaiting_entries(mem_manager, ["c1"])
    parked = _park(token="tok-c1", call_id="c1", row_id=row_id)

    await manager.apply(Decision(park=parked, approved=False, note=None))

    entry = json.loads(next(
        m for m in json.loads(mem_manager.get_tool_context(row_id))
        if m.get("tool_call_id") == "c1"
    )["content"])
    assert entry["result"]["note"] is None
    assert entry["result"]["error"] == DENIAL_INSTRUCTION


@pytest.mark.asyncio
async def test_approval_does_not_carry_the_denial_instruction(
        manager, mem_manager):
    """An approved write reports its real result and nothing else — telling a
    model to wait after a successful action would be worse than saying
    nothing."""
    row_id = _row_with_awaiting_entries(mem_manager, ["c1"])
    parked = _park(token="tok-c1", call_id="c1", row_id=row_id)

    await manager.apply(Decision(park=parked, approved=True))

    blob = mem_manager.get_tool_context(row_id)
    assert DENIAL_INSTRUCTION not in blob


# ---- DP-297 review #3: approved-but-failed is its own outcome -------------

def test_status_separates_the_two_axes():
    """`approved` is what the operator decided; `ok` is whether the tool ran.
    Deriving the durable status from `approved` alone collapsed them, so an
    approved write that then failed was recorded as a plain success."""
    park = _park()
    assert Decision(park=park, approved=True, ok=True).status == "approved"
    assert Decision(park=park, approved=True, ok=False).status == \
        "approved_but_failed"
    assert Decision(park=park, approved=False, ok=False).status == "denied"
    # A denial is a denial regardless of `ok` — nothing ran, so the flag is
    # meaningless on that branch and must not leak into the status.
    assert Decision(park=park, approved=False, ok=True).status == "denied"


@pytest.mark.asyncio
async def test_apply_records_an_approved_failure_distinctly(
        manager, mem_manager):
    """The failure has to be visible to a reader that keys off `status`.

    Before this, history said `approved` and the only statement of the failure
    was the `error` inside `result` — which denial ALSO produces, so `error`
    alone cannot distinguish "the operator refused" from "it broke". That is
    the same defect DENIAL_INSTRUCTION fixes one branch over."""
    row_id = _row_with_awaiting_entries(mem_manager, ["c1"])
    parked = _park(token="tok-c1", call_id="c1", row_id=row_id)
    manager.park(parked)
    manager._tool_manager.execute_tool.side_effect = RuntimeError("zammad 500")

    decision = Decision(park=parked, approved=True)
    await manager.apply(decision)

    payload = json.loads(json.loads(
        mem_manager.get_tool_context(row_id))[1]["content"])
    assert payload["status"] == "approved_but_failed"
    assert payload["status"] != "denied", "the operator did approve it"
    assert "zammad 500" in json.dumps(payload["result"])
    assert decision.ok is False


@pytest.mark.asyncio
async def test_audit_row_carries_the_failed_status(manager, mem_manager):
    """The audit trail keys off the same property, so it must not read
    `approved` for a write that never landed."""
    row_id = _row_with_awaiting_entries(mem_manager, ["c1"])
    parked = _park(token="tok-c1", call_id="c1", row_id=row_id)
    manager.park(parked)
    manager._tool_manager.execute_tool.side_effect = RuntimeError("boom")

    await manager.apply(Decision(park=parked, approved=True))

    conn = mem_manager._get_connection()
    row = conn.execute(
        "SELECT * FROM Audit_Log WHERE event_type='audit_decision'"
    ).fetchone()
    assert row is not None, "apply() must log an audit_decision row"
    assert row["new_state"] == "approved_but_failed"


# ---- the ok-derivation: PARK_STATUS_FAILED was unreachable in production ----


@pytest.mark.asyncio
async def test_a_real_tool_manager_records_a_failed_write_as_failed(
        mem_manager):
    """`approved_but_failed` must be reachable through the REAL ToolManager.

    Every other test producing that status gives a *mocked* `execute_tool` a
    `side_effect`, i.e. asserts a raise the production class cannot emit:
    `ToolManager.execute_tool` catches every handler exception and RETURNS
    `{"error": ...}`. Deriving `decision.ok` from reaching the line after the
    call therefore made it unconditionally True — a Zammad 500 that fired
    *after* the ticket was created was recorded as a plain `approved`, and
    every consumer keyed off `PARK_STATUS_FAILED` (the "approved but FAILED"
    continuation line, `executed_ok` in the audit row) was dead code.

    Drives the real class on purpose. A mock here would re-assert the defect.
    """
    from src.tools.tool_manager import ToolManager

    async def boom(**_kwargs):
        raise RuntimeError("zammad 500 after the ticket was created")

    tool_manager = ToolManager()
    tool_manager.register("update_ticket", boom)
    mgr = ConfirmationManager(lambda: tool_manager, mem_manager)

    parked = _park(token="a", call_id="c1")
    mgr.park(parked)
    decision = Decision(park=parked, approved=True)
    await mgr.apply(decision)

    assert decision.ok is False
    assert decision.status == "approved_but_failed"


@pytest.mark.asyncio
async def test_a_real_tool_manager_records_a_successful_write_as_approved(
        mem_manager):
    """The other half, so the fix is not "everything is a failure now"."""
    from src.tools.tool_manager import ToolManager

    async def fine(**_kwargs):
        return {"ticket": 7}

    tool_manager = ToolManager()
    tool_manager.register("update_ticket", fine)
    mgr = ConfirmationManager(lambda: tool_manager, mem_manager)

    parked = _park(token="a", call_id="c1")
    mgr.park(parked)
    decision = Decision(park=parked, approved=True)
    await mgr.apply(decision)

    assert decision.ok is True
    assert decision.status == "approved"
