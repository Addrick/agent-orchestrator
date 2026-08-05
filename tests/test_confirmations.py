# tests/test_confirmations.py
"""DP-297 — the token-keyed gated-write store.

Unit-level coverage of ConfirmationManager itself: the token index, the
double-resolve guard, in-place history patching, and lazy expiry. The
end-to-end park → approve → summarize flow lives in
tests/integration/test_resume_kernel_convergence.py.
"""

import json
import sqlite3
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from config.global_config import PARK_ROW_RETENTION, PENDING_ACTION_TTL
from src.confirmations import (
    DENIAL_INSTRUCTION, ConfirmationManager, Decision, ParkedWrite,
)
from src.memory.memory_manager import MemoryManager, PARK_DB_UNKNOWN
from src.tools.tool_loop import (
    PARK_STATUS_APPROVED, PARK_STATUS_QUARANTINED,
)


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

# ---- DP-319: durability ---------------------------------------------------
#
# The property under test throughout this section is the one DP-297 could not
# offer: a park outlives the process. Every test here therefore either inspects
# the durable row directly or rebuilds a SECOND manager over the same database,
# because a test that only exercises `manager` proves nothing about a restart —
# the in-memory dict passes it either way.

def _fresh_manager(mem_manager):
    """A manager with EMPTY in-memory state over the same database.

    This is the restart: same DB, no dict, no indexes, no locks.
    """
    tm = MagicMock()
    tm.execute_tool = AsyncMock(return_value={"ok": True})
    mgr = ConfirmationManager(lambda: tm, mem_manager)
    mgr._tool_manager = tm
    return mgr


def _park_row(mem_manager, token):
    conn = mem_manager._get_connection()
    return conn.execute(
        "SELECT * FROM Parked_Writes WHERE token = ?", (token,)).fetchone()


def test_park_writes_a_durable_row(manager, mem_manager):
    manager.park(_park(token="a", row_id=7))

    row = _park_row(mem_manager, "a")
    assert row is not None, "the park must exist outside the process"
    assert row["status"] == "pending"
    assert row["parked_assistant_id"] == 7
    assert json.loads(row["write_call"])["name"] == "update_ticket"


def test_take_claims_the_durable_row(manager, mem_manager):
    manager.park(_park(token="a"))
    manager.take("a")

    assert _park_row(mem_manager, "a")["status"] == "claimed"


def test_restore_releases_the_durable_row(manager, mem_manager):
    manager.park(_park(token="a"))
    manager.restore(manager.take("a"))

    assert _park_row(mem_manager, "a")["status"] == "pending"


def test_a_park_survives_a_restart(manager, mem_manager):
    """The whole ticket, in one test: park, lose the process, resolve it.

    `_fresh_manager` shares only the database, so everything the original
    manager held in memory is gone — which is what a restart does.
    """
    manager.park(_park(token="a", row_id=3))

    revived = _fresh_manager(mem_manager)
    assert revived.list_for("u", "p") == [], "nothing is loaded until rebuild"

    revived.rebuild_from_store()

    parks = revived.list_for("u", "p")
    assert [p.token for p in parks] == ["a"]
    assert parks[0].parked_assistant_id == 3, \
        "without the row id the revived park cannot patch its history entry"
    assert parks[0].write_call["arguments"] == {"x": 1}
    assert revived.take("a") is not None, "the revived park must be resolvable"


def test_a_restart_preserves_duplicate_refs(manager, mem_manager):
    """A suppressed duplicate's pointer has to survive too.

    It is the only thing that will ever correct that duplicate's "still
    awaiting the operator" entry, so losing it on restart leaves history
    asserting a decided action is queued, permanently.
    """
    parked = _park(token="a")
    manager.park(parked)
    manager.note_duplicate_ref(parked, 41, "c-dup")

    revived = _fresh_manager(mem_manager)
    revived.rebuild_from_store()

    assert revived.list_for("u", "p")[0].duplicate_refs == [(41, "c-dup")]


def test_restart_expires_a_park_whose_ttl_passed_while_down(
        manager, mem_manager):
    """The orphaned-`awaiting_human_approval` bug, from the PR #189 review.

    The lazy sweep only walks `self.pending`, so a park a restart never loaded
    is a park it never expires: the DB row keeps saying "awaiting", and the
    model reads that every turn and waits forever for a result no code path can
    produce.
    """
    row_id = _row_with_awaiting_entries(mem_manager, ["c1"])
    manager.park(_park(token="tok-c1", call_id="c1", row_id=row_id,
                       created_at=time.time() - PENDING_ACTION_TTL - 60))

    revived = _fresh_manager(mem_manager)
    counts = revived.rebuild_from_store()

    assert counts["expired"] == 1
    assert revived.list_for("u", "p") == []
    entry = json.loads(json.loads(
        mem_manager.get_tool_context(row_id))[1]["content"])
    assert entry["status"] == "expired", \
        "history must stop claiming the write is awaiting an operator"
    assert _park_row(mem_manager, "tok-c1")["status"] == "expired"


def test_restart_does_not_re_execute_a_claimed_park(manager, mem_manager):
    """A decision in flight when the process died is NOT retried.

    The write may already have run; a gated write is gated because it is
    irreversible, so re-running it on a guess is the failure this subsystem
    exists to prevent. The park is terminated as `interrupted_by_restart` and
    the model is told to re-check state instead.
    """
    row_id = _row_with_awaiting_entries(mem_manager, ["c1"])
    manager.park(_park(token="tok-c1", call_id="c1", row_id=row_id))
    manager.take("tok-c1")  # claimed — the process dies here

    revived = _fresh_manager(mem_manager)
    counts = revived.rebuild_from_store()

    assert counts["interrupted"] == 1
    revived._tool_manager.execute_tool.assert_not_called()
    assert revived.list_for("u", "p") == []
    entry = json.loads(json.loads(
        mem_manager.get_tool_context(row_id))[1]["content"])
    assert entry["status"] == "interrupted_by_restart"
    assert "unknown" in entry["result"]["error"], \
        "the model must not read this as a plain failure and retry"

    conn = mem_manager._get_connection()
    assert conn.execute(
        "SELECT COUNT(*) FROM Audit_Log "
        "WHERE event_type='audit_park_interrupted'").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_resolving_erases_the_payload_but_keeps_the_row(
        manager, mem_manager):
    """A decided park keeps its identity hash and loses its arguments.

    The row survives because the re-execution guard reads it; the arguments do
    not, because nothing needs them once the write has run and this table sits
    on disk for a week.
    """
    row_id = _row_with_awaiting_entries(mem_manager, ["c1"])
    parked = _park(token="tok-c1", call_id="c1", row_id=row_id)
    manager.park(parked)
    manager.take("tok-c1")

    await manager.apply(Decision(park=parked, approved=True))

    row = _park_row(mem_manager, "tok-c1")
    assert row["status"] == "resolved"
    assert row["resolution"] == "approved"
    assert row["write_call"] is None, "arguments must not outlive the decision"
    assert row["audit_info"] is None
    assert row["call_identity"], "the identity hash is what the guard needs"


def test_expiry_finalizes_the_durable_row(manager, mem_manager):
    manager.park(_park(token="a",
                       created_at=time.time() - PENDING_ACTION_TTL - 60))
    manager.sweep_expired()

    row = _park_row(mem_manager, "a")
    assert row["status"] == "expired"
    assert row["write_call"] is None


def test_purge_drops_only_old_terminal_rows(manager, mem_manager):
    manager.park(_park(token="live"))
    manager.park(_park(token="old"))
    manager.take("old")
    mem_manager.finalize_parked_write(
        "old", "resolved", "approved", now=time.time() - 1000)

    assert mem_manager.purge_parked_writes(time.time() - 500) == 1
    assert _park_row(mem_manager, "old") is None
    assert _park_row(mem_manager, "live") is not None, \
        "a pending park has no resolved_at and must never be purged"


def test_an_unserializable_write_call_is_not_persisted(manager, mem_manager):
    """Better in-memory-only than stored lossily.

    The stored payload is what an approved write executes with after a restart,
    so a `default=str` fallback would mean running the tool with the repr of an
    argument instead of the argument itself.
    """
    parked = _park(token="a")
    parked.write_call = {"id": "c1", "name": "update_ticket",
                         "arguments": {"blob": object()}}
    manager.park(parked)

    assert _park_row(mem_manager, "a") is None
    assert [p.token for p in manager.list_for("u", "p")] == ["a"], \
        "it still works for this process; only durability is given up"


# ---- DP-319: the re-execution guard (PR #189 review finding B) -------------

@pytest.mark.asyncio
async def test_an_executed_write_is_not_parkable_again(manager, mem_manager):
    """The double-execution path this ticket closes.

    During the continuation the park has already been taken, so `list_for` is
    blind to it — and that turn is exactly when the model re-proposes, since it
    is re-reading its own tool span.
    """
    parked = _park(token="a", call_id="c1")
    manager.park(parked)
    manager.take("a")
    await manager.apply(Decision(park=parked, approved=True))

    assert manager.list_for("u", "p") == [], "the park is gone from the index"
    hit = manager.already_resolved(("u", "p"), dict(parked.write_call, id="c2"))
    assert hit is not None and hit["resolution"] == "approved"


@pytest.mark.asyncio
async def test_a_denied_write_may_be_proposed_again(manager, mem_manager):
    """Nothing ran, so a second proposal is a new request, not a repeat.

    DP-297 supports this explicitly: an operator who denies a write and then
    asks for it on purpose must be able to reach it.
    """
    parked = _park(token="a", call_id="c1")
    manager.park(parked)
    manager.take("a")
    await manager.apply(Decision(park=parked, approved=False))

    assert manager.already_resolved(("u", "p"), dict(parked.write_call)) is None


@pytest.mark.asyncio
async def test_the_reexecution_guard_expires(manager, mem_manager):
    """Sized for the continuation turn, not for the park's whole TTL.

    A day-wide guard would silently refuse a legitimate repeat of the same
    action hours later.
    """
    from config.global_config import PARK_REEXECUTION_GUARD_WINDOW

    parked = _park(token="a", call_id="c1")
    manager.park(parked)
    manager.take("a")
    await manager.apply(Decision(park=parked, approved=True))

    conn = mem_manager._get_connection()
    conn.execute("UPDATE Parked_Writes SET resolved_at = ? WHERE token = 'a'",
                 (time.time() - PARK_REEXECUTION_GUARD_WINDOW - 60,))
    conn.commit()

    assert manager.already_resolved(("u", "p"), dict(parked.write_call)) is None


@pytest.mark.asyncio
async def test_the_guard_is_scoped_to_one_conversation(manager, mem_manager):
    """Another operator's decision must not suppress this one's proposal."""
    parked = _park(token="a", user="alice", call_id="c1")
    manager.park(parked)
    manager.take("a")
    await manager.apply(Decision(park=parked, approved=True))

    assert manager.already_resolved(
        ("bob", "p"), dict(parked.write_call)) is None


# ---- DP-319 review: the durable store's own failure modes ------------------
#
# Everything below covers a way the durable half can be WORSE than no durable
# half: a park lost from memory while its row says pending, a decided write
# resurrected as approvable, a boot that never completes, secrets kept forever,
# a decision fabricated for a park nobody touched. The suite above does not
# distinguish any of these from correct behaviour.


class _BrokenConnection:
    """A connection whose every statement raises, like a locked database."""

    def cursor(self):
        return self

    def execute(self, *_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    def commit(self):
        raise sqlite3.OperationalError("database is locked")

    def rollback(self):
        return None


def test_take_does_not_raise_when_the_store_is_unavailable(
        manager, mem_manager, monkeypatch):
    """A DB error after the in-memory pop must not lose the park.

    `take` pops first and claims second, so an exception from the claim escapes
    with the park already removed from `pending`: the operator's approval 500s,
    the park is gone from this process entirely, and its row still reads
    `pending` — so it silently reappears as an unanswered proposal on the next
    restart. `claim_parked_write` swallows `sqlite3.Error` for exactly that
    reason; nothing in `stream_resolve_park` would have caught it.
    """
    manager.park(_park(token="a"))
    monkeypatch.setattr(mem_manager, "_get_connection", _BrokenConnection)

    parked = manager.take("a")

    assert parked is not None, "the caller must still get its park"
    assert manager.pending == {}, "and the pop must still have happened"


def test_restore_refuses_to_resurrect_a_decided_park(manager, mem_manager):
    """A terminal row must win over the in-memory restore.

    `release_parked_write` returns False both when the row is missing and when
    it is already terminal, and the missing-row branch re-INSERTs with
    `status='pending'` — which, being `INSERT OR REPLACE`, rewinds `resolved_at`
    and `resolution` to NULL. Treating the two alike turns an irreversible write
    that ALREADY RAN back into an approvable affordance that survives the next
    restart.
    """
    parked = _park(token="a")
    manager.park(parked)
    manager.take("a")
    # Decided by some other path between the take and the restore.
    mem_manager.finalize_parked_write("a", "resolved", "approved", "ran")

    manager.restore(parked)

    assert manager.list_for("u", "p") == [], \
        "a write that already ran must not be offered again"
    row = _park_row(mem_manager, "a")
    assert row["status"] == "resolved"
    assert row["resolution"] == "approved"
    assert row["resolved_at"] is not None, "the decision must not be rewound"


def test_restore_reinserts_a_park_whose_row_vanished(manager, mem_manager):
    """The other half of that branch, so the fix is not merely "never
    re-insert": a genuinely missing row still has to come back, or the restored
    park would vanish on the next restart."""
    parked = _park(token="a")
    manager.park(parked)
    manager.take("a")
    conn = mem_manager._get_connection()
    conn.execute("DELETE FROM Parked_Writes WHERE token = 'a'")
    conn.commit()

    manager.restore(parked)

    assert [p.token for p in manager.list_for("u", "p")] == ["a"]
    assert _park_row(mem_manager, "a")["status"] == "pending"


@pytest.mark.asyncio
async def test_an_approved_but_failed_write_is_not_parkable_again(
        manager, mem_manager):
    """`approved_but_failed` means the tool RAN and then raised.

    A ticket created before the API returned 500; a write that landed before
    the client timed out. Excluding it from the guard reopens the exact
    double-execution hole for the worst case — the operator reads "it failed",
    approves the re-proposal, and gets two tickets.
    """
    manager._tool_manager.execute_tool = AsyncMock(
        side_effect=RuntimeError("500 after the ticket was created"))
    parked = _park(token="a", call_id="c1")
    manager.park(parked)
    manager.take("a")
    decision = Decision(park=parked, approved=True)
    await manager.apply(decision)

    assert decision.status == "approved_but_failed"
    hit = manager.already_resolved(("u", "p"), dict(parked.write_call, id="c2"))
    assert hit is not None, "a call that may have taken effect must not re-park"
    assert hit["resolution"] == "approved_but_failed"


# ---- boot must not be the strictest path in the system --------------------

def _corrupt_write_call(mem_manager, token):
    conn = mem_manager._get_connection()
    conn.execute("UPDATE Parked_Writes SET write_call = ? WHERE token = ?",
                 ('{"id": "c1", "name": "update_tic', token))
    conn.commit()


def test_a_row_with_an_undecodable_payload_is_quarantined(
        manager, mem_manager):
    """An unreadable call must not come back as an approvable park.

    Defaulting it to `{}` produces a park that renders a perfectly normal
    approve/deny affordance, executes tool "unknown" if approved, and — having
    no `call_id` — can never have its history entry patched, so that entry
    reads `awaiting_human_approval` forever. That orphan state is precisely what
    the boot expiry branch exists to eliminate.
    """
    manager.park(_park(token="a"))
    _corrupt_write_call(mem_manager, "a")

    revived = _fresh_manager(mem_manager)
    counts = revived.rebuild_from_store()

    assert counts["quarantined"] == 1
    assert counts["restored"] == 0
    assert revived.list_for("u", "p") == []


def test_a_quarantined_row_stops_holding_its_arguments(manager, mem_manager):
    """Skipping the row instead leaves it `pending` FOREVER.

    `purge_parked_writes` only deletes terminal rows, so a row nothing can read
    is never loaded, never expired and never purged — and `write_call` on a
    pending row is the *unscrubbed* payload, because an approved call has to
    execute with the real values. One malformed row therefore parks whatever
    secret that call carried on disk for the life of the database, plus an
    ERROR log line on every single boot.
    """
    manager.park(_park(token="a"))
    _corrupt_write_call(mem_manager, "a")

    _fresh_manager(mem_manager).rebuild_from_store()

    row = _park_row(mem_manager, "a")
    assert row["status"] == "quarantined", "it must reach a terminal state"
    assert row["write_call"] is None, "the payload must stop living on disk"
    assert row["resolved_at"] is not None, "or purge can never collect it"
    assert row["resolution"] == PARK_STATUS_QUARANTINED, (
        "an unreadable row must not be filterable as a decision that was "
        "in flight — that one may have executed, this one provably did not"
    )

    # And a second boot no longer sees it at all.
    assert _fresh_manager(mem_manager).rebuild_from_store()["quarantined"] == 0


def test_quarantining_a_row_leaves_an_audit_trail(manager, mem_manager):
    """Every other park-terminating path writes one; this was the only one that
    did not.

    After `PARK_ROW_RETENTION` the row itself is purged, so without this the
    only durable trace of a proposed irreversible action would be its
    `audit_parked` row — with nothing anywhere saying it was ever terminated,
    or why.
    """
    manager.park(_park(token="a"))
    _corrupt_write_call(mem_manager, "a")

    _fresh_manager(mem_manager).rebuild_from_store()

    events = mem_manager._get_connection().execute(
        "SELECT * FROM Audit_Log WHERE event_type = 'audit_park_quarantined'"
    ).fetchall()
    assert len(events) == 1
    assert events[0]["new_state"] == PARK_STATUS_QUARANTINED
    assert "NOT executed" in events[0]["reason"]


def test_a_row_that_cannot_be_reconciled_is_also_quarantined(
        manager, mem_manager, monkeypatch):
    """Same trap as the unreadable row, reached through the other branch.

    Logging and returning left the row `pending`/`claimed` forever: never
    loaded, never expired, never purged (purge only deletes terminal rows), so
    it failed identically on every subsequent boot while its unscrubbed
    arguments stayed on disk permanently.
    """
    manager.park(_park(token="bad"))
    conn = mem_manager._get_connection()
    conn.execute("UPDATE Parked_Writes SET created_at = ? WHERE token = 'bad'",
                 (time.time() - PENDING_ACTION_TTL - 60,))
    conn.commit()

    revived = _fresh_manager(mem_manager)
    monkeypatch.setattr(
        revived, "patch_parked_entry",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("unreadable")))

    counts = revived.rebuild_from_store()

    assert counts["quarantined"] == 1
    row = _park_row(mem_manager, "bad")
    assert row["status"] == "quarantined"
    assert row["write_call"] is None, "the payload must stop living on disk"


def test_one_unreconcilable_row_does_not_stop_the_boot(manager, mem_manager,
                                                       monkeypatch):
    """`rebuild_from_store` runs inside `create_chat_system`.

    An exception from the terminal work — `patch_parked_entry` reaching into a
    corrupt `tool_context`, a locked store — used to travel straight out into
    the bootstrap, so one bad row meant the bot refused to start AND every park
    reconciled before it was discarded.
    """
    manager.park(_park(token="bad"))
    manager.park(_park(token="good"))
    # Backdated in the DB, not at park() time: `park` sweeps before it inserts,
    # so a park born stale is expired by the very next park() call and would
    # already be terminal before the boot pass ever reads it — the test would
    # then pass against the unguarded code it exists to catch.
    conn = mem_manager._get_connection()
    conn.execute("UPDATE Parked_Writes SET created_at = ? WHERE token = 'bad'",
                 (time.time() - PENDING_ACTION_TTL - 60,))
    conn.commit()

    revived = _fresh_manager(mem_manager)
    real_patch = revived.patch_parked_entry

    def _explode(park, *args, **kwargs):
        if park.token == "bad":
            raise RuntimeError("tool_context is unreadable")
        return real_patch(park, *args, **kwargs)

    monkeypatch.setattr(revived, "patch_parked_entry", _explode)

    counts = revived.rebuild_from_store()

    assert counts["restored"] == 1, "the healthy park must survive the bad one"
    assert [p.token for p in revived.list_for("u", "p")] == ["good"]


# ---- DP-319 review: failure modes that silently reopen the guard ----------

def test_restore_will_not_reinsert_when_the_store_cannot_be_read(
        manager, mem_manager, monkeypatch):
    """"Could not tell" must fail CLOSED, not read as "no row".

    `release_parked_write` answers False on a transient `database is locked`,
    and the status read used to answer None for both that and a genuinely
    missing row. The re-insert branch then rewrote `status='pending'` and NULLed
    `resolved_at` — resurrecting an already-executed irreversible write as an
    approvable affordance, by way of the branch that looks like the safe one.
    """
    parked = _park(token="a")
    manager.park(parked)
    manager.take("a")
    mem_manager.finalize_parked_write("a", "resolved", PARK_STATUS_APPROVED,
                                      "ran")

    # A real locked store, not a stubbed return value: the defect was that the
    # sqlite3.Error was CAUGHT and reported as None, so stubbing the answer
    # would skip the very code under test.
    real_connection = mem_manager._get_connection

    class _LockedCursor:
        def execute(self, *a, **k):
            raise sqlite3.OperationalError("database is locked")

    class _LockedConnection:
        def cursor(self):
            return _LockedCursor()

        def execute(self, *a, **k):
            raise sqlite3.OperationalError("database is locked")

        def rollback(self):
            pass

    monkeypatch.setattr(mem_manager, "_get_connection",
                        lambda: _LockedConnection())
    try:
        manager.restore(parked)
    finally:
        monkeypatch.setattr(mem_manager, "_get_connection", real_connection)

    assert manager.list_for("u", "p") == [], \
        "an unknown durable state must not become an approvable affordance"
    assert _park_row(mem_manager, "a")["status"] == "resolved"


def test_rebuild_twice_does_not_duplicate_a_park(manager, mem_manager):
    """`_reinstate` appended unconditionally, so `_by_key` grew a second copy
    of the token while `pending` was merely overwritten.

    `list_for` filters on membership, not uniqueness, so it yielded the same
    park twice: two pending chunks sharing one `ephemeral_chunk_id`.
    """
    manager.park(_park(token="a"))

    revived = _fresh_manager(mem_manager)
    revived.rebuild_from_store()
    revived.rebuild_from_store()

    assert [p.token for p in revived.list_for("u", "p")] == ["a"]


def test_the_boot_purge_marks_the_purge_clock(manager, mem_manager):
    """`_last_purge` starts at "never" and only `_purge_due` assigns it, so a
    boot purge that bypassed it left the very next park() or list_for()
    scheduling a second, identical DELETE milliseconds later."""
    revived = _fresh_manager(mem_manager)
    revived.rebuild_from_store()

    assert revived._last_purge > 0
    assert revived._purge_due() is False, \
        "the boot pass already purged; the next read must not repeat it"


def test_a_park_whose_row_was_refused_still_guards_re_execution(
        manager, mem_manager, monkeypatch):
    """`insert_parked_write` refuses a call it cannot serialize losslessly.

    `park()` discarded that answer, so the park went live with no row —
    `finalize_parked_write` then matched nothing and `already_resolved` found
    nothing, silently disabling the double-execution guard for exactly the calls
    that could not be persisted. The park is still offered (pre-DP-319
    behaviour); what must not happen is the guard failing open.
    """
    monkeypatch.setattr(mem_manager, "insert_parked_write",
                        lambda **kwargs: False)
    parked = _park(token="a", call_id="c1")
    manager.park(parked)

    assert parked.persisted is False
    assert [p.token for p in manager.list_for("u", "p")] == ["a"], \
        "a park that could not be stored is still the operator's to decide"

    manager.take("a")
    import asyncio
    asyncio.run(manager.apply(Decision(park=parked, approved=True)))

    hit = manager.already_resolved(("u", "p"),
                                   dict(parked.write_call, id="c2"))
    assert hit is not None, (
        "with no durable row the store answers None forever, so the model's "
        "re-proposal parks a second copy and an approval runs the write twice"
    )
    assert hit["resolution"] == PARK_STATUS_APPROVED


def test_unreadable_duplicate_refs_do_not_destroy_the_park(
        manager, mem_manager, caplog):
    """Not fatal — quarantining would discard an executable park to protect a
    pointer list — but not silent either.

    Every reference lost here is a `duplicate_of_pending` entry claiming the
    action is still awaiting the operator that nothing will ever correct.
    """
    manager.park(_park(token="a"))
    conn = mem_manager._get_connection()
    conn.execute("UPDATE Parked_Writes SET duplicate_refs = ? "
                 "WHERE token = 'a'", ('[[1, "c1"',))
    conn.commit()

    revived = _fresh_manager(mem_manager)
    with caplog.at_level("ERROR"):
        counts = revived.rebuild_from_store()

    assert counts["restored"] == 1
    assert revived.pending["a"].duplicate_refs == []
    assert any("suppressed-duplicate pointers" in r.message
               for r in caplog.records), \
        "a permanently wrong history entry must not be logged nowhere"


# ---- the sweep must not fabricate a decision, or block the loop ------------

def test_the_expiry_sweep_does_not_claim_rows(manager, mem_manager):
    """`claimed` means "an operator's decision was in flight".

    The sweep claiming up front — before handing `expire` to a worker thread —
    made a crash inside that window reboot as `interrupted_by_restart`: history
    patched with "whether it ran is unknown" and an audit row reading "Process
    restarted after the decision was claimed", for a park no operator ever saw
    and that provably never ran. It also put a commit back on the hot path the
    `_take_expired` / `_sweep_off_thread` split exists to keep clear.
    """
    manager.park(_park(token="a",
                       created_at=time.time() - PENDING_ACTION_TTL - 60))

    stale = manager._take_expired()

    assert [p.token for p in stale] == ["a"]
    assert _park_row(mem_manager, "a")["status"] == "pending", \
        "the DB half belongs to the off-thread finish, not to the eviction"


def test_a_crash_mid_sweep_reboots_as_expired_not_interrupted(
        manager, mem_manager):
    """The consequence of the above, stated as the restart it protects."""
    row_id = _row_with_awaiting_entries(mem_manager, ["c1"])
    manager.park(_park(token="tok-c1", call_id="c1", row_id=row_id,
                       created_at=time.time() - PENDING_ACTION_TTL - 60))
    manager._take_expired()  # the process dies here, before `_finish`

    counts = _fresh_manager(mem_manager).rebuild_from_store()

    assert counts["expired"] == 1
    assert counts["interrupted"] == 0, \
        "nobody decided this park; the audit trail must not say they did"
    entry = json.loads(json.loads(
        mem_manager.get_tool_context(row_id))[1]["content"])
    assert entry["status"] == "expired"


def test_retention_is_enforced_while_the_process_runs(manager, mem_manager):
    """The purge ran only in `rebuild_from_store` — i.e. once, at boot.

    So the deployment durability was added for (a bot that stays up for months)
    was the one deployment that never purged anything, and
    `PARK_ROW_RETENTION` went unenforced for as long as it kept running.
    """
    manager.park(_park(token="old"))
    manager.take("old")
    mem_manager.finalize_parked_write(
        "old", "resolved", "approved", "ran",
        now=time.time() - PARK_ROW_RETENTION - 100)

    # A read on the hot path — no restart, no boot pass.
    manager._last_purge = 0.0
    manager.list_for("u", "p")

    assert _park_row(mem_manager, "old") is None


def test_the_purge_is_throttled(manager):
    """It hangs off a sweep that fires once per write call and once per portal
    transcript load, which is far too often for a DELETE."""
    manager._last_purge = 0.0
    assert manager._purge_due() is True
    assert manager._purge_due() is False, "the clock is marked as it answers"


def test_expiry_records_a_filterable_outcome_and_a_separate_reason(
        manager, mem_manager):
    """`resolution` is an enum the guard filters on; the sentence is its own
    column. Writing the prose into `resolution` made every future query over
    expired rows match nothing."""
    manager.park(_park(token="a",
                       created_at=time.time() - PENDING_ACTION_TTL - 60))
    manager.sweep_expired()

    row = _park_row(mem_manager, "a")
    assert row["resolution"] == "expired", "machine-readable"
    assert "No decision within" in row["resolution_reason"], "human-readable"
