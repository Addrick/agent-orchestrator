# tests/integration/test_unprompted_write_deferral.py
"""DP-493/DP-497: a write listed under `allow` in the persona's tool policy
runs without a prompt, still parks the deferral its result declares, and that
deferral resumes the turn's own conversation exactly as one parked after an
approval does (tests/integration/test_resume_kernel_convergence.py). Since
DP-497 the policy is the only gate, for every tool: allow runs, ask parks --
a read under ask included.

Real MemoryManager + ToolLoop, mocked engine; `create_ticket` stands in for
`queue_codr_task` because the mechanism is tool-agnostic (that the real tool
declares a `codr_ticket` deferral is pinned in tests/stagr/test_stagr_tools.py)."""

import json

import pytest

from src.chat_system import DoneEvent, PendingConfirmationEvent
from src.agents.codr_agent import IDLE_WAKE_RESULT
from src.deferral_kinds import DEFERRAL_KIND_CODR_IDLE, DEFERRAL_KIND_CODR_TICKET, declare_deferral
from src.stagr.handler import _codr_busy
from tests.integration.test_resume_kernel_convergence import (
    _confirm_persona, _drain, _set_engine, _text, _calls, _tool_entries,
)

pytestmark = pytest.mark.integration

CALL = {"id": "q1", "name": "create_ticket", "arguments": {"title": "t", "body": "b"}}


def _policy(persona, *, allow=(), ask=()):
    """Move the named tools to `allow` / `ask` in the persona's policy
    (`_confirm_persona` enabled every tool at its default); a name in both
    lists stays in both."""
    current = persona.get_tool_policy()
    moved = set(allow) | set(ask)
    persona.set_tool_policy({
        "allow": [n for n in current.allow if n not in moved] + list(allow),
        "ask": [n for n in current.ask if n not in moved] + list(ask),
    })
    return persona


def _allowed(chat_system):
    return _policy(_confirm_persona(chat_system), allow=["create_ticket"])


def _tool(chat_system, ticket="CD-001", payload=None):
    """Install a handler under the REAL `ToolManager.execute_tool`, so the
    loop and the approval path see the `{"result": ...}` envelope production
    hands them. A fake `execute_tool` returning the handler's dict unwrapped
    is what hid the envelope bug this ticket found (`deferral_kinds._declaration`)."""
    executed = []

    async def handler(**kwargs):
        executed.append("create_ticket")
        return declare_deferral(payload or {"status": "ok", "queued": ticket}, DEFERRAL_KIND_CODR_TICKET, ticket)
    chat_system.tool_manager._handlers["create_ticket"] = handler
    return executed


async def _turn(chat_system, user="adam"):
    _set_engine(chat_system, [_calls(CALL), _text("Queued it.")])
    return await _drain(chat_system.stream_response("test_persona", user, "ops", "queue it"))


@pytest.mark.asyncio
async def test_a_write_under_allow_runs_in_the_turn_and_parks_its_deferral(mocked_chat_system):
    chat_system, mem_manager = mocked_chat_system
    _allowed(chat_system)
    executed = _tool(chat_system)

    events = await _turn(chat_system)

    assert executed == ["create_ticket"], "it ran without a click"
    assert not [e for e in events if isinstance(e, PendingConfirmationEvent)], "no approval prompt"
    assert chat_system.confirmations.list_for("adam", "test_persona") == []
    deferred = chat_system.confirmations.pending.get("CD-001")
    assert deferred is not None and deferred.kind == DEFERRAL_KIND_CODR_TICKET
    assert (deferred.user_identifier, deferred.channel) == ("adam", "ops"), "coordinates come from the turn"
    entry = _tool_entries(mem_manager)["q1"]
    assert entry["status"] == "awaiting:codr_ticket" and entry["result"]["result"]["queued"] == "CD-001"
    assert "_deferral" not in json.dumps(entry), "the declaration is plumbing, not content"


@pytest.mark.asyncio
async def test_its_deferral_resumes_once_with_the_outcome(mocked_chat_system):
    chat_system, mem_manager = mocked_chat_system
    _allowed(chat_system)
    executed = _tool(chat_system)
    await _turn(chat_system)

    _set_engine(chat_system, [_text("CD-001 passed; queueing the next one.")])
    events = await _drain(chat_system.stream_resolve_deferral(
        "CD-001", kind=DEFERRAL_KIND_CODR_TICKET, status="passed", result={"ticket": "CD-001", "state": "passed"}))
    again = await _drain(chat_system.stream_resolve_deferral(
        "CD-001", kind=DEFERRAL_KIND_CODR_TICKET, status="passed", result={}))

    assert [e.text for e in events if isinstance(e, DoneEvent)] == ["CD-001 passed; queueing the next one."]
    assert again == [], "settled once"
    assert executed == ["create_ticket"], "settling never runs the tool again"
    entry = _tool_entries(mem_manager)["q1"]
    assert entry["status"] == "passed" and entry["result"]["state"] == "passed"


@pytest.mark.asyncio
@pytest.mark.parametrize("allow, ask", [
    ((), ["create_ticket"]),                   # under ask
    ((), ()),                                  # enabled at its default: a write asks
    (["create_ticket"], ["create_ticket"]),    # under both: ask wins
], ids=["ask", "default", "both"])
async def test_a_write_not_allowed_parks_for_approval(mocked_chat_system, allow, ask):
    chat_system, _ = mocked_chat_system
    _policy(_confirm_persona(chat_system), allow=allow, ask=ask)
    executed = _tool(chat_system)

    events = await _turn(chat_system)

    assert executed == [] and [e for e in events if isinstance(e, PendingConfirmationEvent)]
    assert len(chat_system.confirmations.list_for("adam", "test_persona")) == 1


READ = {"id": "r1", "name": "get_agent_status", "arguments": {"agent_id": "a"}}


def _read_tool(chat_system):
    ran = []

    async def handler(**kwargs):
        ran.append(kwargs)
        return {"status": "ok", "agents": []}
    chat_system.tool_manager._handlers["get_agent_status"] = handler
    return ran


async def _read_turn(chat_system):
    _set_engine(chat_system, [_calls(READ), _text("Checked.")])
    return await _drain(chat_system.stream_response("test_persona", "adam", "ops", "status?"))


@pytest.mark.asyncio
async def test_a_read_under_allow_runs(mocked_chat_system):
    chat_system, _ = mocked_chat_system
    _policy(_confirm_persona(chat_system), allow=["get_agent_status"])
    ran = _read_tool(chat_system)

    events = await _read_turn(chat_system)

    assert ran == [{"agent_id": "a"}]
    assert not [e for e in events if isinstance(e, PendingConfirmationEvent)]


@pytest.mark.asyncio
async def test_a_read_under_ask_parks_and_runs_once_approved(mocked_chat_system):
    """DP-497: ask on a read is real -- the read waits in the same approval
    park a write uses, and runs only after the click."""
    chat_system, mem_manager = mocked_chat_system
    _policy(_confirm_persona(chat_system), ask=["get_agent_status"])
    ran = _read_tool(chat_system)

    events = await _read_turn(chat_system)

    assert ran == [], "it waited"
    assert [e for e in events if isinstance(e, PendingConfirmationEvent)]
    (park,) = chat_system.confirmations.list_for("adam", "test_persona")
    assert park.write_call["name"] == "get_agent_status"
    assert _tool_entries(mem_manager)["r1"]["status"] == "awaiting_human_approval"

    _set_engine(chat_system, [_text("No agents running.")])
    await _drain(chat_system.stream_resolve_park("adam", "test_persona", park.token, approved=True))

    assert ran == [{"agent_id": "a"}]
    assert _tool_entries(mem_manager)["r1"]["status"] == "approved"


@pytest.mark.asyncio
async def test_a_failed_unprompted_write_parks_nothing(mocked_chat_system):
    chat_system, _ = mocked_chat_system
    _allowed(chat_system)
    _tool(chat_system, payload={"status": "error", "message": "nope"})

    await _turn(chat_system)

    assert "CD-001" not in chat_system.confirmations.pending


@pytest.mark.asyncio
async def test_an_approved_write_through_the_real_envelope_re_parks_its_deferral(mocked_chat_system):
    """The approval path, the tool not under allow: approving runs the write, and
    the deferral its handler declared — inside the envelope — becomes the
    park the codr agent settles. Before DP-493 it was patched as done."""
    chat_system, mem_manager = mocked_chat_system
    _confirm_persona(chat_system)
    executed = _tool(chat_system)
    await _turn(chat_system)
    (park,) = chat_system.confirmations.list_for("adam", "test_persona")

    _set_engine(chat_system, [_text("Queued; I will hear when it ends.")])
    await _drain(chat_system.stream_resolve_park("adam", "test_persona", park.token, approved=True))

    assert executed == ["create_ticket"]
    deferred = chat_system.confirmations.pending.get("CD-001")
    assert deferred is not None and deferred.kind == DEFERRAL_KIND_CODR_TICKET
    entry = _tool_entries(mem_manager)["q1"]
    assert entry["status"] == "awaiting:codr_ticket"
    assert "_deferral" not in json.dumps(entry)


@pytest.mark.asyncio
async def test_a_call_that_found_codr_busy_parks_and_resumes_when_codr_is_idle(mocked_chat_system):
    """DP-517 end to end through the real envelope: the exact result stagr's
    `codr_shell` returns while codr is busy parks a `codr_idle` deferral in
    the turn's conversation, and the codr agent's idle wake settles it once."""
    chat_system, mem_manager = mocked_chat_system
    _allowed(chat_system)
    busy = _codr_busy("the command")
    token = busy["_deferral"]["token"]

    async def handler(**kwargs):
        return dict(busy)
    chat_system.tool_manager._handlers["create_ticket"] = handler

    await _turn(chat_system)

    deferred = chat_system.confirmations.pending.get(token)
    assert deferred is not None and deferred.kind == DEFERRAL_KIND_CODR_IDLE
    assert (deferred.user_identifier, deferred.channel) == ("adam", "ops")
    assert [p.token for p in chat_system.confirmations.pending_of_kind(DEFERRAL_KIND_CODR_IDLE)] == [token]
    assert _tool_entries(mem_manager)["q1"]["status"] == "awaiting:codr_idle"

    _set_engine(chat_system, [_text("codr is free; ran it.")])
    events = await _drain(chat_system.stream_resolve_deferral(
        token, kind=DEFERRAL_KIND_CODR_IDLE, status="idle", result=dict(IDLE_WAKE_RESULT)))
    assert [e.text for e in events if isinstance(e, DoneEvent)] == ["codr is free; ran it."]
    assert chat_system.confirmations.pending_of_kind(DEFERRAL_KIND_CODR_IDLE) == []
    assert _tool_entries(mem_manager)["q1"]["status"] == "idle"


@pytest.mark.asyncio
async def test_several_waits_from_one_turn_settle_as_one_continuation(mocked_chat_system):
    """DP-517: a turn whose calls found codr busy twice gets ONE continuation
    when codr is idle. Two separate resolves fold only if scheduling happens
    to queue the second before the first drains; the batch does not rely on it."""
    chat_system, mem_manager = mocked_chat_system
    _allowed(chat_system)
    waits = [_codr_busy("the command"), _codr_busy("the copy")]

    async def handler(**kwargs):
        return dict(waits.pop(0))
    chat_system.tool_manager._handlers["create_ticket"] = handler
    second = {"id": "q2", "name": "create_ticket", "arguments": {"title": "t2", "body": "b"}}
    _set_engine(chat_system, [_calls(CALL, second), _text("Waiting for codr.")])
    await _drain(chat_system.stream_response("test_persona", "adam", "ops", "prepare codr"))
    tokens = [p.token for p in chat_system.confirmations.pending_of_kind(DEFERRAL_KIND_CODR_IDLE)]
    assert len(tokens) == 2

    _set_engine(chat_system, [_text("Both ran.")])             # a second continuation would find no reply
    events = await _drain(chat_system.stream_resolve_deferrals(
        tokens, kind=DEFERRAL_KIND_CODR_IDLE, status="idle", result=dict(IDLE_WAKE_RESULT)))

    assert [e.text for e in events if isinstance(e, DoneEvent)] == ["Both ran."]
    assert chat_system.confirmations.pending_of_kind(DEFERRAL_KIND_CODR_IDLE) == []
    entries = _tool_entries(mem_manager)
    assert entries["q1"]["status"] == entries["q2"]["status"] == "idle"


# --- DP-518: a settled deferral's continuation is owed until one completes ---


class _ProcessDied(BaseException):
    """Stands in for the process dying mid-continuation: nothing in the turn
    pipeline catches a BaseException, so it unwinds like a kill would."""


def _engine_fails(chat_system, exc):
    async def fake_generate_response(persona_config, history_object, *a, **k):
        raise exc
    chat_system.text_engine.generate_response.side_effect = fake_generate_response


def _owed_row(mem_manager, token):
    cursor = mem_manager._get_connection().cursor()
    cursor.execute("SELECT status, continuation_owed, write_call FROM Parked_Writes WHERE token = ?", (token,))
    return tuple(cursor.fetchone())


def _restart(chat_system):
    """A fresh ConfirmationManager over the same DB, rebuilt as boot does."""
    from src.confirmations import ConfirmationManager
    chat_system.confirmations = ConfirmationManager(
        lambda: chat_system.tool_manager, chat_system.memory_manager, chat_system._offered_tool_names,
        persona_lookup=lambda name: chat_system.personas.get(name))
    chat_system.confirmations.rebuild_from_store()
    return chat_system.confirmations


@pytest.mark.asyncio
async def test_a_continuation_that_errors_is_owed_and_runs_again_exactly_once(mocked_chat_system):
    """The model server is down when codr's ticket ends: the deferral is settled
    (history patched) but stagr's turn errors. It used to be gone for good; now
    the codr agent can ask again, once that turn completes nothing is owed, and
    the tool never runs again."""
    from src.engine import LLMCommunicationError
    chat_system, mem_manager = mocked_chat_system
    _allowed(chat_system)
    executed = _tool(chat_system)
    await _turn(chat_system)

    _engine_fails(chat_system, LLMCommunicationError("Cannot connect to the model server"))
    events = await _drain(chat_system.stream_resolve_deferral(
        "CD-001", kind=DEFERRAL_KIND_CODR_TICKET, status="passed", result={"ticket": "CD-001", "state": "passed"}))
    assert not [e for e in events if isinstance(e, DoneEvent)]
    assert _tool_entries(mem_manager)["q1"]["status"] == "passed", "settled: the result is in history"
    owed = chat_system.confirmations.owed_of_kind(DEFERRAL_KIND_CODR_TICKET)
    assert [d.park.token for d in owed] == ["CD-001"]
    assert _owed_row(mem_manager, "CD-001")[:2] == ("resolved", 1)

    _set_engine(chat_system, [_text("CD-001 passed; queueing the next one.")])
    events = await _drain(chat_system.stream_resume_continuations(["CD-001"], kind=DEFERRAL_KIND_CODR_TICKET))
    assert [e.text for e in events if isinstance(e, DoneEvent)] == ["CD-001 passed; queueing the next one."]
    sent = json.dumps(chat_system.text_engine.generate_response.call_args.args[1])
    assert "create_ticket (codr_ticket): passed" in sent and "cut off before it finished" in sent

    assert chat_system.confirmations.owed_of_kind(DEFERRAL_KIND_CODR_TICKET) == []
    assert _owed_row(mem_manager, "CD-001") == ("resolved", 0, None), "closed, payload erased"
    assert await _drain(chat_system.stream_resume_continuations(
        ["CD-001"], kind=DEFERRAL_KIND_CODR_TICKET)) == [], "no second continuation"
    assert await _drain(chat_system.stream_resolve_deferral(
        "CD-001", kind=DEFERRAL_KIND_CODR_TICKET, status="passed", result={})) == [], "settled once"
    assert _restart(chat_system).owed_of_kind(DEFERRAL_KIND_CODR_TICKET) == [], "nor after a restart"
    assert executed == ["create_ticket"]


@pytest.mark.asyncio
async def test_a_restart_during_the_continuation_reruns_it_once(mocked_chat_system):
    """derpr dies after the deferral is settled and before stagr's turn
    finishes. The boot reload finds the continuation still owed (master's
    `rebuild_from_store` read only pending/claimed rows, so it was lost)."""
    chat_system, mem_manager = mocked_chat_system
    _allowed(chat_system)
    executed = _tool(chat_system)
    await _turn(chat_system)

    _engine_fails(chat_system, _ProcessDied())
    with pytest.raises(_ProcessDied):
        await _drain(chat_system.stream_resolve_deferral(
            "CD-001", kind=DEFERRAL_KIND_CODR_TICKET, status="passed", result={"state": "passed"}))

    confirmations = _restart(chat_system)
    (owed,) = confirmations.owed_of_kind(DEFERRAL_KIND_CODR_TICKET)
    assert (owed.park.token, owed.park.user_identifier, owed.park.channel) == ("CD-001", "adam", "ops")
    assert owed.status == "passed" and owed.patched, "read back from the row and from history"
    assert "CD-001" not in confirmations.pending, "not settleable again"

    _set_engine(chat_system, [_text("CD-001 passed.")])
    events = await _drain(chat_system.stream_resume_continuations(["CD-001"], kind=DEFERRAL_KIND_CODR_TICKET))
    assert [e.text for e in events if isinstance(e, DoneEvent)] == ["CD-001 passed."]
    assert _restart(chat_system).owed_of_kind(DEFERRAL_KIND_CODR_TICKET) == []
    assert _tool_entries(mem_manager)["q1"]["status"] == "passed"
    assert executed == ["create_ticket"]


@pytest.mark.asyncio
async def test_an_approval_whose_reply_errors_is_never_owed(mocked_chat_system):
    """Approval semantics are untouched: the approved write ran once, its row
    is resolved with its payload erased, and nothing asks for the reply again."""
    from src.engine import LLMCommunicationError
    chat_system, mem_manager = mocked_chat_system
    _confirm_persona(chat_system)
    ran = []

    async def handler(**kwargs):
        ran.append(kwargs)
        return {"status": "ok"}
    chat_system.tool_manager._handlers["create_ticket"] = handler
    await _turn(chat_system)
    (park,) = chat_system.confirmations.list_for("adam", "test_persona")

    _engine_fails(chat_system, LLMCommunicationError("down"))
    await _drain(chat_system.stream_resolve_park("adam", "test_persona", park.token, approved=True))

    assert len(ran) == 1
    assert _owed_row(mem_manager, park.token) == ("resolved", 0, None)
    assert chat_system.confirmations.owed_of_kind("approval") == []
    assert _restart(chat_system).owed_of_kind("approval") == []
