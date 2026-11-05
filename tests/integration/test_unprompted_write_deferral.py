# tests/integration/test_unprompted_write_deferral.py
"""DP-493/DP-496: a write the persona may run without a prompt (tool
`auto_approvable`, listed under `allow` in the persona's tool policy) still
parks the deferral its result declares, and
that deferral resumes the turn's own conversation exactly as one parked after
an approval does (tests/integration/test_resume_kernel_convergence.py).

Real MemoryManager + ToolLoop, mocked engine; `create_ticket` stands in for
`queue_codr_task` because the mechanism is tool-agnostic (that the real tool
declares a `codr_ticket` deferral is pinned in tests/stagr/test_stagr_tools.py)."""

import json

import pytest

from src.chat_system import DoneEvent, PendingConfirmationEvent
from src.deferral_kinds import DEFERRAL_KIND_CODR_TICKET, declare_deferral
from src.tools import tool_loop
from tests.integration.test_resume_kernel_convergence import (
    _confirm_persona, _drain, _set_engine, _text, _calls, _tool_entries,
)

pytestmark = pytest.mark.integration

CALL = {"id": "q1", "name": "create_ticket", "arguments": {"title": "t", "body": "b"}}


@pytest.fixture
def auto_approvable(monkeypatch):
    monkeypatch.setattr(tool_loop, "is_auto_approvable", lambda name: name == "create_ticket")


def _policy(persona, *, allow=(), ask=()):
    """The persona's tool policy (DP-496): the wildcard `_confirm_persona`
    set, plus `allow`/`ask` listings by name."""
    persona.set_tool_policy({"default": "allow", "allow": ["*", *allow], "ask": list(ask)})
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
async def test_a_write_under_allow_runs_in_the_turn_and_parks_its_deferral(mocked_chat_system, auto_approvable):
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
async def test_its_deferral_resumes_once_with_the_outcome(mocked_chat_system, auto_approvable):
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
    ((), ()),                                  # only the wildcard: not an explicit allow
    (["create_ticket"], ["create_ticket"]),    # under both: ask wins
], ids=["ask", "wildcard", "both"])
async def test_a_write_not_explicitly_allowed_parks_for_approval(mocked_chat_system, auto_approvable, allow, ask):
    chat_system, _ = mocked_chat_system
    _policy(_confirm_persona(chat_system), allow=allow, ask=ask)
    executed = _tool(chat_system)

    events = await _turn(chat_system)

    assert executed == [] and [e for e in events if isinstance(e, PendingConfirmationEvent)]
    assert len(chat_system.confirmations.list_for("adam", "test_persona")) == 1


@pytest.mark.asyncio
async def test_allow_does_not_reach_a_write_that_is_not_auto_approvable(mocked_chat_system):
    """Scope (DP-496): only the marked writes follow allow/ask; every other
    write still prompts, whatever the policy lists."""
    chat_system, _ = mocked_chat_system
    _allowed(chat_system)
    executed = _tool(chat_system)

    events = await _turn(chat_system)

    assert executed == [] and [e for e in events if isinstance(e, PendingConfirmationEvent)]


@pytest.mark.asyncio
async def test_a_failed_unprompted_write_parks_nothing(mocked_chat_system, auto_approvable):
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
