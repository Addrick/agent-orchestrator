# tests/integration/test_resume_kernel_convergence.py
#
# DP-124 converged the approve/deny continuation onto the _orchestrate kernel
# instead of re-implementing it. DP-297 then made parking non-blocking, so the
# same convergence now has to hold for N proposals resolvable in any order.
#
# Pinned here:
#
#   1. approved -> the continuation runs a *further* tool loop
#   2. denied   -> clean close, no write executed
#   3. the persisted assistant row carries the real channel (not channel="")
#   4. no turn-context leak across the continuation
#   5. a burst of writes in one turn all survive, with distinct tokens
#   6. out-of-order resolution works
#   7. concurrent approvals serialize into ONE continuation
#
# All run with a real MemoryManager + TextEngine (generate_response mocked), so
# the continuation exercises the genuine ToolLoop path.

import asyncio
import json
import time

import pytest

from src.chat_system import (
    ResponseType, PendingConfirmationEvent, DoneEvent,
)
from src.confirmations import ParkedWrite
from src.persona import ExecutionMode
from src.tools.turn_context import get_turn_context
from config.global_config import PENDING_ACTION_TTL

pytestmark = pytest.mark.integration


async def _drain(stream):
    return [ev async for ev in stream]


def _set_engine(chat_system, scripted):
    """Point mocked generate_response at a sequence of (result, payload) tuples."""
    it = iter(scripted)

    async def fake_generate_response(persona_config, history_object, *a, **k):
        return next(it)

    chat_system.text_engine.generate_response.side_effect = fake_generate_response


def _text(content):
    return ({"type": "text", "content": content}, {})


def _calls(*call_dicts):
    return ({"type": "tool_calls", "calls": list(call_dicts)}, {})


async def _park_writes(chat_system, *, user, channel, write_calls,
                       closing_text="Proposed."):
    """Drive turn 1 so it gates `write_calls`, then ends with text.

    Since DP-297 a gated write does not end the turn, so the script needs a
    trailing text response — that extra step IS the behaviour change.
    """
    _set_engine(chat_system, [
        _calls(*write_calls),
        _text(closing_text),
    ])
    await _drain(
        chat_system.stream_response("test_persona", user, channel, "do the thing")
    )
    parks = chat_system.confirmations.list_for(user, "test_persona")
    assert len(parks) == len(write_calls)
    assert get_turn_context() is None
    return [p.token for p in parks]


def _confirm_persona(chat_system):
    persona = chat_system.personas["test_persona"]
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(["*"])
    return persona


def _recording_tool_manager(chat_system, result=None):
    executed = []

    async def fake_execute(name, **kwargs):
        executed.append(name)
        return result if result is not None else {"ok": True}
    chat_system.tool_manager.execute_tool = fake_execute  # type: ignore[assignment]
    return executed


@pytest.mark.asyncio
async def test_approval_runs_further_tool_loop(mocked_chat_system):
    """Approval continues through the full kernel: the approved write executes
    and the model's follow-up read runs a *further* loop iteration."""
    chat_system, _ = mocked_chat_system
    _confirm_persona(chat_system)
    executed = _recording_tool_manager(chat_system)

    (token,) = await _park_writes(
        chat_system, user="u1", channel="c1",
        write_calls=[{"id": "w1", "name": "create_ticket",
                      "arguments": {"title": "t", "body": "b"}}],
    )

    # Continuation: a read tool call, then a text answer.
    _set_engine(chat_system, [
        _calls({"id": "r1", "name": "get_agent_status",
                "arguments": {"agent_id": "a"}}),
        _text("Ticket opened and status checked."),
    ])

    text, rtype, assistant_id, uid = await chat_system.resolve_park(
        "u1", "test_persona", token, approved=True,
    )

    assert rtype == ResponseType.LLM_GENERATION
    assert text == "Ticket opened and status checked."
    assert "create_ticket" in executed, "approved write was not executed"
    assert "get_agent_status" in executed, "continuation did not run a further tool loop"
    assert assistant_id is not None
    assert uid is None
    assert chat_system.confirmations.list_for("u1", "test_persona") == []
    assert get_turn_context() is None, "turn scope leaked after continuation"


@pytest.mark.asyncio
async def test_denial_closes_cleanly(mocked_chat_system):
    """Denial returns the model's close-out text; the write never executes."""
    chat_system, _ = mocked_chat_system
    _confirm_persona(chat_system)
    executed = _recording_tool_manager(chat_system)

    (token,) = await _park_writes(
        chat_system, user="u2", channel="c2",
        write_calls=[{"id": "w1", "name": "create_ticket",
                      "arguments": {"title": "t", "body": "b"}}],
    )

    _set_engine(chat_system, [_text("Understood, I won't create the ticket.")])

    text, rtype, assistant_id, uid = await chat_system.resolve_park(
        "u2", "test_persona", token, approved=False,
    )

    assert rtype == ResponseType.LLM_GENERATION
    assert "won't create" in text
    assert "create_ticket" not in executed, "denied write must not execute"
    assert chat_system.confirmations.list_for("u2", "test_persona") == []
    assert get_turn_context() is None


@pytest.mark.asyncio
async def test_continuation_persists_assistant_on_correct_channel(mocked_chat_system):
    """The continuation's assistant row is logged on the parked channel — not
    the channel="" the old implementation hardcoded."""
    chat_system, mem_manager = mocked_chat_system
    _confirm_persona(chat_system)
    _recording_tool_manager(chat_system)

    (token,) = await _park_writes(
        chat_system, user="u3", channel="team-chan",
        write_calls=[{"id": "w1", "name": "create_ticket",
                      "arguments": {"title": "t", "body": "b"}}],
    )

    _set_engine(chat_system, [_text("Done.")])
    await chat_system.resolve_park("u3", "test_persona", token, approved=True)

    conn = mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT channel, content FROM User_Interactions WHERE author_role='assistant'"
    )
    rows = cursor.fetchall()
    assert any(r["channel"] == "team-chan" and "Done" in r["content"] for r in rows), \
        "assistant row not persisted on the parked channel"
    assert not any(r["channel"] == "" for r in rows), \
        "assistant row persisted with hardcoded empty channel"


@pytest.mark.asyncio
async def test_approved_write_result_is_patched_into_history(mocked_chat_system):
    """The approved write's real result replaces its awaiting-approval entry in
    the ALREADY-COMMITTED parked row, and replays on later turns.

    This is DP-297's replacement for re-sealing the span on resume. The park's
    entry is patched in place, so the model reads the actual outcome — and
    exactly once, since nothing re-writes the span.
    """
    chat_system, mem_manager = mocked_chat_system
    _confirm_persona(chat_system)
    _recording_tool_manager(chat_system, result={"ticket_id": 42})

    (token,) = await _park_writes(
        chat_system, user="u6", channel="c6",
        write_calls=[{"id": "w1", "name": "create_ticket",
                      "arguments": {"title": "t", "body": "b"}}],
        closing_text="Ticket proposed.",
    )

    conn = mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT tool_context FROM User_Interactions "
        "WHERE author_role='assistant' AND content='Ticket proposed.'"
    )
    parked_ctx = json.loads(cursor.fetchone()["tool_context"])
    awaiting = next(m for m in parked_ctx
                    if m.get("role") == "tool" and m.get("tool_call_id") == "w1")
    assert json.loads(awaiting["content"])["status"] == "awaiting_human_approval"

    _set_engine(chat_system, [_text("Ticket created.")])
    await chat_system.resolve_park("u6", "test_persona", token, approved=True)

    cursor.execute(
        "SELECT tool_context FROM User_Interactions "
        "WHERE author_role='assistant' AND content='Ticket proposed.'"
    )
    patched = json.loads(cursor.fetchone()["tool_context"])
    entry = next(m for m in patched
                 if m.get("role") == "tool" and m.get("tool_call_id") == "w1")
    payload = json.loads(entry["content"])
    assert payload["status"] == "approved"
    assert payload["result"] == {"ticket_id": 42}
    assert payload["token"] == token

    # The write's call and its patched result both still replay into history.
    # The user row matters: DP-296 drops a tool block with no user turn ahead
    # of it, because that shape opens the wire array with a function call and
    # Gemini rejects the request.
    replayed = chat_system.request_builder.format_raw_history_for_llm(
        [{"author_role": "user", "author_name": "Alice", "content": "make a ticket"},
         {"author_role": "assistant", "author_name": "test_persona",
          "content": "Ticket proposed.",
          "tool_context": json.dumps(patched)}],
        memory_mode="channel", persona_name="test_persona", server_id=None,
    )
    assert any(m.get("role") == "tool" and m.get("name") == "create_ticket"
               for m in replayed)


@pytest.mark.asyncio
async def test_continuation_pins_scope_and_resets(mocked_chat_system):
    """The continuation runs with the parked scope pinned (so engine-side tools
    inherit persona/user/channel) and the ContextVar is reset on exit."""
    chat_system, _ = mocked_chat_system
    _confirm_persona(chat_system)

    seen = {}

    async def fake_execute(name, **kwargs):
        seen["ctx"] = get_turn_context()
        return {"ok": True}
    chat_system.tool_manager.execute_tool = fake_execute  # type: ignore[assignment]

    (token,) = await _park_writes(
        chat_system, user="u4", channel="c4",
        write_calls=[{"id": "w1", "name": "create_ticket",
                      "arguments": {"title": "t", "body": "b"}}],
    )

    _set_engine(chat_system, [
        _calls({"id": "r1", "name": "get_agent_status",
                "arguments": {"agent_id": "a"}}),
        _text("ok"),
    ])
    await chat_system.resolve_park("u4", "test_persona", token, approved=True)

    assert seen["ctx"] is not None, "continuation ran with no turn context"
    assert seen["ctx"].user_identifier == "u4"
    assert seen["ctx"].persona_name == "test_persona"
    assert seen["ctx"].channel == "c4"
    assert get_turn_context() is None, "turn context not reset after continuation"


@pytest.mark.asyncio
async def test_expired_park_closes_out(mocked_chat_system):
    """An expired park closes out without re-entering the kernel."""
    chat_system, _ = mocked_chat_system

    chat_system.confirmations.park(ParkedWrite(
        token="stale-token",
        write_call={"id": "w1", "name": "create_ticket", "arguments": {}},
        audit_info={"actions": []},
        confirmation_text="Approve?",
        user_identifier="u5",
        persona_name="test_persona",
        channel="c5",
        created_at=time.time() - PENDING_ACTION_TTL - 10,
    ))

    text, rtype, assistant_id, uid = await chat_system.resolve_park(
        "u5", "test_persona", "stale-token", approved=True,
    )

    assert rtype == ResponseType.DEV_COMMAND
    assert "expired" in text.lower()
    assert assistant_id is None
    assert chat_system.confirmations.list_for("u5", "test_persona") == []
    assert get_turn_context() is None


@pytest.mark.asyncio
async def test_unknown_token_closes_out(mocked_chat_system):
    """Resolving a token that was never parked returns the not-found close-out."""
    chat_system, _ = mocked_chat_system
    text, rtype, assistant_id, uid = await chat_system.resolve_park(
        "nobody", "test_persona", "no-such-token", approved=True,
    )
    assert rtype == ResponseType.DEV_COMMAND
    assert "No such pending action" in text
    assert assistant_id is None


# -------- DP-297: bursts, ordering, and concurrency ------------------------


@pytest.mark.asyncio
async def test_burst_of_writes_all_survive_with_distinct_tokens(mocked_chat_system):
    """THE regression this ticket exists for.

    One turn proposing three writes must yield three live proposals. Under the
    old 1:1 `(user, persona)` key, parks 1 and 2 were evicted (each emitting an
    `audit_parked_evicted` row) and only the last survived — which is why the
    2026-07-26 Discord session left proposals dangling with no affordance.
    """
    chat_system, mem_manager = mocked_chat_system
    _confirm_persona(chat_system)
    _recording_tool_manager(chat_system)

    # Three writes across three iterations: the loop must keep going after each.
    _set_engine(chat_system, [
        _calls({"id": "w1", "name": "create_ticket", "arguments": {"title": "a"}}),
        _calls({"id": "w2", "name": "update_ticket", "arguments": {"ticket_id": 1}}),
        _calls({"id": "w3", "name": "merge_tickets",
                "arguments": {"source_ticket_id": 2, "target_ticket_id": 3}}),
        _text("Proposed three actions."),
    ])
    events = await _drain(
        chat_system.stream_response("test_persona", "burst", "c", "do three things")
    )

    parks = chat_system.confirmations.list_for("burst", "test_persona")
    assert [p.write_call["name"] for p in parks] == [
        "create_ticket", "update_ticket", "merge_tickets",
    ]
    assert len({p.token for p in parks}) == 3

    # One approve/deny affordance per proposal, all before the terminal event.
    pce = [e for e in events if isinstance(e, PendingConfirmationEvent)]
    assert [e.token for e in pce] == [p.token for p in parks]
    done_idx = next(i for i, e in enumerate(events) if isinstance(e, DoneEvent))
    assert all(i < done_idx for i, e in enumerate(events)
               if isinstance(e, PendingConfirmationEvent))

    # The turn still ended with the model's own text.
    assert events[done_idx].text == "Proposed three actions."

    # Nothing was evicted.
    conn = mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) AS n FROM Audit_Log WHERE event_type='audit_parked_evicted'"
    )
    assert cursor.fetchone()["n"] == 0
    cursor.execute(
        "SELECT COUNT(*) AS n FROM Audit_Log WHERE event_type='audit_parked'"
    )
    assert cursor.fetchone()["n"] == 3


@pytest.mark.asyncio
async def test_out_of_order_resolution(mocked_chat_system):
    """Proposals resolve independently and in any order."""
    chat_system, _ = mocked_chat_system
    _confirm_persona(chat_system)
    executed = _recording_tool_manager(chat_system)

    _set_engine(chat_system, [
        _calls({"id": "w1", "name": "create_ticket", "arguments": {"title": "a"}}),
        _calls({"id": "w2", "name": "update_ticket", "arguments": {"ticket_id": 1}}),
        _text("Proposed two."),
    ])
    await _drain(
        chat_system.stream_response("test_persona", "ooo", "c", "two things")
    )
    first, second = [p.token
                     for p in chat_system.confirmations.list_for("ooo", "test_persona")]

    # Resolve the SECOND one first.
    _set_engine(chat_system, [_text("Updated.")])
    await chat_system.resolve_park("ooo", "test_persona", second, approved=True)
    assert executed == ["update_ticket"]
    # The first is untouched and still live.
    remaining = chat_system.confirmations.list_for("ooo", "test_persona")
    assert [p.token for p in remaining] == [first]

    _set_engine(chat_system, [_text("Created.")])
    await chat_system.resolve_park("ooo", "test_persona", first, approved=True)
    assert executed == ["update_ticket", "create_ticket"]
    assert chat_system.confirmations.list_for("ooo", "test_persona") == []


@pytest.mark.asyncio
async def test_concurrent_approvals_serialize(mocked_chat_system):
    """Two approvals fired at once must never run two tool loops over the same
    conversation concurrently.

    This is the property the per-conversation lock exists for. Without it both
    continuations mutate one history list and each writes its own assistant row
    for the same logical turn — the interleaving that made per-approval
    re-entry unsafe once a burst became possible.

    Note what is NOT promised: that both fold into a single summary. Acquiring
    an uncontended asyncio.Lock does not suspend, so the winner is typically
    already inside its continuation before the loser is scheduled. Coalescing
    is best-effort (decisions arriving mid-execution do get folded in);
    serialization is the guarantee.
    """
    chat_system, mem_manager = mocked_chat_system
    _confirm_persona(chat_system)

    concurrent = {"now": 0, "max": 0}

    async def fake_execute(name, **kwargs):
        concurrent["now"] += 1
        concurrent["max"] = max(concurrent["max"], concurrent["now"])
        await asyncio.sleep(0)  # give the other task a chance to interleave
        concurrent["now"] -= 1
        return {"ok": True}
    chat_system.tool_manager.execute_tool = fake_execute  # type: ignore[assignment]

    _set_engine(chat_system, [
        _calls({"id": "w1", "name": "create_ticket", "arguments": {"title": "a"}}),
        _calls({"id": "w2", "name": "update_ticket", "arguments": {"ticket_id": 1}}),
        _text("Proposed two."),
    ])
    await _drain(
        chat_system.stream_response("test_persona", "race", "c", "two things")
    )
    tokens = [p.token
              for p in chat_system.confirmations.list_for("race", "test_persona")]

    _set_engine(chat_system, [_text("Handled."), _text("Handled.")])

    await asyncio.gather(*(
        chat_system.resolve_park("race", "test_persona", t, approved=True)
        for t in tokens
    ))

    assert concurrent["max"] == 1, "two approvals executed tools concurrently"
    assert chat_system.confirmations.list_for("race", "test_persona") == []

    # Every resolved proposal is accounted for exactly once in history.
    conn = mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) AS n FROM Audit_Log "
        "WHERE event_type='audit_decision' AND new_state='approved'"
    )
    assert cursor.fetchone()["n"] == 2


@pytest.mark.asyncio
async def test_park_yields_pending_confirmation_event(mocked_chat_system):
    """A gated write surfaces a PendingConfirmationEvent (structured call +
    token) before the terminal DoneEvent, so a surface can render approve/deny.
    The token matches the stored park."""
    chat_system, _ = mocked_chat_system
    _confirm_persona(chat_system)

    _set_engine(chat_system, [
        _calls({"id": "w1", "name": "create_ticket",
                "arguments": {"title": "t", "body": "b"}}),
        _text("Proposed."),
    ])
    events = await _drain(
        chat_system.stream_response("test_persona", "u6", "c6", "do it")
    )

    pce = [e for e in events if isinstance(e, PendingConfirmationEvent)]
    assert len(pce) == 1, "exactly one PendingConfirmationEvent expected"
    ev = pce[0]
    assert ev.persona_name == "test_persona"
    assert ev.write_calls[0]["name"] == "create_ticket"
    assert ev.token, "park event must carry a token"

    parks = chat_system.confirmations.list_for("u6", "test_persona")
    assert ev.token == parks[0].token, "event token must match the stored park"

    pce_idx = next(i for i, e in enumerate(events)
                   if isinstance(e, PendingConfirmationEvent))
    done_idx = next(i for i, e in enumerate(events) if isinstance(e, DoneEvent))
    assert pce_idx < done_idx, "park event must precede the terminal DoneEvent"


@pytest.mark.asyncio
async def test_stale_token_leaves_other_parks_intact(mocked_chat_system):
    """A resolve with an unknown token executes nothing and leaves live parks
    alone, so a correct-token retry still goes through."""
    chat_system, _ = mocked_chat_system
    _confirm_persona(chat_system)
    executed = _recording_tool_manager(chat_system)

    (token,) = await _park_writes(
        chat_system, user="u7", channel="c7",
        write_calls=[{"id": "w1", "name": "create_ticket",
                      "arguments": {"title": "t", "body": "b"}}],
    )

    events = await _drain(chat_system.stream_resolve_park(
        "u7", "test_persona", "not-the-token", approved=True,
    ))
    done = [e for e in events if isinstance(e, DoneEvent)][-1]
    assert done.response_type == ResponseType.DEV_COMMAND
    assert "no such pending action" in done.text.lower()
    assert "create_ticket" not in executed, "stale resolve must not execute the write"

    parks = chat_system.confirmations.list_for("u7", "test_persona")
    assert [p.token for p in parks] == [token], \
        "stale token must leave the real park intact"


@pytest.mark.asyncio
async def test_valid_token_executes_write(mocked_chat_system):
    """A streaming resolve with a live token consumes the park, executes the
    write, and streams the continuation."""
    chat_system, _ = mocked_chat_system
    _confirm_persona(chat_system)
    executed = _recording_tool_manager(chat_system)

    (token,) = await _park_writes(
        chat_system, user="u8", channel="c8",
        write_calls=[{"id": "w1", "name": "create_ticket",
                      "arguments": {"title": "t", "body": "b"}}],
    )

    _set_engine(chat_system, [_text("Ticket opened.")])
    events = await _drain(chat_system.stream_resolve_park(
        "u8", "test_persona", token, approved=True,
    ))

    done = [e for e in events if isinstance(e, DoneEvent)][-1]
    assert done.response_type == ResponseType.LLM_GENERATION
    assert done.text == "Ticket opened."
    assert executed == ["create_ticket"]
    assert chat_system.confirmations.list_for("u8", "test_persona") == []


# ---- DP-297 duplicate-proposal guard, wired end to end -------------------
#
# The tool-loop unit tests drive `pending_lookup` with a fake. These assert the
# real closure in `_orchestrate` is actually PASSED and actually consults the
# live store -- a guard that exists but is never reached is the failure mode
# this file exists to catch.


@pytest.mark.asyncio
async def test_reproposal_across_turns_does_not_create_a_second_park(
        mocked_chat_system):
    """A later turn re-proposing a still-pending write gets no second park.

    The cross-turn case is the one that bites: the park survives the turn, the
    model re-reads it as pending, and proposes again.
    """
    chat_system, _ = mocked_chat_system
    _confirm_persona(chat_system)
    _recording_tool_manager(chat_system)

    write = {"id": "w1", "name": "create_ticket",
             "arguments": {"title": "t", "body": "b"}}
    (token,) = await _park_writes(
        chat_system, user="u9", channel="c9", write_calls=[write],
    )

    # Turn 2: same action, different provider call id (as a real re-proposal
    # would have).
    _set_engine(chat_system, [
        _calls({"id": "w2", "name": "create_ticket",
                "arguments": {"title": "t", "body": "b"}}),
        _text("Still waiting on you."),
    ])
    await _drain(chat_system.stream_response(
        "test_persona", "u9", "c9", "do it again"))

    parks = chat_system.confirmations.list_for("u9", "test_persona")
    assert len(parks) == 1, "re-proposal must not queue a second affordance"
    assert parks[0].token == token, "the original park is the survivor"


@pytest.mark.asyncio
async def test_reproposal_after_resolution_parks_again(mocked_chat_system):
    """The guard keys on PENDING, not on history.

    Once the operator has decided, the action is no longer queued, so proposing
    it again is a legitimate new request -- a denied write the user then asks
    for explicitly must be able to reach them a second time.
    """
    chat_system, _ = mocked_chat_system
    _confirm_persona(chat_system)
    _recording_tool_manager(chat_system)

    write = {"id": "w1", "name": "create_ticket",
             "arguments": {"title": "t", "body": "b"}}
    (token,) = await _park_writes(
        chat_system, user="u10", channel="c10", write_calls=[write],
    )

    _set_engine(chat_system, [_text("Denied, understood.")])
    await _drain(chat_system.stream_resolve_park(
        "u10", "test_persona", token, approved=False,
    ))
    assert chat_system.confirmations.list_for("u10", "test_persona") == []

    _set_engine(chat_system, [
        _calls({"id": "w2", "name": "create_ticket",
                "arguments": {"title": "t", "body": "b"}}),
        _text("Proposed again."),
    ])
    await _drain(chat_system.stream_response(
        "test_persona", "u10", "c10", "actually, do it"))

    parks = chat_system.confirmations.list_for("u10", "test_persona")
    assert len(parks) == 1
    assert parks[0].token != token, "a new decision needs a new token"


@pytest.mark.asyncio
async def test_reproposal_during_the_continuation_cannot_double_execute(
        mocked_chat_system):
    """DP-319: the write already RAN, so a second park is a second execution.

    This is the hole `pending_lookup` structurally cannot cover. The
    continuation begins after `take()` has removed the park, so both scopes the
    pending guard consults are empty — and the continuation is precisely when
    the model re-proposes, because it is re-reading its own tool span. Before
    DP-319 a fresh park appeared here and approving it ran the irreversible
    write a second time.
    """
    chat_system, _ = mocked_chat_system
    _confirm_persona(chat_system)
    executed = _recording_tool_manager(chat_system)

    write = {"id": "w1", "name": "create_ticket",
             "arguments": {"title": "t", "body": "b"}}
    (token,) = await _park_writes(
        chat_system, user="u11", channel="c11", write_calls=[write],
    )

    # The continuation re-proposes the identical action before summarizing.
    _set_engine(chat_system, [
        _calls({"id": "w2", "name": "create_ticket",
                "arguments": {"title": "t", "body": "b"}}),
        _text("Done."),
    ])
    await _drain(chat_system.stream_resolve_park(
        "u11", "test_persona", token, approved=True,
    ))

    assert executed == ["create_ticket"], "the write must run exactly once"
    assert chat_system.confirmations.list_for("u11", "test_persona") == [], \
        "the re-proposal must not leave a second approvable affordance"


def _tool_entries(mem_manager):
    """Every sealed tool entry across the conversation, keyed by call id."""
    cursor = mem_manager._get_connection().cursor()
    cursor.execute(
        "SELECT tool_context FROM User_Interactions "
        "WHERE tool_context IS NOT NULL ORDER BY interaction_id"
    )
    out = {}
    for (blob,) in cursor.fetchall():
        for msg in json.loads(blob):
            if msg.get("role") == "tool":
                out[msg.get("tool_call_id")] = json.loads(msg["content"])
    return out


@pytest.mark.asyncio
async def test_resolving_a_park_also_patches_its_suppressed_duplicate(
        mocked_chat_system):
    """A suppressed duplicate must not keep claiming the action is pending.

    The duplicate's entry says "still awaiting the operator". Nothing else
    would ever correct it, so once the original is decided history would assert
    a decided action is queued — the same verdict/record split that the denial
    instruction fixed elsewhere.

    Note the duplicate lands in a DIFFERENT assistant row than the park (the
    re-proposal is a later turn), so this also pins that the patch follows
    `duplicate_refs` across rows rather than scanning the park's own row.
    """
    chat_system, mem_manager = mocked_chat_system
    _confirm_persona(chat_system)
    _recording_tool_manager(chat_system)

    (token,) = await _park_writes(
        chat_system, user="u12", channel="c12",
        write_calls=[{"id": "w1", "name": "create_ticket",
                      "arguments": {"title": "t", "body": "b"}}],
    )

    # Turn 2 re-proposes it; the guard answers inline with `duplicate_of_pending`.
    _set_engine(chat_system, [
        _calls({"id": "w2", "name": "create_ticket",
                "arguments": {"title": "t", "body": "b"}}),
        _text("Still waiting."),
    ])
    await _drain(chat_system.stream_response(
        "test_persona", "u12", "c12", "do it again"))

    before = _tool_entries(mem_manager)
    assert before["w2"]["status"] == "duplicate_of_pending"

    _set_engine(chat_system, [_text("Ticket opened.")])
    await _drain(chat_system.stream_resolve_park(
        "u12", "test_persona", token, approved=True))

    after = _tool_entries(mem_manager)
    assert after["w1"]["status"] == "approved"
    assert after["w2"]["status"] == "approved", (
        "the suppressed duplicate still claims the action is pending"
    )
    # Marked, so the transcript does not read as the action having run twice.
    assert after["w2"]["duplicate_of"] == "w1"


@pytest.mark.asyncio
async def test_duplicate_in_the_same_turn_is_also_patched(mocked_chat_system):
    """The in-turn case: park and duplicate share one row.

    `_register_duplicates` runs AFTER `_register_parks` precisely so a duplicate
    can find a park minted moments earlier in the same turn.
    """
    chat_system, mem_manager = mocked_chat_system
    _confirm_persona(chat_system)
    _recording_tool_manager(chat_system)

    _set_engine(chat_system, [
        _calls({"id": "w1", "name": "create_ticket",
                "arguments": {"title": "t", "body": "b"}}),
        _calls({"id": "w2", "name": "create_ticket",
                "arguments": {"title": "t", "body": "b"}}),
        _text("Proposed once."),
    ])
    await _drain(chat_system.stream_response(
        "test_persona", "u13", "c13", "open a ticket"))

    parks = chat_system.confirmations.list_for("u13", "test_persona")
    assert len(parks) == 1
    assert _tool_entries(mem_manager)["w2"]["status"] == "duplicate_of_pending"

    _set_engine(chat_system, [_text("Denied.")])
    await _drain(chat_system.stream_resolve_park(
        "u13", "test_persona", parks[0].token, approved=False))

    after = _tool_entries(mem_manager)
    assert after["w1"]["status"] == "denied"
    assert after["w2"]["status"] == "denied"


@pytest.mark.asyncio
async def test_distinct_writes_across_turns_both_park(mocked_chat_system):
    """The guard must not swallow a genuinely different second proposal."""
    chat_system, _ = mocked_chat_system
    _confirm_persona(chat_system)
    _recording_tool_manager(chat_system)

    (first,) = await _park_writes(
        chat_system, user="u11", channel="c11",
        write_calls=[{"id": "w1", "name": "create_ticket",
                      "arguments": {"title": "t", "body": "b"}}],
    )

    _set_engine(chat_system, [
        _calls({"id": "w2", "name": "create_ticket",
                "arguments": {"title": "OTHER", "body": "b"}}),
        _text("Two for review."),
    ])
    await _drain(chat_system.stream_response(
        "test_persona", "u11", "c11", "and another"))

    parks = chat_system.confirmations.list_for("u11", "test_persona")
    assert len(parks) == 2
    assert {p.token for p in parks} >= {first}
