# tests/integration/test_resume_kernel_convergence.py
#
# DP-124: resume_pending_confirmation re-enters the _orchestrate kernel with
# the parked turn instead of re-implementing it. These tests pin the four
# behaviours that convergence unlocks / must preserve:
#
#   1. approved -> the continuation runs a *further* tool loop
#   2. denied   -> clean close, no write executed
#   3. the persisted assistant row carries the real channel (not channel="")
#   4. no turn-context leak across the resumed turn (scope pinned during the
#      continuation, reset afterwards)
#
# plus the parked-turn timeout/expiry behaviour, which must survive the
# refactor. All run with a real MemoryManager + TextEngine (generate_response
# mocked), so the continuation exercises the genuine ToolLoop path.

import time

import pytest

from src.chat_system import (
    ResponseType, PendingConfirmationEvent, DoneEvent,
)
from src.confirmations import PendingConfirmation
from src.persona import ExecutionMode
from src.tools.turn_context import get_turn_context
from config.global_config import PENDING_CONFIRMATION_TIMEOUT

pytestmark = pytest.mark.integration


async def _drain(stream):
    return [ev async for ev in stream]


def _set_engine(chat_system, scripted):
    """Point mocked generate_response at a sequence of (result, payload) tuples."""
    it = iter(scripted)

    async def fake_generate_response(persona_config, history_object, *a, **k):
        return next(it)

    chat_system.text_engine.generate_response.side_effect = fake_generate_response


async def _park_write(chat_system, *, user, channel, write_call):
    """Drive turn 1 to park a write-confirmation, drained clean."""
    _set_engine(chat_system, [
        ({"type": "tool_calls", "calls": [write_call]}, {}),
    ])
    await _drain(chat_system.stream_response("test_persona", user, channel, "do the thing"))
    assert (user, "test_persona") in chat_system.confirmations.pending
    assert get_turn_context() is None


@pytest.mark.asyncio
async def test_resume_approved_runs_further_tool_loop(mocked_chat_system):
    """Approval continues the turn through the full kernel: the approved write
    executes and the model's follow-up read tool call runs a *further* loop
    iteration — the capability the old partial re-implementation lacked."""
    chat_system, _ = mocked_chat_system
    persona = chat_system.personas["test_persona"]
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(["*"])

    executed = []

    async def fake_execute(name, **kwargs):
        executed.append(name)
        return {"ok": True}
    chat_system.tool_manager.execute_tool = fake_execute  # type: ignore[assignment]

    await _park_write(
        chat_system, user="u1", channel="c1",
        write_call={"id": "w1", "name": "create_ticket",
                    "arguments": {"title": "t", "body": "b"}},
    )

    # Continuation: a read tool call, then a text answer.
    _set_engine(chat_system, [
        ({"type": "tool_calls", "calls": [
            {"id": "r1", "name": "get_agent_status", "arguments": {"agent_id": "a"}}]}, {}),
        ({"type": "text", "content": "Ticket opened and status checked."}, {}),
    ])

    text, rtype, assistant_id, uid = await chat_system.resume_pending_confirmation(
        "u1", "test_persona", approved=True,
    )

    assert rtype == ResponseType.LLM_GENERATION
    assert text == "Ticket opened and status checked."
    assert "create_ticket" in executed, "approved write was not executed"
    assert "get_agent_status" in executed, "continuation did not run a further tool loop"
    assert assistant_id is not None
    assert uid is None
    assert ("u1", "test_persona") not in chat_system.confirmations.pending
    assert get_turn_context() is None, "turn scope leaked after resume"


@pytest.mark.asyncio
async def test_resume_denied_closes_cleanly(mocked_chat_system):
    """Denial feeds synthetic denial results to the model and returns its
    close-out text; the rejected write never executes."""
    chat_system, _ = mocked_chat_system
    persona = chat_system.personas["test_persona"]
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(["*"])

    executed = []

    async def fake_execute(name, **kwargs):
        executed.append(name)
        return {"ok": True}
    chat_system.tool_manager.execute_tool = fake_execute  # type: ignore[assignment]

    await _park_write(
        chat_system, user="u2", channel="c2",
        write_call={"id": "w1", "name": "create_ticket",
                    "arguments": {"title": "t", "body": "b"}},
    )

    _set_engine(chat_system, [
        ({"type": "text", "content": "Understood, I won't create the ticket."}, {}),
    ])

    text, rtype, assistant_id, uid = await chat_system.resume_pending_confirmation(
        "u2", "test_persona", approved=False,
    )

    assert rtype == ResponseType.LLM_GENERATION
    assert "won't create" in text
    assert "create_ticket" not in executed, "denied write must not execute"
    assert ("u2", "test_persona") not in chat_system.confirmations.pending
    assert get_turn_context() is None


@pytest.mark.asyncio
async def test_resume_persists_assistant_on_correct_channel(mocked_chat_system):
    """The continuation's assistant row is logged on the parked channel — not
    the channel="" the old implementation hardcoded."""
    chat_system, mem_manager = mocked_chat_system
    persona = chat_system.personas["test_persona"]
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(["*"])

    async def fake_execute(name, **kwargs):
        return {"ok": True}
    chat_system.tool_manager.execute_tool = fake_execute  # type: ignore[assignment]

    await _park_write(
        chat_system, user="u3", channel="team-chan",
        write_call={"id": "w1", "name": "create_ticket",
                    "arguments": {"title": "t", "body": "b"}},
    )

    _set_engine(chat_system, [
        ({"type": "text", "content": "Done."}, {}),
    ])
    await chat_system.resume_pending_confirmation("u3", "test_persona", approved=True)

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
async def test_resume_persists_write_into_tool_context(mocked_chat_system):
    """The approved write and its result must be captured in the assistant row's
    tool_context so they replay on later turns. Regression: the resumed loop used
    to set history_start *after* the parked tool calls, dropping the executed
    write from history entirely."""
    chat_system, mem_manager = mocked_chat_system
    persona = chat_system.personas["test_persona"]
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(["*"])

    async def fake_execute(name, **kwargs):
        return {"ticket_id": 42}
    chat_system.tool_manager.execute_tool = fake_execute  # type: ignore[assignment]

    await _park_write(
        chat_system, user="u6", channel="c6",
        write_call={"id": "w1", "name": "create_ticket",
                    "arguments": {"title": "t", "body": "b"}},
    )

    _set_engine(chat_system, [
        ({"type": "text", "content": "Ticket created."}, {}),
    ])
    await chat_system.resume_pending_confirmation("u6", "test_persona", approved=True)

    conn = mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT tool_context FROM User_Interactions "
        "WHERE author_role='assistant' AND content='Ticket created.'"
    )
    row = cursor.fetchone()
    assert row is not None and row["tool_context"], \
        "assistant turn persisted without tool_context"

    import json
    tool_ctx = json.loads(row["tool_context"])
    # The parked write tool_call and its execution result must both be present.
    assert any(
        m.get("role") == "assistant" and any(
            c.get("name") == "create_ticket" for c in m.get("tool_calls", [])
        ) for m in tool_ctx
    ), "approved write tool_call missing from replayed tool_context"
    assert any(
        m.get("role") == "tool" and m.get("name") == "create_ticket" for m in tool_ctx
    ), "approved write result missing from replayed tool_context"

    # And it actually replays into the next turn's formatted history.
    replayed = chat_system.request_builder.format_raw_history_for_llm(
        [{"author_role": "assistant", "author_name": "test_persona",
          "content": "Ticket created.", "tool_context": row["tool_context"]}],
        memory_mode="channel", persona_name="test_persona", server_id=None,
    )
    assert any(m.get("role") == "tool" and m.get("name") == "create_ticket" for m in replayed)


@pytest.mark.asyncio
async def test_resume_pins_scope_during_continuation_and_resets(mocked_chat_system):
    """The resumed turn runs with the parked scope pinned (so engine-side tools
    inherit persona/user/channel) and the ContextVar is reset on exit."""
    chat_system, _ = mocked_chat_system
    persona = chat_system.personas["test_persona"]
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(["*"])

    seen = {}

    async def fake_execute(name, **kwargs):
        seen["ctx"] = get_turn_context()
        return {"ok": True}
    chat_system.tool_manager.execute_tool = fake_execute  # type: ignore[assignment]

    await _park_write(
        chat_system, user="u4", channel="c4",
        write_call={"id": "w1", "name": "create_ticket",
                    "arguments": {"title": "t", "body": "b"}},
    )

    # Continuation issues a read so fake_execute fires inside the resumed turn.
    _set_engine(chat_system, [
        ({"type": "tool_calls", "calls": [
            {"id": "r1", "name": "get_agent_status", "arguments": {"agent_id": "a"}}]}, {}),
        ({"type": "text", "content": "ok"}, {}),
    ])
    await chat_system.resume_pending_confirmation("u4", "test_persona", approved=True)

    assert seen["ctx"] is not None, "continuation ran with no turn context"
    assert seen["ctx"].user_identifier == "u4"
    assert seen["ctx"].persona_name == "test_persona"
    assert seen["ctx"].channel == "c4"
    assert get_turn_context() is None, "turn context not reset after resume"


@pytest.mark.asyncio
async def test_resume_expired_confirmation(mocked_chat_system):
    """An expired parked confirmation closes out without re-entering the kernel."""
    chat_system, _ = mocked_chat_system

    chat_system.confirmations.pending[("u5", "test_persona")] = PendingConfirmation(
        write_calls=[{"id": "w1", "name": "create_ticket", "arguments": {}}],
        conversation_history=[],
        persona_name="test_persona",
        tools_for_llm=[],
        image_url=None,
        channel="c5",
        created_at=time.time() - PENDING_CONFIRMATION_TIMEOUT - 10,
    )

    text, rtype, assistant_id, uid = await chat_system.resume_pending_confirmation(
        "u5", "test_persona", approved=True,
    )

    assert rtype == ResponseType.DEV_COMMAND
    assert "expired" in text.lower()
    assert assistant_id is None
    assert ("u5", "test_persona") not in chat_system.confirmations.pending
    assert get_turn_context() is None


@pytest.mark.asyncio
async def test_resume_no_pending_confirmation(mocked_chat_system):
    """Resume with nothing parked returns the not-found close-out."""
    chat_system, _ = mocked_chat_system
    text, rtype, assistant_id, uid = await chat_system.resume_pending_confirmation(
        "nobody", "test_persona", approved=True,
    )
    assert rtype == ResponseType.DEV_COMMAND
    assert "No pending confirmation" in text
    assert assistant_id is None


# -------- DP-127: portal-facing park event + tokenised streaming resume --------


@pytest.mark.asyncio
async def test_park_yields_pending_confirmation_event(mocked_chat_system):
    """When a write parks, the stream surfaces a PendingConfirmationEvent
    (structured calls + resume token) before the terminal DoneEvent, so an
    interactive surface can render approve/deny. The token matches the park."""
    chat_system, _ = mocked_chat_system
    persona = chat_system.personas["test_persona"]
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(["*"])

    _set_engine(chat_system, [
        ({"type": "tool_calls", "calls": [
            {"id": "w1", "name": "create_ticket",
             "arguments": {"title": "t", "body": "b"}}]}, {}),
    ])
    events = await _drain(
        chat_system.stream_response("test_persona", "u6", "c6", "do it")
    )

    pce = [e for e in events if isinstance(e, PendingConfirmationEvent)]
    assert len(pce) == 1, "exactly one PendingConfirmationEvent expected"
    ev = pce[0]
    assert ev.persona_name == "test_persona"
    assert ev.write_calls[0]["name"] == "create_ticket"
    assert ev.token, "park event must carry a resume token"

    parked = chat_system.confirmations.pending[("u6", "test_persona")]
    assert ev.token == parked.token, "event token must match the stored park"

    pce_idx = next(i for i, e in enumerate(events)
                   if isinstance(e, PendingConfirmationEvent))
    done_idx = next(i for i, e in enumerate(events) if isinstance(e, DoneEvent))
    assert pce_idx < done_idx, "park event must precede the terminal DoneEvent"


@pytest.mark.asyncio
async def test_stream_resume_token_mismatch_preserves_park(mocked_chat_system):
    """A streaming resume with a stale token is rejected and leaves the real
    park intact, so a correct-token retry can still go through."""
    chat_system, _ = mocked_chat_system
    persona = chat_system.personas["test_persona"]
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(["*"])

    executed = []

    async def fake_execute(name, **kwargs):
        executed.append(name)
        return {"ok": True}
    chat_system.tool_manager.execute_tool = fake_execute  # type: ignore[assignment]

    await _park_write(
        chat_system, user="u7", channel="c7",
        write_call={"id": "w1", "name": "create_ticket",
                    "arguments": {"title": "t", "body": "b"}},
    )

    events = await _drain(chat_system.stream_resume_confirmation(
        "u7", "test_persona", approved=True, expected_token="not-the-token",
    ))
    done = [e for e in events if isinstance(e, DoneEvent)][-1]
    assert done.response_type == ResponseType.DEV_COMMAND
    assert "no longer valid" in done.text.lower()
    assert "create_ticket" not in executed, "stale resume must not execute the write"
    assert ("u7", "test_persona") in chat_system.confirmations.pending, \
        "stale token must leave the park intact"


@pytest.mark.asyncio
async def test_stream_resume_valid_token_executes_write(mocked_chat_system):
    """A streaming resume with the matching token consumes the park, executes
    the write, and streams the continuation."""
    chat_system, _ = mocked_chat_system
    persona = chat_system.personas["test_persona"]
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(["*"])

    executed = []

    async def fake_execute(name, **kwargs):
        executed.append(name)
        return {"ok": True}
    chat_system.tool_manager.execute_tool = fake_execute  # type: ignore[assignment]

    await _park_write(
        chat_system, user="u8", channel="c8",
        write_call={"id": "w1", "name": "create_ticket",
                    "arguments": {"title": "t", "body": "b"}},
    )
    token = chat_system.confirmations.pending[("u8", "test_persona")].token

    _set_engine(chat_system, [
        ({"type": "text", "content": "Ticket opened."}, {}),
    ])
    events = await _drain(chat_system.stream_resume_confirmation(
        "u8", "test_persona", approved=True, expected_token=token,
    ))
    done = [e for e in events if isinstance(e, DoneEvent)][-1]
    assert done.response_type == ResponseType.LLM_GENERATION
    assert done.text == "Ticket opened."
    assert "create_ticket" in executed
    assert ("u8", "test_persona") not in chat_system.confirmations.pending
    assert get_turn_context() is None
