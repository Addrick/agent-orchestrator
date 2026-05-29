# tests/integration/test_tool_loop_exit_invariants.py
#
# Exit-path × cleanup-invariant coverage for the agent tool-use loop.
#
# The tool loop has ~5 ways to terminate and the orchestrator (_orchestrate)
# ~8; each exit owes the same handful of obligations. The bugs these tests
# pin down are cells in that matrix that nothing previously checked. See the
# design reference: docs/tool_loop_exits.md
#
# Invariants every exit must satisfy:
#   I1  get_turn_context() is None after the turn      (no ContextVar leak)
#   I2  user turn persisted even on mid-flight error
#   I3  assistant persisted iff final_text and LLM_GENERATION
#   I4  _conversation_taints written with correct value
#   I5  exactly one terminal event (DoneEvent xor ErrorEvent)
#
# RED-FIRST: against current master, the tests for findings #1/#2/#4 fail
# (they document the bugs); the fixes flip them green. The #3 concurrency
# test is green only once _execute_calls parallelizes read groups.

import asyncio
import copy
import time

import pytest
from unittest.mock import AsyncMock

from src.chat_system import (
    DoneEvent, ErrorEvent, ToolCallResultEvent, ResponseType,
)
from src.tools.turn_context import get_turn_context

pytestmark = pytest.mark.integration


async def _drain(stream):
    return [ev async for ev in stream]


def _script_engine(chat_system, scripted, *, capture=None):
    """Point mocked generate_response at a sequence of (result, payload)
    tuples. Optionally records live turn context + a deep copy of each
    per-iteration message history (so in-place appends can't mutate the
    snapshot out from under assertions)."""
    it = iter(scripted)

    async def fake_generate_response(persona_config, history_object, *a, **k):
        if capture is not None:
            capture["turn_ctx_seen"].append(get_turn_context())
            capture["histories"].append(
                copy.deepcopy(history_object.get("message_history"))
            )
        return next(it)

    chat_system.text_engine.generate_response.side_effect = fake_generate_response


# --------------------------------------------------------------------------
# Finding #1 — turn-context ContextVar leak across exit paths.
# Each test asserts I1 (get_turn_context() is None after the turn).
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ctx_reset_when_postloop_persistence_raises(mocked_chat_system):
    """Normal LLM_GENERATION finish, but assistant persistence raises after
    the loop. reset_turn_context must still run."""
    chat_system, _ = mocked_chat_system
    chat_system.personas["test_persona"].set_enabled_tools(["*"])
    _script_engine(chat_system, [({"type": "text", "content": "hello"}, {})])

    def boom(*a, **k):
        raise RuntimeError("simulated DB failure in post-loop persistence")
    chat_system._commit_or_update_assistant = boom  # type: ignore[assignment]

    assert get_turn_context() is None
    with pytest.raises(RuntimeError):
        await _drain(chat_system.stream_response(
            "test_persona", "u1", "c1", "hi",
        ))
    assert get_turn_context() is None, "I1: ContextVar leaked after post-loop exception"


@pytest.mark.asyncio
async def test_ctx_reset_on_early_consumer_break(mocked_chat_system):
    """A consumer that stops iterating right after DoneEvent (realistic SSE
    client) must not leave the ContextVar set."""
    chat_system, _ = mocked_chat_system
    chat_system.personas["test_persona"].set_enabled_tools(["*"])
    _script_engine(chat_system, [({"type": "text", "content": "answer"}, {})])

    assert get_turn_context() is None
    stream = chat_system.stream_response("test_persona", "u2", "c2", "hi")
    async for ev in stream:
        if isinstance(ev, DoneEvent):
            break
    await stream.aclose()  # what GC/StreamingResponse teardown does
    assert get_turn_context() is None, "I1: ContextVar leaked after early consumer break"


@pytest.mark.asyncio
async def test_ctx_reset_when_audit_logging_raises(mocked_chat_system):
    """PENDING_CONFIRMATION exit: log_audit_event after parking is unguarded.
    If it raises, reset_turn_context must still run."""
    chat_system, _ = mocked_chat_system
    chat_system.personas["test_persona"].set_enabled_tools(["*"])
    chat_system.tool_manager.enrich_audit_action = AsyncMock(return_value=None)  # type: ignore[assignment]
    _script_engine(chat_system, [
        ({"type": "tool_calls", "calls": [
            {"id": "w1", "name": "create_ticket",
             "arguments": {"title": "x", "body": "y"}}]}, {}),
    ])

    def boom(*a, **k):
        raise RuntimeError("simulated audit-log failure")
    chat_system.memory_manager.log_audit_event = boom  # type: ignore[assignment]

    assert get_turn_context() is None
    with pytest.raises(RuntimeError):
        await _drain(chat_system.stream_response(
            "test_persona", "u3", "c3", "open a ticket",
        ))
    assert get_turn_context() is None, "I1: ContextVar leaked after audit-log exception"


@pytest.mark.asyncio
async def test_ctx_reset_when_user_turn_logging_raises(mocked_chat_system):
    """_log_user_turn runs before the loop's try-block. If it raises, the
    ContextVar (already set) must still be reset."""
    chat_system, _ = mocked_chat_system
    chat_system.personas["test_persona"].set_enabled_tools(["*"])
    _script_engine(chat_system, [({"type": "text", "content": "hi"}, {})])

    def boom(*a, **k):
        raise RuntimeError("simulated user-turn log failure")
    chat_system._log_user_turn = boom  # type: ignore[assignment]

    assert get_turn_context() is None
    with pytest.raises(RuntimeError):
        await _drain(chat_system.stream_response(
            "test_persona", "u4", "c4", "hi",
        ))
    assert get_turn_context() is None, "I1: ContextVar leaked after user-turn-log exception"


# --------------------------------------------------------------------------
# Finding #2 — tool result message must carry a stable tool_call_id that
# matches the lifecycle event, even when the provider omits 'id'.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_idless_tool_call_gets_stable_matching_tool_call_id(mocked_chat_system):
    """A provider that returns a tool call without an 'id' must still produce
    a tool-result history message whose tool_call_id is non-null and equals
    the ToolCallResultEvent.call_id — otherwise the next iteration sends the
    model unpaired call/result blocks."""
    chat_system, _ = mocked_chat_system
    chat_system.personas["test_persona"].set_enabled_tools(["*"])

    capture = {"turn_ctx_seen": [], "histories": []}
    _script_engine(chat_system, [
        # iteration 0: tool call with NO 'id' field
        ({"type": "tool_calls", "calls": [
            {"name": "get_agent_status", "arguments": {"agent_id": "z"}}]}, {}),
        # iteration 1: final answer
        ({"type": "text", "content": "done"}, {}),
    ], capture=capture)

    async def fake_execute(name, **kwargs):
        return {"status": "running"}
    chat_system.tool_manager.execute_tool = fake_execute  # type: ignore[assignment]

    events = await _drain(chat_system.stream_response(
        "test_persona", "u5", "c5", "status?",
    ))

    # The tool message the model sees on iteration 1.
    iter1 = capture["histories"][1]
    tool_msgs = [m for m in iter1 if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    history_call_id = tool_msgs[0]["tool_call_id"]
    assert history_call_id is not None, "#2: tool_call_id is None for id-less provider call"

    result_evs = [e for e in events if isinstance(e, ToolCallResultEvent)]
    assert len(result_evs) == 1
    assert result_evs[0].call_id == history_call_id, (
        "#2: event call_id and history tool_call_id diverge"
    )


# --------------------------------------------------------------------------
# Finding #4 — resume_pending_confirmation must set (and reset) turn context
# for the post-confirmation continuation.
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resume_sets_and_resets_turn_context(mocked_chat_system):
    """The continuation generate_response in resume must run with the turn's
    scope pinned, and the ContextVar must be reset afterward."""
    chat_system, _ = mocked_chat_system
    chat_system.personas["test_persona"].set_enabled_tools(["*"])
    chat_system.tool_manager.enrich_audit_action = AsyncMock(return_value=None)  # type: ignore[assignment]
    chat_system._execute_write_calls = AsyncMock()  # type: ignore[assignment]

    # Turn 1: drive a write tool to park a PendingConfirmation, drained clean.
    _script_engine(chat_system, [
        ({"type": "tool_calls", "calls": [
            {"id": "w1", "name": "create_ticket",
             "arguments": {"title": "t", "body": "b"}}]}, {}),
    ])
    await _drain(chat_system.stream_response(
        "test_persona", "u6", "c6", "open a ticket",
    ))
    assert ("u6", "test_persona") in chat_system._pending_confirmations
    assert get_turn_context() is None  # clean after turn 1

    # Resume: record live ctx during the continuation.
    seen = {}

    async def fake_continuation(persona_config, history_object, *a, **k):
        seen["ctx"] = get_turn_context()
        return ({"type": "text", "content": "Ticket created."}, {})
    chat_system.text_engine.generate_response.side_effect = fake_continuation

    await chat_system.resume_pending_confirmation("u6", "test_persona", approved=True)

    assert seen["ctx"] is not None, "#4: resume continuation ran with no turn context"
    assert seen["ctx"].user_identifier == "u6"
    assert seen["ctx"].persona_name == "test_persona"
    assert get_turn_context() is None, "#4: turn context not reset after resume"


# --------------------------------------------------------------------------
# Finding #3 — read calls sharing a group_id should execute concurrently.
# (Green only after _execute_calls parallelizes the read group.)
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_group_executes_concurrently(mocked_chat_system):
    """Two read calls returned in one provider response share a group_id and
    should run concurrently — their executions must overlap in time."""
    chat_system, _ = mocked_chat_system
    chat_system.personas["test_persona"].set_enabled_tools(["*"])
    _script_engine(chat_system, [
        ({"type": "tool_calls", "calls": [
            {"id": "r1", "name": "get_agent_status", "arguments": {"agent_id": "a"}},
            {"id": "r2", "name": "get_agent_history", "arguments": {"agent_id": "a"}},
        ]}, {}),
        ({"type": "text", "content": "done"}, {}),
    ])

    spans = {}

    async def fake_execute(name, **kwargs):
        start = time.perf_counter()
        await asyncio.sleep(0.1)
        spans[name] = (start, time.perf_counter())
        return {"ok": True}
    chat_system.tool_manager.execute_tool = fake_execute  # type: ignore[assignment]

    await _drain(chat_system.stream_response(
        "test_persona", "u7", "c7", "check both",
    ))

    (s1, e1) = spans["get_agent_status"]
    (s2, e2) = spans["get_agent_history"]
    # Concurrent ⇒ the second starts before the first finishes.
    assert s2 < e1, "#3: read group ran serially, not concurrently"
