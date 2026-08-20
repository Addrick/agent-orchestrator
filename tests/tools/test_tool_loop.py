# tests/tools/test_tool_loop.py

import json
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.engine import LLMCommunicationError
from src.generation_events import (
    ErrorEvent, ResponseType, TokenEvent,
    ToolCallResultEvent, ToolCallStartEvent,
)
from src.persona import ExecutionMode
from src.tools.tool_loop import (
    PARK_STATUS_AWAITING, PARK_STATUS_DUPLICATE, ToolLoop, WriteParkedEvent,
    _ApiPayloadEvent, _LoopFinishedEvent, _ToolContextEvent,
    render_max_iteration_text, write_call_identity,
)


def _make_persona(execution_mode=ExecutionMode.AUTONOMOUS):
    p = MagicMock()
    p.get_config_for_engine.return_value = {"model_name": "local"}
    p.get_prompt.return_value = "You are a test assistant."
    p.get_execution_mode.return_value = execution_mode
    return p


def _stream(events: List[Dict[str, Any]]):
    """Build an async iterator that yields the given provider events."""
    async def gen() -> AsyncIterator[Dict[str, Any]]:
        for ev in events:
            yield ev
    return gen()


def _make_engine(streams: List[List[Dict[str, Any]]]):
    """Mock TextEngine whose stream_messages returns each scripted stream
    in order across loop iterations."""
    engine = MagicMock()
    iterator = iter(streams)

    def stream_messages(*args, **kwargs):
        return _stream(next(iterator))
    engine.stream_messages.side_effect = stream_messages
    return engine


def _make_tool_manager(results: Dict[str, Any]):
    manager = MagicMock()
    async def execute(name, **kwargs):
        return results.get(name, {"result": "ok"})
    manager.execute_tool = AsyncMock(side_effect=execute)
    manager.enrich_audit_action = AsyncMock(return_value=None)
    return manager


async def _drain(loop_run):
    """Collect events from a ToolLoop.run() iterator."""
    out = []
    async for ev in loop_run:
        out.append(ev)
    return out


@pytest.mark.asyncio
async def test_single_tool_call_then_text():
    """One tool call, then the model produces text and exits."""
    engine = _make_engine([
        [
            {"type": "api_payload", "payload": {"req": 1}},
            {"type": "tool_calls", "calls": [
                {"id": "abc123", "name": "search_tool", "arguments": {"q": "x"}}
            ]},
            {"type": "done", "full_text": ""},
        ],
        [
            {"type": "api_payload", "payload": {"req": 2}},
            {"type": "text_delta", "text": "hello "},
            {"type": "text_delta", "text": "world"},
            {"type": "done", "full_text": "hello world"},
        ],
    ])
    tools = _make_tool_manager({"search_tool": {"result": "found"}})
    loop = ToolLoop(engine, tools, max_iterations=5)
    history: List[Dict[str, Any]] = []

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=history,
        params=MagicMock(), tools=[],
    ))

    types = [type(e).__name__ for e in events]
    assert types == [
        "_ApiPayloadEvent",
        "ToolCallStartEvent",
        "ToolCallResultEvent",
        "TokenEvent", "TokenEvent",
        "_ApiPayloadEvent",
        "_LoopFinishedEvent",
    ]

    start = events[1]
    assert isinstance(start, ToolCallStartEvent)
    assert start.tool_name == "search_tool"
    assert start.call_id == "abc123"
    assert start.arguments == {"q": "x"}

    result = events[2]
    assert isinstance(result, ToolCallResultEvent)
    assert result.call_id == "abc123"
    assert json.loads(result.result) == {"result": "found"}
    assert result.error is None

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.final_text == "hello world"
    assert finished.response_type == ResponseType.LLM_GENERATION
    assert finished.tool_context_json is not None  # contains the assistant+tool turns

    # History was mutated to contain assistant tool_calls + tool result.
    assert history[0]["role"] == "assistant"
    assert history[0]["tool_calls"][0]["name"] == "search_tool"
    assert history[1]["role"] == "tool"


@pytest.mark.asyncio
async def test_multiple_sequential_tool_calls():
    """Two iterations of tool calls before text settles."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "c1", "name": "tool_a", "arguments": {}}
            ]},
            {"type": "done", "full_text": ""},
        ],
        [
            {"type": "tool_calls", "calls": [
                {"id": "c2", "name": "tool_b", "arguments": {"k": 1}}
            ]},
            {"type": "done", "full_text": ""},
        ],
        [
            {"type": "text_delta", "text": "done"},
            {"type": "done", "full_text": "done"},
        ],
    ])
    tools = _make_tool_manager({"tool_a": {"result": "a"}, "tool_b": {"result": "b"}})
    loop = ToolLoop(engine, tools, max_iterations=5)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
    results = [e for e in events if isinstance(e, ToolCallResultEvent)]
    assert [s.tool_name for s in starts] == ["tool_a", "tool_b"]
    assert [r.tool_name for r in results] == ["tool_a", "tool_b"]
    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.final_text == "done"


@pytest.mark.asyncio
async def test_group_id_shared_per_iter_unique_across_iters():
    """portal_tool_trace_ui Phase A: every ToolCall*Event minted in the
    same iteration shares one group_id; a new iter mints a fresh one.
    Carries the forward-compat plumbing for parallel-call rendering."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "c1", "name": "tool_a", "arguments": {}},
                {"id": "c2", "name": "tool_b", "arguments": {}},
            ]},
            {"type": "done", "full_text": ""},
        ],
        [
            {"type": "tool_calls", "calls": [
                {"id": "c3", "name": "tool_a", "arguments": {}}
            ]},
            {"type": "done", "full_text": ""},
        ],
        [
            {"type": "text_delta", "text": "ok"},
            {"type": "done", "full_text": "ok"},
        ],
    ])
    tools = _make_tool_manager({"tool_a": {"r": 1}, "tool_b": {"r": 2}})
    loop = ToolLoop(engine, tools, max_iterations=5)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    tool_evs = [e for e in events
                if isinstance(e, (ToolCallStartEvent, ToolCallResultEvent))]
    # iter0 produced 2 calls → 4 events (2 start + 2 result), all same group
    iter0 = tool_evs[:4]
    iter1 = tool_evs[4:6]
    assert all(e.group_id for e in tool_evs), "group_id must be populated"
    assert len({e.group_id for e in iter0}) == 1
    assert len({e.group_id for e in iter1}) == 1
    assert iter0[0].group_id != iter1[0].group_id


@pytest.mark.asyncio
async def test_tool_error_surfaces_in_result_event():
    """A tool whose result dict contains 'error' is surfaced via the
    event's `error` field; the loop continues so the LLM can adapt."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "c1", "name": "broken_tool", "arguments": {}}
            ]},
            {"type": "done", "full_text": ""},
        ],
        [
            {"type": "text_delta", "text": "recovered"},
            {"type": "done", "full_text": "recovered"},
        ],
    ])
    tools = _make_tool_manager({"broken_tool": {"error": "boom"}})
    loop = ToolLoop(engine, tools, max_iterations=5)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    [result] = [e for e in events if isinstance(e, ToolCallResultEvent)]
    assert result.error == "boom"
    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.final_text == "recovered"


@pytest.mark.asyncio
async def test_a_soft_failure_also_surfaces_in_result_event():
    """DP-323: a handler that reports failure by RETURNING must colour the
    ToolCard too, not only one that raises.

    Read tools use the same `{"status": "error"}` shape their write siblings
    do (`proxmox.handler._err`), so before this the portal rendered a failed
    remote command as a clean success and the operator had to read the JSON.
    Same predicate as the gated-write path — one definition of "this failed".
    """
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "c1", "name": "soft_fail", "arguments": {}}
            ]},
            {"type": "done", "full_text": ""},
        ],
        [
            {"type": "text_delta", "text": "recovered"},
            {"type": "done", "full_text": "recovered"},
        ],
    ])
    tools = _make_tool_manager({
        "soft_fail": {"result": {"status": "error",
                                 "message": "ssh failed: timeout"}},
    })
    loop = ToolLoop(engine, tools, max_iterations=5)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    [result] = [e for e in events if isinstance(e, ToolCallResultEvent)]
    assert result.error == "ssh failed: timeout"


@pytest.mark.asyncio
async def test_llm_communication_error_yields_error_event():
    """Provider errors terminate the loop with ErrorEvent."""
    async def boom(*args, **kwargs):
        raise LLMCommunicationError("upstream 500", api_payload={"req": 1})
        yield  # pragma: no cover — make this an async generator
    engine = MagicMock()
    engine.stream_messages.side_effect = lambda *a, **k: boom()
    tools = _make_tool_manager({})
    loop = ToolLoop(engine, tools)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    # ApiPayloadEvent with the error's payload, then ErrorEvent.
    assert any(isinstance(e, _ApiPayloadEvent) for e in events)
    assert isinstance(events[-1], ErrorEvent)
    assert "upstream 500" in events[-1].message


@pytest.mark.asyncio
async def test_max_iterations_cap():
    """If the model never stops calling tools, loop bails after the cap."""
    one_call_stream = lambda i: [
        {"type": "tool_calls", "calls": [
            {"id": f"c{i}", "name": "spinner", "arguments": {}}
        ]},
        {"type": "done", "full_text": ""},
    ]
    engine = _make_engine([one_call_stream(i) for i in range(3)])
    tools = _make_tool_manager({"spinner": {"result": "spin"}})
    loop = ToolLoop(engine, tools, max_iterations=3)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.response_type == ResponseType.DEV_COMMAND
    # DP-335: the cap-hit text lists what the turn actually spent its budget
    # on. The old canned sentence named no tool, so a user (and the next
    # session) could not tell an exhausted budget from a broken tool.
    assert "3 of my tool steps" in finished.final_text
    assert "`spinner`" in finished.final_text
    assert "(same call as #1)" in finished.final_text


@pytest.mark.asyncio
async def test_confirm_mode_parks_write_calls():
    """A write tool is gated via a WriteParkedEvent, not executed."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "w1", "name": "create_ticket", "arguments": {"title": "x"}}
            ]},
            {"type": "done", "full_text": ""},
        ],
        [
            {"type": "done", "full_text": "Queued that for you."},
        ],
    ])
    tools = _make_tool_manager({})
    loop = ToolLoop(engine, tools)

    events = await _drain(loop.run(
        persona=_make_persona(execution_mode=ExecutionMode.CONFIRM),
        conversation_history=[], params=MagicMock(), tools=[],
    ))

    parks = [e for e in events if isinstance(e, WriteParkedEvent)]
    assert len(parks) == 1
    assert parks[0].write_call["name"] == "create_ticket"
    assert parks[0].token
    # Write tool was NOT executed — manager should not have been called for it.
    tools.execute_tool.assert_not_called()


@pytest.mark.asyncio
async def test_park_does_not_end_the_turn():
    """DP-297: the loop keeps going after gating a write, so the turn ends
    with the model's own text instead of dying on the proposal.

    This is the regression that motivated the ticket: the model physically
    could not propose a second write, because the first ended the turn.
    """
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "w1", "name": "create_ticket", "arguments": {"title": "a"}}
            ]},
            {"type": "done", "full_text": ""},
        ],
        [
            {"type": "tool_calls", "calls": [
                {"id": "w2", "name": "update_ticket", "arguments": {"state": "closed"}}
            ]},
            {"type": "done", "full_text": ""},
        ],
        [
            {"type": "done", "full_text": "Proposed two actions."},
        ],
    ])
    loop = ToolLoop(engine, _make_tool_manager({}))

    events = await _drain(loop.run(
        persona=_make_persona(execution_mode=ExecutionMode.CONFIRM),
        conversation_history=[], params=MagicMock(), tools=[],
    ))

    parks = [e for e in events if isinstance(e, WriteParkedEvent)]
    assert [p.write_call["name"] for p in parks] == [
        "create_ticket", "update_ticket",
    ]
    # Distinct tokens — one dialog per proposal, independently resolvable.
    assert len({p.token for p in parks}) == 2

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.response_type == ResponseType.LLM_GENERATION
    assert finished.final_text == "Proposed two actions."


@pytest.mark.asyncio
async def test_parked_write_is_answered_inline_in_history():
    """Each gated write gets a real synthetic tool result appended.

    Not cosmetic: an assistant message whose tool_calls have no matching
    result is rejected by Anthropic and Gemini on the next iteration, so this
    is what makes continuing past a park possible at all. It is also the entry
    ConfirmationManager patches when the operator decides.
    """
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "w1", "name": "create_ticket", "arguments": {"title": "x"}}
            ]},
            {"type": "done", "full_text": ""},
        ],
        [
            {"type": "done", "full_text": "done"},
        ],
    ])
    history: List[Dict[str, Any]] = []
    loop = ToolLoop(engine, _make_tool_manager({}))

    events = await _drain(loop.run(
        persona=_make_persona(execution_mode=ExecutionMode.CONFIRM),
        conversation_history=history, params=MagicMock(), tools=[],
    ))
    token = [e for e in events if isinstance(e, WriteParkedEvent)][0].token

    result_msg = next(m for m in history
                      if m.get("role") == "tool" and m.get("tool_call_id") == "w1")
    payload = json.loads(result_msg["content"])
    assert payload["status"] == PARK_STATUS_AWAITING
    assert payload["token"] == token
    # The re-proposal guard rides in the result, not the system prompt, so it
    # reaches every provider identically.
    assert "not re-submit" in payload["instruction"].lower()


# --- DP-296: tool context must survive every loop exit ---------------------
#
# Before DP-296 the park, error and iteration-cap exits all persisted
# tool_context_json=None, so a gated or errored turn left the model with no
# record of its own tool calls on the next request.


def _tool_context(finished) -> List[Dict[str, Any]]:
    assert finished.tool_context_json is not None
    return json.loads(finished.tool_context_json)


def _assert_calls_all_answered(msgs: List[Dict[str, Any]]) -> None:
    """Every tool_call in the slice has a matching tool result. Anthropic and
    Gemini both reject unpaired blocks, so an unsealed slice is unsendable."""
    answered = {m.get("tool_call_id") for m in msgs if m.get("role") == "tool"}
    for m in msgs:
        for call in m.get("tool_calls") or []:
            assert call["id"] in answered, f"unpaired tool_call {call['id']}"


@pytest.mark.asyncio
async def test_park_seals_reads_and_pending_write():
    """A turn that gated a write persists the reads it ran plus the gated
    write's awaiting-approval entry, sealed so the slice is replayable."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "r1", "name": "get_ticket_details", "arguments": {"id": 1}}
            ]},
            {"type": "done", "full_text": ""},
        ],
        [
            {"type": "tool_calls", "calls": [
                {"id": "w1", "name": "update_ticket", "arguments": {"state": "closed"}}
            ]},
            {"type": "done", "full_text": ""},
        ],
        [
            {"type": "done", "full_text": "Proposed a close."},
        ],
    ])
    tools = _make_tool_manager({"get_ticket_details": {"title": "printer"}})
    loop = ToolLoop(engine, tools)

    events = await _drain(loop.run(
        persona=_make_persona(execution_mode=ExecutionMode.CONFIRM),
        conversation_history=[], params=MagicMock(), tools=[],
    ))

    finished = events[-1]
    assert finished.response_type == ResponseType.LLM_GENERATION
    msgs = _tool_context(finished)
    _assert_calls_all_answered(msgs)

    # The executed read carries its real result.
    read_result = next(m for m in msgs if m.get("tool_call_id") == "r1")
    assert "printer" in read_result["content"]

    # The gated write carries the REAL synthetic result the loop appended —
    # not a placeholder the seal invented. That distinction is what lets the
    # entry be patched in place later when the operator decides.
    write_result = next(m for m in msgs if m.get("tool_call_id") == "w1")
    assert json.loads(write_result["content"])["status"] == PARK_STATUS_AWAITING


@pytest.mark.asyncio
async def test_error_exit_emits_sealed_tool_context():
    """A provider error after a completed tool call still surfaces that call's
    context, so the next turn can see what was attempted."""
    streams = iter([
        [
            {"type": "tool_calls", "calls": [
                {"id": "r1", "name": "search_tickets", "arguments": {}}
            ]},
            {"type": "done", "full_text": ""},
        ],
    ])

    def stream_messages(*args, **kwargs):
        try:
            return _stream(next(streams))
        except StopIteration:
            async def boom():
                raise LLMCommunicationError("upstream 500")
                yield  # pragma: no cover
            return boom()

    engine = MagicMock()
    engine.stream_messages.side_effect = stream_messages
    tools = _make_tool_manager({"search_tickets": {"hits": 3}})
    loop = ToolLoop(engine, tools)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    assert isinstance(events[-1], ErrorEvent)
    ctx = next(e for e in events if isinstance(e, _ToolContextEvent))
    msgs = json.loads(ctx.tool_context_json)
    _assert_calls_all_answered(msgs)
    assert any(m.get("tool_call_id") == "r1" for m in msgs)


@pytest.mark.asyncio
async def test_error_before_any_tool_call_seals_nothing():
    """No tool calls yet — nothing to persist, and the event must not invent
    an empty assistant row."""
    async def boom(*args, **kwargs):
        raise LLMCommunicationError("upstream 500", api_payload={"req": 1})
        yield  # pragma: no cover
    engine = MagicMock()
    engine.stream_messages.side_effect = lambda *a, **k: boom()
    loop = ToolLoop(engine, _make_tool_manager({}))

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    ctx = next(e for e in events if isinstance(e, _ToolContextEvent))
    assert ctx.tool_context_json is None


@pytest.mark.asyncio
async def test_max_iterations_seals_tool_context():
    """The iteration cap is an exit like any other — it must not drop the
    calls the model made on the way there."""
    one_call_stream = lambda i: [
        {"type": "tool_calls", "calls": [
            {"id": f"c{i}", "name": "spinner", "arguments": {}}
        ]},
        {"type": "done", "full_text": ""},
    ]
    engine = _make_engine([one_call_stream(i) for i in range(3)])
    tools = _make_tool_manager({"spinner": {"result": "spin"}})
    loop = ToolLoop(engine, tools, max_iterations=3)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    msgs = _tool_context(events[-1])
    _assert_calls_all_answered(msgs)
    assert {m.get("tool_call_id") for m in msgs if m.get("role") == "tool"} == {
        "c0", "c1", "c2",
    }


@pytest.mark.asyncio
async def test_seal_respects_history_start_override():
    """A resumed turn seals from the park boundary, so the whole span (parked
    read, approved write, continuation) lands in one tool_context."""
    prior = [
        {"role": "user", "content": "close it"},
        {"role": "assistant", "tool_calls": [
            {"id": "w1", "name": "update_ticket", "arguments": {}}
        ]},
        {"role": "tool", "tool_call_id": "w1", "name": "update_ticket",
         "content": '{"ok": true}'},
    ]
    engine = _make_engine([[{"type": "done", "full_text": "Closed it."}]])
    loop = ToolLoop(engine, _make_tool_manager({}))

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=prior,
        params=MagicMock(), tools=[], history_start_override=1,
    ))

    msgs = _tool_context(events[-1])
    _assert_calls_all_answered(msgs)
    # Starts at the park boundary — the user turn stays out of the block.
    assert msgs[0].get("tool_calls")[0]["id"] == "w1"
    assert len(msgs) == 2


@pytest.mark.asyncio
async def test_clean_exit_does_not_seal_with_the_error_reason():
    """DP-296 review: the no-more-tool-calls exit is a *normal* completion, but
    it sealed with SEAL_ERROR. Dormant while every read is answered by
    _execute_calls — but if a result ever failed to land, a cleanly finished
    turn would tell the model its own successful call errored."""
    from src.tools.tool_loop import SEAL_ERROR

    # A call whose result never landed, replayed into a turn that then finishes
    # cleanly: exactly the shape the reason string describes.
    prior = [
        {"role": "user", "content": "close it"},
        {"role": "assistant", "tool_calls": [
            {"id": "w1", "name": "update_ticket", "arguments": {}}
        ]},
    ]
    engine = _make_engine([[{"type": "done", "full_text": "All set."}]])
    loop = ToolLoop(engine, _make_tool_manager({}))

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=prior,
        params=MagicMock(), tools=[], history_start_override=1,
    ))

    msgs = _tool_context(events[-1])
    _assert_calls_all_answered(msgs)
    sealed = next(m for m in msgs if m.get("role") == "tool")
    assert json.loads(sealed["content"])["reason"] != SEAL_ERROR
    assert json.loads(sealed["content"]) == {
        "status": "not_executed", "reason": "unknown",
    }


# ---- DP-297 duplicate-proposal guard -------------------------------------

def test_write_call_identity_ignores_the_provider_call_id():
    """Two re-proposals of one action always carry different ids, so the id
    must not participate — otherwise the guard never fires."""
    a = {"id": "call_1", "name": "create_ticket", "arguments": {"t": "x"}}
    b = {"id": "call_2", "name": "create_ticket", "arguments": {"t": "x"}}
    assert write_call_identity(a) == write_call_identity(b)


def test_write_call_identity_is_argument_order_independent():
    """Argument key order is not stable across iterations; a reordered dict is
    the same proposal."""
    a = {"name": "create_ticket", "arguments": {"a": 1, "b": 2}}
    b = {"name": "create_ticket", "arguments": {"b": 2, "a": 1}}
    assert write_call_identity(a) == write_call_identity(b)


def test_write_call_identity_separates_different_arguments():
    """The guard must not collapse genuinely different proposals — six ticket
    updates in one turn are six proposals, not one."""
    a = {"name": "create_ticket", "arguments": {"t": "x"}}
    b = {"name": "create_ticket", "arguments": {"t": "y"}}
    assert write_call_identity(a) != write_call_identity(b)


@pytest.mark.asyncio
async def test_reproposed_write_is_answered_but_not_parked_twice():
    """A model that ignores the do-not-resubmit instruction gets one park.

    Without this the operator sees N identical affordances, each of which
    executes the write again if approved.
    """
    call = {"id": "w1", "name": "create_ticket", "arguments": {"title": "x"}}
    again = {"id": "w2", "name": "create_ticket", "arguments": {"title": "x"}}
    engine = _make_engine([
        [{"type": "tool_calls", "calls": [dict(call)]},
         {"type": "done", "full_text": ""}],
        [{"type": "tool_calls", "calls": [dict(again)]},
         {"type": "done", "full_text": ""}],
        [{"type": "done", "full_text": "Waiting on you."}],
    ])
    tools = _make_tool_manager({})
    loop = ToolLoop(engine, tools)

    parked: List[Dict[str, Any]] = []

    def pending_lookup(wc):
        for p in parked:
            if write_call_identity(p["call"]) == write_call_identity(wc):
                return p["token"]
        return None

    history: List[Dict[str, Any]] = []
    events = []
    async for ev in loop.run(
        persona=_make_persona(execution_mode=ExecutionMode.CONFIRM),
        conversation_history=history, params=MagicMock(), tools=[],
        pending_lookup=pending_lookup,
    ):
        if isinstance(ev, WriteParkedEvent):
            parked.append({"call": ev.write_call, "token": ev.token})
        events.append(ev)

    parks = [e for e in events if isinstance(e, WriteParkedEvent)]
    assert len(parks) == 1, "the re-proposal must not create a second park"

    # The duplicate call is still ANSWERED — an assistant tool_calls block with
    # no matching result is rejected outright by Anthropic and Gemini.
    dup = [m for m in history
           if m.get("role") == "tool" and m.get("tool_call_id") == "w2"]
    assert len(dup) == 1
    content = json.loads(dup[0]["content"])
    assert content["status"] == PARK_STATUS_DUPLICATE
    # It points at the live proposal, so the model can reason about which one.
    assert content["token"] == parks[0].token

    tools.execute_tool.assert_not_called()


@pytest.mark.asyncio
async def test_distinct_writes_in_one_iteration_all_park():
    """The guard is about repetition, not volume: several different writes
    proposed together are several proposals."""
    engine = _make_engine([
        [{"type": "tool_calls", "calls": [
            {"id": "w1", "name": "create_ticket", "arguments": {"t": "a"}},
            {"id": "w2", "name": "create_ticket", "arguments": {"t": "b"}},
            {"id": "w3", "name": "create_ticket", "arguments": {"t": "c"}},
        ]},
         {"type": "done", "full_text": ""}],
        [{"type": "done", "full_text": "Three for review."}],
    ])
    tools = _make_tool_manager({})
    loop = ToolLoop(engine, tools)

    parked: List[Dict[str, Any]] = []

    def pending_lookup(wc):
        for p in parked:
            if write_call_identity(p["call"]) == write_call_identity(wc):
                return p["token"]
        return None

    events = []
    async for ev in loop.run(
        persona=_make_persona(execution_mode=ExecutionMode.CONFIRM),
        conversation_history=[], params=MagicMock(), tools=[],
        pending_lookup=pending_lookup,
    ):
        if isinstance(ev, WriteParkedEvent):
            parked.append({"call": ev.write_call, "token": ev.token})
        events.append(ev)

    parks = [e for e in events if isinstance(e, WriteParkedEvent)]
    assert len(parks) == 3
    assert len({p.token for p in parks}) == 3


@pytest.mark.asyncio
async def test_no_pending_lookup_parks_everything():
    """`pending_lookup` is optional — callers that do not supply it (tests,
    any future embedder) must keep the pre-guard behaviour."""
    engine = _make_engine([
        [{"type": "tool_calls", "calls": [
            {"id": "w1", "name": "create_ticket", "arguments": {"t": "a"}}]},
         {"type": "done", "full_text": ""}],
        [{"type": "tool_calls", "calls": [
            {"id": "w2", "name": "create_ticket", "arguments": {"t": "a"}}]},
         {"type": "done", "full_text": ""}],
        [{"type": "done", "full_text": "ok"}],
    ])
    tools = _make_tool_manager({})
    loop = ToolLoop(engine, tools)

    events = await _drain(loop.run(
        persona=_make_persona(execution_mode=ExecutionMode.CONFIRM),
        conversation_history=[], params=MagicMock(), tools=[],
    ))
    assert len([e for e in events if isinstance(e, WriteParkedEvent)]) == 2


def test_write_call_identity_never_matches_when_arguments_are_unserializable():
    """The fallback identity has to be genuinely unique.

    It was `repr(object())`, which reads like "a value nothing can equal" but
    is not: CPython reuses the address of the object it just freed, so two
    evaluations return the same string. The guard's fail-open branch therefore
    matched ALWAYS — two different unserializable writes collapsed into one,
    the second was suppressed as a duplicate, and no park (and no operator
    affordance) was ever created for it."""
    a = {"name": "create_ticket", "arguments": {("tuple", "key"): 1}}
    b = {"name": "create_ticket", "arguments": {("other", "key"): 2}}

    ia, ib = write_call_identity(a), write_call_identity(b)

    assert ia != ib
    # Same call twice must also not match itself: the arguments are
    # incomparable, so "already pending" is unknowable and parking is the only
    # fail-closed answer.
    assert write_call_identity(a) != write_call_identity(a)
    assert ia[0] == "create_ticket"


def test_write_call_identity_fallback_is_unique_in_bulk():
    """Uniqueness has to hold every time, not usually.

    `repr(object())` collides whenever the allocator hands back the address it
    just freed — which it does often enough that a single-pair assertion is
    flaky in both directions. Sampling makes the defect deterministic: 200
    fallback identities from an address-derived value always contain
    duplicates, from a uuid never do."""
    call = {"name": "create_ticket", "arguments": {("k",): 1}}
    idents = [write_call_identity(call)[1] for _ in range(200)]
    assert len(set(idents)) == 200


# --- DP-335: the iteration-cap message ---------------------------------------

def _call_msg(call_id, name, args):
    return {"role": "assistant", "tool_calls": [
        {"id": call_id, "name": name, "arguments": args},
    ]}


def _result_msg(call_id, name, payload):
    return {
        "role": "tool", "tool_call_id": call_id, "name": name,
        "content": json.dumps(payload),
    }


def _cap_history():
    """The shape of the prod turn that motivated DP-335: reads, a verbatim
    repeat, a park, a failure, and one call the cap cut off unanswered."""
    return [
        _call_msg("c0", "pve_status", {}),
        _result_msg("c0", "pve_status", {"result": {"status": "ok"}}),
        _call_msg("c1", "hf_search", {"query": "qwen 27b"}),
        _result_msg("c1", "hf_search", {"result": {"models": []}}),
        _call_msg("c2", "pve_status", {}),
        _result_msg("c2", "pve_status", {"result": {"status": "ok"}}),
        _call_msg("c3", "install_model", {"repo": "unsloth/X"}),
        _result_msg("c3", "install_model",
                    {"status": PARK_STATUS_AWAITING, "token": "t"}),
        _call_msg("c4", "gpu_status", {}),
        _result_msg("c4", "gpu_status", {"error": "ssh timeout after 30s"}),
        _call_msg("c5", "list_models", {}),
    ]


def test_max_iteration_text_lists_every_call_with_its_outcome():
    """The whole point of DP-335: the user can see what the budget bought.

    Every tool returned `ok` in the turn this came from, so the canned "stuck
    in a loop" sentence described a malfunction that had not happened, and the
    one fact that mattered — which calls ate the budget — was only recoverable
    by reading the sealed tool_context out of the database."""
    text = render_max_iteration_text(_cap_history(), 0, 10)

    assert "`pve_status`" in text
    assert '`hf_search` {"query": "qwen 27b"}' in text
    assert '`install_model` {"repo": "unsloth/X"}' in text
    # Outcomes are distinguished, not flattened to "ran".
    assert "waiting for your approval" in text
    assert "failed" in text and "ssh timeout after 30s" in text
    # The cap cut c5 off before its result landed; sealing synthesizes one
    # later, so at render time it is genuinely unanswered.
    assert "`list_models` — no result" in text
    assert "10 of my tool steps" in text


def test_max_iteration_text_marks_verbatim_repeats():
    """A repeated read is the common shape of this exit (4 of 10 iterations in
    the DP-335 turn) and is invisible in a flat list of names."""
    text = render_max_iteration_text(_cap_history(), 0, 10)

    assert "3. `pve_status` — ok (same call as #1)" in text
    # Only the repeat is marked — the first occurrence and the distinct calls
    # around it must not be.
    assert text.count("same call as") == 1


def test_max_iteration_text_respects_the_history_slice():
    """`start` is the turn boundary — earlier turns' calls are not this turn's
    spend, and listing them would misattribute the budget."""
    history = [
        _call_msg("old", "list_models", {}),
        _result_msg("old", "list_models", {"result": {"models": []}}),
    ] + _cap_history()

    text = render_max_iteration_text(history, 2, 10)

    assert text.count("`list_models`") == 1
    assert "1. `pve_status`" in text


def test_max_iteration_text_scrubs_arguments():
    """Arguments are model-authored and this string goes to a surface, so it
    crosses the same egress boundary tool results do (DP-225)."""
    from src.security.scrubber import get_scrubber, reset_scrubber

    reset_scrubber()
    get_scrubber().register("hunter2-supersecret", "TEST_KEY")
    try:
        history = [_call_msg("c0", "set_active_model",
                             {"token": "hunter2-supersecret"})]
        text = render_max_iteration_text(history, 0, 10)
    finally:
        reset_scrubber()

    assert "hunter2-supersecret" not in text


def test_max_iteration_text_falls_back_when_no_calls_recorded():
    """No tool calls in the slice means there is nothing to show; the message
    must still be a sentence, not an empty list."""
    text = render_max_iteration_text([], 0, 10)

    assert "10 tool steps" in text
    assert text.strip().endswith("?")
