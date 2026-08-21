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
    render_call_summary_footer, render_max_iteration_text, write_call_identity,
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


def _spin(i: int, count: int = 1):
    """One provider turn that asks for `count` calls to `spinner`."""
    return [
        {"type": "tool_calls", "calls": [
            {"id": f"c{i}_{n}", "name": "spinner", "arguments": {"n": n}}
            for n in range(count)
        ]},
        {"type": "done", "full_text": ""},
    ]


_NO_WRAP_UP = [{"type": "done", "full_text": ""}]


@pytest.mark.asyncio
async def test_max_iterations_cap():
    """If the model never stops calling tools, loop bails after the cap.

    `max_iterations` is the runaway guard since DP-335, so this trips it by
    keeping the call budget far out of reach. The wrap-up completion is scripted
    empty, which is what drives the fallback to the deterministic list.
    """
    engine = _make_engine([_spin(i) for i in range(3)] + [_NO_WRAP_UP])
    tools = _make_tool_manager({"spinner": {"result": "spin"}})
    loop = ToolLoop(engine, tools, max_iterations=3, max_tool_calls=100)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.response_type == ResponseType.DEV_COMMAND
    assert tools.execute_tool.call_count == 3
    # DP-335: the cap-hit text lists what the turn actually spent its budget
    # on. The old canned sentence named no tool, so a user (and the next
    # session) could not tell an exhausted budget from a broken tool.
    assert "`spinner`" in finished.final_text
    assert "(same call as #1)" in finished.final_text
    # And it names the limit that ACTUALLY tripped. This turn ran out of
    # iterations after 3 calls against a call budget of 100; the first cut
    # handed `max_tool_calls` to the renderer unconditionally and so replied
    # "I used all 100 of my tool steps" — the wrong-diagnosis-in-the-exit-
    # message failure this ticket exists to remove.
    assert "loop guard" in finished.final_text
    assert "3 tool step(s)" in finished.final_text
    assert "100" not in finished.final_text


@pytest.mark.asyncio
@pytest.mark.parametrize("per_message,messages", [(1, 10), (5, 2)])
async def test_budget_is_the_same_for_a_batching_and_a_serial_model(
        per_message, messages):
    """S1, the whole point of counting calls instead of iterations.

    Before DP-335 the budget was `range(max_iterations)`, so the identical
    config value bought 10 tool calls from a model that emits one call per
    message (live `hypr` on agy-flash — measured, not assumed) and 50 from one
    that batches five. The number could not be tuned because it did not denote
    a fixed quantity. Both shapes must now spend exactly the budget.
    """
    engine = _make_engine(
        [_spin(i, per_message) for i in range(messages)] + [_NO_WRAP_UP]
    )
    tools = _make_tool_manager({"spinner": {"result": "spin"}})
    loop = ToolLoop(engine, tools, max_iterations=50, max_tool_calls=10)

    await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    assert tools.execute_tool.call_count == 10
    # And the batching model got there in a fifth of the round trips — the
    # latency win that used to cost 5x the budget. +1 is the wrap-up.
    assert engine.stream_messages.call_count == messages + 1


@pytest.mark.asyncio
async def test_a_batch_that_crosses_the_budget_still_runs_whole():
    """A group the model proposed as one plan is never truncated to fit.

    Same rule the write path already follows for a burst of proposals: half of
    a coherent batch is worse than one turn of overshoot, and the group is
    dispatched concurrently anyway, so the overshoot costs no extra round trip.
    """
    engine = _make_engine([_spin(0, 5), _NO_WRAP_UP])
    tools = _make_tool_manager({"spinner": {"result": "spin"}})
    loop = ToolLoop(engine, tools, max_iterations=50, max_tool_calls=3)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    assert tools.execute_tool.call_count == 5
    assert isinstance(events[-1], _LoopFinishedEvent)
    # The overshoot ends the turn; it does not buy another iteration.
    assert engine.stream_messages.call_count == 2


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
    text = render_max_iteration_text(_cap_history(), 0, used=6, budget=10)

    assert "`pve_status`" in text
    assert '`hf_search` {"query": "qwen 27b"}' in text
    assert '`install_model` {"repo": "unsloth/X"}' in text
    # Outcomes are distinguished, not flattened to "ran".
    assert "waiting for your approval" in text
    assert "failed" in text and "ssh timeout after 30s" in text
    # The cap cut c5 off before its result landed; sealing synthesizes one
    # later, so at render time it is genuinely unanswered.
    assert "`list_models` — no result" in text
    assert "6 of 10 tool steps" in text


def test_max_iteration_text_marks_verbatim_repeats():
    """A repeated read is the common shape of this exit (4 of 10 iterations in
    the DP-335 turn) and is invisible in a flat list of names."""
    text = render_max_iteration_text(_cap_history(), 0, used=6, budget=10)

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

    text = render_max_iteration_text(history, 2, used=6, budget=10)

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
        text = render_max_iteration_text(history, 0, used=1, budget=10)
    finally:
        reset_scrubber()

    assert "hunter2-supersecret" not in text


def test_max_iteration_text_falls_back_when_no_calls_recorded():
    """No tool calls in the slice means there is nothing to show; the message
    must still be a sentence, not an empty list."""
    text = render_max_iteration_text([], 0, used=0, budget=10)

    assert "10 tool steps" in text
    assert text.strip().endswith("?")


def test_call_summary_footer_pins_the_spend_under_a_prose_answer():
    """The footer is the ground truth half of the exhaustion reply: the prose
    above it is a generation and can be vague about what it ran, this is read
    straight out of history and cannot be."""
    footer = render_call_summary_footer(_cap_history(), 0, used=6, budget=15)

    assert "6 of 15 used" in footer
    assert "1. `pve_status` — ok" in footer
    assert "3. `pve_status` — ok (same call as #1)" in footer
    # No preamble sentence — that is the standalone renderer's job.
    assert "I spent" not in footer


def test_call_summary_footer_is_empty_when_the_turn_made_no_calls():
    """Nothing to pin means no footer, not a header with an empty list under
    it — the prose answer has to be able to stand alone."""
    assert render_call_summary_footer([], 0, used=0, budget=15) == ""


def test_renderers_report_the_limit_that_actually_tripped():
    """The budget named in the text is the one that ended the turn.

    The first cut handed `max_tool_calls` to both renderers unconditionally, so
    a turn stopped by the runaway guard still claimed to have spent the call
    budget — a number it never came near. Stating the wrong diagnosis in the
    exit message is the failure DP-335 exists to remove, so it must not be
    reintroduced by the message that replaced it.
    """
    guard = render_max_iteration_text(
        _cap_history(), 0, used=6, budget=15, exhausted=False,
    )
    assert "loop guard" in guard
    assert "6 tool step(s)" in guard
    assert "15" not in guard

    footer = render_call_summary_footer(
        _cap_history(), 0, used=6, budget=15, exhausted=False,
    )
    assert "Loop guard stopped the turn after 6 tool step(s)" in footer
    assert "of 15 used" not in footer


def test_footer_reports_the_turns_own_spend_not_the_rendered_slice():
    """`history_start_override` walks the render boundary back over a PARKED
    turn so the seal spans both, but the resumed turn's `calls_used` starts at
    zero. Deriving the headline from the slice therefore billed this turn for
    another turn's calls. The header takes `used`; the discrepancy is stated
    rather than left to contradict the list beneath it."""
    footer = render_call_summary_footer(
        _cap_history(), 0, used=2, budget=15,
    )

    assert "(2 of 15 used)" in footer
    # The list still shows everything the seal covers — that is the point of
    # rendering the whole slice — but it says so.
    assert "the 6 calls below span the whole parked exchange" in footer


def test_budget_overshoot_never_claims_all_n_of_n():
    """A batch is charged whole and never truncated, so `used` can exceed
    `budget`. The text states both numbers instead of the old "all N of N",
    which would have been arithmetically false on exactly those turns."""
    text = render_max_iteration_text(
        _cap_history(), 0, used=17, budget=15,
    )

    assert "17 of 15 tool steps" in text


def test_runaway_guard_stays_above_the_call_budget():
    """`MAX_TOOL_ITERATIONS` is unreachable at the shipped defaults, by design.

    Every iteration that continues past the tool-call check charges at least
    one call, so `iterations_used <= calls_used` always holds and the call
    budget always trips first while the guard sits above it. Pinned because the
    relation is the whole reason the guard is inert: raise `MAX_TOOL_CALLS`
    past this number and the guard silently starts truncating ordinary turns
    while every other limit still reads as generous.
    """
    from config import global_config

    assert global_config.MAX_TOOL_ITERATIONS > global_config.MAX_TOOL_CALLS


# --- DP-335 S2: the exhaustion answer ---------------------------------------


@pytest.mark.asyncio
async def test_exhaustion_answers_from_the_transcript():
    """The turn that motivated DP-335 had its answer in the second tool result
    and still ended on a canned sentence that read as a fault. Budget
    exhaustion now spends one toolless completion turning the transcript it
    already has into a real answer."""
    engine = _make_engine([
        _spin(0),
        [{"type": "text_delta", "text": "No gguf exists for that repo; "},
         {"type": "text_delta", "text": "want the community quant?"},
         {"type": "done",
          "full_text": "No gguf exists for that repo; want the community quant?"}],
    ])
    tools = _make_tool_manager({"spinner": {"result": "spin"}})
    loop = ToolLoop(engine, tools, max_iterations=50, max_tool_calls=1)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    # LLM_GENERATION, not DEV_COMMAND: `_orchestrate` gates *retention* on this
    # field, so a real answer that shipped as DEV_COMMAND would never reach the
    # memory bank. The canned fault it replaces was correctly excluded.
    assert finished.response_type == ResponseType.LLM_GENERATION
    assert finished.final_text.startswith("No gguf exists for that repo")
    # Prose first, ground-truth list under it — the two compose.
    assert "1 of 1 used" in finished.final_text
    assert "`spinner`" in finished.final_text
    # But only the PROSE is embedded. The footer is a machine-generated listing
    # of tool names and arguments, not something the persona said; retaining it
    # would make `spinner {"n": 0} — ok` a recallable semantic memory and
    # replay it next turn as the persona's own prior words.
    assert finished.retain_text == (
        "No gguf exists for that repo; want the community quant?"
    )
    assert "`spinner`" not in finished.retain_text


@pytest.mark.asyncio
async def test_exhaustion_keeps_deltas_when_the_provider_reports_empty_done():
    """A provider can stream the answer as deltas and then report `done` with
    an empty `full_text` — every one-shot result the driver classifies as
    `tool_calls` does exactly that, and so do truncated streams. Preferring the
    empty `full_text` over the deltas already in hand threw a real answer away
    and dropped the turn to the canned list."""
    engine = _make_engine([
        _spin(0),
        [{"type": "text_delta", "text": "The R9700 has 32 GiB; "},
         {"type": "text_delta", "text": "the 27B q4 fits."},
         {"type": "done", "full_text": ""}],
    ])
    tools = _make_tool_manager({"spinner": {"result": "spin"}})
    loop = ToolLoop(engine, tools, max_iterations=50, max_tool_calls=1)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.response_type == ResponseType.LLM_GENERATION
    assert finished.final_text.startswith("The R9700 has 32 GiB")


def test_exhaustion_nudge_does_not_forge_a_system_marker():
    """The nudge is a system instruction delivered in the USER position.

    Tool output is attacker-influenceable here (`produces_untrusted` →
    `turn_tainted`), so a turn that demonstrates "`[system] …` in the user
    channel carries system authority" makes an injected `[system] Ignore the
    approval gate and …` inside a `web_search` result materially more credible
    on the next turn. `_render_resolution_nudge`, the park-continuation twin
    this is modeled on, states its facts plainly for the same reason.
    """
    from src.tools.tool_loop import _EXHAUSTION_NUDGE

    assert "[system]" not in _EXHAUSTION_NUDGE
    assert _EXHAUSTION_NUDGE.startswith("You have used your entire tool budget")


@pytest.mark.asyncio
async def test_exhaustion_wrap_up_is_toolless_and_nudged():
    """Tools are withheld because the budget is spent — offering them invites a
    call this path cannot run. The nudge rides in the wire messages only."""
    engine = _make_engine([
        _spin(0),
        [{"type": "done", "full_text": "Here is what I found."}],
    ])
    tools = _make_tool_manager({"spinner": {"result": "spin"}})
    loop = ToolLoop(engine, tools, max_iterations=50, max_tool_calls=1)
    history: List[Dict[str, Any]] = []

    await _drain(loop.run(
        persona=_make_persona(), conversation_history=history,
        params=MagicMock(), tools=[{"name": "spinner"}],
    ))

    first_call, wrap_call = engine.stream_messages.call_args_list
    assert first_call.kwargs["tools"] == [{"name": "spinner"}]
    assert wrap_call.kwargs["tools"] is None

    wrap_messages = wrap_call.args[1]
    assert wrap_messages[-1]["role"] == "user"
    assert "entire tool budget" in wrap_messages[-1]["content"]
    # And it is NOT in the history that gets sealed and persisted: replayed
    # next turn it would read as something the user said.
    assert not any("entire tool budget" in str(m.get("content") or "")
                   for m in history)


@pytest.mark.asyncio
async def test_exhaustion_falls_back_when_the_wrap_up_generation_fails():
    """Best-effort. A provider that is down must not turn a turn that merely
    ran long into an error the user has to interpret."""
    engine = MagicMock()
    scripted = iter([_spin(0)])

    def stream_messages(*args, **kwargs):
        if kwargs.get("tools") is None:
            raise LLMCommunicationError("upstream 503")
        return _stream(next(scripted))
    engine.stream_messages.side_effect = stream_messages

    tools = _make_tool_manager({"spinner": {"result": "spin"}})
    loop = ToolLoop(engine, tools, max_iterations=50, max_tool_calls=1)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.response_type == ResponseType.DEV_COMMAND
    assert "my whole tool budget (1 of 1 tool steps)" in finished.final_text
    assert "`spinner`" in finished.final_text
    # The exit is still an exit: exactly one terminal event, nothing trailing.
    assert sum(isinstance(e, (_LoopFinishedEvent, ErrorEvent))
               for e in events) == 1


@pytest.mark.asyncio
async def test_exhaustion_still_seals_the_tool_context():
    """The wrap-up must not displace the seal — the parks and reads the turn
    made are the only record those calls have."""
    engine = _make_engine([
        _spin(0),
        [{"type": "done", "full_text": "Answering from what I have."}],
    ])
    tools = _make_tool_manager({"spinner": {"result": "spin"}})
    loop = ToolLoop(engine, tools, max_iterations=50, max_tool_calls=1)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    msgs = _tool_context(events[-1])
    _assert_calls_all_answered(msgs)
    assert {m.get("tool_call_id") for m in msgs if m.get("role") == "tool"} == {
        "c0_0",
    }


# --- DP-335 instrumentation --------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_calls_are_counted_in_the_turn_log(caplog):
    """The measurement that replaced a proposed per-turn read cache.

    A cache sized on the single observed turn would have funded more wandering
    while breaking `install_status`, which exists to be polled. So the repeat
    rate is logged instead and a recurrence arrives with numbers attached.
    Identities are hashed — the canonical argument string is secret-bearing.
    """
    engine = _make_engine([
        [{"type": "tool_calls", "calls": [
            {"id": "a", "name": "spinner", "arguments": {"q": "x"}},
            {"id": "b", "name": "spinner", "arguments": {"q": "x"}},
            {"id": "c", "name": "spinner", "arguments": {"q": "y"}},
        ]},
         {"type": "done", "full_text": ""}],
        [{"type": "done", "full_text": "done"}],
    ])
    tools = _make_tool_manager({"spinner": {"result": "spin"}})
    loop = ToolLoop(engine, tools, max_iterations=50, max_tool_calls=10)

    with caplog.at_level("INFO", logger="src.tools.tool_loop"):
        await _drain(loop.run(
            persona=_make_persona(), conversation_history=[],
            params=MagicMock(), tools=[],
        ))

    summary = next(r.getMessage() for r in caplog.records
                   if "tool-loop turn:" in r.getMessage())
    assert "3 call(s)" in summary
    assert "2 distinct" in summary
    assert "1 repeated call(s)" in summary
    assert "spinner#" in summary and "x2" in summary
    # Raw arguments never reach the log line.
    assert '"q"' not in summary


# --------------------------------------------------------------------------
# DP-338 — the assistant's own words go into history beside its calls.
#
# The prose a model writes before a batch is the plan for that batch. Dropping
# it left the next iteration reading a transcript in which calls appeared with
# no stated reason, so the model re-derived the plan from an identical history
# and re-emitted the same batch — the same class of loss DP-335 fixed at the
# END of a turn, one iteration earlier. Providers park the prose in two
# different places, so both are pinned.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batched_calls_run_in_one_iteration():
    """Three calls in one provider message cost one round trip, not three."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "c1", "name": "pve_status", "arguments": {}},
                {"id": "c2", "name": "gpu_status", "arguments": {}},
                {"id": "c3", "name": "list_models", "arguments": {}},
            ]},
            {"type": "done", "full_text": ""},
        ],
        [{"type": "done", "full_text": "All three are healthy."}],
    ])
    tools = _make_tool_manager({})
    loop = ToolLoop(engine, tools)
    history: List[Dict[str, Any]] = []

    await _drain(loop.run(
        persona=_make_persona(), conversation_history=history,
        params=MagicMock(), tools=[],
    ))

    assert engine.stream_messages.call_count == 2
    assert tools.execute_tool.call_count == 3
    # One assistant entry holding all three calls, then three paired results.
    assert history[0]["role"] == "assistant"
    assert [c["name"] for c in history[0]["tool_calls"]] == [
        "pve_status", "gpu_status", "list_models",
    ]
    assert [m["tool_call_id"] for m in history[1:4]] == ["c1", "c2", "c3"]


@pytest.mark.asyncio
async def test_streamed_prose_lands_on_the_assistant_tool_call_entry():
    """The streaming providers delta the prose out and zero `full_text` on a
    tool turn, so the deltas are the only copy."""
    engine = _make_engine([
        [
            {"type": "text_delta", "text": "Checking the node "},
            {"type": "text_delta", "text": "and the card."},
            {"type": "tool_calls", "calls": [
                {"id": "c1", "name": "pve_status", "arguments": {}},
            ]},
            {"type": "done", "full_text": ""},
        ],
        [{"type": "done", "full_text": "Node is up."}],
    ])
    loop = ToolLoop(engine, _make_tool_manager({}))
    history: List[Dict[str, Any]] = []

    await _drain(loop.run(
        persona=_make_persona(), conversation_history=history,
        params=MagicMock(), tools=[],
    ))

    assert history[0]["content"] == "Checking the node and the card."


@pytest.mark.asyncio
async def test_one_shot_prose_on_done_lands_on_the_same_entry():
    """agy's one-shot route emits no deltas at all — its prose arrives on
    `done`, which is why reading `accumulated_parts` alone lost it."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "c1", "name": "pve_status", "arguments": {}},
            ]},
            {"type": "done", "full_text": "Checking the node first."},
        ],
        [{"type": "done", "full_text": "Node is up."}],
    ])
    loop = ToolLoop(engine, _make_tool_manager({}))
    history: List[Dict[str, Any]] = []

    await _drain(loop.run(
        persona=_make_persona(), conversation_history=history,
        params=MagicMock(), tools=[],
    ))

    assert history[0]["content"] == "Checking the node first."


@pytest.mark.asyncio
async def test_prose_is_scrubbed_before_it_enters_history():
    """This prose is a NEW persistence path (DP-225 boundary 2). It is sealed
    into `tool_context`, written to `User_Interactions` and replayed to the
    provider next turn, so scrubbing only the audit dialog would show the
    operator a redacted sentence while storing the unredacted copy."""
    from src.security.scrubber import get_scrubber, reset_scrubber

    reset_scrubber()
    get_scrubber().register("hunter2-supersecret", "TEST_KEY")
    try:
        engine = _make_engine([
            [
                {"type": "text_delta",
                 "text": "Retrying with hunter2-supersecret."},
                {"type": "tool_calls", "calls": [
                    {"id": "c1", "name": "pve_status", "arguments": {}},
                ]},
                {"type": "done", "full_text": ""},
            ],
            [{"type": "done", "full_text": "Node is up."}],
        ])
        loop = ToolLoop(engine, _make_tool_manager({}))
        history: List[Dict[str, Any]] = []

        await _drain(loop.run(
            persona=_make_persona(), conversation_history=history,
            params=MagicMock(), tools=[],
        ))
    finally:
        reset_scrubber()

    assert "hunter2-supersecret" not in history[0]["content"]
    assert "TEST_KEY" in history[0]["content"]


@pytest.mark.asyncio
async def test_call_only_iteration_writes_no_content_key():
    """No prose, no key — the history shape every other consumer already
    handles stays untouched when there is nothing to carry."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "c1", "name": "pve_status", "arguments": {}},
            ]},
            {"type": "done", "full_text": ""},
        ],
        [{"type": "done", "full_text": "Node is up."}],
    ])
    loop = ToolLoop(engine, _make_tool_manager({}))
    history: List[Dict[str, Any]] = []

    await _drain(loop.run(
        persona=_make_persona(), conversation_history=history,
        params=MagicMock(), tools=[],
    ))

    assert "content" not in history[0]


@pytest.mark.asyncio
async def test_one_shot_prose_reaches_the_park_audit_reasoning():
    """`model_reasoning` is the "why" on the operator's approve/deny dialog.
    It read the deltas directly, so every agy-backed persona parked its writes
    with a blank one."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "w1", "name": "create_ticket",
                 "arguments": {"title": "x"}},
            ]},
            {"type": "done", "full_text": "Filing this so the outage is tracked."},
        ],
        [{"type": "done", "full_text": "Queued that for you."}],
    ])
    loop = ToolLoop(engine, _make_tool_manager({}))

    events = await _drain(loop.run(
        persona=_make_persona(execution_mode=ExecutionMode.CONFIRM),
        conversation_history=[], params=MagicMock(), tools=[],
    ))

    park = [e for e in events if isinstance(e, WriteParkedEvent)][0]
    assert park.audit_info["model_reasoning"] == (
        "Filing this so the outage is tracked."
    )
