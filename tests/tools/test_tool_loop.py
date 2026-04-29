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
    ToolLoop, _ApiPayloadEvent, _LoopFinishedEvent,
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
    assert "stuck in a loop" in finished.final_text


@pytest.mark.asyncio
async def test_confirm_mode_parks_write_calls():
    """CONFIRM-mode persona with a write tool: loop parks via
    pending_writes on the terminal event, no execution."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "w1", "name": "create_ticket", "arguments": {"title": "x"}}
            ]},
            {"type": "done", "full_text": ""},
        ],
    ])
    tools = _make_tool_manager({})
    loop = ToolLoop(engine, tools)

    events = await _drain(loop.run(
        persona=_make_persona(execution_mode=ExecutionMode.CONFIRM),
        conversation_history=[], params=MagicMock(), tools=[],
    ))

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.response_type == ResponseType.PENDING_CONFIRMATION
    assert finished.pending_writes is not None
    assert finished.pending_writes[0]["name"] == "create_ticket"
    # Write tool was NOT executed — manager should not have been called for it.
    tools.execute_tool.assert_not_called()
