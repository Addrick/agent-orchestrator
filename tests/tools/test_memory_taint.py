# tests/tools/test_memory_taint.py
"""Phase 5: Memory taint propagation integration tests.

Verifies the core invariant: when memory retrieval surfaces summaries with
the `untrusted` flag, `turn_tainted` is set on the LoopFinishedEvent and
taint provenance (taint_sources) includes "memory_recall".
"""

import json
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.generation_events import ResponseType
from src.persona import ExecutionMode
from src.tools.tool_loop import (
    ToolLoop, _LoopFinishedEvent,
)


def _make_persona(execution_mode=ExecutionMode.AUTONOMOUS):
    p = MagicMock()
    p.get_config_for_engine.return_value = {"model_name": "local"}
    p.get_prompt.return_value = "You are a test assistant."
    p.get_execution_mode.return_value = execution_mode
    return p


def _stream(events: List[Dict[str, Any]]):
    async def gen() -> AsyncIterator[Dict[str, Any]]:
        for ev in events:
            yield ev
    return gen()


def _make_engine(streams: List[List[Dict[str, Any]]]):
    engine = MagicMock()
    iterator = iter(streams)
    def stream_messages(*args, **kwargs):
        return _stream(next(iterator))
    engine.stream_messages.side_effect = stream_messages
    return engine


def _make_tool_manager(results: Dict[str, Any] | None = None):
    manager = MagicMock()
    async def execute(name, **kwargs):
        if results and name in results:
            return results[name]
        return {"result": "ok"}
    manager.execute_tool = AsyncMock(side_effect=execute)
    return manager


async def _drain(loop_run):
    out = []
    async for ev in loop_run:
        out.append(ev)
    return out


@pytest.mark.asyncio
async def test_initial_taint_sources_propagates_to_finished_event():
    """When initial_taint_sources=['memory_recall'] is passed in,
    the _LoopFinishedEvent includes it in the taint_sources (via audit_info)
    and sets turn_tainted=True when write tools are present."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "w1", "name": "create_ticket", "arguments": {"title": "test"}}
            ]},
            {"type": "done", "full_text": ""},
        ],
    ])
    tools = _make_tool_manager()
    loop = ToolLoop(engine, tools)

    events = await _drain(loop.run(
        persona=_make_persona(),
        conversation_history=[], params=MagicMock(), tools=[],
        turn_tainted=True,
        initial_taint_sources=["memory_recall"],
    ))

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.response_type == ResponseType.PENDING_CONFIRMATION
    assert finished.turn_tainted is True
    assert finished.audit_info is not None
    assert finished.audit_info["tainted"] is True
    assert "memory_recall" in finished.audit_info["taint_sources"]


@pytest.mark.asyncio
async def test_initial_taint_sources_empty_no_taint():
    """When no initial_taint_sources and no untrusted tools,
    turn_tainted stays False and no taint_sources are reported."""
    engine = _make_engine([
        [
            {"type": "text_delta", "text": "hello"},
            {"type": "done", "full_text": "hello"},
        ],
    ])
    tools = _make_tool_manager()
    loop = ToolLoop(engine, tools)

    events = await _drain(loop.run(
        persona=_make_persona(),
        conversation_history=[], params=MagicMock(), tools=[],
    ))

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.turn_tainted is False


@pytest.mark.asyncio
async def test_memory_taint_combines_with_tool_taint():
    """When initial_taint_sources includes 'memory_recall' AND a read tool
    also produces untrusted content, both sources appear in the audit."""
    engine = _make_engine([
        # Iteration 1: read tool (web_search) + write tool (create_ticket)
        [
            {"type": "tool_calls", "calls": [
                {"id": "r1", "name": "web_search", "arguments": {"query": "test"}},
                {"id": "w1", "name": "create_ticket", "arguments": {"title": "test"}},
            ]},
            {"type": "done", "full_text": ""},
        ],
    ])
    tools = _make_tool_manager({"web_search": {"result": []}})
    loop = ToolLoop(engine, tools)

    events = await _drain(loop.run(
        persona=_make_persona(),
        conversation_history=[], params=MagicMock(), tools=[],
        turn_tainted=True,
        initial_taint_sources=["memory_recall"],
    ))

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.audit_info is not None
    assert "memory_recall" in finished.audit_info["taint_sources"]
    assert "web_search" in finished.audit_info["taint_sources"]


@pytest.mark.asyncio
async def test_memory_taint_text_only_no_audit_surface():
    """When memory taint is set but the model only produces text (no writes),
    turn_tainted is true but no audit_info is generated (no write calls)."""
    engine = _make_engine([
        [
            {"type": "text_delta", "text": "Here is what I found"},
            {"type": "done", "full_text": "Here is what I found"},
        ],
    ])
    tools = _make_tool_manager()
    loop = ToolLoop(engine, tools)

    events = await _drain(loop.run(
        persona=_make_persona(),
        conversation_history=[], params=MagicMock(), tools=[],
        turn_tainted=True,
        initial_taint_sources=["memory_recall"],
    ))

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.turn_tainted is True
    assert finished.audit_info is None  # No write calls → no audit surface
    assert finished.response_type == ResponseType.LLM_GENERATION


@pytest.mark.asyncio
async def test_initial_taint_sources_defaults_empty():
    """ToolLoop.run() works without initial_taint_sources (backward compat)."""
    engine = _make_engine([
        [
            {"type": "text_delta", "text": "ok"},
            {"type": "done", "full_text": "ok"},
        ],
    ])
    tools = _make_tool_manager()
    loop = ToolLoop(engine, tools)

    # No initial_taint_sources kwarg at all
    events = await _drain(loop.run(
        persona=_make_persona(),
        conversation_history=[], params=MagicMock(), tools=[],
    ))

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.final_text == "ok"
