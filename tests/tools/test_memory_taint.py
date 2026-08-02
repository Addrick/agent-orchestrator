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
    ToolLoop, WriteParkedEvent, _LoopFinishedEvent,
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
    manager.enrich_audit_action = AsyncMock(return_value=None)
    return manager


async def _drain(loop_run):
    out = []
    async for ev in loop_run:
        out.append(ev)
    return out


@pytest.mark.asyncio
async def test_initial_taint_sources_propagates_to_park_event():
    """initial_taint_sources=['memory_recall'] reaches the gated write's
    audit_info, and turn_tainted stays set on the terminal event.

    audit_info moved from the terminal event to the per-write park event in
    DP-297: a turn can gate several writes and still end with text, so there
    is no longer one audit surface per turn.
    """
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "w1", "name": "create_ticket", "arguments": {"title": "test"}}
            ]},
            {"type": "done", "full_text": ""},
        ],
        [
            {"type": "done", "full_text": "queued"},
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

    park = next(e for e in events if isinstance(e, WriteParkedEvent))
    assert park.audit_info["tainted"] is True
    assert "memory_recall" in park.audit_info["taint_sources"]

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.turn_tainted is True


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
        [
            {"type": "done", "full_text": "queued"},
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

    park = next(e for e in events if isinstance(e, WriteParkedEvent))
    assert "memory_recall" in park.audit_info["taint_sources"]
    assert "web_search" in park.audit_info["taint_sources"]


@pytest.mark.asyncio
async def test_memory_taint_text_only_no_audit_surface():
    """When memory taint is set but the model only produces text (no writes),
    turn_tainted is true and no park — hence no audit surface — is emitted."""
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
    assert not [e for e in events if isinstance(e, WriteParkedEvent)]
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
