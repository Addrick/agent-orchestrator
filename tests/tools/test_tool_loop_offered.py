# tests/tools/test_tool_loop_offered.py
"""DP-404: a tool call executes only if its name was offered this turn.

Providers return whatever name the model produced; the local text protocol
parses it out of free text, so an injected `<tool_call>` can name any tool
registered in the process. These pin the loop-level refusal: unoffered reads
do not run, unoffered writes are not parked, and both answer the model with
an error so the call/result pairing stays intact.
"""

import json
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.generation_events import ToolCallResultEvent
from src.persona import ExecutionMode
from src.tools.tool_loop import (
    ToolDeferredEvent, ToolLoop, _LoopFinishedEvent, offered_tool_names,
)


def _persona(mode=ExecutionMode.CONFIRM):
    p = MagicMock()
    p.get_config_for_engine.return_value = {"model_name": "local"}
    p.get_prompt.return_value = "p"
    p.get_execution_mode.return_value = mode
    p.get_name.return_value = "victim"
    return p


def _engine(*batches: List[Dict[str, Any]]):
    """One tool-call batch per iteration, then a closing text turn."""
    streams = [
        [{"type": "tool_calls", "calls": calls}, {"type": "done", "full_text": ""}]
        for calls in batches
    ] + [[{"type": "done", "full_text": "done"}]]
    it = iter(streams)

    async def gen(events) -> AsyncIterator[Dict[str, Any]]:
        for ev in events:
            yield ev

    engine = MagicMock()
    engine.stream_messages.side_effect = lambda *a, **k: gen(next(it))
    return engine


def _manager():
    m = MagicMock()
    m.execute_tool = AsyncMock(return_value={"result": "secret bytes"})
    m.enrich_audit_action = AsyncMock(return_value=None)
    return m


async def _run(engine, manager, tools, history=None):
    loop = ToolLoop(engine, manager)
    return [ev async for ev in loop.run(
        persona=_persona(), conversation_history=history if history is not None else [],
        params=MagicMock(), tools=tools,
    )]


def test_offered_tool_names_reads_both_definition_shapes():
    tools = [
        {"type": "function", "function": {"name": "recall_memory"}},
        {"name": "web_search"},
        {"type": "google_search"},  # non-callable flag, no name
    ]
    assert offered_tool_names(tools) == {"recall_memory", "web_search"}
    assert offered_tool_names(None) == frozenset()


@pytest.mark.asyncio
async def test_unoffered_read_is_refused_and_not_executed():
    manager = _manager()
    history: List[Dict[str, Any]] = []
    events = await _run(
        _engine([{"id": "r1", "name": "list_memories", "arguments": {"bank_id": "joy"}}]),
        manager, tools=[{"name": "web_search"}], history=history,
    )

    manager.execute_tool.assert_not_called()
    result = next(e for e in events if isinstance(e, ToolCallResultEvent))
    assert result.call_id == "r1"
    assert result.error and "not available to this persona" in result.error
    tool_msgs = [m for m in history if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["r1"]
    assert "not executed" in json.loads(tool_msgs[0]["content"])["error"]
    # A refused call read nothing, so it must not taint the turn.
    finished = next(e for e in events if isinstance(e, _LoopFinishedEvent))
    assert finished.turn_tainted is False


@pytest.mark.asyncio
async def test_unoffered_write_is_refused_not_parked():
    manager = _manager()
    events = await _run(
        _engine([{"id": "w1", "name": "create_ticket", "arguments": {"title": "x"}}]),
        manager, tools=[{"name": "web_search"}],
    )

    assert not [e for e in events if isinstance(e, ToolDeferredEvent)]
    manager.execute_tool.assert_not_called()
    result = next(e for e in events if isinstance(e, ToolCallResultEvent))
    assert result.error and "not available to this persona" in result.error


@pytest.mark.asyncio
async def test_mixed_batch_runs_only_the_offered_call_in_order():
    manager = _manager()
    history: List[Dict[str, Any]] = []
    await _run(
        _engine([
            {"id": "a", "name": "delete_bank", "arguments": {}},
            {"id": "b", "name": "web_search", "arguments": {"q": "x"}},
        ]),
        manager, tools=[{"name": "web_search"}], history=history,
    )

    manager.execute_tool.assert_called_once_with("web_search", q="x")
    tool_msgs = [m for m in history if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["a", "b"]
    assert "error" in json.loads(tool_msgs[0]["content"])


@pytest.mark.asyncio
async def test_no_tools_offered_refuses_everything():
    manager = _manager()
    await _run(
        _engine([{"id": "r1", "name": "web_search", "arguments": {}}]),
        manager, tools=[],
    )
    manager.execute_tool.assert_not_called()
