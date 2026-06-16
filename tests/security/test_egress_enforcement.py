# tests/security/test_egress_enforcement.py
"""DP-225 Sprint 2 — egress enforcement at the LLM boundaries.

The scrubber is wired at three boundaries in the tool loop / turn persistence.
These tests register a known secret with the process-global scrubber, then drive
each boundary and assert the secret is redacted to ``[REDACTED:TEST_KEY]`` in
every place the model / audit log / inspector can read it back.
"""

import json
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.generation_events import (
    ResponseType, ToolCallResultEvent,
)
from src.persona import ExecutionMode
from src.security.scrubber import get_scrubber, reset_scrubber
from src.tools.tool_loop import ToolLoop, _LoopFinishedEvent
from src.turn_persistence import TurnPersistence

SECRET = "supersecretvalue123"
REDACTED = "[REDACTED:TEST_KEY]"


@pytest.fixture(autouse=True)
def _scrubber_with_secret():
    """Fresh scrubber holding one registered secret for every test."""
    reset_scrubber()
    get_scrubber().register(SECRET, "TEST_KEY")
    yield
    reset_scrubber()


# ---- shared harness (mirrors tests/tools/test_tool_loop.py) ---------------

def _make_persona(execution_mode=ExecutionMode.AUTONOMOUS):
    p = MagicMock()
    p.get_config_for_engine.return_value = {"model_name": "local"}
    p.get_prompt.return_value = "You are a test assistant."
    p.get_execution_mode.return_value = execution_mode
    p.get_self_edit.return_value = False
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


def _make_tool_manager(results: Dict[str, Any]):
    manager = MagicMock()

    async def execute(name, **kwargs):
        return results.get(name, {"result": "ok"})
    manager.execute_tool = AsyncMock(side_effect=execute)
    manager.enrich_audit_action = AsyncMock(return_value=None)
    return manager


async def _drain(loop_run):
    out = []
    async for ev in loop_run:
        out.append(ev)
    return out


# ---- Boundary 1: tool results -> history + ToolCallResultEvent ------------

@pytest.mark.asyncio
async def test_boundary1_tool_result_scrubbed_in_history_and_event():
    """A read tool returns a secret in its result; both the appended
    conversation_history tool message and the emitted ToolCallResultEvent
    must be redacted (and identical), so the model and UI never see it."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "r1", "name": "search_tool", "arguments": {"q": "x"}}
            ]},
            {"type": "done", "full_text": ""},
        ],
        [
            {"type": "text_delta", "text": "done"},
            {"type": "done", "full_text": "done"},
        ],
    ])
    tools = _make_tool_manager(
        {"search_tool": {"result": f"the key is {SECRET} ok"}}
    )
    loop = ToolLoop(engine, tools, max_iterations=5)
    history: List[Dict[str, Any]] = []

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=history,
        params=MagicMock(), tools=[],
    ))

    result_events = [e for e in events if isinstance(e, ToolCallResultEvent)]
    assert len(result_events) == 1
    result_str = result_events[0].result
    assert SECRET not in result_str
    assert REDACTED in result_str

    tool_msgs = [m for m in history if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    content = tool_msgs[0]["content"]
    assert SECRET not in content
    assert REDACTED in content

    # History content and emitted event must match (scrubbed once, shared).
    assert content == result_str


@pytest.mark.asyncio
async def test_boundary1_tool_error_scrubbed_in_event():
    """A tool whose error message embeds a secret: ToolCallResultEvent.error is
    surfaced raw in the portal SSE / ToolCard, so it must be redacted too — the
    sibling result field being scrubbed is not sufficient."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "r1", "name": "search_tool", "arguments": {"q": "x"}}
            ]},
            {"type": "done", "full_text": ""},
        ],
        [
            {"type": "text_delta", "text": "done"},
            {"type": "done", "full_text": "done"},
        ],
    ])
    tools = _make_tool_manager(
        {"search_tool": {"error": f"auth failed: {SECRET}"}}
    )
    loop = ToolLoop(engine, tools, max_iterations=5)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    result_events = [e for e in events if isinstance(e, ToolCallResultEvent)]
    assert len(result_events) == 1
    err = result_events[0].error
    assert err is not None
    assert SECRET not in err
    assert REDACTED in err


# ---- Boundary 2: audit args -> Agent_Actions + confirmation text ----------

@pytest.mark.asyncio
async def test_boundary2_model_reasoning_scrubbed_in_audit():
    """model_reasoning (joined model text) is persisted in audit_info and shown
    in the UI; a secret the model echoes there must be redacted too."""
    engine = _make_engine([
        [
            {"type": "text_delta", "text": f"thinking about {SECRET} now"},
            {"type": "tool_calls", "calls": [
                {"id": "w1", "name": "create_ticket",
                 "arguments": {"title": "t"}}
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
    assert finished.audit_info is not None
    reasoning = finished.audit_info["model_reasoning"]
    assert reasoning is not None
    assert SECRET not in reasoning
    assert REDACTED in reasoning


@pytest.mark.asyncio
async def test_boundary2_write_args_scrubbed_in_audit_and_confirmation():
    """A CONFIRM-mode write call whose arguments embed a secret: the parked
    audit_info actions and the human confirmation final_text must redact it."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "w1", "name": "create_ticket",
                 "arguments": {"title": "t", "api_key": SECRET}}
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
    assert finished.audit_info is not None

    args = finished.audit_info["actions"][0]["arguments"]
    assert args["api_key"] == REDACTED
    assert SECRET not in json.dumps(args)

    # The human-readable confirmation renders the (scrubbed) args.
    assert SECRET not in finished.final_text
    assert REDACTED in finished.final_text


# ---- Boundary 3: cached api_payload -> /assemble inspector ----------------

def test_boundary3_cached_payload_scrubbed():
    """store_api_request must scrub the payload before it lands in
    last_api_requests / last_api_iterations (surfaced by the inspector)."""
    tp = TurnPersistence(memory_manager=MagicMock(), memory_backend=MagicMock())

    payload: Dict[str, Any] = {
        "model": "local",
        "messages": [{"role": "user", "content": f"token={SECRET}"}],
    }
    tp.store_api_request("user1", "personaA", payload, is_first_iteration=True)

    cached = tp.last_api_requests["user1"]["personaA"]
    assert cached is not None
    blob = json.dumps(cached)
    assert SECRET not in blob
    assert REDACTED in blob

    iters = tp.last_api_iterations["user1"]["personaA"]
    assert SECRET not in json.dumps(iters)
    assert REDACTED in json.dumps(iters)
