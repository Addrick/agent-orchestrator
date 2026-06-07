# tests/tools/test_tool_loop_edge_cases.py
"""DP-199 Batch 1 — CONFIRM-mode tool_loop edge cases.

Mirrors the patterns in test_tool_loop.py — scripted engine streams,
mock ToolManager, `_LoopFinishedEvent` / `PendingConfirmation` assertions.
No production code changes; missing features → `pytest.skip`.
"""

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.generation_events import (
    ErrorEvent, ResponseType, TokenEvent,
    ToolCallResultEvent, ToolCallStartEvent,
)
from src.persona import ExecutionMode
from src.tools.tool_loop import (
    ToolLoop, _ApiPayloadEvent, _LoopFinishedEvent,
)

# Reuse the helpers from the sibling test module via direct import.
from tests.tools.test_tool_loop import (
    _make_persona, _make_engine, _make_tool_manager, _drain,
)


# --- Loop-internal edge cases ---------------------------------------------

@pytest.mark.asyncio
async def test_tool_empty_result_serialization():
    """Tool returns None / empty dict / whitespace string — must still
    serialize cleanly and surface a ToolCallResultEvent (no crash)."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "c1", "name": "empty_a", "arguments": {}},
                {"id": "c2", "name": "empty_b", "arguments": {}},
                {"id": "c3", "name": "empty_c", "arguments": {}},
            ]},
            {"type": "done", "full_text": ""},
        ],
        [
            {"type": "text_delta", "text": "ok"},
            {"type": "done", "full_text": "ok"},
        ],
    ])
    # None, empty dict, whitespace string — all valid tool outputs to dump
    tools = _make_tool_manager({"empty_a": None, "empty_b": {}, "empty_c": "   "})
    loop = ToolLoop(engine, tools, max_iterations=5)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    results = [e for e in events if isinstance(e, ToolCallResultEvent)]
    assert len(results) == 3
    # Each result is a valid JSON string; no error field set for plain outputs
    for r in results:
        json.loads(r.result)  # must round-trip
        assert r.error is None
    assert isinstance(events[-1], _LoopFinishedEvent)
    assert events[-1].final_text == "ok"


@pytest.mark.asyncio
async def test_event_sequence_integrity_on_error():
    """When the provider raises mid-stream, _ApiPayloadEvent (if any) must
    precede the terminal ErrorEvent, and nothing should follow ErrorEvent."""
    from src.engine import LLMCommunicationError

    async def boom(*args, **kwargs):
        raise LLMCommunicationError("upstream 500", api_payload={"req": 1})
        yield  # pragma: no cover

    engine = MagicMock()
    engine.stream_messages.side_effect = lambda *a, **k: boom()
    tools = _make_tool_manager({})
    loop = ToolLoop(engine, tools)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    # Last event is ErrorEvent
    assert isinstance(events[-1], ErrorEvent)
    # No further events emitted after ErrorEvent
    error_idx = next(i for i, e in enumerate(events) if isinstance(e, ErrorEvent))
    assert error_idx == len(events) - 1
    # ApiPayload (if present) came before ErrorEvent
    payload_idxs = [i for i, e in enumerate(events) if isinstance(e, _ApiPayloadEvent)]
    assert all(i < error_idx for i in payload_idxs)


@pytest.mark.asyncio
async def test_single_terminal_event_invariant():
    """Exactly one terminal event (_LoopFinishedEvent OR ErrorEvent)
    per loop.run() call — never both, never duplicated."""
    engine = _make_engine([
        [
            {"type": "text_delta", "text": "hi"},
            {"type": "done", "full_text": "hi"},
        ],
    ])
    tools = _make_tool_manager({})
    loop = ToolLoop(engine, tools)

    events = await _drain(loop.run(
        persona=_make_persona(), conversation_history=[],
        params=MagicMock(), tools=[],
    ))

    terminals = [e for e in events
                 if isinstance(e, (_LoopFinishedEvent, ErrorEvent))]
    assert len(terminals) == 1
    assert isinstance(terminals[0], _LoopFinishedEvent)
    # And the terminal is the last event in the stream
    assert events[-1] is terminals[0]


@pytest.mark.asyncio
async def test_retry_confirm_mode_with_tools():
    """A CONFIRM-mode persona on a retry turn still parks write calls.
    is_retry is a ChatSystem concern; from ToolLoop's POV the behavior
    must be identical — write tools never auto-execute in CONFIRM."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "w1", "name": "update_ticket",
                 "arguments": {"ticket_id": 1, "state": "closed"}}
            ]},
            {"type": "done", "full_text": ""},
        ],
    ])
    tools = _make_tool_manager({})
    loop = ToolLoop(engine, tools)

    # Simulate retry context by pre-seeding history with prior assistant turn
    history: List[Dict[str, Any]] = [
        {"role": "user", "content": "close it"},
        {"role": "assistant", "content": "discarded previous attempt"},
    ]

    events = await _drain(loop.run(
        persona=_make_persona(execution_mode=ExecutionMode.CONFIRM),
        conversation_history=history, params=MagicMock(), tools=[],
    ))

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.response_type == ResponseType.PENDING_CONFIRMATION
    assert finished.pending_writes is not None
    assert finished.pending_writes[0]["name"] == "update_ticket"
    tools.execute_tool.assert_not_called()


@pytest.mark.asyncio
async def test_resume_taint_propagation():
    """initial_taint_sources from a prior memory recall must be reflected
    on the terminal event's audit_info and turn_tainted flag when writes
    are parked. Confirms taint flows memory → tool_loop park."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "w1", "name": "update_ticket",
                 "arguments": {"ticket_id": 1, "state": "closed"}}
            ]},
            {"type": "done", "full_text": ""},
        ],
    ])
    tools = _make_tool_manager({})
    loop = ToolLoop(engine, tools)

    events = await _drain(loop.run(
        persona=_make_persona(execution_mode=ExecutionMode.CONFIRM),
        conversation_history=[], params=MagicMock(), tools=[],
        turn_tainted=True, initial_taint_sources=["memory_recall"],
    ))

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.turn_tainted is True
    assert finished.audit_info is not None
    assert finished.audit_info["tainted"] is True
    assert "memory_recall" in finished.audit_info["taint_sources"]
    # The human-readable confirmation prompt should mention taint
    assert "untrusted" in finished.final_text.lower()


@pytest.mark.asyncio
async def test_audit_info_flag_combinations():
    """audit_info.actions[*] must report binding/sensitivity/irreversible/
    always_confirm flags from definitions. delete_user (ALWAYS_CONFIRM)
    + irreversible — both flags should be set."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "w1", "name": "delete_user", "arguments": {"user_id": 42}}
            ]},
            {"type": "done", "full_text": "I will delete user 42"},
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
    audit = finished.audit_info
    assert audit is not None
    assert audit["execution_mode"] == "CONFIRM"
    action = audit["actions"][0]
    assert action["tool"] == "delete_user"
    assert action["always_confirm"] is True
    # delete_user is irreversible per definitions
    assert action["irreversible"] is True
    # And the rendered final_text surfaces the flags
    assert "HIGH-IMPACT" in finished.final_text
    assert "IRREVERSIBLE" in finished.final_text
