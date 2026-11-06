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
    ToolLoop, ToolDeferredEvent, _LoopFinishedEvent,
)
from src.chat_system import PendingConfirmationEvent
from src.engine import TextEngine
from src.memory.memory_manager import MemoryManager
from src.persona import MemoryMode, Persona
from src.tools.tool_loop import _deferred_result
from tests.helpers import make_chat_system, offer_tools, route_stream_through_generate_response

# DP-404: the loop refuses a call whose name was not offered, so these
# tests offer every tool name they script.
_OFFERED = [{"type": "function", "function": {"name": n}} for n in ("create_ticket", "web_search")]


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

    async def execute(name, _offered, **kwargs):
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
        conversation_history=[], params=MagicMock(), tools=_OFFERED,
        turn_tainted=True,
        initial_taint_sources=["memory_recall"],
    ))

    park = next(e for e in events if isinstance(e, ToolDeferredEvent))
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
        conversation_history=[], params=MagicMock(), tools=_OFFERED,
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
        conversation_history=[], params=MagicMock(), tools=_OFFERED,
        turn_tainted=True,
        initial_taint_sources=["memory_recall"],
    ))

    park = next(e for e in events if isinstance(e, ToolDeferredEvent))
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
        conversation_history=[], params=MagicMock(), tools=_OFFERED,
        turn_tainted=True,
        initial_taint_sources=["memory_recall"],
    ))

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.turn_tainted is True
    assert not [e for e in events if isinstance(e, ToolDeferredEvent)]
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
        conversation_history=[], params=MagicMock(), tools=_OFFERED,
    ))

    finished = events[-1]
    assert isinstance(finished, _LoopFinishedEvent)
    assert finished.final_text == "ok"


@pytest.mark.asyncio
async def test_tool_loop_deduplicates_repeated_read_tool_taint():
    """If a read tool with produces_untrusted=True runs multiple times,
    taint_sources does not have duplicate entries."""
    engine = _make_engine([
        [
            {"type": "tool_calls", "calls": [
                {"id": "r1", "name": "web_search", "arguments": {"query": "q1"}},
                {"id": "r2", "name": "web_search", "arguments": {"query": "q2"}},
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
        conversation_history=[], params=MagicMock(), tools=_OFFERED,
        turn_tainted=True,
        initial_taint_sources=["memory_recall"],
    ))

    park = next(e for e in events if isinstance(e, ToolDeferredEvent))
    assert park.audit_info["taint_sources"] == ["memory_recall", "web_search"]
    assert park.taint_sources == ["memory_recall", "web_search"]


@pytest.mark.asyncio
async def test_auto_approved_deferral_carries_taint_sources():
    """Auto-approved deferral carries taint_sources on ToolDeferredEvent and audit_info."""
    from unittest.mock import patch
    r = {"call_id": "c1", "name": "auto_tool", "args": {"foo": "bar"}}
    tool_result = {"status": "ok"}
    with patch("src.tools.tool_loop.declared_deferral", return_value=("node_job", "tok-auto")):
        res = _deferred_result(r, tool_result, turn_tainted=True, taint_sources=["memory_recall", "web_search", "memory_recall"])
        assert res is not None
        msg, ev = res
        assert ev.turn_tainted is True
        assert ev.taint_sources == ["memory_recall", "web_search"]
        assert ev.audit_info["taint_sources"] == ["memory_recall", "web_search"]


@pytest.mark.asyncio
async def test_park_preserves_taint_sources_across_resume_end_to_end():
    """End-to-end orchestration test:
    Park a tainted write turn with >= 2 distinct taint sources ('memory_recall' + 'web_search')
    -> resolve/resume
    -> continuation proposes a second gated write
    -> assert full taint_sources list survives into continuation's PendingConfirmationEvent
       (audit_info['taint_sources'] and confirmation_text) and audit_decision metadata.
    """
    mem = MemoryManager(db_path=":memory:")
    mem.create_schema()
    try:
        text_engine = TextEngine()
        route_stream_through_generate_response(text_engine)

        persona = Persona(
            persona_name="test_persona", model_name="mock_model", prompt="Test",
            enabled_tools=["*"], memory_mode=MemoryMode.CHANNEL_ISOLATED,
        )
        persona.set_execution_mode(ExecutionMode.CONFIRM)

        chat_system = make_chat_system(
            memory_manager=mem, text_engine=text_engine,
            personas={"test_persona": persona},
        )
        offer_tools(
            chat_system, "test_persona",
            "create_ticket", "update_ticket", "web_search",
        )

        chat_system.request_builder.retrieve_memory_block = AsyncMock(
            return_value=("Untrusted memory summary", True)
        )

        scripted = [
            # Turn 1, iter 1: web_search + create_ticket
            ({"type": "tool_calls", "calls": [
                {"id": "r1", "name": "web_search", "arguments": {"query": "derpr"}},
                {"id": "w1", "name": "create_ticket", "arguments": {"title": "ticket 1"}},
            ]}, {}),
            # Turn 1, iter 2: closing text
            ({"type": "text", "content": "Proposed ticket 1."}, {}),
            # Continuation, iter 1: proposes update_ticket
            ({"type": "tool_calls", "calls": [
                {"id": "w2", "name": "update_ticket", "arguments": {"ticket_id": "1", "note": "fix"}},
            ]}, {}),
            # Continuation, iter 2: closing text
            ({"type": "text", "content": "Proposed update."}, {}),
        ]
        iterator = iter(scripted)

        async def fake_generate(*args, **kwargs):
            return next(iterator)
        text_engine.generate_response = AsyncMock(side_effect=fake_generate)

        # Execute Turn 1
        turn1_events = []
        async for ev in chat_system.stream_response("test_persona", "user1", "chan1", "do work"):
            turn1_events.append(ev)

        park1_events = [e for e in turn1_events if isinstance(e, PendingConfirmationEvent)]
        assert len(park1_events) == 1
        park1 = park1_events[0]
        assert "memory_recall" in park1.audit_info["taint_sources"]
        assert "web_search" in park1.audit_info["taint_sources"]
        assert "⚠️ Context contains untrusted content from: " in park1.text
        assert "memory_recall" in park1.text
        assert "web_search" in park1.text

        live_parks = chat_system.confirmations.list_for("user1", "test_persona")
        assert len(live_parks) == 1
        assert live_parks[0].token == park1.token
        assert set(live_parks[0].taint_sources) == {"memory_recall", "web_search"}

        continuation_events = []
        async for ev in chat_system.stream_resolve_park(
            "user1", "test_persona", park1.token, approved=True,
        ):
            continuation_events.append(ev)

        cont_parks = [e for e in continuation_events if isinstance(e, PendingConfirmationEvent)]
        assert len(cont_parks) == 1
        park2 = cont_parks[0]
        assert "memory_recall" in park2.audit_info["taint_sources"]
        assert "web_search" in park2.audit_info["taint_sources"]
        assert "⚠️ Context contains untrusted content from: " in park2.text
        assert "memory_recall" in park2.text
        assert "web_search" in park2.text

        row = mem._get_connection().execute(
            "SELECT metadata FROM Audit_Log WHERE event_type = 'audit_decision' ORDER BY rowid DESC"
        ).fetchone()
        assert row is not None
        meta = json.loads(row["metadata"])
        assert set(meta["taint_sources"]) == {"memory_recall", "web_search"}
    finally:
        mem.close()
