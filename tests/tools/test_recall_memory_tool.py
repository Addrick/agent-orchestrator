"""DP-113 §5: tests for the `recall_memory` tool wiring + handler.

Asserts:
- Tool definition is present in ALL_TOOL_DEFINITIONS with correct
  capability flags (`produces_untrusted=True`, `irreversible=False`).
- Handler invokes `MemoryBackend.recall` with bank_id == active turn's
  persona and a tag_filter sourced from the active TurnContext.
- Handler returns formatted hits (dict shape) the engine can pass to the
  LLM, with the `untrusted` bit preserved so the security framework's
  taint propagation hooks the result back into the turn.
- When invoked with no active TurnContext (e.g. mis-wired test), the
  handler returns an empty list rather than calling the backend with a
  bogus bank_id.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.backend.base import MemoryHit
from src.tools.definitions import ALL_TOOL_DEFINITIONS, get_tool_capabilities
from src.tools.tool_manager import MemoryRecallHandler, ToolManager
from src.tools.turn_context import TurnContext, set_turn_context, reset_turn_context


def test_recall_memory_tool_definition_present() -> None:
    names = {t.get("function", {}).get("name") for t in ALL_TOOL_DEFINITIONS}
    assert "recall_memory" in names


def test_recall_memory_capabilities_match_security_framework() -> None:
    caps = get_tool_capabilities("recall_memory")
    assert caps["produces_untrusted"] is True
    assert caps["irreversible"] is False


@pytest.mark.asyncio
async def test_recall_memory_handler_invokes_backend_with_turn_scope() -> None:
    backend = MagicMock()
    backend.recall = AsyncMock(return_value=[
        MemoryHit(
            id="42", content="alice mentioned the deploy", score=0.88,
            untrusted=True, tags=["channel:c1", "persona:alice"],
            timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
        ),
    ])
    manager = ToolManager()
    MemoryRecallHandler(backend).register(manager)

    token = set_turn_context(TurnContext(
        persona_name="alice", user_identifier="u1",
        channel="c1", server_id="s9",
    ))
    try:
        out = await manager.execute_tool("recall_memory", query="deploy", limit=3)
    finally:
        reset_turn_context(token)

    backend.recall.assert_awaited_once()
    kwargs = backend.recall.await_args.kwargs
    assert kwargs["bank_id"] == "alice"
    assert kwargs["query"] == "deploy"
    assert kwargs["k"] == 3
    assert "channel:c1" in kwargs["tag_filter"]
    assert "user:u1" in kwargs["tag_filter"]
    assert "server:s9" in kwargs["tag_filter"]

    result = out["result"]
    assert result and isinstance(result, list)
    hit = result[0]
    assert hit["id"] == "42"
    assert hit["content"] == "alice mentioned the deploy"
    assert hit["untrusted"] is True
    assert hit["timestamp"] == "2026-05-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_recall_memory_handler_omits_server_tag_when_absent() -> None:
    """DP-115: turn context with no server_id (DM, ticket interface) must
    not synthesize an empty `server:` tag — that would mis-scope recall."""
    backend = MagicMock()
    backend.recall = AsyncMock(return_value=[])
    manager = ToolManager()
    MemoryRecallHandler(backend).register(manager)

    token = set_turn_context(TurnContext(
        persona_name="alice", user_identifier="u1",
        channel="dm:u1", server_id=None,
    ))
    try:
        await manager.execute_tool("recall_memory", query="x")
    finally:
        reset_turn_context(token)

    tag_filter = backend.recall.await_args.kwargs["tag_filter"]
    assert "channel:dm:u1" in tag_filter
    assert "user:u1" in tag_filter
    assert not any(t.startswith("server:") for t in tag_filter)


@pytest.mark.asyncio
async def test_recall_memory_handler_bank_id_follows_active_persona() -> None:
    """DP-115: bank_id is not cached on the handler — every invocation
    reads the active TurnContext, so per-persona scope isolation holds
    across consecutive turns from different personas."""
    backend = MagicMock()
    backend.recall = AsyncMock(return_value=[])
    manager = ToolManager()
    MemoryRecallHandler(backend).register(manager)

    for persona in ("alice", "bob", "charlie"):
        token = set_turn_context(TurnContext(
            persona_name=persona, user_identifier="u1",
            channel="c1", server_id=None,
        ))
        try:
            await manager.execute_tool("recall_memory", query="q")
        finally:
            reset_turn_context(token)

    banks = [c.kwargs["bank_id"] for c in backend.recall.await_args_list]
    assert banks == ["alice", "bob", "charlie"]


@pytest.mark.asyncio
async def test_recall_memory_handler_no_turn_context_returns_empty() -> None:
    backend = MagicMock()
    backend.recall = AsyncMock(return_value=[])
    manager = ToolManager()
    MemoryRecallHandler(backend).register(manager)

    out = await manager.execute_tool("recall_memory", query="anything")

    backend.recall.assert_not_awaited()
    assert out["result"] == []
