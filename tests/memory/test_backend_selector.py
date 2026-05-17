# tests/memory/test_backend_selector.py
"""DP-115: MemoryManager backend selector.

`SEMANTIC_BACKEND` in global_config picks the concrete `MemoryBackend`
the engine uses for new-shape recall/retain_turn. After the DP-114
default flip, the production default is `hindsight` — regression-protect
both branches plus the explicit-override path.
"""
from __future__ import annotations

from unittest.mock import patch

from src.memory.backend.hindsight import HindsightBackend
from src.memory.backend.sqlite import SqliteSemanticBackend
from src.memory.memory_manager import MemoryManager


def test_selector_picks_hindsight_when_configured() -> None:
    with patch("src.memory.memory_manager.SEMANTIC_BACKEND", "hindsight"), \
         patch("src.memory.memory_manager.HINDSIGHT_URL", "http://stub:8888"):
        mm = MemoryManager(db_path=":memory:")
    assert isinstance(mm.backend, HindsightBackend)
    assert mm.backend.url == "http://stub:8888"


def test_selector_picks_sqlite_when_configured() -> None:
    with patch("src.memory.memory_manager.SEMANTIC_BACKEND", "sqlite"):
        mm = MemoryManager(db_path=":memory:")
    assert isinstance(mm.backend, SqliteSemanticBackend)


def test_selector_explicit_backend_overrides_config() -> None:
    """DI path used by tests + the parallel-agent harness: a caller-supplied
    backend must short-circuit the config-driven branch."""
    sentinel = object()
    with patch("src.memory.memory_manager.SEMANTIC_BACKEND", "hindsight"):
        mm = MemoryManager(db_path=":memory:", backend=sentinel)
    assert mm.backend is sentinel


def test_action_log_routes_to_sqlite_under_hindsight() -> None:
    """Agent-action telemetry stays in sqlite even when the semantic backend
    is Hindsight. The ReminderAgent/DispatchAgent operational audit trail
    must keep working post-cutover (caller hits MemoryManager, not backend)."""
    with patch("src.memory.memory_manager.SEMANTIC_BACKEND", "hindsight"), \
         patch("src.memory.memory_manager.HINDSIGHT_URL", "http://stub:8888"):
        mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    assert isinstance(mm.backend, HindsightBackend)
    assert isinstance(mm._action_log, SqliteSemanticBackend)

    aid = mm.log_agent_action("dispatch", "test", outcome="pending")
    assert aid > 0
    mm.update_agent_action_outcome(aid, "success", "ok")
    hits = mm.get_relevant_agent_actions("dispatch")
    assert len(hits) == 1
    assert hits[0]["outcome"] == "success"
