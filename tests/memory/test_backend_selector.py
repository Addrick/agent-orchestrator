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


def test_selector_default_in_production_is_hindsight() -> None:
    """Guards the DP-114 cutover: shipping `sqlite` again would silently
    take the engine off Hindsight without a config change in user envs."""
    from config import global_config
    assert global_config.SEMANTIC_BACKEND == "hindsight"


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
