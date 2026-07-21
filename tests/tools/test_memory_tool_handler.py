# tests/tools/test_memory_tool_handler.py

import pytest
from unittest.mock import MagicMock

from src.tools.tool_manager import MemoryToolHandler, ToolManager


class DummySqliteBackend:
    pass


class DummyHindsightBackend:
    pass


def test_memory_tool_handler_registers_sqlite_only() -> None:
    # 1. Test registration with SqliteSemanticBackend
    sqlite_backend = DummySqliteBackend()
    sqlite_backend.__class__.__name__ = "SqliteSemanticBackend"
    
    mock_mm_sqlite = MagicMock()
    mock_mm_sqlite.backend = sqlite_backend
    
    manager = ToolManager()
    handler = MemoryToolHandler(mock_mm_sqlite)
    handler.register(manager)
    
    registered_tools = manager.get_tool_definitions()
    tool_names = [t.get("function", {}).get("name") for t in registered_tools]
    
    assert "drill_down_memory" in tool_names
    assert "update_core_memory" in tool_names


def test_memory_tool_handler_ignores_hindsight() -> None:
    # 2. Test registration with HindsightBackend
    hindsight_backend = DummyHindsightBackend()
    hindsight_backend.__class__.__name__ = "HindsightBackend"
    
    mock_mm_hindsight = MagicMock()
    mock_mm_hindsight.backend = hindsight_backend
    
    manager = ToolManager()
    handler = MemoryToolHandler(mock_mm_hindsight)
    handler.register(manager)
    
    registered_tools = manager.get_tool_definitions()
    tool_names = [t.get("function", {}).get("name") for t in registered_tools]
    
    assert "drill_down_memory" not in tool_names
    assert "update_core_memory" not in tool_names


def test_memory_tool_handler_ignores_missing_backend() -> None:
    # 3. Test registration when backend attribute is missing or None
    mock_mm_no_backend = MagicMock(spec=[])  # no backend attribute
    
    manager = ToolManager()
    handler = MemoryToolHandler(mock_mm_no_backend)
    handler.register(manager)
    
    registered_tools = manager.get_tool_definitions()
    tool_names = [t.get("function", {}).get("name") for t in registered_tools]
    
    assert "drill_down_memory" not in tool_names
    assert "update_core_memory" not in tool_names
