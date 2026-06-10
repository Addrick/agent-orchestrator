# tests/helpers.py
"""Shared construction helpers for tests (DP-201).

`make_chat_system` builds a ChatSystem directly from explicit dependencies —
no filesystem reads and no `patch('src.bootstrap.load_personas_from_file')`
ritual. Tests state exactly the personas/tools they need; anything omitted
gets a hermetic default.

Production code and entrypoints keep using `src.bootstrap.create_chat_system`;
tests whose *subject* is the bootstrap wiring (tests/integration/
test_startup_wiring.py) should keep exercising bootstrap too.
"""

from typing import Any, Dict, Optional, Set
from unittest.mock import MagicMock

from src.bootstrap import build_tool_manager
from src.chat_system import ChatSystem
from src.engine import TextEngine
from src.memory.memory_manager import MemoryManager
from src.persona import Persona


def make_chat_system(
    memory_manager: Optional[Any] = None,
    text_engine: Optional[Any] = None,
    personas: Optional[Dict[str, Persona]] = None,
    tool_manager: Optional[Any] = None,
    embedding_service: Optional[Any] = None,
    *,
    system_persona_names: Optional[Set[str]] = None,
    models_available: Optional[Dict[str, Any]] = None,
) -> ChatSystem:
    """Build a ChatSystem with explicit deps and hermetic defaults.

    Defaults: a spec'd MagicMock MemoryManager (with a stub `.backend`,
    which spec= would otherwise hide — it's an instance attribute), a spec'd
    MagicMock TextEngine, an empty persona map, an empty model catalog, and a
    real ToolManager with the core handlers registered (same registration as
    bootstrap, so tool dispatch behaves like production).
    """
    if memory_manager is None:
        memory_manager = MagicMock(spec=MemoryManager)
        memory_manager.backend = MagicMock()
    if text_engine is None:
        text_engine = MagicMock(spec=TextEngine)
    if personas is None:
        personas = {}
    if tool_manager is None:
        tool_manager = build_tool_manager(memory_manager, personas)
    return ChatSystem(
        memory_manager=memory_manager,
        text_engine=text_engine,
        embedding_service=embedding_service,
        personas=personas,
        system_persona_names=system_persona_names or set(),
        tool_manager=tool_manager,
        models_available=models_available,
    )
