# src/bootstrap/__init__.py
"""Composition root for the chat pipeline (DP-200 slice B).

ChatSystem used to load personas and register tool handlers inside its own
``__init__`` — making it a service locator as well as an orchestrator. This
module owns that wiring instead: it loads the persona set, builds a fully
registered ToolManager, and hands ChatSystem its real dependencies. Only
entrypoints (main.py, test/eval fixtures, scripts) should import this module;
the boundary test forbids src/ modules from reaching into it.
"""

from typing import Dict, Optional, Set, Tuple

from config import global_config
from src.chat_system import ChatSystem
from src.embedding_service import EmbeddingService
from src.engine import TextEngine
from src.memory.memory_manager import MemoryManager
from src.persona import Persona
from src.tools.ingest_path import IngestPathHandler
from src.tools.tool_manager import (
    MemoryRecallHandler, MemoryToolHandler, ToolManager, WebSearchHandler,
)
from src.utils.save_utils import (
    load_personas_from_file, load_system_personas_from_file,
)


def load_all_personas() -> Tuple[Dict[str, Persona], Set[str]]:
    """Load user personas plus system personas (callable but not listed)."""
    personas: Dict[str, Persona] = load_personas_from_file() or {}
    system_persona_names: Set[str] = set()
    system_personas = load_system_personas_from_file()
    if system_personas:
        personas.update(system_personas)
        system_persona_names.update(system_personas.keys())
    return personas, system_persona_names


def build_tool_manager(
    memory_manager: MemoryManager, personas: Dict[str, Persona],
) -> ToolManager:
    """Build a ToolManager with the core (non-service) handlers registered.

    Service-bound tools (Zammad, agents) are registered later via
    ``ChatSystem.register_service`` — they depend on optional clients the
    entrypoint constructs conditionally.
    """
    tool_manager = ToolManager()
    WebSearchHandler().register(tool_manager)
    MemoryToolHandler(memory_manager).register(tool_manager)
    MemoryRecallHandler(memory_manager.backend).register(tool_manager)
    IngestPathHandler(
        memory_manager.backend,
        cache_dir=global_config.INGEST_CACHE_DIR,
        persona_lookup=personas.get,
    ).register(tool_manager)
    return tool_manager


def create_chat_system(
    memory_manager: MemoryManager,
    text_engine: TextEngine,
    embedding_service: Optional[EmbeddingService] = None,
) -> ChatSystem:
    """Standard ChatSystem assembly: file-loaded personas + core tool handlers.

    Mirrors the old ``ChatSystem(memory_manager, text_engine, embedding_service)``
    signature so call sites swap the constructor for this factory 1:1.
    """
    personas, system_persona_names = load_all_personas()
    tool_manager = build_tool_manager(memory_manager, personas)
    return ChatSystem(
        memory_manager=memory_manager,
        text_engine=text_engine,
        embedding_service=embedding_service,
        personas=personas,
        system_persona_names=system_persona_names,
        tool_manager=tool_manager,
    )
