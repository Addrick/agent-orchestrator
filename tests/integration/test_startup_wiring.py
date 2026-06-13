# tests/integration/test_startup_wiring.py
#
# Verifies that the startup path in main.py correctly wires all
# ServiceIntegrations so every tool definition has a registered handler.

import os
import time
import random

import pytest
from unittest.mock import MagicMock, patch

from src.agents.agent_manager import AgentManager
from src.agents.agent_service import AgentServiceIntegration
from src.bootstrap import create_chat_system
from src.clients.zammad_service import ZammadIntegration
from memory.memory_manager import MemoryManager
from src.engine import TextEngine
from src.persona import Persona, MemoryMode
from src.tools.definitions import ALL_TOOL_DEFINITIONS
from config.global_config import TEST_MEMORY_DATABASE_FILE
from tests.helpers import make_chat_system

pytestmark = pytest.mark.integration


@pytest.fixture(scope="function")
def wired_system():
    """ChatSystem wired the same way main() does, with mocked externals."""
    db_path = f"{TEST_MEMORY_DATABASE_FILE}.wiring.{random.randint(1000, 9999)}"
    if os.path.exists(db_path):
        os.remove(db_path)

    memory_manager = MemoryManager(db_path=db_path)
    memory_manager.create_schema()
    text_engine = MagicMock(spec=TextEngine)

    test_personas = {
        "test_persona": Persona(
            persona_name="test_persona", model_name="gemini-2.5-flash",
            prompt="test", enabled_tools=["*"],
            memory_mode=MemoryMode.CHANNEL_ISOLATED, history_messages=10,
            service_bindings=["zammad", "agents"],
        ),
    }

    mock_zammad = MagicMock()

    with patch("src.bootstrap.load_personas_from_file", return_value=test_personas):
        chat_system = create_chat_system(memory_manager=memory_manager, text_engine=text_engine)

    # Mirror main.py wiring: register Zammad, then AgentManager + AgentService
    chat_system.register_service(ZammadIntegration(mock_zammad))
    agent_manager = AgentManager(chat_system=chat_system, memory_manager=memory_manager)
    chat_system.register_service(AgentServiceIntegration(agent_manager, memory_manager))

    try:
        yield chat_system
    finally:
        memory_manager.close()
        time.sleep(0.1)
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError:
                pass


def test_all_tool_definitions_have_registered_handlers(wired_system):
    """Every callable tool in ALL_TOOL_DEFINITIONS must have a handler in ToolManager."""
    all_defined = {
        t["function"]["name"]
        for t in ALL_TOOL_DEFINITIONS
        if t.get("type") == "function"
    }
    registered = {
        t["function"]["name"]
        for t in wired_system.tool_manager.get_tool_definitions()
        if t.get("type") == "function"
    }
    missing = all_defined - registered
    assert not missing, f"Tool definitions without registered handlers: {missing}"


def test_all_service_bindings_have_registered_services(wired_system):
    """Every service_binding referenced in tool definitions has a matching registered service."""
    bindings_in_defs = {
        t["service_binding"]
        for t in ALL_TOOL_DEFINITIONS
        if t.get("service_binding")
    }
    missing = {b for b in bindings_in_defs if wired_system.get_service(b) is None}
    assert not missing, f"Service bindings without registered services: {missing}"


def test_models_available_injected_by_bootstrap():
    """create_chat_system populates models_available from the model cache
    (DP-201): ChatSystem itself no longer reads the cache file at construction,
    so the composition root must inject it."""
    mm = MagicMock(spec=MemoryManager)
    mm.backend = MagicMock()
    with patch("src.bootstrap.load_personas_from_file", return_value={}), \
            patch("src.bootstrap.load_system_personas_from_file", return_value={}), \
            patch("src.bootstrap.get_model_list", return_value={"Local": ["local"]}):
        system = create_chat_system(
            memory_manager=mm, text_engine=MagicMock(spec=TextEngine),
        )
    assert system.models_available == {"Local": ["local"]}


def test_get_service_returns_registered_integration(wired_system):
    """Public service lookup replaces reaching into ChatSystem._services."""
    zammad = wired_system.get_service("zammad")
    assert isinstance(zammad, ZammadIntegration)
    assert wired_system.get_service("nonexistent") is None


def test_agent_manager_injects_zammad_client_via_public_accessors(wired_system):
    """Convention-based DI resolves zammad_client through get_service + .client,
    not private attribute reaches."""
    class _NeedsZammad:
        agent_name = "needs_zammad"

        def __init__(self, chat_system, zammad_client):
            self.chat_system = chat_system
            self.zammad_client = zammad_client

    manager = AgentManager(chat_system=wired_system, memory_manager=wired_system.memory_manager)
    instance = manager._build_agent_instance("needs_zammad", _NeedsZammad, {})
    zammad = wired_system.get_service("zammad")
    assert instance.zammad_client is zammad.client


def test_embedding_service_property_exposes_injected_service(wired_system):
    """ChatSystem.embedding_service is the public read path (None when not
    injected; construction-time injection is the only write path)."""
    assert wired_system.embedding_service is None
    mock_emb = MagicMock()
    system = make_chat_system(embedding_service=mock_emb)
    assert system.embedding_service is mock_emb


def test_persona_with_all_bindings_sees_all_tools(wired_system):
    """A persona bound to all services gets every callable tool."""
    persona = wired_system.personas["test_persona"]
    filtered = wired_system.request_builder.filter_tools_for_persona(persona)
    filtered_names = {t["function"]["name"] for t in filtered if t.get("type") == "function"}

    all_defined = {
        t["function"]["name"]
        for t in ALL_TOOL_DEFINITIONS
        if t.get("type") == "function"
    }

    # Exclude model-incompatible tools from the expectation
    from src.tools.definitions import MODEL_INCOMPATIBLE_TOOLS
    from src.utils.model_utils import get_model_prefix
    model_prefix = get_model_prefix(persona.get_model_name())
    incompatible = {
        name for name, prefixes in MODEL_INCOMPATIBLE_TOOLS.items()
        if model_prefix in prefixes
    }
    expected = all_defined - incompatible

    missing = expected - filtered_names
    assert not missing, f"Tools missing from persona's filtered set: {missing}"
