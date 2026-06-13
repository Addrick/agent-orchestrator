# tests/integration/conftest.py

import os
import time
import random

import pytest
from unittest.mock import AsyncMock, MagicMock

from memory.memory_manager import MemoryManager
from src.engine import TextEngine
from tests.helpers import make_chat_system
from src.persona import Persona, MemoryMode
from config.global_config import TEST_MEMORY_DATABASE_FILE


@pytest.fixture(scope="function")
def mocked_chat_system():
    """
    Sets up a ChatSystem with real MemoryManager and TextEngine.
    No external service credentials required. Phase C routes through
    `text_engine.stream_messages`, which wraps the mocked
    `generate_response` for non-local models.
    """
    db_path = f"{TEST_MEMORY_DATABASE_FILE}.{random.randint(1000, 9999)}"
    if os.path.exists(db_path):
        os.remove(db_path)

    memory_manager = MemoryManager(db_path=db_path)
    memory_manager.create_schema()

    text_engine = TextEngine()
    text_engine.generate_response = AsyncMock(  # type: ignore[method-assign]
        return_value=({'type': 'text', 'content': ''}, {}),
    )

    test_personas = {
        "test_persona": Persona(
            persona_name="test_persona", model_name="mock_model", prompt="You are a test persona.",
            enabled_tools=['*'], memory_mode=MemoryMode.CHANNEL_ISOLATED, history_messages=10
        ),
        "capped_persona": Persona(
            persona_name='capped_persona', model_name='mock', prompt='talk', history_messages=100
        )
    }

    chat_system = make_chat_system(
        memory_manager=memory_manager, text_engine=text_engine,
        personas=test_personas,
    )

    try:
        yield chat_system, memory_manager
    finally:
        memory_manager.close()
        time.sleep(0.1)
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError as e:
                print(f"\n[TEARDOWN WARNING] Could not remove test database file: {e}")
