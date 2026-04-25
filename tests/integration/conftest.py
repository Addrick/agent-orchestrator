# tests/integration/conftest.py

import os
import time
import random

import pytest
from unittest.mock import MagicMock, patch

from memory.memory_manager import MemoryManager
from src.chat_system import ChatSystem
from src.engine import TextEngine
from src.persona import Persona, MemoryMode
from config.global_config import TEST_MEMORY_DATABASE_FILE


@pytest.fixture(scope="function")
def mocked_chat_system():
    """
    Sets up a ChatSystem with real MemoryManager and TextEngine.
    No external service credentials required.
    """
    db_path = f"{TEST_MEMORY_DATABASE_FILE}.{random.randint(1000, 9999)}"
    if os.path.exists(db_path):
        os.remove(db_path)

    memory_manager = MemoryManager(db_path=db_path)
    memory_manager.create_schema()

    text_engine = MagicMock(spec=TextEngine)

    test_personas = {
        "test_persona": Persona(
            persona_name="test_persona", model_name="mock_model", prompt="You are a test persona.",
            enabled_tools=['*'], memory_mode=MemoryMode.CHANNEL_ISOLATED, context_length=10
        ),
        "capped_persona": Persona(
            persona_name='capped_persona', model_name='mock', prompt='talk', context_length=100
        )
    }

    with patch('src.chat_system.load_personas_from_file', return_value=test_personas):
        chat_system = ChatSystem(
            memory_manager=memory_manager, text_engine=text_engine,
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
