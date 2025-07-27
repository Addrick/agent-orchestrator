# tests/conftest.py
import pytest
import sqlite3 # Make sure to import sqlite3
from unittest.mock import AsyncMock

from src.persona import Persona
from src.database.context_manager import ContextManager
from src.engine import TextEngine
from src.chat_system import ChatSystem

@pytest.fixture
def default_persona():
    # ... (this fixture is fine) ...
    return Persona(
        persona_name="tester",
        model_name="test-model",
        prompt="You are a test bot."
    )

@pytest.fixture(scope="function")
def memory_db_manager():
    """
    Creates a ContextManager using a single, persistent in-memory SQLite connection
    for the duration of one test function. This ensures all operations within a
    test happen on the same database.
    """
    # 1. Create a single connection to an in-memory database
    connection = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
    connection.row_factory = sqlite3.Row

    # 2. Instantiate the manager (the db_path is now just for show)
    manager = ContextManager(db_path=":memory:")

    # 3. Monkeypatch the manager's _get_connection method to always return our single connection
    manager._get_connection = lambda: connection

    # 4. Now, perform setup. Both calls will use the same, persistent connection.
    manager.create_schema()
    manager._initialize_db()

    # 5. Yield the manager to the test
    yield manager

    # 6. Teardown: close the connection after the test is done
    connection.close()

# The other fixtures (mock_text_engine, chat_system_with_db) are fine and do not need changes.
@pytest.fixture
def mock_text_engine(mocker):
    # ... (no changes needed) ...
    engine = TextEngine()
    mocker.patch.object(engine, 'generate_response', new_callable=AsyncMock)
    return engine

@pytest.fixture
def chat_system_with_db(memory_db_manager, mock_text_engine):
    # ... (no changes needed) ...
    personas = {
        "tester": Persona(
            persona_name="tester",
            model_name="mock-model",
            prompt="You are a test persona.",
            token_limit=100,
            context_length=10
        )
    }
    cs = ChatSystem(context_manager=memory_db_manager, text_engine=mock_text_engine)
    cs.personas = personas
    return cs
