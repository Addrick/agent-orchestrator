# tests/test_message_handler.py

import pytest
from unittest.mock import MagicMock

from src.message_handler import BotLogic
from src.persona import Persona
from src.chat_system import ChatSystem


@pytest.fixture
def mock_chat_system_with_state():
    """Creates a mock ChatSystem that has a real dictionary for personas."""
    chat_system = MagicMock(spec=ChatSystem)
    # Start with a real dictionary to track state changes
    chat_system.personas = {
        "derpr": Persona("derpr", "gpt-4", "You are derpr."),
        "testbot": Persona("testbot", "gpt-3", "You are testbot.")
    }
    return chat_system


@pytest.fixture
def bot_logic(mock_chat_system_with_state):
    """Creates a BotLogic instance connected to the stateful mock ChatSystem."""
    return BotLogic(mock_chat_system_with_state)


# --- Test Cases ---

@pytest.mark.asyncio
async def test_handle_add_persona_success(bot_logic, mock_chat_system_with_state):
    """
    Tests that the 'add' command successfully adds a new persona to the chat system's state.
    This test would have failed with the old, broken implementation.
    """
    assert "new_persona" not in mock_chat_system_with_state.personas

    result = await bot_logic.preprocess_message("derpr", "user1", "add new_persona")

    assert "new_persona" in mock_chat_system_with_state.personas
    assert isinstance(mock_chat_system_with_state.personas["new_persona"], Persona)
    assert result is not None
    assert result["mutated"] is True
    assert "Added 'new_persona'" in result["response"]


@pytest.mark.asyncio
async def test_handle_add_persona_already_exists(bot_logic, mock_chat_system_with_state):
    """Tests that adding a persona that already exists returns an error and does not mutate state."""
    initial_persona_count = len(mock_chat_system_with_state.personas)

    result = await bot_logic.preprocess_message("derpr", "user1", "add derpr")

    assert len(mock_chat_system_with_state.personas) == initial_persona_count
    assert result is not None
    assert result["mutated"] is False
    assert "Error: Persona 'derpr' already exists." in result["response"]


@pytest.mark.asyncio
async def test_handle_delete_persona_success(bot_logic, mock_chat_system_with_state):
    """
    Tests that the 'delete' command successfully removes a persona from the chat system's state.
    This test would have appeared to work but was misleading with the old implementation.
    """
    assert "testbot" in mock_chat_system_with_state.personas

    result = await bot_logic.preprocess_message("derpr", "user1", "delete testbot")

    assert "testbot" not in mock_chat_system_with_state.personas
    assert result is not None
    assert result["mutated"] is True
    assert "Deleted persona 'testbot'." in result["response"]


@pytest.mark.asyncio
async def test_handle_delete_persona_not_found(bot_logic, mock_chat_system_with_state):
    """Tests that deleting a non-existent persona returns an error and does not mutate state."""
    initial_persona_count = len(mock_chat_system_with_state.personas)

    result = await bot_logic.preprocess_message("derpr", "user1", "delete fake_persona")

    assert len(mock_chat_system_with_state.personas) == initial_persona_count
    assert result is not None
    assert result["mutated"] is False
    assert "Error: Persona 'fake_persona' not found." in result["response"]


@pytest.mark.asyncio
async def test_command_fall_through_on_bad_syntax(bot_logic):
    """
    Tests that a command with incorrect syntax (e.g., missing arguments) returns None,
    allowing it to be processed by the LLM instead.
    """
    result = await bot_logic.preprocess_message("derpr", "user1", "add")  # 'add' requires a name
    assert result is None, "Command should fall through to LLM if syntax is invalid"

    result = await bot_logic.preprocess_message("derpr", "user1", "set")  # 'set' requires a sub-command
    assert result is None, "Command should fall through to LLM if syntax is invalid"


@pytest.mark.asyncio
async def test_non_mutating_command(bot_logic):
    """Tests that a read-only command like 'detail' does not set the mutated flag."""
    result = await bot_logic.preprocess_message("derpr", "user1", "detail")
    assert result is not None
    assert result["mutated"] is False
