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
        "derpr": Persona("derpr", "gpt-4", "You are derpr.", context_length=20),
        "testbot": Persona("testbot", "gpt-3", "You are testbot.")
    }
    return chat_system


@pytest.fixture
def bot_logic(mock_chat_system_with_state):
    """Creates a BotLogic instance connected to the stateful mock ChatSystem."""
    return BotLogic(mock_chat_system_with_state)


# --- Test Cases for State Mutation (add/delete) ---

@pytest.mark.asyncio
async def test_handle_add_persona_success(bot_logic, mock_chat_system_with_state):
    """Tests that the 'add' command successfully adds a new persona to the chat system's state."""
    assert "new_persona" not in mock_chat_system_with_state.personas
    result = await bot_logic.preprocess_message("derpr", "user1", "add new_persona")
    assert "new_persona" in mock_chat_system_with_state.personas
    assert isinstance(mock_chat_system_with_state.personas["new_persona"], Persona)
    assert result is not None and result["mutated"] is True and "Added 'new_persona'" in result["response"]


@pytest.mark.asyncio
async def test_handle_add_persona_already_exists(bot_logic, mock_chat_system_with_state):
    """Tests that adding a persona that already exists returns an error and does not mutate state."""
    initial_persona_count = len(mock_chat_system_with_state.personas)
    result = await bot_logic.preprocess_message("derpr", "user1", "add derpr")
    assert len(mock_chat_system_with_state.personas) == initial_persona_count
    assert result is not None and result["mutated"] is False and "already exists" in result["response"]


@pytest.mark.asyncio
async def test_handle_delete_persona_success(bot_logic, mock_chat_system_with_state):
    """Tests that the 'delete' command successfully removes a persona from the chat system's state."""
    assert "testbot" in mock_chat_system_with_state.personas
    result = await bot_logic.preprocess_message("derpr", "user1", "delete testbot")
    assert "testbot" not in mock_chat_system_with_state.personas
    assert result is not None and result["mutated"] is True and "Deleted persona 'testbot'" in result["response"]


@pytest.mark.asyncio
async def test_handle_delete_persona_not_found(bot_logic, mock_chat_system_with_state):
    """Tests that deleting a non-existent persona returns an error and does not mutate state."""
    initial_persona_count = len(mock_chat_system_with_state.personas)
    result = await bot_logic.preprocess_message("derpr", "user1", "delete fake_persona")
    assert len(mock_chat_system_with_state.personas) == initial_persona_count
    assert result is not None and result["mutated"] is False and "not found" in result["response"]


@pytest.mark.asyncio
async def test_command_fall_through_on_bad_syntax(bot_logic):
    """Tests that a command with incorrect syntax returns None, allowing it to be processed by the LLM."""
    assert await bot_logic.preprocess_message("derpr", "user1", "add") is None
    assert await bot_logic.preprocess_message("derpr", "user1", "set") is None


@pytest.mark.asyncio
async def test_non_mutating_command(bot_logic):
    """Tests that a read-only command like 'detail' does not set the mutated flag."""
    result = await bot_logic.preprocess_message("derpr", "user1", "detail")
    assert result is not None and result["mutated"] is False


# --- Test Cases for Dynamic Context Management ---

def test_dynamic_context_lifecycle(bot_logic, mock_chat_system_with_state):
    """
    Tests the full lifecycle of dynamic context: hello, growth, and goodbye.
    """
    persona = mock_chat_system_with_state.personas["derpr"]
    assert persona.get_context_length() == 20, "Initial state should be the default context length."

    bot_logic._handle_start_conversation([], persona, "user1")
    assert persona.get_context_length() == 0, "First call after 'hello' should return 0."
    assert persona.get_context_length() == 2, "Second call should return 2."
    assert persona.get_context_length() == 4, "Third call should return 4."

    bot_logic._handle_stop_conversation([], persona, "user1")
    assert persona.get_context_length() == 20, "After 'goodbye', should revert to default."
    assert persona.get_context_length() == 20, "Should stay at default after reverting."


def test_set_context_command_variations(bot_logic, mock_chat_system_with_state):
    """Tests the enhanced `set context` command for static and dynamic modes."""
    persona = mock_chat_system_with_state.personas["derpr"]

    # Test 1: Set a new static context length
    bot_logic._set_context(["context", "50"], persona)
    assert persona.get_current_effective_context_length() == 50
    assert persona.is_in_dynamic_context() is False

    # Test 2: Switch to dynamic mode, inheriting the current value (50)
    bot_logic._set_context(["context", "dynamic"], persona)
    assert persona.is_in_dynamic_context() is True
    assert persona.get_context_length() == 50, "It should start at the captured value of 50."
    assert persona.get_context_length() == 52, "Then it should grow to 52."

    # Test 3: Set a new static context, which should disable dynamic mode
    bot_logic._set_context(["context", "30"], persona)
    assert persona.is_in_dynamic_context() is False
    assert persona.get_current_effective_context_length() == 30

    # Test 4: Switch to dynamic mode with a specific start value
    bot_logic._set_context(["context", "dynamic", "8"], persona)
    assert persona.is_in_dynamic_context() is True
    assert persona.get_context_length() == 8
    assert persona.get_context_length() == 10

    # Test 5 (Your Refinement): Verify dynamic start from a dynamic value
    # The current context is 12 (from the previous step: 8 -> 10 -> 12)
    assert persona.get_current_effective_context_length() == 12
    # Now, set dynamic again. It should capture 12.
    bot_logic._set_context(["context", "dynamic"], persona)
    assert persona.is_in_dynamic_context() is True
    assert persona.get_context_length() == 12
    assert persona.get_context_length() == 14
