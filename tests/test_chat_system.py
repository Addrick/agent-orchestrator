# tests/test_chat_system.py

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from src.chat_system import ChatSystem, ResponseType
from src.persona import Persona


# Since ChatSystem loads personas on init, we patch it globally for all tests
@pytest.fixture(autouse=True)
def mock_load_personas():
    """Auto-used fixture to patch load_personas_from_file."""
    mock_persona = Persona(
        persona_name='test_persona',
        model_name='test-model',
        prompt='You are a test persona.'
    )
    with patch('src.chat_system.load_personas_from_file', return_value={'test_persona': mock_persona}) as mocked_load:
        yield mocked_load


@pytest.fixture
def mock_context_manager() -> MagicMock:
    """Fixture for a mocked ContextManager."""
    manager = MagicMock()
    manager.get_context_for_generation.return_value = (1, 101, False)  # contact_id, ticket_id, is_new
    manager.build_prompt_context_object.return_value = {
        "user": {"name": "Test User"},
        "business": {"name": "TestCorp"},
        "history": []
    }
    return manager


@pytest.fixture
def mock_text_engine() -> AsyncMock:
    """Fixture for a mocked TextEngine."""
    engine = AsyncMock()
    engine.generate_response.return_value = ("This is a test response.", {"payload": "data"})
    return engine


@pytest.fixture
def mock_bot_logic() -> AsyncMock:
    """Fixture for a mocked BotLogic."""
    logic = AsyncMock()
    logic.preprocess_message.return_value = None
    return logic


@pytest.fixture
def chat_system(mock_context_manager, mock_text_engine, mock_bot_logic) -> ChatSystem:
    """Fixture to initialize ChatSystem with mocked dependencies."""
    cs = ChatSystem(context_manager=mock_context_manager, text_engine=mock_text_engine)
    cs.bot_logic = mock_bot_logic
    return cs


@pytest.mark.asyncio
async def test_generate_response_dev_command(chat_system: ChatSystem, mock_bot_logic: AsyncMock,
                                             mock_text_engine: AsyncMock):
    """Test that a dev command is handled correctly and bypasses the LLM engine."""
    mock_bot_logic.preprocess_message.return_value = {"response": "Dev command output", "mutated": False}

    response, response_type = await chat_system.generate_response("test_persona", "user1", "channel1", "help")

    assert response == "Dev command output"
    assert response_type == ResponseType.DEV_COMMAND
    mock_text_engine.generate_response.assert_not_called()


@pytest.mark.asyncio
@patch('src.chat_system.save_personas_to_file')
async def test_generate_response_dev_command_mutated(mock_save_personas, chat_system: ChatSystem,
                                                     mock_bot_logic: AsyncMock):
    """Test that a mutating dev command triggers a save."""
    mock_bot_logic.preprocess_message.return_value = {"response": "Persona updated", "mutated": True}

    await chat_system.generate_response("test_persona", "user1", "channel1", "set prompt new prompt")

    mock_save_personas.assert_called_once_with(chat_system.personas)


@pytest.mark.asyncio
async def test_generate_response_llm_generation(chat_system: ChatSystem, mock_context_manager: MagicMock,
                                                mock_text_engine: AsyncMock):
    """Test the standard LLM generation flow for an existing user."""
    response, response_type = await chat_system.generate_response(
        persona_name="test_persona",
        user_identifier="user1",
        channel="channel1",
        message="Hello, bot!"
    )

    assert response == "This is a test response."
    assert response_type == ResponseType.LLM_GENERATION

    mock_context_manager.get_context_for_generation.assert_called_once_with("user1", "channel1")
    mock_context_manager.build_prompt_context_object.assert_called_once()
    mock_text_engine.generate_response.assert_called_once()

    # Check that interactions are logged for both inbound and outbound messages
    assert mock_context_manager.log_interaction.call_count == 2
    mock_context_manager.log_interaction.assert_any_call(101, 'inbound', 'Hello, bot!', 'channel1', None)
    mock_context_manager.log_interaction.assert_any_call(101, 'outbound', 'This is a test response.', 'channel1')


@pytest.mark.asyncio
@patch('src.chat_system.asyncio.create_task', new_callable=AsyncMock)
async def test_generate_response_new_contact_triggers_guess(mock_create_task, chat_system: ChatSystem,
                                                            mock_context_manager: MagicMock):
    """Test that a new contact interaction triggers the business guessing task."""
    mock_context_manager.get_context_for_generation.return_value = (2, 102, True)  # is_new = True

    _, _ = await chat_system.generate_response(
        persona_name="test_persona",
        user_identifier="new_user",
        channel="channel1",
        message="I am new here"
    )

    mock_create_task.assert_called_once()
    # Ensure the coroutine passed to create_task is the correct one
    created_coro = mock_create_task.call_args[0][0]
    assert created_coro.__name__ == '_guess_and_record_business'

    # Await the background task to prevent the RuntimeWarning
    # In a real scenario, the event loop runs this, but in tests we must do it manually.
    assert len(chat_system.background_tasks) == 1
    task = chat_system.background_tasks.pop()
    await task


@pytest.mark.asyncio
async def test_guess_and_record_business(chat_system: ChatSystem, mock_context_manager: MagicMock,
                                         mock_text_engine: AsyncMock):
    """Test the business guessing logic directly."""
    mock_context_manager.get_all_businesses.return_value = [{'business_id': 1, 'business_name': 'TestCorp'}]
    mock_text_engine.generate_response.return_value = ("ID: 1, Reason: The email domain matches.", {})

    await chat_system._guess_and_record_business(contact_id=5, user_identifier="test@testcorp.com")

    mock_text_engine.generate_response.assert_called_once()
    # Check that the prompt for guessing is constructed correctly
    call_context = mock_text_engine.generate_response.call_args[0][1]
    assert "Analyze user identifier 'test@testcorp.com'" in call_context['persona_prompt']
    assert "'TestCorp' (ID: 1)" in call_context['persona_prompt']

    mock_context_manager.record_business_guess.assert_called_once_with(5, 1, "The email domain matches.")


@pytest.mark.asyncio
async def test_generate_response_unknown_persona(chat_system: ChatSystem, mock_text_engine: AsyncMock):
    """Test that requesting a non-existent persona returns an error."""
    response, response_type = await chat_system.generate_response(
        persona_name="unknown_persona",
        user_identifier="user1",
        channel="channel1",
        message="Anybody home?"
    )

    assert "Error: Persona not found." in response
    assert response_type == ResponseType.DEV_COMMAND
    mock_text_engine.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_generate_response_engine_exception(chat_system: ChatSystem, mock_text_engine: AsyncMock):
    """Test that an exception from the text engine is handled gracefully."""
    mock_text_engine.generate_response.side_effect = Exception("API is down")

    response, response_type = await chat_system.generate_response(
        persona_name="test_persona",
        user_identifier="user1",
        channel="channel1",
        message="Does this work?"
    )

    assert "An internal error occurred" in response
    assert response_type == ResponseType.DEV_COMMAND
