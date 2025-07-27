# tests/test_message_handler.py

import pytest
import json
from unittest.mock import MagicMock, patch

from src.message_handler import BotLogic
from src.persona import Persona
from src.chat_system import ChatSystem


@pytest.fixture
def mock_persona() -> MagicMock:
    """Fixture for a mocked Persona instance."""
    persona = MagicMock(spec=Persona)
    persona.get_name.return_value = 'test_persona'
    persona.get_prompt.return_value = 'You are a test persona.'
    persona.get_model_name.return_value = 'test-model-v1'
    persona.get_context_length.return_value = 10
    persona.get_response_token_limit.return_value = 512
    persona.get_temperature.return_value = 0.7
    return persona


@pytest.fixture
def mock_chat_system(mock_persona: MagicMock) -> MagicMock:
    """Fixture for a mocked ChatSystem instance."""
    chat_system = MagicMock(spec=ChatSystem)
    chat_system.personas = {'test_persona': mock_persona}
    chat_system.last_api_requests = {
        'user123': {
            'test_persona': {'model': 'test-model-v1', 'prompt': 'test'}
        }
    }
    chat_system.models_available = {'TestProvider': ['test-model-v1', 'test-model-v2']}
    return chat_system


@pytest.fixture
def bot_logic(mock_chat_system: MagicMock) -> BotLogic:
    """Fixture for the BotLogic instance under test."""
    return BotLogic(mock_chat_system)


@pytest.mark.asyncio
async def test_preprocess_non_command(bot_logic: BotLogic):
    """Test that a regular message without a command keyword returns None."""
    result = await bot_logic.preprocess_message('test_persona', 'user123', "this is just a normal message")
    assert result is None


@pytest.mark.asyncio
async def test_preprocess_invalid_command(bot_logic: BotLogic):
    """Test that a message with an unknown command returns None."""
    result = await bot_logic.preprocess_message('test_persona', 'user123', "foobar some arguments")
    assert result is None


def test_handle_help(bot_logic: BotLogic, mock_persona: MagicMock):
    """Test the 'help' command directly."""
    response, mutated = bot_logic._handle_help([], mock_persona, 'user123')
    assert "Talk to a specific persona" in response
    assert "Bot commands:" in response
    assert not mutated


def test_handle_detail(bot_logic: BotLogic, mock_persona: MagicMock):
    """Test the 'detail' command directly."""
    response, mutated = bot_logic._handle_detail([], mock_persona, 'user123')
    assert f"Details for Persona: {mock_persona.get_name()}" in response
    assert f"Model: {mock_persona.get_model_name()}" in response
    assert f"Prompt:\n{mock_persona.get_prompt()}" in response
    assert not mutated


@pytest.mark.asyncio
@pytest.mark.parametrize("command, setter_method_name, expected_value", [
    ("set temp 0.8", "set_temperature", 0.8),
    ("set tokens 2048", "set_response_token_limit", 2048),
    ("set context 15", "set_context_length", 15),
    ("set top_p 0.9", "set_top_p", 0.9),
    ("set top_k 50", "set_top_k", 50)
])
async def test_set_value_success(bot_logic: BotLogic, mock_persona: MagicMock, command, setter_method_name,
                                 expected_value):
    """Test successful 'set' commands for numeric values."""
    setter_mock = getattr(mock_persona, setter_method_name)

    result = await bot_logic.preprocess_message('test_persona', 'user123', command)

    setter_mock.assert_called_once_with(expected_value)
    assert result['mutated'] is True
    assert "Error:" not in result['response']


@pytest.mark.asyncio
@pytest.mark.parametrize("command, setter_method_name", [
    ("set temp abc", "set_temperature"),
    ("set tokens xyz", "set_response_token_limit"),
    ("set context foo", "set_context_length")
])
async def test_set_value_invalid(bot_logic: BotLogic, mock_persona: MagicMock, command, setter_method_name):
    """Test 'set' commands with invalid non-numeric input."""
    setter_mock = getattr(mock_persona, setter_method_name)

    result = await bot_logic.preprocess_message('test_persona', 'user123', command)

    # The setter is still called, but with a value that will resolve to None inside the Persona
    setter_mock.assert_called_once()
    assert result['mutated'] is True
    assert "Non-numeric" in result['response']


@pytest.mark.asyncio
async def test_set_model(bot_logic: BotLogic, mock_persona: MagicMock):
    """Test 'set model' command for both success and failure cases."""
    # Success case
    with patch('src.message_handler.model_utils.check_model_available', return_value=True) as mock_check:
        result = await bot_logic.preprocess_message('test_persona', 'user123', "set model new-model-v2")
        mock_check.assert_called_once_with('new-model-v2')
        mock_persona.set_model_name.assert_called_once_with('new-model-v2')
        assert result['response'] == "Model for test_persona set to 'new-model-v2'."
        assert result['mutated'] is True

    mock_persona.set_model_name.reset_mock()

    # Failure case
    with patch('src.message_handler.model_utils.check_model_available', return_value=False) as mock_check:
        result = await bot_logic.preprocess_message('test_persona', 'user123', "set model non-existent-model")
        mock_check.assert_called_once_with('non-existent-model')
        mock_persona.set_model_name.assert_not_called()
        assert result['response'] == "Error: Model 'non-existent-model' does not exist."
        assert result['mutated'] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("command, getter_method_name, expected_substring", [
    ("what prompt", "get_prompt", "You are a test persona."),
    ("what model", "get_model_name", "test-model-v1"),
    ("what context", "get_context_length", "looks back 10 previous messages"),
    ("what tokens", "get_response_token_limit", "limited to 512 response tokens"),
    ("what temp", "get_temperature", "Temperature for test_persona is set to 0.7")
])
async def test_what_commands(bot_logic: BotLogic, command, getter_method_name, expected_substring):
    """Test various 'what' commands to ensure they retrieve state correctly."""
    result = await bot_logic.preprocess_message('test_persona', 'user123', command)
    assert expected_substring in result['response']
    assert result['mutated'] is False


@pytest.mark.asyncio
async def test_what_models(bot_logic: BotLogic, mock_chat_system: MagicMock):
    """Test the 'what models' command specifically."""
    result = await bot_logic.preprocess_message('test_persona', 'user123', 'what models')
    expected_json = json.dumps(mock_chat_system.models_available, indent=2)
    assert result['response'] == f"Available model options: {expected_json}"
    assert result['mutated'] is False


@pytest.mark.asyncio
async def test_handle_dump_last_success(bot_logic: BotLogic, mock_chat_system: MagicMock):
    """Test 'dump_last' when a previous request exists."""
    result = await bot_logic.preprocess_message('test_persona', 'user123', 'dump_last')

    assert "Last API Request Payload" in result['response']
    # The response should be a formatted JSON string
    assert '"model": "test-model-v1"' in result['response']
    assert '"prompt": "test"' in result['response']
    assert result['mutated'] is False


@pytest.mark.asyncio
async def test_handle_dump_last_no_request(bot_logic: BotLogic):
    """Test 'dump_last' when no previous request exists for the user."""
    # Use a different user identifier that is not in the mock data
    result = await bot_logic.preprocess_message('test_persona', 'user456', 'dump_last')

    assert "No previous request to dump" in result['response']
    assert result['mutated'] is False
