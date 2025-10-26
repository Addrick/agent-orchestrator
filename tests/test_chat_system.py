# tests/test_chat_system.py

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
import json

from src.chat_system import ChatSystem
from src.database.memory_manager import MemoryManager
from src.engine import TextEngine, LLMCommunicationError
from src.clients.zammad_client import ZammadClient
from src.persona import Persona, ExecutionMode, MemoryMode
from config.global_config import SUPPORT_CHANNELS


@pytest.fixture
def chat_system_with_mocks():
    """
    Provides a ChatSystem instance with its primary dependencies mocked.
    Helper methods on the ChatSystem itself are NOT mocked here.
    """
    mock_memory_manager = MagicMock(spec=MemoryManager)
    mock_text_engine = MagicMock(spec=TextEngine)
    mock_zammad_client = MagicMock(spec=ZammadClient)
    # Add the api_url attribute to make the mock a higher-fidelity representation
    mock_zammad_client.api_url = "http://zammad.local"
    mock_tool_manager = AsyncMock()

    mock_text_engine.generate_response = AsyncMock(return_value=({'type': 'text', 'content': 'LLM Reply'}, {}))

    mock_persona = Persona('test_persona', 'mock_model', 'prompt')

    with patch('src.chat_system.load_personas_from_file', return_value={"test_persona": mock_persona}), \
            patch('src.chat_system.ToolManager', return_value=mock_tool_manager):
        system = ChatSystem(
            memory_manager=mock_memory_manager,
            text_engine=mock_text_engine,
            zammad_client=mock_zammad_client
        )
        # Mock bot_logic by default to isolate ChatSystem logic
        system.bot_logic.preprocess_message = AsyncMock(return_value=None)

        yield (system, mock_memory_manager, mock_text_engine, mock_zammad_client,
               mock_persona, mock_tool_manager)


# --- Unit Tests for Helper Methods ---

@pytest.mark.parametrize("channel, expected", [
    ("support", True),
    ("SUPPORT", True),
    ("general", False),
    ("random-support-channel", False),
])
@patch('src.chat_system.SUPPORT_CHANNELS', ['support', 'helpdesk'])
def test_should_create_ticket(channel, expected, chat_system_with_mocks):
    system, _, _, _, _, _ = chat_system_with_mocks
    assert system._should_create_ticket(channel, "any message") == expected


@pytest.mark.parametrize("message, expected", [
    ("Help with [Ticket#12345]", 12345),
    ("[ticket#54321] is the one", 54321),
    ("No ticket here", None),
    ("Invalid format [Ticket#abc]", None),
])
def test_find_ticket_number_in_message(message, expected, chat_system_with_mocks):
    system, _, _, _, _, _ = chat_system_with_mocks
    assert system._find_ticket_number_in_message(message) == expected


@pytest.mark.asyncio
async def test_get_ticket_id_from_number_success(chat_system_with_mocks):
    system, _, _, zammad_mock, _, _ = chat_system_with_mocks
    zammad_mock.search_tickets.return_value = [{'id': 999}]
    result = await system._get_ticket_id_from_number(12345)
    zammad_mock.search_tickets.assert_called_once_with(query="number:12345")
    assert result == 999


@pytest.mark.asyncio
async def test_get_ticket_id_from_number_not_found(chat_system_with_mocks):
    system, _, _, zammad_mock, _, _ = chat_system_with_mocks
    zammad_mock.search_tickets.return_value = []
    result = await system._get_ticket_id_from_number(12345)
    assert result is None


@pytest.mark.asyncio
async def test_get_or_create_zammad_user_existing_real_email(chat_system_with_mocks):
    system, _, _, zammad_mock, _, _ = chat_system_with_mocks
    zammad_mock.search_user.return_value = [{'id': 101, 'email': 'test@example.com'}]
    user_id, email = await system._get_or_create_zammad_user("Test User <test@example.com>", "gmail")
    zammad_mock.search_user.assert_called_once_with('test@example.com')
    zammad_mock.create_user.assert_not_called()
    assert user_id == 101
    assert email == 'test@example.com'


@pytest.mark.asyncio
async def test_get_or_create_zammad_user_new_non_email(chat_system_with_mocks):
    system, _, _, zammad_mock, _, _ = chat_system_with_mocks
    zammad_mock.search_user.return_value = []
    zammad_mock.create_user.return_value = {'id': 102, 'email': 'discord-12345@zammad.local'}
    user_id, email = await system._get_or_create_zammad_user("12345", "discord", user_display_name="DiscordUser")

    expected_email = f"discord-12345@{zammad_mock.api_url.split('//')[1]}"
    zammad_mock.search_user.assert_called_once_with(expected_email)
    zammad_mock.create_user.assert_called_once()
    assert user_id == 102
    assert email == 'discord-12345@zammad.local'


# --- Tests for generate_response Core Logic ---

@pytest.mark.asyncio
async def test_generate_response_handles_dev_command(chat_system_with_mocks):
    system, _, text_engine_mock, _, _, _ = chat_system_with_mocks
    system.bot_logic.preprocess_message.return_value = {"response": "Dev command output", "mutated": False}
    response, _, _ = await system.generate_response("test_persona", "user", "channel", "what model")
    assert response == "Dev command output"
    text_engine_mock.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_generate_response_handles_persona_not_found(chat_system_with_mocks):
    system, _, text_engine_mock, _, _, _ = chat_system_with_mocks
    response, _, _ = await system.generate_response("unknown_persona", "user", "channel", "test")
    assert "Error: Persona not found" in response
    text_engine_mock.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_generate_response_handles_llm_communication_error(chat_system_with_mocks):
    system, _, text_engine_mock, _, _, _ = chat_system_with_mocks
    text_engine_mock.generate_response.side_effect = LLMCommunicationError("API is down")
    response, _, _ = await system.generate_response("test_persona", "user", "channel", "test")
    assert "I'm having trouble connecting" in response


@pytest.mark.asyncio
async def test_generate_response_handles_generic_exception(chat_system_with_mocks):
    system, memory_manager, _, _, _, _ = chat_system_with_mocks
    memory_manager.get_channel_history.side_effect = Exception("DB is locked")
    response, _, _ = await system.generate_response("test_persona", "user", "channel", "test")
    assert "An internal error occurred" in response


@pytest.mark.asyncio
async def test_generate_response_exits_after_max_tool_calls(chat_system_with_mocks):
    system, _, text_engine_mock, _, persona, _ = chat_system_with_mocks
    persona.set_enabled_tools(['*'])
    tool_call = {'type': 'tool_calls', 'calls': [{'id': 'c1', 'name': 'test_tool', 'arguments': {}}]}
    # Make the text engine always return a tool call
    text_engine_mock.generate_response.return_value = (tool_call, {})
    response, _, _ = await system.generate_response("test_persona", "user", "channel", "test")
    assert "stuck in a loop" in response
    # Called exactly MAX_TOOL_CALLS times
    assert text_engine_mock.generate_response.call_count == 5


# --- Existing High-Level and Formatting Tests ---

@pytest.mark.asyncio
async def test_ticket_mode_uses_id_from_message(chat_system_with_mocks):
    """Tests that an explicit ticket ID in a message is used."""
    system, _, _, zammad_mock, _, _, = chat_system_with_mocks

    with patch.object(system, '_should_create_ticket', return_value=True), \
            patch.object(system, '_get_or_create_zammad_user', new_callable=AsyncMock,
                         return_value=(101, "user@example.com")), \
            patch.object(system, '_find_ticket_number_in_message', return_value=9999), \
            patch.object(system, '_get_ticket_id_from_number', new_callable=AsyncMock, return_value=999), \
            patch.object(system, '_find_active_ticket_for_user', new_callable=AsyncMock) as mock_find_active:
        await system.generate_response('test_persona', 'user_gmail_2', 'gmail', "For [Ticket#9999]",
                                       user_display_name='Another User')

        mock_find_active.assert_not_awaited()
        zammad_mock.add_article_to_ticket.assert_any_call(ticket_id=999, body="For [Ticket#9999]",
                                                          impersonate_email="user@example.com")


@pytest.mark.asyncio
async def test_tool_use_in_silent_analysis_mode(chat_system_with_mocks):
    """Tests that in SILENT_ANALYSIS, tool calls are logged as internal notes but not executed."""
    system, _, text_engine_mock, zammad_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.SILENT_ANALYSIS)
    persona.set_enabled_tools(['*'])

    tool_call = {'type': 'tool_calls',
                 'calls': [{'id': 'call_1', 'name': 'update_ticket', 'arguments': {'state': 'closed'}}]}
    final_response = {'type': 'text', 'content': 'I have closed the ticket.'}
    text_engine_mock.generate_response.side_effect = [(tool_call, {}), (final_response, {})]

    with patch.object(system, '_should_create_ticket', return_value=True), \
            patch.object(system, '_get_or_create_zammad_user', new_callable=AsyncMock,
                         return_value=(101, "user@example.com")), \
            patch.object(system, '_find_ticket_number_in_message', return_value=12345), \
            patch.object(system, '_get_ticket_id_from_number', new_callable=AsyncMock, return_value=123):
        await system.generate_response('test_persona', 'user', 'support', 'close ticket [Ticket#12345]')

    tool_manager_mock.execute_tool.assert_not_called()
    log_note_call = call(ticket_id=123,
                         body='SILENT ANALYSIS: Persona \'test_persona\' intended to call tool \'update_ticket\' with arguments: {"state": "closed"}',
                         internal=True)
    zammad_mock.add_article_to_ticket.assert_has_calls([log_note_call], any_order=True)


@pytest.mark.parametrize("history, mode, server_id, persona, expected_role, expected_content", [
    ([{'author_role': 'user', 'author_name': 'OtherUser', 'content': 'Hi'}], "channel", "server1", "test_persona",
     "user", "OtherUser: Hi"),
    ([{'author_role': 'assistant', 'author_name': 'OtherBot', 'content': 'Hi'}], "server", "server1", "test_persona",
     "user", "OtherBot: Hi"),
    ([{'author_role': 'assistant', 'author_name': 'test_persona', 'content': 'Hi'}], "channel", "server1",
     "test_persona", "assistant", "Hi"),
    ([{'author_role': 'user', 'author_name': 'TicketUser', 'content': 'Hi'}], "ticket", None, "test_persona", "user",
     "TicketUser: Hi"),
    ([{'author_role': 'assistant', 'author_name': 'OtherBot', 'content': 'Hi'}], "ticket", None, "test_persona",
     "user", "OtherBot: Hi"),
    ([{'author_role': 'user', 'author_name': 'GmailUser', 'content': 'Hi'}], "channel", None, "test_persona", "user",
     "Hi"),
])
def test_format_raw_history_for_llm(chat_system_with_mocks, history, mode, server_id, persona, expected_role,
                                    expected_content):
    system, _, _, _, _, _ = chat_system_with_mocks
    formatted = system._format_raw_history_for_llm(history, mode, persona, server_id)
    assert len(formatted) == 1
    assert formatted[0]['role'] == expected_role
    assert formatted[0]['content'] == expected_content
