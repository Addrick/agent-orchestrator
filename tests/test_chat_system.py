# tests/test_chat_system.py

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
import json

from src.chat_system import ChatSystem, MAX_TOOL_CALLS
from src.database.memory_manager import MemoryManager
from src.engine import TextEngine, LLMCommunicationError
from src.clients.zammad_client import ZammadClient
from src.persona import ExecutionMode


@pytest.fixture
def chat_system_with_mocks():
    """Provides a ChatSystem instance with mocked dependencies for each test."""
    mock_memory_manager = MagicMock(spec=MemoryManager)
    mock_text_engine = MagicMock(spec=TextEngine)
    mock_zammad_client = MagicMock(spec=ZammadClient)
    mock_zammad_client.api_url = "http://test.zammad.local"
    mock_tool_manager = AsyncMock()

    mock_text_engine.generate_response = AsyncMock(return_value=({'type': 'text', 'content': 'LLM Reply'}, {}))
    mock_zammad_client.create_ticket = MagicMock(return_value={'id': 12345})
    mock_zammad_client.add_article_to_ticket = MagicMock()
    mock_zammad_client.search_user = MagicMock()
    mock_zammad_client.create_user = MagicMock()

    with patch('src.chat_system.load_personas_from_file') as mock_load_personas, \
            patch('src.chat_system.ToolManager', return_value=mock_tool_manager):
        mock_persona = MagicMock()
        mock_persona.get_name.return_value = "test_persona"
        mock_persona.get_config_for_engine.return_value = {"model_name": "test-model"}
        mock_persona.get_prompt.return_value = "You are a test bot."
        mock_persona.get_context_length.return_value = 10
        mock_persona.get_enabled_tools.return_value = []
        mock_persona.get_execution_mode.return_value = ExecutionMode.SILENT_ANALYSIS

        mock_load_personas.return_value = {"test_persona": mock_persona}

        system = ChatSystem(
            memory_manager=mock_memory_manager,
            text_engine=mock_text_engine,
            zammad_client=mock_zammad_client
        )
        system.bot_logic.preprocess_message = AsyncMock(return_value=None)

        system._should_create_ticket = MagicMock()
        system._get_or_create_zammad_user = AsyncMock(return_value=(101, "user@example.com"))
        system._find_ticket_id_in_message = MagicMock()
        system._find_active_ticket_for_user = AsyncMock(return_value=None)

        yield system, mock_memory_manager, mock_text_engine, mock_zammad_client, mock_persona, mock_tool_manager


# --- High-Level Flow Tests ---

@pytest.mark.asyncio
async def test_casual_mode_flow(chat_system_with_mocks):
    """Tests that non-ticket channels only log to memory."""
    system, _, _, zammad_mock, _, _ = chat_system_with_mocks
    system._should_create_ticket.return_value = False

    await system.generate_response('test_persona', 'user_discord_1', 'discord', 'Just chatting')

    system._get_or_create_zammad_user.assert_not_called()
    zammad_mock.create_ticket.assert_not_called()
    zammad_mock.add_article_to_ticket.assert_not_called()


@pytest.mark.asyncio
async def test_ticket_mode_new_ticket(chat_system_with_mocks):
    """Tests that ticket channels create a new Zammad ticket with the initial message."""
    system, _, _, zammad_mock, _, _ = chat_system_with_mocks
    system._should_create_ticket.return_value = True
    system._find_ticket_id_in_message.return_value = None
    system._find_active_ticket_for_user.return_value = None

    await system.generate_response('test_persona', 'user_gmail_1', 'gmail', 'Help me!', user_display_name='Test User')

    system._get_or_create_zammad_user.assert_awaited_once_with('user_gmail_1', 'gmail', 'Test User')
    zammad_mock.create_ticket.assert_called_once()
    _, call_kwargs = zammad_mock.create_ticket.call_args
    assert call_kwargs['title'] == 'New request from Test User via gmail'
    assert call_kwargs['article_body'] == 'Help me!'


@pytest.mark.asyncio
async def test_ticket_mode_updates_existing_open_ticket(chat_system_with_mocks):
    """Tests that if an active open ticket is found, the message is added to it."""
    system, _, _, zammad_mock, _, _ = chat_system_with_mocks
    system._should_create_ticket.return_value = True
    system._find_ticket_id_in_message.return_value = None
    system._find_active_ticket_for_user.return_value = 555

    await system.generate_response('test_persona', 'user_ticket', 'gmail', 'More info', user_display_name='Test User')

    system._find_active_ticket_for_user.assert_awaited_once_with(101)
    zammad_mock.create_ticket.assert_not_called()
    zammad_mock.add_article_to_ticket.assert_any_call(ticket_id=555, body='More info',
                                                      impersonate_email="user@example.com")


@pytest.mark.asyncio
async def test_ticket_mode_uses_id_from_message(chat_system_with_mocks):
    """Tests that an explicit ticket ID in a message is used."""
    system, _, _, zammad_mock, _, _ = chat_system_with_mocks
    system._should_create_ticket.return_value = True
    system._find_ticket_id_in_message.return_value = 999

    await system.generate_response('test_persona', 'user_gmail_2', 'gmail', "For [Ticket#999]",
                                   user_display_name='Another User')

    zammad_mock.create_ticket.assert_not_called()
    system._find_active_ticket_for_user.assert_not_awaited()
    assert zammad_mock.add_article_to_ticket.call_count == 2
    zammad_mock.add_article_to_ticket.assert_any_call(ticket_id=999, body="For [Ticket#999]",
                                                      impersonate_email="user@example.com")
    zammad_mock.add_article_to_ticket.assert_any_call(ticket_id=999, body="LLM Reply")


# --- Tool-Use Scenario Tests ---

@pytest.mark.asyncio
async def test_tool_use_in_silent_analysis_mode(chat_system_with_mocks):
    """Tests that in SILENT_ANALYSIS, tool calls are logged as internal notes but not executed."""
    system, _, text_engine_mock, zammad_mock, persona_mock, tool_manager_mock = chat_system_with_mocks

    persona_mock.get_execution_mode.return_value = ExecutionMode.SILENT_ANALYSIS
    persona_mock.get_enabled_tools.return_value = ['*']
    system._should_create_ticket.return_value = True
    system._find_ticket_id_in_message.return_value = 123

    tool_call = {'type': 'tool_calls',
                 'calls': [{'id': 'call_1', 'name': 'update_ticket', 'arguments': {'state': 'closed'}}]}
    final_response = {'type': 'text', 'content': 'I have closed the ticket.'}
    text_engine_mock.generate_response.side_effect = [(tool_call, {}), (final_response, {})]

    await system.generate_response('test_persona', 'user', 'support', 'close ticket 123')

    tool_manager_mock.execute_tool.assert_not_called()

    log_note_call = call(ticket_id=123,
                         body='SILENT ANALYSIS: Persona \'test_persona\' intended to call tool \'update_ticket\' with arguments: {"state": "closed"}',
                         internal=True)
    zammad_mock.add_article_to_ticket.assert_has_calls([log_note_call], any_order=True)

    assert text_engine_mock.generate_response.call_count == 2


@pytest.mark.asyncio
async def test_tool_use_in_assisted_dispatch_mode(chat_system_with_mocks):
    """Tests that in ASSISTED_DISPATCH, tool calls are executed."""
    system, _, text_engine_mock, zammad_mock, persona_mock, tool_manager_mock = chat_system_with_mocks

    persona_mock.get_execution_mode.return_value = ExecutionMode.ASSISTED_DISPATCH
    persona_mock.get_enabled_tools.return_value = ['*']
    system._should_create_ticket.return_value = True
    system._find_ticket_id_in_message.return_value = 123
    tool_manager_mock.execute_tool.return_value = {"status": "success"}

    tool_call = {'type': 'tool_calls',
                 'calls': [{'id': 'call_1', 'name': 'update_ticket', 'arguments': {'state': 'closed'}}]}
    final_response = {'type': 'text', 'content': 'I have closed the ticket.'}
    text_engine_mock.generate_response.side_effect = [(tool_call, {}), (final_response, {})]

    await system.generate_response('test_persona', 'user', 'support', 'close ticket 123')

    tool_manager_mock.execute_tool.assert_awaited_once_with('update_ticket', state='closed')
    assert text_engine_mock.generate_response.call_count == 2
    second_call_args = text_engine_mock.generate_response.call_args_list[1]
    # FIX: Access the second positional argument (index 1) from the .args tuple
    history_arg = second_call_args.args[1]['history']
    assert history_arg[-1]['role'] == 'tool'
    assert json.loads(history_arg[-1]['content']) == {"status": "success"}


@pytest.mark.asyncio
async def test_tool_use_exceeds_max_calls(chat_system_with_mocks):
    """Tests that the tool use loop terminates and returns an error."""
    system, _, text_engine_mock, _, persona_mock, _ = chat_system_with_mocks
    persona_mock.get_enabled_tools.return_value = ['*']

    tool_call = {'type': 'tool_calls', 'calls': [{'id': 'call_1', 'name': 'foo', 'arguments': {}}]}
    text_engine_mock.generate_response.return_value = (tool_call, {})

    response, _, _ = await system.generate_response('test_persona', 'user', 'support', 'do something')

    assert "stuck in a loop" in response
    assert text_engine_mock.generate_response.call_count == MAX_TOOL_CALLS


# --- Helper Method Tests ---

@pytest.mark.parametrize("history, mode, persona, expected_role, expected_content", [
    ([{'author_role': 'user', 'author_name': 'OtherUser', 'content': 'Hi'}], "channel", "test_persona", "user",
     "OtherUser: Hi"),
    ([{'author_role': 'assistant', 'author_name': 'OtherBot', 'content': 'Hi'}], "channel", "test_persona", "user",
     "OtherBot: Hi"),
    ([{'author_role': 'assistant', 'author_name': 'test_persona', 'content': 'Hi'}], "channel", "test_persona",
     "assistant", "Hi"),
    ([{'author_role': 'user', 'author_name': 'TicketUser', 'content': 'Hi'}], "ticket", "test_persona", "user", "Hi"),
])
def test_format_raw_history_for_llm(chat_system_with_mocks, history, mode, persona, expected_role, expected_content):
    system, _, _, _, _, _ = chat_system_with_mocks
    formatted = system._format_raw_history_for_llm(history, mode, persona)
    assert len(formatted) == 1
    assert formatted[0]['role'] == expected_role
    assert formatted[0]['content'] == expected_content


@pytest.mark.asyncio
@patch('src.chat_system.asyncio.to_thread')
async def test_get_or_create_zammad_user_finds_existing_user(mock_to_thread, chat_system_with_mocks):
    """Tests that an existing user is found and not re-created."""
    mock_to_thread.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
    system, _, _, zammad_mock, _, _ = chat_system_with_mocks
    system._get_or_create_zammad_user.side_effect = system.__class__._get_or_create_zammad_user.__get__(system)

    zammad_mock.search_user.return_value = [{'id': 505, 'email': 'found@example.com'}]

    user_id, email = await system._get_or_create_zammad_user("user", "channel", "display_name")

    assert user_id == 505
    assert email == 'found@example.com'
    zammad_mock.search_user.assert_called_once()
    zammad_mock.create_user.assert_not_called()


@pytest.mark.asyncio
@patch('src.chat_system.asyncio.to_thread')
async def test_get_or_create_zammad_user_uses_display_name(mock_to_thread, chat_system_with_mocks):
    """Tests that display name is parsed correctly for new users."""
    mock_to_thread.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
    system, _, _, zammad_mock, _, _ = chat_system_with_mocks
    system._get_or_create_zammad_user.side_effect = system.__class__._get_or_create_zammad_user.__get__(system)

    zammad_mock.search_user.return_value = []
    zammad_mock.create_user.return_value = {'id': 909, 'email': 'discord-123@test.zammad.local'}

    await system._get_or_create_zammad_user("123", "discord", "Discord User Name")

    zammad_mock.create_user.assert_called_once_with(
        email='discord-123@test.zammad.local', firstname="Discord", lastname="User Name",
        note='Auto-generated user from Discord. Original identifier: 123'
    )


# --- Error and Command Handling Tests ---

@pytest.mark.asyncio
async def test_generate_response_handles_llm_error(chat_system_with_mocks):
    """Tests that a user-facing error is returned if the text engine fails."""
    system, _, text_engine_mock, _, _, _ = chat_system_with_mocks
    text_engine_mock.generate_response.side_effect = LLMCommunicationError("API is down")

    response, _, _ = await system.generate_response('test_persona', 'user', 'channel', 'message')

    assert "I'm having trouble connecting" in response


@pytest.mark.asyncio
async def test_generate_response_handles_dev_command(chat_system_with_mocks):
    """Tests that dev commands bypass the main generation logic."""
    system, _, text_engine_mock, zammad_mock, _, _ = chat_system_with_mocks
    system.bot_logic.preprocess_message.return_value = {'response': 'Dev command processed', 'mutated': False}

    response, _, _ = await system.generate_response('test_persona', 'user', 'channel', 'help')

    assert response == 'Dev command processed'
    text_engine_mock.generate_response.assert_not_called()
    zammad_mock.create_ticket.assert_not_called()
