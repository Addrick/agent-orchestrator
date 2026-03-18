# tests/test_chat_system.py

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
import json

from src.chat_system import ChatSystem, ResponseType
from src.database.memory_manager import MemoryManager
from src.engine import TextEngine, LLMCommunicationError
from src.clients.zammad_client import ZammadClient
from src.persona import Persona, ExecutionMode, MemoryMode


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
    mock_tool_manager.get_tool_definitions = MagicMock(return_value=[])

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
    assert "Error while generating a response:" in response


@pytest.mark.asyncio
async def test_generate_response_stores_payload_on_llm_error(chat_system_with_mocks):
    """
    Ensures that if the LLM raises an error, the prepared API payload is
    still stored for debugging purposes via `dump_last`.
    """
    system, _, text_engine_mock, _, _, _ = chat_system_with_mocks
    failed_payload = {"model": "mock_model", "messages": ["This is the context"]}
    # Configure the mock TextEngine to raise an error that includes the payload
    text_engine_mock.generate_response.side_effect = LLMCommunicationError(
        "API is down",
        api_payload=failed_payload
    )

    # We don't need the response, just to trigger the error handling
    await system.generate_response("test_persona", "user123", "channel", "test message")

    # Assert that the payload from the exception was stored
    assert "user123" in system.last_api_requests
    assert "test_persona" in system.last_api_requests["user123"]
    assert system.last_api_requests["user123"]["test_persona"] == failed_payload


@pytest.mark.asyncio
async def test_generate_response_handles_generic_exception(chat_system_with_mocks):
    system, memory_manager, _, _, _, _ = chat_system_with_mocks
    memory_manager.get_channel_history.side_effect = Exception("DB is locked")
    response, _, _ = await system.generate_response("test_persona", "user", "channel", "test")
    assert "An internal error occurred" in response


@pytest.mark.asyncio
async def test_generate_response_exits_after_max_tool_calls(chat_system_with_mocks):
    system, _, text_engine_mock, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])
    tool_call = {'type': 'tool_calls', 'calls': [{'id': 'c1', 'name': 'test_tool', 'arguments': {}}]}
    # Make the text engine always return a tool call
    text_engine_mock.generate_response.return_value = (tool_call, {})
    tool_manager_mock.execute_tool.return_value = {"result": "ok"}
    response, _, _ = await system.generate_response("test_persona", "user", "channel", "test")
    assert "stuck in a loop" in response
    # Called exactly MAX_TOOL_CALLS times
    assert text_engine_mock.generate_response.call_count == 5


# --- Existing High-Level and Formatting Tests ---

@pytest.mark.asyncio
async def test_ticket_mode_uses_id_from_message(chat_system_with_mocks):
    """Tests that an explicit ticket ID in a message is used."""
    system, _, _, zammad_mock, persona, _ = chat_system_with_mocks
    persona.set_zammad_aware(True)

    with patch.object(system, '_get_or_create_zammad_user', new_callable=AsyncMock,
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
async def test_tool_use_in_autonomous_mode(chat_system_with_mocks):
    """Tests that in AUTONOMOUS mode, tool calls are executed immediately."""
    system, _, text_engine_mock, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.AUTONOMOUS)
    persona.set_enabled_tools(['*'])

    tool_call = {'type': 'tool_calls',
                 'calls': [{'id': 'call_1', 'name': 'update_ticket', 'arguments': {'state': 'closed'}}]}
    final_response = {'type': 'text', 'content': 'I have closed the ticket.'}
    text_engine_mock.generate_response.side_effect = [(tool_call, {}), (final_response, {})]
    tool_manager_mock.execute_tool.return_value = {"result": {"id": 123, "state": "closed"}}

    response, _, _ = await system.generate_response('test_persona', 'user', 'channel', 'close ticket')

    tool_manager_mock.execute_tool.assert_called_once_with('update_ticket', state='closed')
    assert response == 'I have closed the ticket.'


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


# --- CONFIRM Mode Tests ---

@pytest.mark.asyncio
async def test_confirm_mode_returns_pending_for_write_tools(chat_system_with_mocks):
    """In CONFIRM mode, write tool calls should return PENDING_CONFIRMATION instead of executing."""
    system, _, text_engine_mock, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    tool_call = {'type': 'tool_calls',
                 'calls': [{'id': 'call_1', 'name': 'update_ticket', 'arguments': {'state': 'closed'}}]}
    text_engine_mock.generate_response.return_value = (tool_call, {})

    response, response_type, _ = await system.generate_response('test_persona', 'user', 'channel', 'close it')

    assert response_type == ResponseType.PENDING_CONFIRMATION
    assert 'update_ticket' in response
    tool_manager_mock.execute_tool.assert_not_called()


@pytest.mark.asyncio
async def test_confirm_mode_auto_executes_read_only_tools(chat_system_with_mocks):
    """In CONFIRM mode, read-only tools should execute immediately without confirmation."""
    system, _, text_engine_mock, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    tool_call = {'type': 'tool_calls',
                 'calls': [{'id': 'call_1', 'name': 'search_tickets', 'arguments': {'query': 'test'}}]}
    final_response = {'type': 'text', 'content': 'Found 3 tickets.'}
    text_engine_mock.generate_response.side_effect = [(tool_call, {}), (final_response, {})]
    tool_manager_mock.execute_tool.return_value = {"result": [{"id": 1}, {"id": 2}, {"id": 3}]}

    response, response_type, _ = await system.generate_response('test_persona', 'user', 'channel', 'search tickets')

    assert response_type == ResponseType.LLM_GENERATION
    assert response == 'Found 3 tickets.'
    tool_manager_mock.execute_tool.assert_called_once_with('search_tickets', query='test')


@pytest.mark.asyncio
async def test_confirm_mode_mixed_tools_executes_reads_and_pends_writes(chat_system_with_mocks):
    """In CONFIRM mode with mixed read+write tools, reads execute and writes pend."""
    system, _, text_engine_mock, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    tool_call = {'type': 'tool_calls',
                 'calls': [
                     {'id': 'call_1', 'name': 'search_tickets', 'arguments': {'query': 'test'}},
                     {'id': 'call_2', 'name': 'update_ticket', 'arguments': {'ticket_id': 1, 'state': 'closed'}}
                 ]}
    text_engine_mock.generate_response.return_value = (tool_call, {})
    tool_manager_mock.execute_tool.return_value = {"result": [{"id": 1}]}

    response, response_type, _ = await system.generate_response('test_persona', 'user', 'channel', 'find and close')

    assert response_type == ResponseType.PENDING_CONFIRMATION
    assert 'update_ticket' in response
    # Read tool was executed, write tool was not
    tool_manager_mock.execute_tool.assert_called_once_with('search_tickets', query='test')


@pytest.mark.asyncio
async def test_resume_pending_confirmation_approved(chat_system_with_mocks):
    """Approving a pending confirmation should execute the write tools and return the final response."""
    system, _, text_engine_mock, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    # First call: trigger pending confirmation
    tool_call = {'type': 'tool_calls',
                 'calls': [{'id': 'call_1', 'name': 'update_ticket', 'arguments': {'state': 'closed'}}]}
    text_engine_mock.generate_response.return_value = (tool_call, {})
    await system.generate_response('test_persona', 'user', 'channel', 'close it')

    # Resume with approval
    tool_manager_mock.execute_tool.return_value = {"result": {"id": 1, "state": "closed"}}
    final_response = {'type': 'text', 'content': 'Done, ticket closed.'}
    text_engine_mock.generate_response.return_value = (final_response, {})

    response, response_type, _ = await system.resume_pending_confirmation('user', 'test_persona', approved=True)

    assert response_type == ResponseType.LLM_GENERATION
    assert response == 'Done, ticket closed.'
    tool_manager_mock.execute_tool.assert_called_with('update_ticket', state='closed')


@pytest.mark.asyncio
async def test_resume_pending_confirmation_denied(chat_system_with_mocks):
    """Denying a pending confirmation should feed denial to LLM and return its response."""
    system, _, text_engine_mock, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    # First call: trigger pending confirmation
    tool_call = {'type': 'tool_calls',
                 'calls': [{'id': 'call_1', 'name': 'update_ticket', 'arguments': {'state': 'closed'}}]}
    text_engine_mock.generate_response.return_value = (tool_call, {})
    await system.generate_response('test_persona', 'user', 'channel', 'close it')

    # Resume with denial
    final_response = {'type': 'text', 'content': 'Understood, I won\'t close the ticket.'}
    text_engine_mock.generate_response.return_value = (final_response, {})

    response, response_type, _ = await system.resume_pending_confirmation('user', 'test_persona', approved=False)

    assert response_type == ResponseType.LLM_GENERATION
    tool_manager_mock.execute_tool.assert_not_called()
    assert "won't close" in response
