# tests/test_chat_system.py

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
import json

from src.chat_system import ChatSystem, ResponseType, ZammadContext, _get_model_prefix
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


# --- Model Prefix Helper Tests ---

@pytest.mark.parametrize("model_name, expected_prefix", [
    ("gpt-4o", "gpt"),
    ("gpt-3.5-turbo", "gpt"),
    ("claude-3-opus-20240229", "claude"),
    ("claude-3.5-sonnet", "claude"),
    ("gemma-3-27b-it", "gemma"),
    ("gemini-2.5-flash", "gemini"),
    ("gemini-2.5-pro", "gemini"),
    ("gemini-3.1-flash", "gemini-3.1"),
    ("gemini-3.1-pro", "gemini-3.1"),
    ("local", "local"),
    ("some-unknown-model", "unknown"),
])
def test_get_model_prefix(model_name, expected_prefix):
    assert _get_model_prefix(model_name) == expected_prefix


# --- Model Compatibility Filter Tests ---

@pytest.mark.asyncio
async def test_grounding_filtered_for_non_gemini_25_models(chat_system_with_mocks):
    """google_grounding_search should be filtered out for non-Gemini-2.5 models."""
    system, _, text_engine_mock, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])

    grounding_tool = {
        "type": "google_grounding",
        "function": {"name": "google_grounding_search", "description": "Grounding"}
    }
    web_search_tool = {
        "type": "function",
        "function": {"name": "web_search", "description": "Web search", "parameters": {}}
    }
    tool_manager_mock.get_tool_definitions.return_value = [grounding_tool, web_search_tool]

    # Test with GPT model — grounding should be filtered
    persona.set_model_name("gpt-4o")
    await system.generate_response("test_persona", "user", "channel", "test")
    call_args = text_engine_mock.generate_response.call_args
    tools_sent = call_args[1].get('tools', call_args[0][2] if len(call_args[0]) > 2 else [])
    tool_names = [t['function']['name'] for t in tools_sent]
    assert "google_grounding_search" not in tool_names
    assert "web_search" in tool_names


@pytest.mark.asyncio
async def test_grounding_kept_for_gemini_25_models(chat_system_with_mocks):
    """google_grounding_search should be kept for Gemini 2.5 models."""
    system, _, text_engine_mock, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])

    grounding_tool = {
        "type": "google_grounding",
        "function": {"name": "google_grounding_search", "description": "Grounding"}
    }
    web_search_tool = {
        "type": "function",
        "function": {"name": "web_search", "description": "Web search", "parameters": {}}
    }
    tool_manager_mock.get_tool_definitions.return_value = [grounding_tool, web_search_tool]

    persona.set_model_name("gemini-2.5-flash")
    await system.generate_response("test_persona", "user", "channel", "test")
    call_args = text_engine_mock.generate_response.call_args
    tools_sent = call_args[1].get('tools', call_args[0][2] if len(call_args[0]) > 2 else [])
    tool_names = [t['function']['name'] for t in tools_sent]
    assert "google_grounding_search" in tool_names
    assert "web_search" in tool_names


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [
    "gpt-4o", "claude-3-opus-20240229", "gemma-3-27b-it", "gemini-3.1-flash", "local",
])
async def test_grounding_filtered_for_incompatible_models(chat_system_with_mocks, model_name):
    """Grounding should be filtered for all incompatible model prefixes."""
    system, _, text_engine_mock, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])
    persona.set_model_name(model_name)

    grounding_tool = {
        "type": "google_grounding",
        "function": {"name": "google_grounding_search", "description": "Grounding"}
    }
    tool_manager_mock.get_tool_definitions.return_value = [grounding_tool]

    await system.generate_response("test_persona", "user", "channel", "test")
    call_args = text_engine_mock.generate_response.call_args
    tools_sent = call_args[1].get('tools', call_args[0][2] if len(call_args[0]) > 2 else [])
    assert len(tools_sent) == 0


# --- Unit Tests for Extracted Methods ---

@pytest.mark.asyncio
async def test_resolve_zammad_context_not_aware(chat_system_with_mocks):
    """Non-Zammad-aware persona returns empty context."""
    system, _, _, _, persona, _ = chat_system_with_mocks
    persona.set_zammad_aware(False)
    ctx = await system._resolve_zammad_context(persona, 'user', 'channel', 'hi', None)
    assert ctx.is_aware is False
    assert ctx.customer_id is None
    assert ctx.ticket_id is None


@pytest.mark.asyncio
async def test_resolve_zammad_context_with_ticket_number(chat_system_with_mocks):
    """Zammad-aware persona resolves ticket from message number."""
    system, _, _, _, persona, _ = chat_system_with_mocks
    persona.set_zammad_aware(True)

    with patch.object(system, '_get_or_create_zammad_user', new_callable=AsyncMock,
                      return_value=(101, 'u@test.com')), \
         patch.object(system, '_get_ticket_id_from_number', new_callable=AsyncMock, return_value=555):
        ctx = await system._resolve_zammad_context(
            persona, 'user', 'channel', 'See [Ticket#999]', None
        )

    assert ctx.is_aware is True
    assert ctx.customer_id == 101
    assert ctx.ticket_id == 555
    assert ctx.user_facing_ticket_number == 999


@pytest.mark.asyncio
async def test_resolve_zammad_context_falls_back_to_active_ticket(chat_system_with_mocks):
    """Without ticket number in message, resolves via active ticket search."""
    system, _, _, _, persona, _ = chat_system_with_mocks
    persona.set_zammad_aware(True)

    with patch.object(system, '_get_or_create_zammad_user', new_callable=AsyncMock,
                      return_value=(101, 'u@test.com')), \
         patch.object(system, '_find_active_ticket_for_user', new_callable=AsyncMock, return_value=777):
        ctx = await system._resolve_zammad_context(
            persona, 'user', 'channel', 'help me', None
        )

    assert ctx.ticket_id == 777
    assert ctx.user_facing_ticket_number is None


def test_build_conversation_history_channel_mode(chat_system_with_mocks):
    """Channel-isolated mode fetches channel history."""
    system, memory_mock, _, _, persona, _ = chat_system_with_mocks
    persona.set_memory_mode(MemoryMode.CHANNEL_ISOLATED)
    memory_mock.get_channel_history.return_value = [
        {'author_role': 'user', 'author_name': 'Alice', 'content': 'Hello'}
    ]
    zammad_ctx = ZammadContext()

    history = system._build_conversation_history(persona, zammad_ctx, 'user', 'general', 'srv1', None)

    memory_mock.get_channel_history.assert_called_once()
    assert len(history) == 1
    assert history[0]['content'] == 'Alice: Hello'


def test_build_conversation_history_ticket_mode_adds_system_message(chat_system_with_mocks):
    """Ticket-isolated mode adds system context message."""
    system, memory_mock, _, _, persona, _ = chat_system_with_mocks
    persona.set_memory_mode(MemoryMode.TICKET_ISOLATED)
    memory_mock.get_ticket_history.return_value = []
    zammad_ctx = ZammadContext(ticket_id=42, user_facing_ticket_number=100)

    history = system._build_conversation_history(persona, zammad_ctx, 'user', 'ch', None, None)

    assert history[0]['role'] == 'system'
    assert '#100' in history[0]['content']


def test_build_conversation_history_personal_mode(chat_system_with_mocks):
    """Personal mode fetches personal history."""
    system, memory_mock, _, _, persona, _ = chat_system_with_mocks
    persona.set_memory_mode(MemoryMode.PERSONAL)
    memory_mock.get_personal_history.return_value = []
    zammad_ctx = ZammadContext()

    system._build_conversation_history(persona, zammad_ctx, 'user123', 'ch', None, None)

    memory_mock.get_personal_history.assert_called_once_with('user123', 'test_persona', persona.get_context_length())


def test_filter_tools_wildcard(chat_system_with_mocks):
    """Wildcard enabled_tools returns all compatible tools."""
    system, _, _, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])
    tool_a = {"type": "function", "function": {"name": "tool_a", "parameters": {}}}
    tool_b = {"type": "function", "function": {"name": "tool_b", "parameters": {}}}
    tool_manager_mock.get_tool_definitions.return_value = [tool_a, tool_b]

    result = system._filter_tools_for_persona(persona, is_zammad_aware=True)
    assert len(result) == 2


def test_filter_tools_specific_names(chat_system_with_mocks):
    """Specific enabled_tools filters to only those tools."""
    system, _, _, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['tool_a'])
    tool_a = {"type": "function", "function": {"name": "tool_a", "parameters": {}}}
    tool_b = {"type": "function", "function": {"name": "tool_b", "parameters": {}}}
    tool_manager_mock.get_tool_definitions.return_value = [tool_a, tool_b]

    result = system._filter_tools_for_persona(persona, is_zammad_aware=True)
    assert len(result) == 1
    assert result[0]['function']['name'] == 'tool_a'


def test_filter_tools_removes_zammad_when_not_aware(chat_system_with_mocks):
    """Zammad tools are removed when persona is not Zammad-aware."""
    system, _, _, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])
    from src.tools.definitions import ZAMMAD_TOOLS
    zammad_tool = {"type": "function", "function": {"name": list(ZAMMAD_TOOLS)[0], "parameters": {}}}
    other_tool = {"type": "function", "function": {"name": "web_search", "parameters": {}}}
    tool_manager_mock.get_tool_definitions.return_value = [zammad_tool, other_tool]

    result = system._filter_tools_for_persona(persona, is_zammad_aware=False)
    tool_names = [t['function']['name'] for t in result]
    assert list(ZAMMAD_TOOLS)[0] not in tool_names
    assert 'web_search' in tool_names


@pytest.mark.asyncio
async def test_execute_write_calls_injects_customer_id(chat_system_with_mocks):
    """create_ticket gets customer_id injected from ZammadContext."""
    system, _, _, _, _, tool_manager_mock = chat_system_with_mocks
    tool_manager_mock.execute_tool.return_value = {"result": {"id": 50}}

    write_calls = [{"id": "c1", "name": "create_ticket", "arguments": {"title": "Test"}}]
    history = []
    ctx = ZammadContext(customer_id=101)

    await system._execute_write_calls(write_calls, history, ctx)

    tool_manager_mock.execute_tool.assert_called_once_with('create_ticket', title='Test', customer_id=101)
    assert ctx.ticket_id == 50
    assert len(history) == 1


@pytest.mark.asyncio
async def test_execute_write_calls_preserves_explicit_customer_id(chat_system_with_mocks):
    """If arguments already contain customer_id, it is not overwritten."""
    system, _, _, _, _, tool_manager_mock = chat_system_with_mocks
    tool_manager_mock.execute_tool.return_value = {"result": {"id": 60}}

    write_calls = [{"id": "c1", "name": "create_ticket", "arguments": {"title": "T", "customer_id": 999}}]
    history = []
    ctx = ZammadContext(customer_id=101)

    await system._execute_write_calls(write_calls, history, ctx)

    tool_manager_mock.execute_tool.assert_called_once_with('create_ticket', title='T', customer_id=999)


def test_append_denied_tool_results():
    """Denied results are appended for each write call."""
    write_calls = [
        {"id": "c1", "name": "update_ticket"},
        {"id": "c2", "name": "create_ticket"},
    ]
    history = []
    ChatSystem._append_denied_tool_results(write_calls, history)
    assert len(history) == 2
    assert json.loads(history[0]['content'])['error'] == "Tool call denied by user"
    assert history[1]['name'] == "create_ticket"


@pytest.mark.asyncio
async def test_mirror_message_to_zammad_when_aware(chat_system_with_mocks):
    """Message is mirrored to Zammad when aware and ticket exists."""
    system, _, _, zammad_mock, _, _ = chat_system_with_mocks
    ctx = ZammadContext(is_aware=True, ticket_id=42, zammad_email='u@test.com')

    await system._mirror_message_to_zammad(ctx, "Hello")

    zammad_mock.add_article_to_ticket.assert_called_once_with(
        ticket_id=42, body="Hello", impersonate_email='u@test.com'
    )


@pytest.mark.asyncio
async def test_mirror_message_to_zammad_skipped_when_not_aware(chat_system_with_mocks):
    """Mirroring is skipped when not Zammad-aware."""
    system, _, _, zammad_mock, _, _ = chat_system_with_mocks
    ctx = ZammadContext(is_aware=False)

    await system._mirror_message_to_zammad(ctx, "Hello")

    zammad_mock.add_article_to_ticket.assert_not_called()


@pytest.mark.asyncio
async def test_run_tool_loop_text_response(chat_system_with_mocks):
    """Tool loop returns immediately on text response."""
    system, _, text_engine_mock, _, persona, _ = chat_system_with_mocks
    text_engine_mock.generate_response.return_value = ({'type': 'text', 'content': 'Hi'}, {})
    ctx = ZammadContext()

    text, rtype = await system._run_tool_loop(
        persona, [{"role": "user", "content": "hi"}], [], ctx, 'user', 'test_persona', None
    )

    assert rtype == ResponseType.LLM_GENERATION
    assert text == 'Hi'


@pytest.mark.asyncio
async def test_run_tool_loop_max_calls_exceeded(chat_system_with_mocks):
    """Tool loop returns stuck message after MAX_TOOL_CALLS iterations."""
    system, _, text_engine_mock, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])
    tool_call = {'type': 'tool_calls', 'calls': [{'id': 'c1', 'name': 'web_search', 'arguments': {}}]}
    text_engine_mock.generate_response.return_value = (tool_call, {})
    tool_manager_mock.execute_tool.return_value = {"result": "ok"}
    ctx = ZammadContext()

    text, rtype = await system._run_tool_loop(
        persona, [{"role": "user", "content": "search"}], [], ctx, 'user', 'test_persona', None
    )

    assert rtype == ResponseType.DEV_COMMAND
    assert "stuck in a loop" in text
