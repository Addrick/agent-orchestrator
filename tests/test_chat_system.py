# tests/test_chat_system.py

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import json

from src.chat_system import ChatSystem, ResponseType, RequestContext, _get_model_prefix
from src.database.memory_manager import MemoryManager
from src.engine import TextEngine, LLMCommunicationError
from src.clients.zammad_client import ZammadClient
from src.persona import Persona, ExecutionMode, MemoryMode
from src.clients.service_integration import ServiceIntegration


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


# --- Service Integration Tests ---

def test_register_service(chat_system_with_mocks):
    """register_service adds a service to the registry."""
    system, _, _, _, _, _ = chat_system_with_mocks
    mock_service = MagicMock(spec=ServiceIntegration)
    mock_service.name = "test_service"
    system.register_service(mock_service)
    assert "test_service" in system._services


@pytest.mark.asyncio
async def test_resolve_service_contexts_calls_bound_services(chat_system_with_mocks):
    """_resolve_service_contexts calls resolve_context on each bound service."""
    system, _, _, _, persona, _ = chat_system_with_mocks
    mock_service = MagicMock(spec=ServiceIntegration)
    mock_service.name = "test_svc"
    mock_service.resolve_context = AsyncMock(return_value={"key": "value"})
    system._services["test_svc"] = mock_service
    persona.set_service_bindings(["test_svc"])

    ctx = RequestContext(
        persona=persona, persona_name='test_persona', user_identifier='user',
        channel='ch', message='hi',
    )
    await system._resolve_service_contexts(ctx)

    assert ctx.service_data["test_svc"] == {"key": "value"}
    mock_service.resolve_context.assert_called_once_with('user', 'ch', 'hi', None)


@pytest.mark.asyncio
async def test_notify_services_calls_on_message(chat_system_with_mocks):
    """_notify_services calls on_message on each service in service_data."""
    system, _, _, _, _, _ = chat_system_with_mocks
    mock_service = MagicMock(spec=ServiceIntegration)
    mock_service.name = "test_svc"
    mock_service.on_message = AsyncMock()
    system._services["test_svc"] = mock_service

    service_data = {"test_svc": {"ticket_id": 42}}
    await system._notify_services(service_data, "Hello")

    mock_service.on_message.assert_called_once_with({"ticket_id": 42}, "Hello")


# --- Conversation History Tests ---

def test_build_conversation_history_channel_mode(chat_system_with_mocks):
    """Channel-isolated mode fetches channel history."""
    system, memory_mock, _, _, persona, _ = chat_system_with_mocks
    persona.set_memory_mode(MemoryMode.CHANNEL_ISOLATED)
    memory_mock.get_channel_history.return_value = [
        {'author_role': 'user', 'author_name': 'Alice', 'content': 'Hello'}
    ]

    history = system._build_conversation_history(persona, {}, 'user', 'general', 'srv1', None)

    memory_mock.get_channel_history.assert_called_once()
    assert len(history) == 1
    assert history[0]['content'] == 'Alice: Hello'


def test_build_conversation_history_ticket_mode_with_service_system_messages(chat_system_with_mocks):
    """Service system messages are injected into conversation history."""
    system, memory_mock, _, _, persona, _ = chat_system_with_mocks
    persona.set_memory_mode(MemoryMode.TICKET_ISOLATED)
    memory_mock.get_ticket_history.return_value = []

    mock_service = MagicMock(spec=ServiceIntegration)
    mock_service.name = "zammad"
    mock_service.get_system_messages.return_value = [
        {"role": "system", "content": "This conversation is part of Zammad ticket #100."}
    ]
    system._services["zammad"] = mock_service

    service_data = {"zammad": {"ticket_id": 42, "user_facing_ticket_number": 100}}
    history = system._build_conversation_history(persona, service_data, 'user', 'ch', None, None)

    assert history[0]['role'] == 'system'
    assert '#100' in history[0]['content']


def test_build_conversation_history_personal_mode(chat_system_with_mocks):
    """Personal mode fetches personal history."""
    system, memory_mock, _, _, persona, _ = chat_system_with_mocks
    persona.set_memory_mode(MemoryMode.PERSONAL)
    memory_mock.get_personal_history.return_value = []

    system._build_conversation_history(persona, {}, 'user123', 'ch', None, None)

    memory_mock.get_personal_history.assert_called_once_with('user123', 'test_persona', persona.get_context_length())


# --- Tool Filtering Tests ---

def test_filter_tools_wildcard(chat_system_with_mocks):
    """Wildcard enabled_tools returns all compatible tools."""
    system, _, _, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])
    tool_a = {"type": "function", "function": {"name": "tool_a", "parameters": {}}}
    tool_b = {"type": "function", "function": {"name": "tool_b", "parameters": {}}}
    tool_manager_mock.get_tool_definitions.return_value = [tool_a, tool_b]

    result = system._filter_tools_for_persona(persona)
    assert len(result) == 2


def test_filter_tools_specific_names(chat_system_with_mocks):
    """Specific enabled_tools filters to only those tools."""
    system, _, _, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['tool_a'])
    tool_a = {"type": "function", "function": {"name": "tool_a", "parameters": {}}}
    tool_b = {"type": "function", "function": {"name": "tool_b", "parameters": {}}}
    tool_manager_mock.get_tool_definitions.return_value = [tool_a, tool_b]

    result = system._filter_tools_for_persona(persona)
    assert len(result) == 1
    assert result[0]['function']['name'] == 'tool_a'


def test_filter_tools_removes_unbound_service_tools(chat_system_with_mocks):
    """Tools with a service_binding are removed when the persona lacks that binding."""
    system, _, _, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])
    persona.set_service_bindings([])  # No service bindings
    zammad_tool = {"type": "function", "service_binding": "zammad",
                   "function": {"name": "search_tickets", "parameters": {}}}
    other_tool = {"type": "function", "function": {"name": "web_search", "parameters": {}}}
    tool_manager_mock.get_tool_definitions.return_value = [zammad_tool, other_tool]

    result = system._filter_tools_for_persona(persona)
    tool_names = [t['function']['name'] for t in result]
    assert 'search_tickets' not in tool_names
    assert 'web_search' in tool_names


def test_filter_tools_includes_bound_service_tools(chat_system_with_mocks):
    """Tools with a service_binding are included when the persona has that binding."""
    system, _, _, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])
    persona.set_service_bindings(["zammad"])
    zammad_tool = {"type": "function", "service_binding": "zammad",
                   "function": {"name": "search_tickets", "parameters": {}}}
    other_tool = {"type": "function", "function": {"name": "web_search", "parameters": {}}}
    tool_manager_mock.get_tool_definitions.return_value = [zammad_tool, other_tool]

    result = system._filter_tools_for_persona(persona)
    tool_names = [t['function']['name'] for t in result]
    assert 'search_tickets' in tool_names
    assert 'web_search' in tool_names


# --- Write Call Execution Tests ---

@pytest.mark.asyncio
async def test_execute_write_calls_delegates_to_services(chat_system_with_mocks):
    """Services get to prepare tool args and react to results."""
    system, _, _, _, _, tool_manager_mock = chat_system_with_mocks
    tool_manager_mock.execute_tool.return_value = {"result": {"id": 50}}

    mock_service = MagicMock(spec=ServiceIntegration)
    mock_service.name = "zammad"
    mock_service.prepare_tool_args.return_value = {"title": "Test", "customer_id": 101}
    system._services["zammad"] = mock_service

    write_calls = [{"id": "c1", "name": "create_ticket", "arguments": {"title": "Test"}}]
    history: list = []
    service_data = {"zammad": {"customer_id": 101}}

    await system._execute_write_calls(write_calls, history, service_data)

    mock_service.prepare_tool_args.assert_called_once()
    mock_service.on_tool_result.assert_called_once()
    tool_manager_mock.execute_tool.assert_called_once_with(
        'create_ticket', title='Test', customer_id=101
    )
    assert len(history) == 1


@pytest.mark.asyncio
async def test_execute_write_calls_works_without_services(chat_system_with_mocks):
    """Write calls execute normally when no services are registered."""
    system, _, _, _, _, tool_manager_mock = chat_system_with_mocks
    tool_manager_mock.execute_tool.return_value = {"result": {"id": 60}}

    write_calls = [{"id": "c1", "name": "update_ticket", "arguments": {"state": "closed"}}]
    history: list = []

    await system._execute_write_calls(write_calls, history, {})

    tool_manager_mock.execute_tool.assert_called_once_with('update_ticket', state='closed')
    assert len(history) == 1


def test_append_denied_tool_results():
    """Denied results are appended for each write call."""
    write_calls = [
        {"id": "c1", "name": "update_ticket"},
        {"id": "c2", "name": "create_ticket"},
    ]
    history: list = []
    ChatSystem._append_denied_tool_results(write_calls, history)
    assert len(history) == 2
    assert json.loads(history[0]['content'])['error'] == "Tool call denied by user"
    assert history[1]['name'] == "create_ticket"


# --- Tool Loop Tests ---

@pytest.mark.asyncio
async def test_run_tool_loop_text_response(chat_system_with_mocks):
    """Tool loop returns immediately on text response."""
    system, _, text_engine_mock, _, persona, _ = chat_system_with_mocks
    text_engine_mock.generate_response.return_value = ({'type': 'text', 'content': 'Hi'}, {})
    ctx = RequestContext(
        persona=persona, persona_name='test_persona', user_identifier='user',
        channel='ch', message='hi',
        conversation_history=[{"role": "user", "content": "hi"}],
    )

    text, rtype = await system._run_tool_loop(ctx)

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
    ctx = RequestContext(
        persona=persona, persona_name='test_persona', user_identifier='user',
        channel='ch', message='search',
        conversation_history=[{"role": "user", "content": "search"}],
    )

    text, rtype = await system._run_tool_loop(ctx)

    assert rtype == ResponseType.DEV_COMMAND
    assert "stuck in a loop" in text


# --- Orchestration Method Tests ---

@pytest.mark.asyncio
async def test_execute_read_calls(chat_system_with_mocks):
    """Read tool calls are executed and results appended to history."""
    system, _, _, _, _, tool_manager_mock = chat_system_with_mocks
    tool_manager_mock.execute_tool.return_value = {"result": [{"id": 1}]}

    read_calls = [{"id": "c1", "name": "search_tickets", "arguments": {"query": "test"}}]
    history: list = []
    await system._execute_read_calls(read_calls, history)

    tool_manager_mock.execute_tool.assert_called_once_with('search_tickets', query='test')
    assert len(history) == 1
    assert history[0]['role'] == 'tool'
    assert history[0]['tool_call_id'] == 'c1'


@pytest.mark.asyncio
async def test_prepare_request_populates_context(chat_system_with_mocks):
    """_prepare_request resolves service contexts, builds history, filters tools, appends user message."""
    system, memory_mock, _, _, persona, tool_manager_mock = chat_system_with_mocks
    memory_mock.get_channel_history.return_value = []
    ctx = RequestContext(
        persona=persona, persona_name='test_persona', user_identifier='user',
        channel='general', message='hello', server_id='srv1',
    )

    await system._prepare_request(ctx)

    assert ctx.service_data == {}  # No services bound
    assert ctx.conversation_history[-1] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_execute_request_notifies_services(chat_system_with_mocks):
    """_execute_request notifies services of the final LLM response."""
    system, _, text_engine_mock, _, persona, _ = chat_system_with_mocks
    text_engine_mock.generate_response.return_value = ({'type': 'text', 'content': 'Reply'}, {})

    mock_service = MagicMock(spec=ServiceIntegration)
    mock_service.name = "zammad"
    mock_service.on_message = AsyncMock()
    system._services["zammad"] = mock_service

    ctx = RequestContext(
        persona=persona, persona_name='test_persona', user_identifier='user',
        channel='ch', message='hi',
        service_data={"zammad": {"ticket_id": 42, "zammad_email": "u@test.com"}},
        conversation_history=[{"role": "user", "content": "hi"}],
    )

    text, rtype = await system._execute_request(ctx)

    assert text == 'Reply'
    assert rtype == ResponseType.LLM_GENERATION
    mock_service.on_message.assert_called_once_with(
        {"ticket_id": 42, "zammad_email": "u@test.com"}, 'Reply'
    )


@pytest.mark.asyncio
async def test_execute_request_skips_notify_for_pending(chat_system_with_mocks):
    """_execute_request does not notify services for confirmation prompts."""
    system, _, text_engine_mock, _, persona, _ = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])
    tool_call = {'type': 'tool_calls',
                 'calls': [{'id': 'c1', 'name': 'update_ticket', 'arguments': {'state': 'closed'}}]}
    text_engine_mock.generate_response.return_value = (tool_call, {})

    mock_service = MagicMock(spec=ServiceIntegration)
    mock_service.name = "zammad"
    mock_service.on_message = AsyncMock()
    system._services["zammad"] = mock_service

    ctx = RequestContext(
        persona=persona, persona_name='test_persona', user_identifier='user',
        channel='ch', message='close it',
        service_data={"zammad": {"ticket_id": 42}},
        conversation_history=[{"role": "user", "content": "close it"}],
    )

    _, rtype = await system._execute_request(ctx)

    assert rtype == ResponseType.PENDING_CONFIRMATION
    mock_service.on_message.assert_not_called()


@pytest.mark.asyncio
async def test_generate_response_extracts_ticket_id_from_service_data(chat_system_with_mocks):
    """generate_response returns ticket_id from service_data for backward compatibility."""
    system, memory_mock, text_engine_mock, _, persona, _ = chat_system_with_mocks
    memory_mock.get_channel_history.return_value = []
    persona.set_service_bindings(["zammad"])

    mock_service = MagicMock(spec=ServiceIntegration)
    mock_service.name = "zammad"
    mock_service.resolve_context = AsyncMock(return_value={"ticket_id": 42, "customer_id": 101})
    mock_service.on_message = AsyncMock()
    mock_service.get_system_messages.return_value = []
    system._services["zammad"] = mock_service

    _, _, ticket_id = await system.generate_response('test_persona', 'user', 'channel', 'test')

    assert ticket_id == 42
