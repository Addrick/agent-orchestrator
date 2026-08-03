# tests/test_chat_system.py

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import json
import re

from config.global_config import MAX_TOOL_CALLS
from src.confirmations import ConfirmationManager, Decision, ParkedWrite
from tests.helpers import (
    make_chat_system, only_pending_token,
    route_stream_through_generate_response,
)
from src.chat_system import (
    ChatSystem, ResponseType,
    DoneEvent, ErrorEvent, TokenEvent,
    ToolCallResultEvent, ToolCallStartEvent,
)
from src.request_builder import RequestContext
from src.utils.model_utils import get_model_prefix
from memory.memory_manager import MemoryManager
from src.engine import TextEngine, LLMCommunicationError
from src.persona import Persona, ExecutionMode, MemoryMode
from src.clients.service_integration import ServiceIntegration


@pytest.fixture
def chat_system_with_mocks():
    """
    Provides a ChatSystem instance with its primary dependencies mocked.
    Helper methods on the ChatSystem itself are NOT mocked here.

    Uses a real TextEngine with `generate_response` mocked. DP-206b: the
    pipeline streams through the `_stream_response` policy driver; the test
    bridge routes that driver back through the mocked `generate_response`, so
    every assertion on `text_engine.generate_response.*` works unchanged.
    """
    mock_memory_manager = MagicMock(spec=MemoryManager)
    # DP-113: ChatSystem reads memory_manager.backend at construction. spec=
    # restricts to class attributes, so attach a stub backend explicitly.
    mock_memory_manager.backend = MagicMock()
    text_engine = TextEngine()
    text_engine.generate_response = AsyncMock(  # type: ignore[method-assign]
        return_value=({'type': 'text', 'content': 'LLM Reply'}, {}),
    )
    route_stream_through_generate_response(text_engine)
    mock_tool_manager = AsyncMock()
    mock_tool_manager.get_tool_definitions = MagicMock(return_value=[])

    mock_persona = Persona('test_persona', 'mock_model', 'prompt')

    system = make_chat_system(
        memory_manager=mock_memory_manager,
        text_engine=text_engine,
        personas={"test_persona": mock_persona},
        tool_manager=mock_tool_manager,
    )
    # Mock bot_logic by default to isolate ChatSystem logic
    system.bot_logic.preprocess_message = AsyncMock(return_value=None)

    yield (system, mock_memory_manager, text_engine,
           mock_persona, mock_tool_manager)


# --- Tests for generate_response Core Logic ---

@pytest.mark.asyncio
async def test_generate_response_handles_dev_command(chat_system_with_mocks):
    system, _, text_engine_mock, _, _ = chat_system_with_mocks
    system.bot_logic.preprocess_message.return_value = {"response": "Dev command output", "mutated": False}
    response, _, _ , _ = await system.generate_response("test_persona", "user", "channel", "what model")
    assert response == "Dev command output"
    text_engine_mock.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_generate_response_handles_persona_not_found(chat_system_with_mocks):
    system, _, text_engine_mock, _, _ = chat_system_with_mocks
    response, _, _ , _ = await system.generate_response("unknown_persona", "user", "channel", "test")
    assert "Error: Persona not found" in response
    text_engine_mock.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_generate_response_refuses_quarantined_persona(chat_system_with_mocks):
    """DP-128: a security-quarantined persona is refused with an explanatory
    message and never reaches the LLM."""
    system, _, text_engine_mock, mock_persona, _ = chat_system_with_mocks
    mock_persona._security_block_reasons = [
        "Insecure composition: network:read + local:write (potential disk rewrite via injection)"
    ]
    response, rtype, _, _ = await system.generate_response(
        "test_persona", "user", "channel", "do something")
    assert "quarantined" in response.lower()
    assert "network:read + local:write" in response
    assert rtype == ResponseType.DEV_COMMAND
    text_engine_mock.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_quarantined_persona_still_accepts_dev_commands(chat_system_with_mocks):
    """The block gate sits AFTER dev-command preprocessing, so `set tools` can
    still reach a quarantined persona to repair it live."""
    system, _, text_engine_mock, mock_persona, _ = chat_system_with_mocks
    mock_persona._security_block_reasons = ["Insecure composition: ..."]
    system.bot_logic.preprocess_message.return_value = {
        "response": "tools updated", "mutated": True}
    with patch("src.chat_system.save_personas_to_file"):
        response, _, _, _ = await system.generate_response(
            "test_persona", "user", "channel", "set tools get_ticket_details")
    assert response == "tools updated"
    text_engine_mock.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_generate_response_handles_llm_communication_error(chat_system_with_mocks):
    system, _, text_engine_mock, _, _ = chat_system_with_mocks
    text_engine_mock.generate_response.side_effect = LLMCommunicationError("API is down")
    response, _, _ , _ = await system.generate_response("test_persona", "user", "channel", "test")
    assert "Error while generating a response:" in response


@pytest.mark.asyncio
async def test_generate_response_stores_payload_on_llm_error(chat_system_with_mocks):
    """
    Ensures that if the LLM raises an error, the prepared API payload is
    still stored for debugging purposes via `dump_last`.
    """
    system, _, text_engine_mock, _, _ = chat_system_with_mocks
    failed_payload = {"model": "mock_model", "messages": ["This is the context"]}
    # Configure the mock TextEngine to raise an error that includes the payload
    text_engine_mock.generate_response.side_effect = LLMCommunicationError(
        "API is down",
        api_payload=failed_payload
    )

    # We don't need the response, just to trigger the error handling
    await system.generate_response("test_persona", "user123", "channel", "test message")

    # Assert that the payload from the exception was stored. store_api_request
    # adds a `_tools_for_llm` key and (DP-225) scrubs into a fresh dict, so the
    # cached copy is no longer the same object as failed_payload — compare the
    # meaningful content rather than asserting object equality.
    assert "user123" in system.turn_persistence.last_api_requests
    assert "test_persona" in system.turn_persistence.last_api_requests["user123"]
    stored = system.turn_persistence.last_api_requests["user123"]["test_persona"]
    assert stored is not None
    assert stored["model"] == failed_payload["model"]
    assert stored["messages"] == failed_payload["messages"]


def test_store_api_request_preserves_tools_across_iterations(chat_system_with_mocks):
    """tools_for_llm from iteration 0 must survive subsequent stores that pass None."""
    system, _, _, _, _ = chat_system_with_mocks
    tools = [{"type": "function", "function": {"name": "web_search"}}]

    # Iteration 0: stores tools
    system.turn_persistence.store_api_request("user1", "persona1", {"model": "m1"}, tools_for_llm=tools)
    assert system.turn_persistence.last_api_requests["user1"]["persona1"]["_tools_for_llm"] is tools

    # Iteration 1+: tools_for_llm=None, but should carry forward
    system.turn_persistence.store_api_request("user1", "persona1", {"model": "m1"}, tools_for_llm=None)
    assert system.turn_persistence.last_api_requests["user1"]["persona1"]["_tools_for_llm"] is tools


def test_store_api_request_no_tools_when_never_set(chat_system_with_mocks):
    """If tools were never stored, subsequent None calls should not invent them."""
    system, _, _, _, _ = chat_system_with_mocks
    system.turn_persistence.store_api_request("user1", "persona1", {"model": "m1"}, tools_for_llm=None)
    assert "_tools_for_llm" not in system.turn_persistence.last_api_requests["user1"]["persona1"]


@pytest.mark.asyncio
async def test_generate_response_handles_generic_exception(chat_system_with_mocks):
    system, memory_manager, _, _, _ = chat_system_with_mocks
    memory_manager.get_channel_history.side_effect = Exception("DB is locked")
    response, _, _ , _ = await system.generate_response("test_persona", "user", "channel", "test")
    # DP-280: opaque "internal error" gave no signal. The surfaced message now
    # carries the exception class, a correlation ref, and the short detail.
    assert "Internal error" in response
    assert "[Exception]" in response
    assert "DB is locked" in response
    assert re.search(r"ref [0-9a-f]{8}", response)


@pytest.mark.asyncio
async def test_generate_response_exits_after_max_tool_calls(chat_system_with_mocks):
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])
    tool_call = {'type': 'tool_calls', 'calls': [{'id': 'c1', 'name': 'test_tool', 'arguments': {}}]}
    # Make the text engine always return a tool call
    text_engine_mock.generate_response.return_value = (tool_call, {})
    tool_manager_mock.execute_tool.return_value = {"result": "ok"}
    response, _, _ , _ = await system.generate_response("test_persona", "user", "channel", "test")
    assert "stuck in a loop" in response
    # Called exactly MAX_TOOL_CALLS times
    assert text_engine_mock.generate_response.call_count == MAX_TOOL_CALLS


# --- Existing High-Level and Formatting Tests ---

@pytest.mark.asyncio
async def test_tool_use_in_autonomous_mode(chat_system_with_mocks):
    """In AUTONOMOUS mode, write tool calls still gate for audit (universal write-audit model)."""
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.AUTONOMOUS)
    persona.set_enabled_tools(['*'])

    tool_call = {'type': 'tool_calls',
                 'calls': [{'id': 'call_1', 'name': 'update_ticket', 'arguments': {'state': 'closed'}}]}
    text_engine_mock.generate_response.side_effect = [
        (tool_call, {}),
        ({'type': 'text', 'content': 'Proposed.'}, {}),
    ]

    await system.generate_response('test_persona', 'user', 'channel', 'close ticket')

    assert len(system.confirmations.list_for('user', 'test_persona')) == 1
    tool_manager_mock.execute_tool.assert_not_called()


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
    system, _, _, _, _ = chat_system_with_mocks
    formatted = system.request_builder.format_raw_history_for_llm(history, mode, persona, server_id)
    assert len(formatted) == 1
    assert formatted[0]['role'] == expected_role
    assert formatted[0]['content'] == expected_content


# --- CONFIRM Mode Tests ---

@pytest.mark.asyncio
async def test_confirm_mode_gates_write_tools(chat_system_with_mocks):
    """A write tool call is gated, not executed — and the turn still finishes
    with the model's own text (DP-297; it used to end on PENDING_CONFIRMATION).
    """
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    tool_call = {'type': 'tool_calls',
                 'calls': [{'id': 'call_1', 'name': 'update_ticket', 'arguments': {'state': 'closed'}}]}
    text_engine_mock.generate_response.side_effect = [
        (tool_call, {}),
        ({'type': 'text', 'content': 'Queued that for your approval.'}, {}),
    ]

    response, response_type, _ , _ = await system.generate_response('test_persona', 'user', 'channel', 'close it')

    assert response_type == ResponseType.LLM_GENERATION
    assert response == 'Queued that for your approval.'
    tool_manager_mock.execute_tool.assert_not_called()

    parks = system.confirmations.list_for('user', 'test_persona')
    assert len(parks) == 1
    assert parks[0].write_call['name'] == 'update_ticket'
    assert 'update_ticket' in parks[0].confirmation_text


@pytest.mark.asyncio
async def test_confirm_mode_auto_executes_read_only_tools(chat_system_with_mocks):
    """In CONFIRM mode, read-only tools should execute immediately without confirmation."""
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    tool_call = {'type': 'tool_calls',
                 'calls': [{'id': 'call_1', 'name': 'search_tickets', 'arguments': {'query': 'test'}}]}
    final_response = {'type': 'text', 'content': 'Found 3 tickets.'}
    text_engine_mock.generate_response.side_effect = [(tool_call, {}), (final_response, {})]
    tool_manager_mock.execute_tool.return_value = {"result": [{"id": 1}, {"id": 2}, {"id": 3}]}

    response, response_type, _ , _ = await system.generate_response('test_persona', 'user', 'channel', 'search tickets')

    assert response_type == ResponseType.LLM_GENERATION
    assert response == 'Found 3 tickets.'
    tool_manager_mock.execute_tool.assert_called_once_with('search_tickets', query='test')


@pytest.mark.asyncio
async def test_confirm_mode_mixed_tools_executes_reads_and_pends_writes(chat_system_with_mocks):
    """In CONFIRM mode with mixed read+write tools, reads execute and writes pend."""
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    tool_call = {'type': 'tool_calls',
                 'calls': [
                     {'id': 'call_1', 'name': 'search_tickets', 'arguments': {'query': 'test'}},
                     {'id': 'call_2', 'name': 'update_ticket', 'arguments': {'ticket_id': 1, 'state': 'closed'}}
                 ]}
    text_engine_mock.generate_response.side_effect = [
        (tool_call, {}),
        ({'type': 'text', 'content': 'Found one; proposed the close.'}, {}),
    ]
    tool_manager_mock.execute_tool.return_value = {"result": [{"id": 1}]}

    await system.generate_response('test_persona', 'user', 'channel', 'find and close')

    parks = system.confirmations.list_for('user', 'test_persona')
    assert [p.write_call['name'] for p in parks] == ['update_ticket']
    # Read tool was executed, write tool was not
    tool_manager_mock.execute_tool.assert_called_once_with('search_tickets', query='test')


@pytest.mark.asyncio
async def _park_one_write(system, text_engine_mock, tool_name='update_ticket',
                          call_id='call_1', arguments=None):
    """Drive a turn that gates exactly one write, and return its token."""
    text_engine_mock.generate_response.side_effect = [
        ({'type': 'tool_calls',
          'calls': [{'id': call_id, 'name': tool_name,
                     'arguments': arguments if arguments is not None
                     else {'state': 'closed'}}]}, {}),
        ({'type': 'text', 'content': 'Proposed.'}, {}),
    ]
    await system.generate_response('test_persona', 'user', 'channel', 'close it')
    return only_pending_token(system, 'user', 'test_persona')


@pytest.mark.asyncio
async def test_approving_a_park_executes_and_summarizes(chat_system_with_mocks):
    """Approval executes the gated write, then runs a summary turn."""
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    token = await _park_one_write(system, text_engine_mock)

    tool_manager_mock.execute_tool.return_value = {"result": {"id": 1, "state": "closed"}}
    text_engine_mock.generate_response.side_effect = None
    text_engine_mock.generate_response.return_value = (
        {'type': 'text', 'content': 'Done, ticket closed.'}, {},
    )

    response, response_type, _, _ = await system.resolve_park(
        'user', 'test_persona', token, approved=True,
    )

    assert response_type == ResponseType.LLM_GENERATION
    assert response == 'Done, ticket closed.'
    tool_manager_mock.execute_tool.assert_called_with('update_ticket', state='closed')
    # Resolved parks leave the pending set.
    assert system.confirmations.list_for('user', 'test_persona') == []


@pytest.mark.asyncio
async def test_denying_a_park_executes_nothing(chat_system_with_mocks):
    """Denial runs no tool but still drives a summary turn."""
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    token = await _park_one_write(system, text_engine_mock)

    text_engine_mock.generate_response.side_effect = None
    text_engine_mock.generate_response.return_value = (
        {'type': 'text', 'content': "Understood, I won't close the ticket."}, {},
    )

    response, response_type, _, _ = await system.resolve_park(
        'user', 'test_persona', token, approved=False,
    )

    assert response_type == ResponseType.LLM_GENERATION
    tool_manager_mock.execute_tool.assert_not_called()
    assert "won't close" in response


@pytest.mark.asyncio
async def test_continuation_can_chain_a_read(chat_system_with_mocks):
    """After approving a write, the continuation may emit a chained read (e.g.
    a read-back). It must drive ToolLoop to completion so the read executes and
    the final text reaches the user.
    """
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    token = await _park_one_write(system, text_engine_mock)

    tool_manager_mock.execute_tool.side_effect = [
        {"result": {"id": 1, "state": "closed"}},   # update_ticket result
        {"result": {"id": 1, "state": "closed"}},   # get_ticket_details result
    ]
    text_engine_mock.generate_response.side_effect = [
        ({'type': 'tool_calls',
          'calls': [{'id': 'call_2', 'name': 'get_ticket_details', 'arguments': {'ticket_number': 1}}]},
         {}),
        ({'type': 'text', 'content': 'Confirmed: ticket 1 is now closed.'}, {}),
    ]

    response, response_type, _, _ = await system.resolve_park(
        'user', 'test_persona', token, approved=True,
    )

    assert response_type == ResponseType.LLM_GENERATION
    assert response == 'Confirmed: ticket 1 is now closed.'
    # update_ticket (write) + get_ticket_details (chained read) both executed.
    executed = [c.args[0] for c in tool_manager_mock.execute_tool.call_args_list]
    assert executed == ['update_ticket', 'get_ticket_details']


@pytest.mark.asyncio
async def test_resolving_twice_executes_once(chat_system_with_mocks):
    """A double-click cannot execute the same write twice.

    `ConfirmationManager.take` is synchronous, so exactly one caller wins the
    pop; the loser sees the token as already gone.
    """
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    token = await _park_one_write(system, text_engine_mock)

    tool_manager_mock.execute_tool.return_value = {"result": "ok"}
    text_engine_mock.generate_response.side_effect = None
    text_engine_mock.generate_response.return_value = (
        {'type': 'text', 'content': 'Done.'}, {},
    )

    await system.resolve_park('user', 'test_persona', token, approved=True)
    second_text, second_type, _, _ = await system.resolve_park(
        'user', 'test_persona', token, approved=True,
    )

    assert tool_manager_mock.execute_tool.call_count == 1
    assert second_type == ResponseType.DEV_COMMAND
    assert 'No such pending action' in second_text


@pytest.mark.asyncio
async def test_token_from_another_conversation_is_refused(chat_system_with_mocks):
    """A token is only resolvable from the conversation that created it, and a
    refused claim must leave the park live rather than consuming it."""
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_enabled_tools(['*'])

    token = await _park_one_write(system, text_engine_mock)

    text, rtype, _, _ = await system.resolve_park(
        'someone_else', 'test_persona', token, approved=True,
    )

    assert rtype == ResponseType.DEV_COMMAND
    assert 'No such pending action' in text
    tool_manager_mock.execute_tool.assert_not_called()
    # Still live for its real owner.
    assert only_pending_token(system, 'user', 'test_persona') == token


# --- Model Prefix Helper Tests ---

@pytest.mark.parametrize("model_name, expected_prefix", [
    ("gpt-4o", "gpt"),
    ("gpt-3.5-turbo", "gpt"),
    ("claude-3-opus-20240229", "claude"),
    ("claude-3.5-sonnet", "claude"),
    ("gemma-4-31b-it", "gemma"),
    ("gemini-2.5-flash", "gemini"),
    ("gemini-2.5-pro", "gemini"),
    ("gemini-3.1-flash", "gemini-3.1"),
    ("gemini-3.1-pro", "gemini-3.1"),
    ("local", "local"),
    ("some-unknown-model", "unknown"),
])
def test_get_model_prefix(model_name, expected_prefix):
    assert get_model_prefix(model_name) == expected_prefix


# --- Model Compatibility Filter Tests ---

@pytest.mark.asyncio
async def test_grounding_filtered_for_non_grounding_models(chat_system_with_mocks):
    """google_grounding_search should be filtered out for non-grounding-compatible models."""
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
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
@pytest.mark.parametrize("model_name", [
    "gemini-2.5-flash", "gemini-3.1-flash"
])
async def test_grounding_kept_for_compatible_gemini_models(chat_system_with_mocks, model_name):
    """google_grounding_search should be kept for grounding-compatible Gemini models."""
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
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

    persona.set_model_name(model_name)
    await system.generate_response("test_persona", "user", "channel", "test")
    call_args = text_engine_mock.generate_response.call_args
    tools_sent = call_args[1].get('tools', call_args[0][2] if len(call_args[0]) > 2 else [])
    tool_names = [t['function']['name'] for t in tools_sent]
    assert "google_grounding_search" in tool_names
    assert "web_search" in tool_names


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [
    "gpt-4o", "claude-3-opus-20240229", "gemma-4-31b-it", "local",
])
async def test_grounding_filtered_for_incompatible_models(chat_system_with_mocks, model_name):
    """Grounding should be filtered for all incompatible model prefixes."""
    system, _, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])
    persona.set_model_name(model_name)

    grounding_tool = {
        "type": "google_grounding",
        "function": {"name": "google_grounding_search", "description": "Grounding"}
    }
    tool_manager_mock.get_tool_definitions.return_value = [grounding_tool]

    if model_name == "local":
        # DP-206b: local bypasses generate_response entirely — it streams via
        # the engine-owned kobold StreamEngine. Capture tools at that seam.
        captured = {}

        async def _fake_local(config, messages, params, tools=None):
            captured["tools"] = tools
            yield {"type": "api_payload", "payload": {}}
            yield {"type": "done", "full_text": "ok"}

        text_engine_mock.stream_engine = MagicMock()
        text_engine_mock.stream_engine.stream_messages = MagicMock(side_effect=_fake_local)
        await system.generate_response("test_persona", "user", "channel", "test")
        assert len(captured["tools"] or []) == 0
        return

    await system.generate_response("test_persona", "user", "channel", "test")
    call_args = text_engine_mock.generate_response.call_args
    tools_sent = call_args[1].get('tools', call_args[0][2] if len(call_args[0]) > 2 else [])
    assert len(tools_sent) == 0


# --- Service Integration Tests ---

def test_register_service(chat_system_with_mocks):
    """register_service adds a service to the registry and registers its tools."""
    system, _, _, _, _ = chat_system_with_mocks
    mock_service = MagicMock(spec=ServiceIntegration)
    mock_service.name = "test_service"
    system.register_service(mock_service)
    assert system.get_service("test_service") is mock_service
    mock_service.register_tools.assert_called_once_with(system.tool_manager)


# --- Conversation History Tests ---

def test_build_conversation_history_channel_mode(chat_system_with_mocks):
    """Channel-isolated mode fetches channel history."""
    system, memory_mock, _, persona, _ = chat_system_with_mocks
    persona.set_memory_mode(MemoryMode.CHANNEL_ISOLATED)
    memory_mock.get_channel_history.return_value = [
        {'interaction_id': 1, 'author_role': 'user', 'author_name': 'Alice', 'content': 'Hello'}
    ]

    history, oldest_id = system.request_builder.build_conversation_history(persona, 'user', 'general', 'srv1', None)

    memory_mock.get_channel_history.assert_called_once()
    assert len(history) == 1
    assert history[0]['content'] == 'Alice: Hello'
    assert oldest_id == 1


def test_build_conversation_history_ticket_mode_returns_empty(chat_system_with_mocks):
    """TICKET_ISOLATED mode returns empty history (no ticket resolution available)."""
    system, _, _, persona, _ = chat_system_with_mocks
    persona.set_memory_mode(MemoryMode.TICKET_ISOLATED)

    history, oldest_id = system.request_builder.build_conversation_history(persona, 'user', 'ch', None, None)

    assert history == []
    assert oldest_id is None


def test_build_conversation_history_personal_mode(chat_system_with_mocks):
    """Personal mode fetches personal history."""
    system, memory_mock, _, persona, _ = chat_system_with_mocks
    persona.set_memory_mode(MemoryMode.PERSONAL)
    memory_mock.get_personal_history.return_value = []

    system.request_builder.build_conversation_history(persona, 'user123', 'ch', None, None)

    memory_mock.get_personal_history.assert_called_once_with('user123', 'test_persona', persona.get_history_messages())


# --- DP-142: read-only paths must not advance the hello override ---

def test_get_view_history_does_not_advance_override(chat_system_with_mocks):
    """A transcript fetch must not inflate the hello window (DP-142)."""
    system, memory_mock, _, persona, _ = chat_system_with_mocks
    persona.set_memory_mode(MemoryMode.CHANNEL_ISOLATED)
    persona.start_new_conversation(0)
    memory_mock.get_channel_history.return_value = []

    system.get_view_history('test_persona', 'user', 'general', 'srv1')
    system.get_view_history('test_persona', 'user', 'general', 'srv1')

    assert persona.get_current_effective_history_messages() == 0


@pytest.mark.asyncio
async def test_assemble_request_does_not_advance_override(chat_system_with_mocks):
    """The /assemble dry-run must not inflate the hello window (DP-142)."""
    system, memory_mock, _, persona, _ = chat_system_with_mocks
    persona.set_memory_mode(MemoryMode.CHANNEL_ISOLATED)
    persona.start_new_conversation(0)
    memory_mock.get_channel_history.return_value = []

    await system.assemble_request('test_persona', 'user', 'general', 'hi', server_id='srv1')
    await system.assemble_request('test_persona', 'user', 'general', 'hi', server_id='srv1')

    assert persona.get_current_effective_history_messages() == 0


def test_build_conversation_history_live_advances_override(chat_system_with_mocks):
    """The live generation path still advances the hello window (DP-142)."""
    system, memory_mock, _, persona, _ = chat_system_with_mocks
    persona.set_memory_mode(MemoryMode.CHANNEL_ISOLATED)
    persona.start_new_conversation(0)
    memory_mock.get_channel_history.return_value = []

    system.request_builder.build_conversation_history(persona, 'user', 'general', 'srv1', None)
    assert persona.get_current_effective_history_messages() == 2
    system.request_builder.build_conversation_history(persona, 'user', 'general', 'srv1', None)
    assert persona.get_current_effective_history_messages() == 4


# --- DP-142: read-only paths must not advance the hello override ---

def test_get_view_history_does_not_advance_override(chat_system_with_mocks):
    """A transcript fetch must not inflate the hello window (DP-142)."""
    system, memory_mock, _, persona, _ = chat_system_with_mocks
    persona.set_memory_mode(MemoryMode.CHANNEL_ISOLATED)
    persona.start_new_conversation(0)
    memory_mock.get_channel_history.return_value = []

    system.get_view_history('test_persona', 'user', 'general', 'srv1')
    system.get_view_history('test_persona', 'user', 'general', 'srv1')

    assert persona.get_current_effective_history_messages() == 0


@pytest.mark.asyncio
async def test_assemble_request_does_not_advance_override(chat_system_with_mocks):
    """The /assemble dry-run must not inflate the hello window (DP-142)."""
    system, memory_mock, _, persona, _ = chat_system_with_mocks
    persona.set_memory_mode(MemoryMode.CHANNEL_ISOLATED)
    persona.start_new_conversation(0)
    memory_mock.get_channel_history.return_value = []

    await system.assemble_request('test_persona', 'user', 'general', 'hi', server_id='srv1')
    await system.assemble_request('test_persona', 'user', 'general', 'hi', server_id='srv1')

    assert persona.get_current_effective_history_messages() == 0


def test_build_conversation_history_live_advances_override(chat_system_with_mocks):
    """The live generation path still advances the hello window (DP-142)."""
    system, memory_mock, _, persona, _ = chat_system_with_mocks
    persona.set_memory_mode(MemoryMode.CHANNEL_ISOLATED)
    persona.start_new_conversation(0)
    memory_mock.get_channel_history.return_value = []

    system.request_builder.build_conversation_history(persona, 'user', 'general', 'srv1', None)
    assert persona.get_current_effective_history_messages() == 2
    system.request_builder.build_conversation_history(persona, 'user', 'general', 'srv1', None)
    assert persona.get_current_effective_history_messages() == 4


# --- Tool Filtering Tests ---

def test_filter_tools_wildcard(chat_system_with_mocks):
    """Wildcard enabled_tools returns all compatible tools."""
    system, _, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])
    tool_a = {"type": "function", "function": {"name": "tool_a", "parameters": {}}}
    tool_b = {"type": "function", "function": {"name": "tool_b", "parameters": {}}}
    tool_manager_mock.get_tool_definitions.return_value = [tool_a, tool_b]

    result = system.request_builder.filter_tools_for_persona(persona)
    assert len(result) == 2


def test_filter_tools_specific_names(chat_system_with_mocks):
    """Specific enabled_tools filters to only those tools."""
    system, _, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['tool_a'])
    tool_a = {"type": "function", "function": {"name": "tool_a", "parameters": {}}}
    tool_b = {"type": "function", "function": {"name": "tool_b", "parameters": {}}}
    tool_manager_mock.get_tool_definitions.return_value = [tool_a, tool_b]

    result = system.request_builder.filter_tools_for_persona(persona)
    assert len(result) == 1
    assert result[0]['function']['name'] == 'tool_a'


def test_filter_tools_removes_unbound_service_tools(chat_system_with_mocks):
    """Tools with a service_binding are removed when the persona lacks that binding."""
    system, _, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])
    persona.set_service_bindings([])  # No service bindings
    zammad_tool = {"type": "function", "service_binding": "zammad",
                   "function": {"name": "search_tickets", "parameters": {}}}
    other_tool = {"type": "function", "function": {"name": "web_search", "parameters": {}}}
    tool_manager_mock.get_tool_definitions.return_value = [zammad_tool, other_tool]

    result = system.request_builder.filter_tools_for_persona(persona)
    tool_names = [t['function']['name'] for t in result]
    assert 'search_tickets' not in tool_names
    assert 'web_search' in tool_names


def test_filter_tools_includes_bound_service_tools(chat_system_with_mocks):
    """Tools with a service_binding are included when the persona has that binding."""
    system, _, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])
    persona.set_service_bindings(["zammad"])
    zammad_tool = {"type": "function", "service_binding": "zammad",
                   "function": {"name": "search_tickets", "parameters": {}}}
    other_tool = {"type": "function", "function": {"name": "web_search", "parameters": {}}}
    tool_manager_mock.get_tool_definitions.return_value = [zammad_tool, other_tool]

    result = system.request_builder.filter_tools_for_persona(persona)
    tool_names = [t['function']['name'] for t in result]
    assert 'search_tickets' in tool_names
    assert 'web_search' in tool_names


def test_filter_tools_includes_agent_tools_with_agents_binding(chat_system_with_mocks):
    """Agent tools are included when the persona has the 'agents' service binding."""
    system, _, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])
    persona.set_service_bindings(["agents"])
    agent_tool = {"type": "function", "service_binding": "agents",
                  "function": {"name": "get_agent_status", "parameters": {}}}
    zammad_tool = {"type": "function", "service_binding": "zammad",
                   "function": {"name": "search_tickets", "parameters": {}}}
    universal_tool = {"type": "function", "function": {"name": "web_search", "parameters": {}}}
    tool_manager_mock.get_tool_definitions.return_value = [agent_tool, zammad_tool, universal_tool]

    result = system.request_builder.filter_tools_for_persona(persona)
    tool_names = [t['function']['name'] for t in result]
    assert 'get_agent_status' in tool_names
    assert 'web_search' in tool_names
    assert 'search_tickets' not in tool_names


def test_filter_tools_excludes_agent_tools_without_binding(chat_system_with_mocks):
    """Agent tools are excluded when the persona lacks the 'agents' service binding."""
    system, _, _, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])
    persona.set_service_bindings(["zammad"])
    agent_tool = {"type": "function", "service_binding": "agents",
                  "function": {"name": "get_agent_status", "parameters": {}}}
    zammad_tool = {"type": "function", "service_binding": "zammad",
                   "function": {"name": "search_tickets", "parameters": {}}}
    tool_manager_mock.get_tool_definitions.return_value = [agent_tool, zammad_tool]

    result = system.request_builder.filter_tools_for_persona(persona)
    tool_names = [t['function']['name'] for t in result]
    assert 'get_agent_status' not in tool_names
    assert 'search_tickets' in tool_names


# --- Write Call Execution Tests ---

def _decision(approved=True, tool="update_ticket", args=None):
    park = ParkedWrite(
        token="tok1",
        write_call={"id": "c1", "name": tool,
                    "arguments": args if args is not None else {"state": "closed"}},
        audit_info={"actions": []},
        confirmation_text="",
        user_identifier="user",
        persona_name="test_persona",
    )
    return Decision(park=park, approved=approved)


@pytest.mark.asyncio
async def test_apply_executes_an_approved_write(chat_system_with_mocks):
    """An approved decision runs its tool and records the result."""
    system, _, _, _, tool_manager_mock = chat_system_with_mocks
    tool_manager_mock.execute_tool.return_value = {"result": {"id": 60}}

    decision = _decision(approved=True)
    await system.confirmations.apply(decision)

    tool_manager_mock.execute_tool.assert_called_once_with('update_ticket', state='closed')
    assert decision.ok is True
    assert decision.result == {"result": {"id": 60}}


@pytest.mark.asyncio
async def test_apply_records_a_denial_without_executing(chat_system_with_mocks):
    """A denied decision runs nothing and records the refusal as its result."""
    system, _, _, _, tool_manager_mock = chat_system_with_mocks

    decision = _decision(approved=False)
    await system.confirmations.apply(decision)

    tool_manager_mock.execute_tool.assert_not_called()
    assert decision.ok is False
    assert "denied" in decision.result["error"].lower()


@pytest.mark.asyncio
async def test_apply_survives_a_failing_tool(chat_system_with_mocks):
    """A raising tool must not escape `apply` — the decision is still resolved,
    marked failed, and reported, or the park would be consumed with the
    operator told nothing."""
    system, _, _, _, tool_manager_mock = chat_system_with_mocks
    tool_manager_mock.execute_tool.side_effect = RuntimeError("zammad down")

    decision = _decision(approved=True)
    await system.confirmations.apply(decision)

    assert decision.ok is False
    assert "zammad down" in decision.result["error"]


@pytest.mark.asyncio
async def test_confirmations_see_post_init_tool_manager_swap(chat_system_with_mocks):
    """ConfirmationManager resolves the tool manager per call (lookup closure,
    like RequestBuilder.persona_lookup): a post-init rebind of
    chat_system.tool_manager must be what approved writes execute against."""
    system, _, _, _, original_tm = chat_system_with_mocks
    swapped_tm = AsyncMock()
    swapped_tm.execute_tool.return_value = {"ok": True}
    system.tool_manager = swapped_tm

    await system.confirmations.apply(_decision(approved=True))

    swapped_tm.execute_tool.assert_called_once_with("update_ticket", state="closed")
    original_tm.execute_tool.assert_not_called()


# --- Orchestration Method Tests ---

@pytest.mark.asyncio
async def test_prepare_request_populates_context(chat_system_with_mocks):
    """_prepare_request resolves service contexts, builds history, filters tools, appends user message."""
    system, memory_mock, _, persona, tool_manager_mock = chat_system_with_mocks
    memory_mock.get_channel_history.return_value = []
    ctx = RequestContext(
        persona=persona, persona_name='test_persona', user_identifier='user',
        channel='general', message='hello', server_id='srv1',
    )

    await system.request_builder.prepare_request(ctx)

    assert ctx.conversation_history[-1] == {"role": "user", "content": "hello"}


@pytest.mark.asyncio
async def test_prepare_request_prunes_to_max_context_tokens(chat_system_with_mocks):
    """Phase 3: oversized history is pruned to fit max_context_tokens - response_token_limit."""
    system, memory_mock, _, persona, _ = chat_system_with_mocks
    persona.set_response_token_limit(100)
    persona.set_max_context_tokens(200)

    big = "x" * 600  # 150 tokens char/4
    memory_mock.get_channel_history.return_value = [
        {"author_role": "user", "content": big, "interaction_id": 1, "timestamp": "t", "author_name": "u"},
        {"author_role": "assistant", "content": big, "interaction_id": 2, "timestamp": "t", "author_name": "a"},
        {"author_role": "user", "content": big, "interaction_id": 3, "timestamp": "t", "author_name": "u"},
        {"author_role": "assistant", "content": big, "interaction_id": 4, "timestamp": "t", "author_name": "a"},
    ]
    ctx = RequestContext(
        persona=persona, persona_name='test_persona', user_identifier='user',
        channel='general', message='latest user msg', server_id='srv1',
    )

    await system.request_builder.prepare_request(ctx)

    assert ctx.conversation_history[-1]["content"] == "latest user msg"
    assert len(ctx.conversation_history) < 5  # at least one drop


# --- Orphaned tool head (DP-298) ---

def _orphan_history_rows():
    """A DB window that opens mid-turn: the assistant row's tool_context expands
    into a tool_calls/tool pair whose triggering user row fell off the window."""
    tool_context = json.dumps([
        {"role": "assistant", "tool_calls": [{"id": "c1", "name": "inspect_agents", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "inspect_agents", "content": "{}"},
    ])
    return [
        {"author_role": "assistant", "content": "agent dispatched", "author_name": "test_persona",
         "tool_context": tool_context, "interaction_id": 2, "timestamp": "t"},
        {"author_role": "user", "content": "and now?", "author_name": "u",
         "interaction_id": 3, "timestamp": "t"},
    ]


@pytest.mark.asyncio
async def test_prepare_request_drops_orphaned_tool_head_from_db_window(chat_system_with_mocks):
    """The DB window counts rows, not turns, so it can open on an assistant row
    whose tool_context expands to a tool_calls/tool pair with no preceding user
    turn — Google rejects that with a 400.

    DP-296 now also guards this at the source (`build_conversation_history`
    refuses to replay tool_context that no user/tool turn precedes), so on this
    path the repair is belt-and-braces: the assertion is that nothing unpaired
    reaches the wire, not that this particular guard is the one that acted.
    The path where `drop_orphaned_tool_head` is still load-bearing is
    `client_messages` — see the test below."""
    system, memory_mock, _, persona, _ = chat_system_with_mocks

    memory_mock.get_channel_history.return_value = _orphan_history_rows()
    ctx = RequestContext(
        persona=persona, persona_name='test_persona', user_identifier='user',
        channel='general', message='latest user msg', server_id='srv1',
    )

    await system.request_builder.prepare_request(ctx)

    head = ctx.conversation_history[0]
    assert head.get("role") != "tool"
    assert not head.get("tool_calls")
    assert not any(m.get("role") == "tool" or m.get("tool_calls")
                   for m in ctx.conversation_history)
    assert ctx.conversation_history[-1] == {"role": "user", "content": "latest user msg"}


@pytest.mark.asyncio
async def test_prepare_request_drops_orphaned_tool_head_from_client_messages(chat_system_with_mocks):
    """DP-298: the kobold-lite path supplies its own message array, which
    `prepare_request` uses verbatim apart from stripping ONE leading system
    message. DP-296's source guard lives in the DB branch and does not run here,
    so a client window that opens mid-tool-sequence reaches the provider unless
    the head repair catches it. This is the path the repair is load-bearing on."""
    system, memory_mock, _, persona, _ = chat_system_with_mocks

    memory_mock.get_channel_history.return_value = []
    ctx = RequestContext(
        persona=persona, persona_name='test_persona', user_identifier='user',
        channel='general', message='latest user msg', server_id='srv1',
        client_messages=[
            # kobold-lite's window opened after the user turn that triggered it
            {"role": "tool", "tool_call_id": "c1", "name": "inspect_agents", "content": "{}"},
            {"role": "assistant", "content": "agent dispatched"},
            {"role": "user", "content": "and now?"},
        ],
    )

    await system.request_builder.prepare_request(ctx)

    assert ctx.conversation_history[0].get("role") != "tool"
    assert not any(m.get("role") == "tool" or m.get("tool_calls")
                   for m in ctx.conversation_history)
    assert ctx.conversation_history[-1] == {"role": "user", "content": "latest user msg"}


@pytest.mark.asyncio
async def test_prepare_request_drops_orphaned_tool_head_behind_ltm_block(chat_system_with_mocks):
    """DP-298: the LTM recall block is inserted at index 0 *after* the history is
    built, so it sits in front of the orphan. It is not a real user turn and must
    not shield it — otherwise the exact case this repair exists for (recall hit +
    a window opening on tool traffic) still reaches the provider and 400s."""
    system, memory_mock, _, persona, _ = chat_system_with_mocks

    memory_mock.get_channel_history.return_value = []
    system.request_builder.retrieve_memory_block = AsyncMock(  # type: ignore[method-assign]
        return_value=("[Recalled context] the user asked about agents earlier.", False),
    )
    ctx = RequestContext(
        persona=persona, persona_name='test_persona', user_identifier='user',
        channel='general', message='latest user msg', server_id='srv1',
        client_messages=[
            {"role": "tool", "tool_call_id": "c1", "name": "inspect_agents", "content": "{}"},
            {"role": "assistant", "content": "agent dispatched"},
            {"role": "user", "content": "and now?"},
        ],
    )

    await system.request_builder.prepare_request(ctx)

    # The recall block is preserved at the head — only the orphan is removed.
    assert ctx.conversation_history[0]["content"].startswith("[Recalled context]")
    assert not any(m.get("role") == "tool" or m.get("tool_calls")
                   for m in ctx.conversation_history)
    assert ctx.conversation_history[-1] == {"role": "user", "content": "latest user msg"}


# --- Internal Logging Tests ---

@pytest.mark.asyncio
async def test_generate_response_logs_user_and_assistant(chat_system_with_mocks):
    """generate_response logs both user and assistant messages via memory_manager."""
    system, memory_mock, _, _, _ = chat_system_with_mocks
    memory_mock.get_channel_history.return_value = []
    memory_mock.log_message.return_value = 42

    _, response_type, _, _ = await system.generate_response(
        "test_persona", "user", "channel", "hello",
        platform_message_id="msg_1", user_display_name="Alice"
    )

    assert response_type == ResponseType.LLM_GENERATION
    assert memory_mock.log_message.call_count == 2

    # First call: user message
    user_call = memory_mock.log_message.call_args_list[0]
    assert user_call.kwargs['author_role'] == 'user'
    assert user_call.kwargs['content'] == 'hello'
    assert user_call.kwargs['platform_message_id'] == 'msg_1'
    assert user_call.kwargs['author_name'] == 'Alice'

    # Second call: assistant message
    bot_call = memory_mock.log_message.call_args_list[1]
    assert bot_call.kwargs['author_role'] == 'assistant'
    assert bot_call.kwargs['author_name'] == 'test_persona'
    assert bot_call.kwargs['content'] == 'LLM Reply'


@pytest.mark.asyncio
async def test_generate_response_logs_tool_context(chat_system_with_mocks):
    """Tool loop produces stored tool_context JSON on the assistant log entry."""
    system, memory_mock, text_engine_mock, persona, tool_manager_mock = chat_system_with_mocks
    memory_mock.get_channel_history.return_value = []
    memory_mock.log_message.return_value = 1
    persona.set_enabled_tools(['*'])

    tool_call = {'type': 'tool_calls',
                 'calls': [{'id': 'c1', 'name': 'web_search', 'arguments': {'query': 'test'}}]}
    final_response = {'type': 'text', 'content': 'Search result.'}
    text_engine_mock.generate_response.side_effect = [(tool_call, {}), (final_response, {})]
    tool_manager_mock.execute_tool.return_value = {"result": "data"}

    await system.generate_response("test_persona", "user", "channel", "search")

    # Assistant log call should include tool_context
    bot_call = memory_mock.log_message.call_args_list[1]
    tool_ctx = bot_call.kwargs.get('tool_context')
    assert tool_ctx is not None
    parsed = json.loads(tool_ctx)
    # Should contain the assistant tool_calls message and the tool result
    roles = [m.get('role') for m in parsed]
    assert 'assistant' in roles or any('tool_calls' in m for m in parsed)
    assert 'tool' in roles


@pytest.mark.asyncio
async def test_generate_response_no_tool_context_without_tools(chat_system_with_mocks):
    """When no tools are used, tool_context should be None."""
    system, memory_mock, _, _, _ = chat_system_with_mocks
    memory_mock.get_channel_history.return_value = []
    memory_mock.log_message.return_value = 1

    await system.generate_response("test_persona", "user", "channel", "hello")

    bot_call = memory_mock.log_message.call_args_list[1]
    assert bot_call.kwargs.get('tool_context') is None


@pytest.mark.asyncio
async def test_generate_response_returns_interaction_id(chat_system_with_mocks):
    """generate_response returns the assistant interaction_id for platform_message_id update."""
    system, memory_mock, _, _, _ = chat_system_with_mocks
    memory_mock.get_channel_history.return_value = []
    memory_mock.log_message.side_effect = [10, 42]  # user id, assistant id

    _, _, assistant_id, _ = await system.generate_response(
        "test_persona", "user", "channel", "hello"
    )

    assert assistant_id == 42


def test_format_raw_history_injects_tool_context(chat_system_with_mocks):
    """Tool context JSON is reconstructed into the history before the assistant message."""
    system, _, _, _, _ = chat_system_with_mocks
    tool_ctx = json.dumps([
        {"role": "assistant", "tool_calls": [{"id": "c1", "name": "web_search", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "web_search", "content": '{"result": "ok"}'}
    ])
    raw_history = [
        {'author_role': 'user', 'author_name': 'Alice', 'content': 'search please'},
        {'author_role': 'assistant', 'author_name': 'test_persona', 'content': 'Here are results',
         'tool_context': tool_ctx},
    ]

    formatted = system.request_builder.format_raw_history_for_llm(raw_history, "channel", "test_persona", None)

    # user + tool_call assistant + tool result + final assistant = 4 messages
    assert len(formatted) == 4
    assert formatted[0]['role'] == 'user'
    assert 'tool_calls' in formatted[1]
    assert formatted[2]['role'] == 'tool'
    assert formatted[3] == {'role': 'assistant', 'content': 'Here are results'}


def test_format_raw_history_drops_orphaned_tool_context(chat_system_with_mocks):
    """DP-296: a replayed tool block must never open the wire array.

    The history limit slices raw rows, so an assistant row carrying
    tool_context can end up oldest. Replaying it there puts a function call
    first and Gemini rejects the whole request ("function call turn must come
    immediately after a user turn or after a function response turn"), which
    took out four consecutive attempts in production on 2026-07-26.
    """
    system, _, _, _, _ = chat_system_with_mocks
    tool_ctx = json.dumps([
        {"role": "assistant", "tool_calls": [{"id": "c1", "name": "web_search", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "web_search", "content": '{"result": "ok"}'}
    ])
    # No preceding user row survived the limit.
    raw_history = [
        {'author_role': 'assistant', 'author_name': 'test_persona', 'content': 'Here are results',
         'tool_context': tool_ctx},
        {'author_role': 'user', 'author_name': 'Alice', 'content': 'thanks'},
    ]

    formatted = system.request_builder.format_raw_history_for_llm(raw_history, "channel", "test_persona", None)

    assert formatted[0] == {'role': 'assistant', 'content': 'Here are results'}
    assert not any('tool_calls' in m for m in formatted)


def test_format_raw_history_tool_context_after_assistant_is_dropped(chat_system_with_mocks):
    """Two assistant rows in a row: the second one's tool block would follow an
    assistant turn, which is equally invalid. Drop it too."""
    system, _, _, _, _ = chat_system_with_mocks
    tool_ctx = json.dumps([
        {"role": "assistant", "tool_calls": [{"id": "c1", "name": "web_search", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "web_search", "content": '{"result": "ok"}'}
    ])
    raw_history = [
        {'author_role': 'user', 'author_name': 'Alice', 'content': 'search please'},
        {'author_role': 'assistant', 'author_name': 'test_persona', 'content': 'One moment',
         'tool_context': None},
        {'author_role': 'assistant', 'author_name': 'test_persona', 'content': 'Here are results',
         'tool_context': tool_ctx},
    ]

    formatted = system.request_builder.format_raw_history_for_llm(raw_history, "channel", "test_persona", None)

    assert [m['role'] for m in formatted] == ['user', 'assistant', 'assistant']


def test_format_raw_history_tool_context_after_park_row_is_kept(chat_system_with_mocks):
    """DP-296 review: the provider allows a function-call turn after a user turn
    *or after a function-response turn*, and a park row is exactly the latter —
    it has empty content, so the block it emits ends on a `tool` message.
    Accepting only 'user' would drop the tool context of every row following a
    park, which is the memory this feature exists to keep.
    """
    system, _, _, _, _ = chat_system_with_mocks
    park_ctx = json.dumps([
        {"role": "assistant", "tool_calls": [{"id": "c1", "name": "update_ticket", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "update_ticket",
         "content": '{"status": "not_executed", "reason": "awaiting_approval"}'},
    ])
    resumed_ctx = json.dumps([
        {"role": "assistant", "tool_calls": [{"id": "c2", "name": "update_ticket", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "update_ticket", "content": '{"ok": true}'},
    ])
    raw_history = [
        {'author_role': 'user', 'author_name': 'Alice', 'content': 'close the ticket'},
        # The park: tool context only, no renderable text.
        {'author_role': 'assistant', 'author_name': 'test_persona', 'content': '',
         'tool_context': park_ctx},
        {'author_role': 'assistant', 'author_name': 'test_persona', 'content': 'Closed it.',
         'tool_context': resumed_ctx},
    ]

    formatted = system.request_builder.format_raw_history_for_llm(raw_history, "channel", "test_persona", None)

    assert [m['role'] for m in formatted] == [
        'user', 'assistant', 'tool', 'assistant', 'tool', 'assistant',
    ]
    assert [c['id'] for m in formatted if m.get('tool_calls')
            for c in m['tool_calls']] == ['c1', 'c2']


def test_format_raw_history_no_tool_context(chat_system_with_mocks):
    """NULL tool_context produces no extra messages."""
    system, _, _, _, _ = chat_system_with_mocks
    raw_history = [
        {'author_role': 'assistant', 'author_name': 'test_persona', 'content': 'Hi',
         'tool_context': None},
    ]

    formatted = system.request_builder.format_raw_history_for_llm(raw_history, "channel", "test_persona", None)

    assert len(formatted) == 1
    assert formatted[0] == {'role': 'assistant', 'content': 'Hi'}


# --- Phase C: stream_response + _orchestrate kernel ---

async def _drain_events(stream):
    out = []
    async for ev in stream:
        out.append(ev)
    return out


@pytest.mark.asyncio
async def test_stream_response_yields_token_then_done_for_text(chat_system_with_mocks):
    """Plain text generation surfaces TokenEvent(s) then a DoneEvent."""
    system, memory_mock, _, _, _ = chat_system_with_mocks
    memory_mock.get_channel_history.return_value = []
    memory_mock.log_message.side_effect = [10, 42]

    events = await _drain_events(
        system.stream_response("test_persona", "user", "channel", "hi")
    )
    types = [type(e).__name__ for e in events]
    assert types[0] == "TokenEvent"
    assert types[-1] == "DoneEvent"
    assert events[0].delta == "LLM Reply"
    assert events[-1].text == "LLM Reply"
    assert events[-1].response_type == ResponseType.LLM_GENERATION
    assert events[-1].assistant_id == 42
    assert events[-1].user_interaction_id == 10


@pytest.mark.asyncio
async def test_stream_response_dev_command_skips_token_events(chat_system_with_mocks):
    """Dev command short-circuits before any TokenEvent."""
    system, _, _, _, _ = chat_system_with_mocks
    system.bot_logic.preprocess_message.return_value = {
        "response": "Dev output", "mutated": False,
    }
    events = await _drain_events(
        system.stream_response("test_persona", "user", "channel", "what model")
    )
    assert len(events) == 1
    assert isinstance(events[0], DoneEvent)
    assert events[0].text == "Dev output"
    assert events[0].response_type == ResponseType.DEV_COMMAND




@pytest.mark.asyncio
async def test_stream_response_emits_error_on_llm_failure(chat_system_with_mocks):
    """LLMCommunicationError surfaces as a single ErrorEvent."""
    system, memory_mock, text_engine, _, _ = chat_system_with_mocks
    memory_mock.get_channel_history.return_value = []
    text_engine.generate_response.side_effect = LLMCommunicationError("boom")

    events = await _drain_events(
        system.stream_response("test_persona", "user", "channel", "hi")
    )
    err = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(err) == 1
    assert "Error while generating" in err[0].message
    # No DoneEvent on error
    assert not any(isinstance(e, DoneEvent) for e in events)


@pytest.mark.asyncio
async def test_stream_response_is_retry_archives_via_handle_portal_retry(
    chat_system_with_mocks,
):
    """is_retry=True archives the prior assistant row and updates in place."""
    system, memory_mock, _, _, _ = chat_system_with_mocks
    memory_mock.get_channel_history.return_value = []
    memory_mock.handle_portal_retry.return_value = 99
    memory_mock.update_interaction_content.return_value = True

    events = await _drain_events(
        system.stream_response(
            "test_persona", "portal", "web_ui", "ignored",
            is_retry=True,
        )
    )
    done = [e for e in events if isinstance(e, DoneEvent)][-1]
    memory_mock.handle_portal_retry.assert_called_once_with(
        persona_name="test_persona",
        user_identifier="portal",
        channel="web_ui",
    )
    # Retry forwards the regenerated turn's tool_context (None here — this turn
    # used no tools — which correctly clears any stale tools on the row) and
    # explicitly clears reasoning_content (DP-141: the prior attempt's `<think>`
    # no longer matches the regenerated text; omitting would now preserve it).
    memory_mock.update_interaction_content.assert_called_once_with(
        99, "LLM Reply", reasoning_content=None, tool_context=None)
    # No new user log_message on retry path
    log_calls = [c for c in memory_mock.log_message.call_args_list
                 if c.kwargs.get('author_role') == 'user']
    assert log_calls == []
    assert done.assistant_id == 99
    assert done.user_interaction_id is None


@pytest.mark.asyncio
async def test_stream_response_is_retry_pops_trailing_assistant(chat_system_with_mocks):
    """On retry, history from DB ends with the about-to-be-overwritten
    assistant row. Kernel must pop it from `messages_for_llm` so the model
    regenerates from the user turn instead of continuing its own response,
    and must NOT append a fresh user turn (DB already terminates with one).
    """
    system, memory_mock, text_engine, _, _ = chat_system_with_mocks
    # DB ends with a user/assistant pair — the prior turn being retried.
    memory_mock.get_channel_history.return_value = [
        {"author_role": "user", "author_name": "user", "content": "prior prompt", "interaction_id": 1},
        {"author_role": "assistant", "author_name": "test_persona", "content": "first attempt", "interaction_id": 2},
    ]
    memory_mock.handle_portal_retry.return_value = 2
    memory_mock.update_interaction_content.return_value = True

    await _drain_events(
        system.stream_response(
            "test_persona", "portal", "web_ui", "prior prompt",
            is_retry=True,
        )
    )

    # text_engine.generate_response is the AsyncMock under stream_messages.
    text_engine.generate_response.assert_called_once()
    history_object = text_engine.generate_response.call_args.args[1]
    history = history_object["message_history"]
    # Trailing assistant ("first attempt") must have been popped, leaving
    # the prior user turn as the last message — and no duplicate user.
    assert history[-1].get("role") == "user"
    assert history[-1].get("content") == "prior prompt"
    user_count = sum(1 for m in history if m.get("role") == "user")
    assert user_count == 1
    assistant_count = sum(1 for m in history if m.get("role") == "assistant")
    assert assistant_count == 0


@pytest.mark.asyncio
async def test_stream_response_interleaves_tool_events_with_tokens(chat_system_with_mocks):
    """tool_revamp_v1: tool-enabled persona surfaces ToolCallStart /
    ToolCallResult between TokenEvent runs in a single linear stream."""
    system, memory_mock, text_engine, persona, tool_manager_mock = chat_system_with_mocks
    persona.set_enabled_tools(['*'])
    memory_mock.get_channel_history.return_value = []
    memory_mock.log_message.side_effect = [10, 42]

    tool_call = {'type': 'tool_calls',
                 'calls': [{'id': 'call_xyz', 'name': 'search_tickets',
                            'arguments': {'query': 'open'}}]}
    final = {'type': 'text', 'content': 'Found two tickets.'}
    text_engine.generate_response.side_effect = [(tool_call, {}), (final, {})]
    tool_manager_mock.execute_tool.return_value = {"result": [{"id": 1}, {"id": 2}]}

    events = await _drain_events(
        system.stream_response("test_persona", "user", "channel", "find tickets")
    )
    types = [type(e).__name__ for e in events]
    # ToolCallStart precedes ToolCallResult; both precede the final
    # TokenEvent + DoneEvent (no parallel calls in this scenario).
    start_idx = types.index("ToolCallStartEvent")
    result_idx = types.index("ToolCallResultEvent")
    token_idx = types.index("TokenEvent")
    done_idx = types.index("DoneEvent")
    assert start_idx < result_idx < token_idx < done_idx

    start = events[start_idx]
    result = events[result_idx]
    assert start.tool_name == "search_tickets"
    assert start.arguments == {"query": "open"}
    assert start.call_id == "call_xyz"
    assert result.call_id == start.call_id
    assert result.error is None
    assert events[done_idx].text == "Found two tickets."


@pytest.mark.asyncio
async def test_generate_response_round_trips_through_orchestrate(chat_system_with_mocks):
    """generate_response is now a collect-stream wrapper — same 4-tuple."""
    system, memory_mock, _, _, _ = chat_system_with_mocks
    memory_mock.get_channel_history.return_value = []
    memory_mock.log_message.side_effect = [11, 22]

    text, rtype, aid, uid = await system.generate_response(
        "test_persona", "user", "channel", "hello",
    )
    assert text == "LLM Reply"
    assert rtype == ResponseType.LLM_GENERATION
    assert aid == 22
    assert uid == 11
