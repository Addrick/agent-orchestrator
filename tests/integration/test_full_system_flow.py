# tests/integration/test_full_system_flow.py
#
# Multi-component integration tests with mocked external services.
# Zammad-dependent tests have been moved to tests/live/test_full_system_zammad.py.

import pytest
from datetime import datetime
from unittest.mock import patch, AsyncMock, MagicMock

from src.chat_system import ChatSystem, ResponseType
from src.clients.zammad_service import ZammadIntegration
from src.persona import MemoryMode, ExecutionMode

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_dynamic_context_ignores_dev_commands(mocked_chat_system):
    chat_system, memory_manager = mocked_chat_system
    persona = chat_system.personas['test_persona']
    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=({'type': 'text', 'content': '...'}, {})):
        await chat_system.generate_response("test_persona", "user1", "channel", "hello")
        await chat_system.generate_response("test_persona", "user1", "channel", "first message")
        assert persona.get_current_effective_context_length() == 2
        await chat_system.generate_response("test_persona", "user1", "channel", "what model")
        assert persona.get_current_effective_context_length() == 2
        with patch.object(memory_manager, 'get_channel_history',
                          wraps=memory_manager.get_channel_history) as mock_get_history:
            await chat_system.generate_response("test_persona", "user1", "channel", "second message")
            mock_get_history.assert_called_with("channel", "test_persona", None, 2)


@pytest.mark.asyncio
async def test_context_transformation_and_multi_user_differentiation(mocked_chat_system):
    chat_system, memory_manager = mocked_chat_system
    persona = chat_system.personas['test_persona']
    persona.set_memory_mode(MemoryMode.CHANNEL_ISOLATED)
    channel, user1_id, server_id = "test-channel", "user1", "server1"
    memory_manager.log_message(user1_id, "test_persona", channel, 'user', "UserOne", "msg1", datetime.now(),
                               server_id=server_id)
    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=({'type': 'text', 'content': ''}, {})) as mock_llm_call:
        await chat_system.generate_response("test_persona", user1_id, channel, "msg2", server_id=server_id)
        context = mock_llm_call.call_args[0][1]['history']
        assert len(context) == 2
        assert context[0]['content'] == "UserOne: msg1"
        assert context[1]['content'] == "msg2"


@pytest.mark.asyncio
async def test_end_to_end_message_suppression(mocked_chat_system):
    chat_system, memory_manager = mocked_chat_system
    channel, user_id = "test-channel", "user1"
    memory_manager.log_message(user_id, "test_persona", channel, 'user', user_id, "Message 1", datetime.now(),
                               platform_message_id="p10")
    memory_manager.log_message(user_id, "test_persona", channel, 'user', user_id, "Message to suppress", datetime.now(),
                               platform_message_id="p11_suppress")
    memory_manager.log_message(user_id, "test_persona", channel, 'user', user_id, "Message 3", datetime.now(),
                               platform_message_id="p12")
    assert memory_manager.suppress_message_by_platform_id("p11_suppress") is True
    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=({'type': 'text', 'content': ''}, {})) as mock_llm_call:
        await chat_system.generate_response("test_persona", user_id, channel, "Final message")
        context = mock_llm_call.call_args[0][1]['history']
        assert len(context) == 3
        assert "Message to suppress" not in [c['content'] for c in context]


@pytest.mark.asyncio
async def test_persona_context_length_is_capped_by_history_limit(mocked_chat_system):
    chat_system, memory_manager = mocked_chat_system
    with patch.object(memory_manager, 'get_channel_history',
                      wraps=memory_manager.get_channel_history) as mock_get_history:
        await chat_system.generate_response("capped_persona", "user1", "any", "test", history_limit=5)
        mock_get_history.assert_called_once()
        assert mock_get_history.call_args[0][3] == 5


@pytest.mark.asyncio
async def test_empty_history_is_handled_gracefully(mocked_chat_system):
    chat_system, _ = mocked_chat_system
    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=({'type': 'text', 'content': 'Hello!'}, {})) as mock_llm_call:
        await chat_system.generate_response("test_persona", "new_user_123", "new-channel", "First message ever")
        context = mock_llm_call.call_args[0][1]['history']
        assert len(context) == 1
        assert context[0] == {'role': 'user', 'content': 'First message ever'}


@pytest.mark.asyncio
async def test_history_limit_zero_for_channel_mode(mocked_chat_system):
    chat_system, memory_manager = mocked_chat_system
    user1, channel = "user1", "test-channel"
    memory_manager.log_message(user1, "test_persona", channel, 'user', user1, "An ignored message", datetime.now())
    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=({'type': 'text', 'content': ''}, {})) as mock_llm_call:
        await chat_system.generate_response("test_persona", user1, channel, "A new message", history_limit=0)
        context = mock_llm_call.call_args[0][1]['history']
        assert len(context) == 1
        assert context[0] == {'role': 'user', 'content': 'A new message'}


# =============================================================================
# CONFIRM MODE INTEGRATION TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_confirm_mode_auto_executes_read_only_tools(mocked_chat_system):
    """CONFIRM mode: read-only tools execute immediately without pending confirmation."""
    chat_system, _ = mocked_chat_system
    mock_zammad = MagicMock()
    mock_zammad.api_url = "http://zammad.test"
    mock_zammad.search_tickets.return_value = [{"id": 1, "title": "Open ticket"}]
    mock_zammad.search_user.return_value = []
    chat_system.register_service(ZammadIntegration(mock_zammad))
    persona = chat_system.personas['test_persona']
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_service_bindings(["zammad"])

    tool_call = ({'type': 'tool_calls', 'calls': [
        {'id': 'call_1', 'name': 'search_tickets', 'arguments': {'query': 'state.name:open'}}]}, {})
    final_text = ({'type': 'text', 'content': 'Found some tickets.'}, {})

    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      side_effect=[tool_call, final_text]):
        response, response_type, _, _ = await chat_system.generate_response(
            "test_persona", "user1", "channel", "Search for open tickets"
        )
        assert response_type == ResponseType.LLM_GENERATION
        assert response == 'Found some tickets.'
        assert ("user1", "test_persona") not in chat_system._pending_confirmations


# =============================================================================
# SERVICE BINDING TOOL FILTERING TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_no_service_bindings_excludes_service_tools(mocked_chat_system):
    """No service_bindings: service-bound tools are filtered out even with enabled_tools=['*']."""
    chat_system, _ = mocked_chat_system
    persona = chat_system.personas['test_persona']
    persona.set_service_bindings([])

    zammad_tool_names = {'get_ticket_details', 'update_ticket', 'add_note_to_ticket',
                         'create_ticket', 'search_tickets', 'search_user', 'create_user',
                         'update_user', 'delete_user'}

    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=({'type': 'text', 'content': 'ok'}, {})) as mock_llm_call:
        await chat_system.generate_response("test_persona", "user1", "channel", "test")
        # Phase C: stream_messages forwards tools positionally to generate_response.
        tools_passed = mock_llm_call.call_args.args[2] if len(mock_llm_call.call_args.args) > 2 else []
        tool_names_passed = {t.get('function', {}).get('name') for t in tools_passed}
        assert tool_names_passed.isdisjoint(zammad_tool_names), \
            f"Zammad tools should be excluded but found: {tool_names_passed & zammad_tool_names}"


@pytest.mark.asyncio
async def test_zammad_service_binding_includes_zammad_tools(mocked_chat_system):
    """service_bindings=["zammad"]: Zammad tools are included in tools_for_llm."""
    chat_system, _ = mocked_chat_system
    mock_zammad = MagicMock()
    mock_zammad.api_url = "http://zammad.test"
    chat_system.register_service(ZammadIntegration(mock_zammad))
    persona = chat_system.personas['test_persona']
    persona.set_service_bindings(["zammad"])

    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=({'type': 'text', 'content': 'ok'}, {})) as mock_llm_call:
        await chat_system.generate_response("test_persona", "user1", "channel", "test")
        # Phase C: stream_messages forwards tools positionally to generate_response.
        tools_passed = mock_llm_call.call_args.args[2] if len(mock_llm_call.call_args.args) > 2 else []
        tool_names_passed = {t.get('function', {}).get('name') for t in tools_passed}
        assert 'create_ticket' in tool_names_passed
        assert 'search_tickets' in tool_names_passed
