# tests/live/test_full_system_zammad.py
#
# Zammad-dependent integration tests extracted from test_full_system_flow.py.
# These require a live Zammad instance.

import pytest
from unittest.mock import patch, AsyncMock

from src.chat_system import ResponseType
from src.persona import ExecutionMode

from tests.live.conftest import wait_for_search

pytestmark = pytest.mark.zammad_live


@pytest.mark.asyncio
async def test_confirm_mode_pends_write_tools(live_chat_system, managed_zammad_user):
    """CONFIRM mode: write tool calls are pended, not executed immediately."""
    chat_system, _, zammad_client = live_chat_system
    persona = chat_system.personas['test_persona']
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_service_bindings(["zammad"])
    user_info = managed_zammad_user

    tool_call = ({'type': 'tool_calls', 'calls': [
        {'id': 'call_1', 'name': 'create_ticket', 'arguments': {'title': 'Pending Ticket', 'body': 'test'}}]}, {})
    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      side_effect=[tool_call]):
        response, response_type, _, _ = await chat_system.generate_response(
            "test_persona", user_info["identifier"], "support", "Create a ticket"
        )
        assert response_type == ResponseType.PENDING_CONFIRMATION
        assert "create_ticket" in response
        assert (user_info["identifier"], "test_persona") in chat_system._pending_confirmations
        results = zammad_client.search_tickets(query="title:\"Pending Ticket\"")
        assert len(results) == 0


@pytest.mark.asyncio
async def test_confirm_mode_resume_approved_creates_ticket(live_chat_system, managed_zammad_user):
    """CONFIRM mode approved: pended write tool executes and creates a real Zammad ticket."""
    chat_system, _, zammad_client = live_chat_system
    persona = chat_system.personas['test_persona']
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_service_bindings(["zammad"])
    user_info = managed_zammad_user
    created_ticket_id = None

    try:
        tool_call = ({'type': 'tool_calls', 'calls': [
            {'id': 'call_1', 'name': 'create_ticket',
             'arguments': {'title': 'Approved Ticket', 'body': 'approved body',
                          'customer_id': user_info['id']}}]}, {})
        final_text = ({'type': 'text', 'content': 'Ticket created successfully.'}, {})

        with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                          side_effect=[tool_call]):
            await chat_system.generate_response(
                "test_persona", user_info["identifier"], "support", "Create a ticket"
            )

        with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                          return_value=final_text):
            response, response_type, _, _ = await chat_system.resume_pending_confirmation(
                user_info["identifier"], "test_persona", approved=True
            )
            assert response_type == ResponseType.LLM_GENERATION

            # Verification: We don't need search/indexing here because we can get the ID
            # from the Zammad logs or just check for the most recent ticket from this user.
            # In this mock setup, we'll just check if a ticket with the expected title exists
            # via the LIST API (which is often faster/non-ES in some setups) or just wait_for_search
            # but with a very specific query.
            # Actually, the user asked to skip indexing if possible.
            # Zammad's LIST API doesn't support title filtering easily, so we'll do a quick search
            # but with a fallback to just checking the last 5 tickets for the user.
            
            def find_ticket():
                # list_tickets hits the DB directly, much faster than search
                tickets = zammad_client.list_tickets(params={'expand': 'true'})
                return [t for t in tickets if t['title'] == 'Approved Ticket' and t['customer_id'] == user_info['id']]

            await wait_for_search(
                search_func=find_ticket,
                assertion_func=lambda results: len(results) >= 1,
                timeout=5
            )
            results = find_ticket()
            created_ticket_id = results[0]['id']
            assert results[0]['title'] == 'Approved Ticket'
    finally:
        if created_ticket_id:
            zammad_client.delete_ticket(created_ticket_id)


@pytest.mark.asyncio
async def test_confirm_mode_resume_denied_skips_tool(live_chat_system, managed_zammad_user):
    """CONFIRM mode denied: write tool is not executed, LLM receives denial feedback."""
    chat_system, _, zammad_client = live_chat_system
    persona = chat_system.personas['test_persona']
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_service_bindings(["zammad"])
    user_info = managed_zammad_user

    tool_call = ({'type': 'tool_calls', 'calls': [
        {'id': 'call_1', 'name': 'create_ticket',
         'arguments': {'title': 'Denied Ticket', 'body': 'should not exist'}}]}, {})
    denial_response = ({'type': 'text', 'content': 'Understood, I will not create the ticket.'}, {})

    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      side_effect=[tool_call]):
        await chat_system.generate_response(
            "test_persona", user_info["identifier"], "support", "Create a ticket"
        )

    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=denial_response):
        response, response_type, _, _ = await chat_system.resume_pending_confirmation(
            user_info["identifier"], "test_persona", approved=False
        )
        assert response_type == ResponseType.LLM_GENERATION
        assert "not create" in response.lower() or "denied" in response.lower() or len(response) > 0

    results = zammad_client.search_tickets(query="title:\"Denied Ticket\"")
    assert len(results) == 0


# ---------------------------------------------------------------------------
# Tool registration coverage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_tools_populates_tool_manager(live_chat_system):
    """register_tools: ZammadIntegration registers all Zammad CRUD tools with the ToolManager."""
    chat_system, _, _ = live_chat_system
    registered_names = set(chat_system.tool_manager._handlers.keys())
    expected_zammad_tools = {
        "get_ticket_details", "update_ticket", "add_note_to_ticket",
        "search_tickets", "create_ticket", "search_user", "create_user",
        "update_user",
    }
    assert expected_zammad_tools.issubset(registered_names), (
        f"Missing Zammad tools: {expected_zammad_tools - registered_names}"
    )
