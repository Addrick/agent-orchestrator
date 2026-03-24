# tests/live/test_full_system_zammad.py
#
# Zammad-dependent integration tests extracted from test_full_system_flow.py.
# These require a live Zammad instance.

import asyncio
import pytest
from datetime import datetime
from unittest.mock import patch, AsyncMock

from src.chat_system import ChatSystem, ResponseType
from src.persona import MemoryMode, ExecutionMode

from tests.live.conftest import wait_for_search

pytestmark = pytest.mark.zammad_live


@pytest.mark.asyncio
async def test_tool_driven_ticket_creation_flow(live_chat_system, managed_zammad_user):
    """AUTONOMOUS mode: LLM returns create_ticket tool call -> tool executes -> ticket created in Zammad."""
    chat_system, _, zammad_client = live_chat_system
    persona = chat_system.personas['test_persona']
    persona.set_execution_mode(ExecutionMode.AUTONOMOUS)
    persona.set_service_bindings(["zammad"])
    user_info = managed_zammad_user
    created_ticket_id = None
    try:
        tool_call = ({'type': 'tool_calls', 'calls': [
            {'name': 'create_ticket', 'arguments': {'title': 'New Problem', 'body': '..._body_...'}}]}, {})
        final_text = ({'type': 'text', 'content': 'Ticket created.'}, {})
        with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                          side_effect=[tool_call, final_text]):
            _, _, ticket_id = await chat_system.generate_response(
                "test_persona", user_info["identifier"], "support", "Help me"
            )
            assert ticket_id is not None
            created_ticket_id = ticket_id
            ticket_data = zammad_client.get_ticket(created_ticket_id)
            assert ticket_data['customer_id'] == user_info["id"]
            assert ticket_data['title'] == 'New Problem'
    finally:
        if created_ticket_id:
            zammad_client.delete_ticket(created_ticket_id)


@pytest.mark.asyncio
async def test_zammad_user_creation_for_non_email_identifier(live_chat_system):
    """zammad_aware persona: non-email user identifier triggers lazy Zammad user creation."""
    chat_system, _, zammad_client = live_chat_system
    chat_system.personas['test_persona'].set_service_bindings(["zammad"])
    from urllib.parse import urlparse
    static_user_identifier = "pytest_user_to_delete"
    expected_email = f"support-{static_user_identifier}@{urlparse(zammad_client.api_url).hostname}"
    created_zammad_user_id = None

    try:
        existing_users = zammad_client.search_user(query=expected_email)
        for user in existing_users:
            tickets = zammad_client.search_tickets(query=f"customer_id:{user['id']}")
            for ticket in tickets:
                zammad_client.delete_ticket(ticket['id'])
            zammad_client.delete_user(user['id'])

        with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                          return_value=({'type': 'text', 'content': '...'}, {})):
            await chat_system.generate_response(
                "test_persona", static_user_identifier, "support", "test message", user_display_name="New Test User"
            )

        await wait_for_search(
            search_func=lambda: zammad_client.search_user(query=expected_email),
            assertion_func=lambda results: len(results) == 1
        )
        created_users = zammad_client.search_user(query=expected_email)
        assert len(created_users) == 1
        created_zammad_user_id = created_users[0]['id']

    finally:
        if created_zammad_user_id:
            tickets_to_delete = zammad_client.search_tickets(query=f"customer_id:{created_zammad_user_id}")
            for ticket in tickets_to_delete:
                zammad_client.delete_ticket(ticket['id'])
            import time
            if tickets_to_delete:
                time.sleep(1)
            zammad_client.delete_user(created_zammad_user_id)


@pytest.mark.asyncio
async def test_ticket_history_is_used_when_mode_is_ticket(live_chat_system, managed_zammad_user):
    """TICKET_ISOLATED mode with zammad_aware: history is scoped to a specific Zammad ticket."""
    chat_system, memory_manager, zammad_client = live_chat_system
    persona = chat_system.personas['test_persona']
    persona.set_memory_mode(MemoryMode.TICKET_ISOLATED)
    persona.set_service_bindings(["zammad"])
    user_info = managed_zammad_user
    ticket_id = None
    try:
        ticket_data = zammad_client.create_ticket(
            title="Test",
            group="Users",
            customer_id=user_info['id'],
            article_body="Initial article. This message exists ONLY in Zammad."
        )
        ticket_id = ticket_data['id']
        ticket_number = ticket_data['number']

        await wait_for_search(
            search_func=lambda: zammad_client.search_tickets(query=f"number:{ticket_number}"),
            assertion_func=lambda results: len(results) == 1 and results[0]['id'] == ticket_id
        )

        memory_manager.log_message(user_info['identifier'], "test_persona", "support", 'user', "User", "msg1",
                                   datetime.now(), zammad_ticket_id=ticket_id)
        with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                          return_value=({'type': 'text', 'content': ''}, {})) as mock_llm_call:
            await chat_system.generate_response(
                "test_persona", user_info['identifier'], "support", f"Follow up for [Ticket#{ticket_number}]"
            )
            context = mock_llm_call.call_args[0][1]['history']
            assert len(context) == 3
            assert "part of Zammad ticket" in context[0]['content']
            assert context[1]['content'] == "User: msg1"
            assert context[2]['role'] == 'user'
    finally:
        if ticket_id:
            zammad_client.delete_ticket(ticket_id)


@pytest.mark.asyncio
async def test_context_transformation_in_ticket_mode(live_chat_system, managed_zammad_user):
    """TICKET_ISOLATED mode: history includes username-prefixed messages from the ticket."""
    chat_system, memory_manager, zammad_client = live_chat_system
    persona = chat_system.personas['test_persona']
    persona.set_memory_mode(MemoryMode.TICKET_ISOLATED)
    persona.set_service_bindings(["zammad"])
    user_info = managed_zammad_user
    ticket_id = None
    try:
        ticket_data = zammad_client.create_ticket(title="Test", group="Users", customer_id=user_info['id'])
        ticket_id = ticket_data['id']
        ticket_number = ticket_data['number']

        await wait_for_search(
            search_func=lambda: zammad_client.search_tickets(query=f"number:{ticket_number}"),
            assertion_func=lambda results: len(results) == 1 and results[0]['id'] == ticket_id
        )

        memory_manager.log_message(user_info['identifier'], "test_persona", "support", 'user', "SpecificUserName",
                                   "Hello world", datetime.now(), zammad_ticket_id=ticket_id)
        with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                          return_value=({'type': 'text', 'content': ''}, {})) as mock_llm_call:
            await chat_system.generate_response("test_persona", user_info['identifier'], "support",
                                                f"Another for [Ticket#{ticket_data['number']}]")
            context = mock_llm_call.call_args[0][1]['history']
            assert len(context) == 3
            assert context[1]['content'] == "SpecificUserName: Hello world"
    finally:
        if ticket_id:
            zammad_client.delete_ticket(ticket_id)


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
        response, response_type, ticket_id = await chat_system.generate_response(
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
             'arguments': {'title': 'Approved Ticket', 'body': 'approved body'}}]}, {})
        final_text = ({'type': 'text', 'content': 'Ticket created successfully.'}, {})

        with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                          side_effect=[tool_call]):
            await chat_system.generate_response(
                "test_persona", user_info["identifier"], "support", "Create a ticket"
            )

        with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                          return_value=final_text):
            response, response_type, ticket_id = await chat_system.resume_pending_confirmation(
                user_info["identifier"], "test_persona", approved=True
            )
            assert response_type == ResponseType.LLM_GENERATION
            assert ticket_id is not None
            created_ticket_id = ticket_id

            ticket_data = zammad_client.get_ticket(created_ticket_id)
            assert ticket_data['title'] == 'Approved Ticket'
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
        response, response_type, _ = await chat_system.resume_pending_confirmation(
            user_info["identifier"], "test_persona", approved=False
        )
        assert response_type == ResponseType.LLM_GENERATION
        assert "not create" in response.lower() or "denied" in response.lower() or len(response) > 0

    results = zammad_client.search_tickets(query="title:\"Denied Ticket\"")
    assert len(results) == 0


# ---------------------------------------------------------------------------
# ServiceIntegration hook coverage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_tools_populates_tool_manager(live_chat_system):
    """register_tools hook: ZammadIntegration registers all Zammad CRUD tools with the ToolManager."""
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


@pytest.mark.asyncio
async def test_get_tracking_id_returns_ticket_id(live_chat_system, managed_zammad_user):
    """get_tracking_id hook: generate_response returns the ticket ID resolved by the service."""
    chat_system, _, zammad_client = live_chat_system
    persona = chat_system.personas['test_persona']
    persona.set_service_bindings(["zammad"])
    user_info = managed_zammad_user
    ticket_id = None
    try:
        ticket_data = zammad_client.create_ticket(
            title="TrackingID Test", group="Users", customer_id=user_info['id'],
            article_body="Seed article."
        )
        ticket_id = ticket_data['id']
        ticket_number = ticket_data['number']

        await wait_for_search(
            search_func=lambda: zammad_client.search_tickets(query=f"number:{ticket_number}"),
            assertion_func=lambda results: len(results) == 1
        )

        # Ticket referenced by number: get_tracking_id should surface the resolved ID.
        with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                          return_value=({'type': 'text', 'content': 'Got it.'}, {})):
            _, _, returned_ticket_id = await chat_system.generate_response(
                "test_persona", user_info["identifier"], "support",
                f"Checking [Ticket#{ticket_number}]"
            )
            assert returned_ticket_id == ticket_id

        # No ticket reference: active ticket found for user should still surface.
        with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                          return_value=({'type': 'text', 'content': 'Still here.'}, {})):
            _, _, returned_ticket_id = await chat_system.generate_response(
                "test_persona", user_info["identifier"], "support",
                "Follow-up without ticket reference"
            )
            assert returned_ticket_id == ticket_id
    finally:
        if ticket_id:
            zammad_client.delete_ticket(ticket_id)


@pytest.mark.asyncio
async def test_on_message_mirrors_to_zammad_ticket(live_chat_system, managed_zammad_user):
    """on_message hook: user message and bot response are mirrored as articles on the Zammad ticket."""
    chat_system, _, zammad_client = live_chat_system
    persona = chat_system.personas['test_persona']
    persona.set_service_bindings(["zammad"])
    user_info = managed_zammad_user
    ticket_id = None
    try:
        ticket_data = zammad_client.create_ticket(
            title="Mirror Test", group="Users", customer_id=user_info['id'],
            article_body="Seed article."
        )
        ticket_id = ticket_data['id']
        ticket_number = ticket_data['number']

        await wait_for_search(
            search_func=lambda: zammad_client.search_tickets(query=f"number:{ticket_number}"),
            assertion_func=lambda results: len(results) == 1
        )

        user_msg = "Please help with mirroring test"
        bot_reply = "Sure, I can help with that."

        with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                          return_value=({'type': 'text', 'content': bot_reply}, {})):
            await chat_system.generate_response(
                "test_persona", user_info["identifier"], "support",
                f"{user_msg} [Ticket#{ticket_number}]"
            )

        # Zammad article creation is synchronous (via asyncio.to_thread),
        # so articles should exist immediately — but give a tiny buffer.
        await asyncio.sleep(0.5)

        articles = zammad_client.get_ticket_articles(ticket_id)
        article_bodies = [a.get('body', '') for a in articles]

        # First article is the seed; user message and bot reply should follow.
        assert any(user_msg in body for body in article_bodies), (
            f"User message not mirrored. Article bodies: {article_bodies}"
        )
        assert any(bot_reply in body for body in article_bodies), (
            f"Bot reply not mirrored. Article bodies: {article_bodies}"
        )
    finally:
        if ticket_id:
            zammad_client.delete_ticket(ticket_id)
