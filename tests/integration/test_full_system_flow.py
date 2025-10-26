# tests/integration/test_full_system_flow.py

import pytest
import os
import time
import asyncio
from datetime import datetime
from unittest.mock import patch, AsyncMock
import random
from urllib.parse import urlparse
from typing import Callable, Coroutine, Any, List

# Mark all tests in this file as 'integration'.
pytestmark = pytest.mark.integration

from src.chat_system import ChatSystem
from src.database.memory_manager import MemoryManager
from src.engine import TextEngine
from src.clients.zammad_client import ZammadClient
from config.global_config import TEST_MEMORY_DATABASE_FILE, TEST_DATABASE_DIR
from src.persona import Persona, MemoryMode, ExecutionMode

PERSISTENT_TEST_USER_EMAIL = "pytest-integration-user@zammad.local"


async def _wait_for_search(search_func: Callable[..., List[Any]], assertion_func: Callable[[List[Any]], bool],
                           timeout: int = 5, interval: float = 0.5):
    """
    Repeatedly calls a search function and checks an assertion until it passes or times out.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        results = await asyncio.to_thread(search_func)
        if assertion_func(results):
            return  # Success
        await asyncio.sleep(interval)
    pytest.fail(f"Search assertion did not pass within {timeout} seconds.")


@pytest.fixture(scope="function")
def live_chat_system():
    """
    Sets up a self-contained, fully integrated ChatSystem instance for each test function.
    """
    db_path = f"{TEST_MEMORY_DATABASE_FILE}.{random.randint(1000, 9999)}"
    if os.path.exists(db_path):
        os.remove(db_path)

    memory_manager = MemoryManager(db_path=db_path)
    memory_manager.create_schema()

    try:
        zammad_client = ZammadClient()
        zammad_client.get_self()
    except Exception as e:
        pytest.skip(f"Skipping integration tests: Zammad client setup failed. Error: {e}")

    text_engine = TextEngine()

    # Programmatically create personas for a self-contained test environment
    test_personas = {
        "test_persona": Persona(
            persona_name="test_persona", model_name="mock_model", prompt="You are a test persona.",
            enabled_tools=['*'], memory_mode=MemoryMode.CHANNEL_ISOLATED, context_length=10
        ),
        "capped_persona": Persona(
            persona_name='capped_persona', model_name='mock', prompt='talk', context_length=100
        )
    }

    with patch('src.chat_system.load_personas_from_file', return_value=test_personas):
        chat_system = ChatSystem(
            memory_manager=memory_manager, text_engine=text_engine, zammad_client=zammad_client
        )

    try:
        yield chat_system, memory_manager, zammad_client
    finally:
        memory_manager.close()
        time.sleep(0.1)
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except PermissionError as e:
                print(f"\n[TEARDOWN WARNING] Could not remove test database file: {e}")


@pytest.fixture(scope="function")
def managed_zammad_user(live_chat_system):
    """Finds or creates a single persistent user for Zammad-related tests."""
    _, _, zammad_client = live_chat_system
    users = zammad_client.search_user(query=PERSISTENT_TEST_USER_EMAIL)
    if users:
        user_id = users[0]['id']
    else:
        user_data = zammad_client.create_user(email=PERSISTENT_TEST_USER_EMAIL, firstname="Pytest", lastname="User")
        user_id = user_data['id']

    tickets = zammad_client.search_tickets(query=f"customer_id:{user_id}")
    for ticket in tickets:
        zammad_client.delete_ticket(ticket['id'])

    yield {"id": user_id, "identifier": f"Pytest User <{PERSISTENT_TEST_USER_EMAIL}>"}


@pytest.mark.asyncio
@patch('src.chat_system.ChatSystem._should_create_ticket', return_value=True)
async def test_tool_driven_ticket_creation_flow(mock_should_create, live_chat_system, managed_zammad_user):
    chat_system, _, zammad_client = live_chat_system
    chat_system.personas['test_persona'].set_execution_mode(ExecutionMode.ASSISTED_DISPATCH)
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
@patch('src.chat_system.ChatSystem._should_create_ticket', return_value=True)
async def test_zammad_user_creation_for_non_email_identifier(mock_should_create, live_chat_system):
    chat_system, _, zammad_client = live_chat_system
    # Use a static, predictable identifier to prevent creating numerous orphaned users on failure
    static_user_identifier = "pytest_user_to_delete"
    expected_email = f"support-{static_user_identifier}@{urlparse(zammad_client.api_url).hostname}"
    created_zammad_user_id = None

    try:
        # Initial cleanup to ensure a clean slate from any previous failed runs
        existing_users = zammad_client.search_user(query=expected_email)
        for user in existing_users:
            tickets = zammad_client.search_tickets(query=f"customer_id:{user['id']}")
            for ticket in tickets:
                zammad_client.delete_ticket(ticket['id'])
            zammad_client.delete_user(user['id'])

        # Main test logic
        with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                          return_value=({'type': 'text', 'content': '...'}, {})):
            await chat_system.generate_response(
                "test_persona", static_user_identifier, "support", "test message", user_display_name="New Test User"
            )

        # Wait for the user to be indexed by Zammad search
        await _wait_for_search(
            search_func=lambda: zammad_client.search_user(query=expected_email),
            assertion_func=lambda results: len(results) == 1
        )
        created_users = zammad_client.search_user(query=expected_email)
        assert len(created_users) == 1
        created_zammad_user_id = created_users[0]['id']

    finally:
        # Robust cleanup: delete dependent tickets first, then the user
        if created_zammad_user_id:
            tickets_to_delete = zammad_client.search_tickets(query=f"customer_id:{created_zammad_user_id}")
            for ticket in tickets_to_delete:
                zammad_client.delete_ticket(ticket['id'])

            # Add a small delay to ensure ticket deletion is processed before user deletion
            if tickets_to_delete:
                time.sleep(1)

            zammad_client.delete_user(created_zammad_user_id)

@pytest.mark.asyncio
async def test_dynamic_context_ignores_dev_commands(live_chat_system):
    chat_system, memory_manager, _ = live_chat_system
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
@patch('src.chat_system.ChatSystem._should_create_ticket', return_value=True)
async def test_ticket_history_is_used_when_mode_is_ticket(mock_should_create, live_chat_system, managed_zammad_user):
    chat_system, memory_manager, zammad_client = live_chat_system
    persona = chat_system.personas['test_persona']
    persona.set_memory_mode(MemoryMode.TICKET_ISOLATED)
    user_info = managed_zammad_user
    ticket_id = None
    try:
        # Create a ticket with an initial article to ensure it is indexed by Zammad search.
        ticket_data = zammad_client.create_ticket(
            title="Test",
            group="Users",
            customer_id=user_info['id'],
            article_body="Initial article. This message exists ONLY in Zammad."
        )
        ticket_id = ticket_data['id']
        ticket_number = ticket_data['number']

        # Wait for the ticket to become searchable by its number
        await _wait_for_search(
            search_func=lambda: zammad_client.search_tickets(query=f"number:{ticket_number}"),
            assertion_func=lambda results: len(results) == 1 and results[0]['id'] == ticket_id
        )

        # This message is logged to the local DB and should appear in the context.
        memory_manager.log_message(user_info['identifier'], "test_persona", "support", 'user', "User", "msg1",
                                   datetime.now(), zammad_ticket_id=ticket_id)
        with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                          return_value=({'type': 'text', 'content': ''}, {})) as mock_llm_call:
            await chat_system.generate_response(
                "test_persona", user_info['identifier'], "support", f"Follow up for [Ticket#{ticket_number}]"
            )
            context = mock_llm_call.call_args[0][1]['history']

            # Expected history: system msg, msg1 from local DB, current msg.
            # The initial article is NOT included as it was never logged to MemoryManager.
            assert len(context) == 3
            assert "part of Zammad ticket" in context[0]['content']
            assert context[1]['content'] == "User: msg1"
            assert context[2]['role'] == 'user'
    finally:
        if ticket_id:
            zammad_client.delete_ticket(ticket_id)


@pytest.mark.asyncio
async def test_context_transformation_and_multi_user_differentiation(live_chat_system):
    chat_system, memory_manager, _ = live_chat_system
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
async def test_end_to_end_message_suppression(live_chat_system):
    chat_system, memory_manager, _ = live_chat_system
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
async def test_persona_context_length_is_capped_by_history_limit(live_chat_system):
    chat_system, memory_manager, _ = live_chat_system
    with patch.object(memory_manager, 'get_channel_history',
                      wraps=memory_manager.get_channel_history) as mock_get_history:
        await chat_system.generate_response("capped_persona", "user1", "any", "test", history_limit=5)
        mock_get_history.assert_called_once()
        assert mock_get_history.call_args[0][3] == 5


@pytest.mark.asyncio
@patch('src.chat_system.ChatSystem._should_create_ticket', return_value=True)
async def test_context_transformation_in_ticket_mode(mock_should_create, live_chat_system, managed_zammad_user):
    chat_system, memory_manager, zammad_client = live_chat_system
    persona = chat_system.personas['test_persona']
    persona.set_memory_mode(MemoryMode.TICKET_ISOLATED)
    user_info = managed_zammad_user
    ticket_id = None
    try:
        ticket_data = zammad_client.create_ticket(title="Test", group="Users", customer_id=user_info['id'])
        ticket_id = ticket_data['id']
        ticket_number = ticket_data['number']

        await _wait_for_search(
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
async def test_empty_history_is_handled_gracefully(live_chat_system):
    chat_system, _, _ = live_chat_system
    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=({'type': 'text', 'content': 'Hello!'}, {})) as mock_llm_call:
        await chat_system.generate_response("test_persona", "new_user_123", "new-channel", "First message ever")
        context = mock_llm_call.call_args[0][1]['history']
        assert len(context) == 1
        assert context[0] == {'role': 'user', 'content': 'First message ever'}


@pytest.mark.asyncio
async def test_history_limit_zero_for_channel_mode(live_chat_system):
    chat_system, memory_manager, _ = live_chat_system
    user1, channel = "user1", "test-channel"
    memory_manager.log_message(user1, "test_persona", channel, 'user', user1, "An ignored message", datetime.now())
    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=({'type': 'text', 'content': ''}, {})) as mock_llm_call:
        await chat_system.generate_response("test_persona", user1, channel, "A new message", history_limit=0)
        context = mock_llm_call.call_args[0][1]['history']
        assert len(context) == 1
        assert context[0] == {'role': 'user', 'content': 'A new message'}
