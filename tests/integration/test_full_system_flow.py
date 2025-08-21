# tests/integration/test_full_system_flow.py

import pytest
import os
import time
from datetime import datetime
from unittest.mock import patch, AsyncMock, MagicMock
from requests.exceptions import RequestException

# Mark all tests in this file as 'integration'.
pytestmark = pytest.mark.integration

from src.chat_system import ChatSystem, ResponseType
from src.database.memory_manager import MemoryManager
from src.engine import TextEngine
from src.clients.zammad_client import ZammadClient
from src.utils.save_utils import load_personas_from_file as real_load_personas
from config.global_config import TEST_MEMORY_DATABASE_FILE, TEST_DATABASE_DIR, TEST_PERSONA_SAVE_FILE

# Use a static, persistent user for all Zammad integration tests.
PERSISTENT_TEST_USER_EMAIL = "pytest-integration-user@zammad.local"


@pytest.fixture(scope="function")
def live_zammad_client():
    """Provides a live Zammad client, skipping tests if unavailable."""
    try:
        zammad_client = ZammadClient()
        zammad_client.get_self()
        return zammad_client
    except (ValueError, RequestException) as e:
        pytest.skip(
            f"Skipping live Zammad test: Could not connect. Please ensure it is running and .env is configured. Error: {e}")


@pytest.fixture(scope="function")
def managed_zammad_user(live_zammad_client):
    """Finds or creates a persistent user for tests that require a live Zammad connection."""
    users = live_zammad_client.search_user(query=PERSISTENT_TEST_USER_EMAIL)
    if users:
        user_id = users[0]['id']
    else:
        user_data = live_zammad_client.create_user(
            email=PERSISTENT_TEST_USER_EMAIL, firstname="Pytest", lastname="PersistentUser"
        )
        user_id = user_data['id']

    orphaned_tickets = live_zammad_client.search_tickets(query=f"customer_id:{user_id}")
    for ticket in orphaned_tickets:
        live_zammad_client.delete_ticket(ticket['id'])

    yield {"id": user_id, "identifier": f"Pytest PersistentUser <{PERSISTENT_TEST_USER_EMAIL}>"}


@pytest.fixture(scope="function")
def chat_system_no_zammad():
    """
    Sets up a ChatSystem with a MOCKED Zammad client.
    This allows testing core logic without a live Zammad connection.
    """
    os.makedirs(TEST_DATABASE_DIR, exist_ok=True)
    if os.path.exists(TEST_MEMORY_DATABASE_FILE):
        os.remove(TEST_MEMORY_DATABASE_FILE)

    memory_manager = MemoryManager(db_path=TEST_MEMORY_DATABASE_FILE)
    memory_manager.create_schema()
    text_engine = TextEngine()

    if not os.path.exists(TEST_PERSONA_SAVE_FILE):
        pytest.skip(f"Skipping tests: Test persona file not found at {TEST_PERSONA_SAVE_FILE}")

    test_personas = real_load_personas(file_path=TEST_PERSONA_SAVE_FILE)

    from src.persona import Persona
    if 'chatter' not in test_personas:
        test_personas['chatter'] = Persona(persona_name='chatter', model_name='mock', prompt='talk',
                                           memory_type='channel', context_length=10)
    if 'private' not in test_personas:
        test_personas['private'] = Persona(persona_name='private', model_name='mock', prompt='private',
                                           memory_type='personal', context_length=10)
    if 'capped_persona' not in test_personas:
        test_personas['capped_persona'] = Persona(persona_name='capped_persona', model_name='mock', prompt='talk',
                                                  context_length=100)

    with patch('src.chat_system.load_personas_from_file', return_value=test_personas):
        chat_system = ChatSystem(
            memory_manager=memory_manager,
            text_engine=text_engine,
            zammad_client=MagicMock(spec=ZammadClient)  # Use a mock client
        )

    try:
        yield chat_system, memory_manager, chat_system.zammad_client
    finally:
        memory_manager.close()
        time.sleep(0.1)
        try:
            if os.path.exists(TEST_MEMORY_DATABASE_FILE):
                os.remove(TEST_MEMORY_DATABASE_FILE)
        except PermissionError as e:
            print(f"\n[TEARDOWN WARNING] Could not remove test database file due to a lock: {e}")


# --- Zammad-Independent Tests ---

@pytest.mark.asyncio
async def test_context_transformation_logic(chat_system_no_zammad):
    chat_system, memory_manager, _ = chat_system_no_zammad
    channel, user_id, user_name = "test-channel", "user1", "Human"

    memory_manager.log_message(user_identifier=user_id, persona_name="chatter", channel=channel, author_role='user',
                               author_name=user_name, content="Hello chatter", timestamp=datetime.now(),
                               platform_message_id="p1")
    time.sleep(0.01)
    memory_manager.log_message(user_identifier=user_id, persona_name="chatter", channel=channel,
                               author_role='assistant', author_name='chatter', content="Hi from chatter!",
                               timestamp=datetime.now(), platform_message_id="p2")
    time.sleep(0.01)
    memory_manager.log_message(user_identifier=user_id, persona_name="private", channel=channel, author_role='user',
                               author_name=user_name, content="Hi private bot", timestamp=datetime.now(),
                               platform_message_id="p3")
    time.sleep(0.01)
    memory_manager.log_message(user_identifier=user_id, persona_name="private", channel=channel,
                               author_role='assistant', author_name='private', content="Hi from private bot.",
                               timestamp=datetime.now(), platform_message_id="p4")

    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=("", {})) as mock_llm_call:
        await chat_system.generate_response(
            persona_name="chatter", user_identifier=user_id, channel=channel, message="Final message"
        )
        context_chatter = mock_llm_call.call_args[0][1]['history']
        assert len(context_chatter) == 4

        mock_llm_call.reset_mock()
        await chat_system.generate_response(
            persona_name="private", user_identifier=user_id, channel=channel, message="Final message"
        )
        context_private = mock_llm_call.call_args[0][1]['history']
        assert len(context_private) == 2


@pytest.mark.asyncio
async def test_memory_type_logic(chat_system_no_zammad):
    chat_system, memory_manager, _ = chat_system_no_zammad
    user1, user2, channel = "user1", "user2", "test-channel"

    memory_manager.log_message(user1, "derpr", channel, 'user', user1, "User1 message for derpr", datetime.now(), "p5")
    time.sleep(0.01)
    memory_manager.log_message(user2, "derpr", channel, 'user', user2, "User2 message for derpr", datetime.now(), "p6")
    time.sleep(0.01)
    memory_manager.log_message(user1, "private", channel, 'user', user1, "User1 message for private", datetime.now(),
                               "p7")

    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=("", {})) as mock_llm_call:
        await chat_system.generate_response(
            persona_name="derpr", user_identifier=user1, channel=channel, message="Hi"
        )
        context_auto = mock_llm_call.call_args[0][1]['history']
        assert len(context_auto) == 3

        mock_llm_call.reset_mock()
        await chat_system.generate_response(
            persona_name="private", user_identifier=user1, channel=channel, message="Hi"
        )
        context_personal = mock_llm_call.call_args[0][1]['history']
        assert len(context_personal) == 1


@pytest.mark.asyncio
async def test_end_to_end_message_suppression(chat_system_no_zammad):
    chat_system, memory_manager, _ = chat_system_no_zammad
    channel, user_id = "test-channel", "user1"

    memory_manager.log_message(user_id, "derpr", channel, 'user', user_id, "Message 1", datetime.now(), "p10")
    time.sleep(0.01)
    memory_manager.log_message(user_id, "derpr", channel, 'user', user_id, "Message to suppress", datetime.now(),
                               "p11_suppress")
    time.sleep(0.01)
    memory_manager.log_message(user_id, "derpr", channel, 'user', user_id, "Message 3", datetime.now(), "p12")
    assert memory_manager.suppress_message_by_platform_id("p11_suppress") is True

    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=("", {})) as mock_llm_call:
        await chat_system.generate_response(
            persona_name="derpr", user_identifier=user_id, channel=channel, message="Final message"
        )
        context = mock_llm_call.call_args[0][1]['history']
        assert len(context) == 2


@pytest.mark.asyncio
async def test_persona_context_length_is_capped_by_history_limit(chat_system_no_zammad):
    chat_system, memory_manager, _ = chat_system_no_zammad

    with patch.object(memory_manager, 'get_channel_history',
                      wraps=memory_manager.get_channel_history) as mock_get_history:
        await chat_system.generate_response(
            persona_name="capped_persona", user_identifier="user1", channel="any",
            message="test", history_limit=5
        )
        mock_get_history.assert_called_once()
        call_args, _ = mock_get_history.call_args
        assert call_args[1] == 5


@pytest.mark.asyncio
async def test_context_transformation_ignores_user_author_name(chat_system_no_zammad):
    chat_system, memory_manager, _ = chat_system_no_zammad
    channel, user_id, user_name = "test-channel", "user1", "SpecificUserName"
    memory_manager.log_message(user_id, "derpr", channel, 'user', user_name, "Hello world", datetime.now())

    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=("", {})) as mock_llm_call:
        await chat_system.generate_response(
            persona_name="derpr", user_identifier=user_id, channel=channel, message="Another message"
        )
        context = mock_llm_call.call_args[0][1]['history']
        assert len(context) == 1
        assert "SpecificUserName:" not in context[0]['content']


@pytest.mark.asyncio
async def test_empty_history_is_handled_gracefully(chat_system_no_zammad):
    chat_system, _, _ = chat_system_no_zammad

    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=("Hello!", {})) as mock_llm_call:
        _, _, _ = await chat_system.generate_response(
            persona_name="derpr", user_identifier="new_user_123",
            channel="new-channel", message="First message ever"
        )
        context = mock_llm_call.call_args[0][1]['history']
        assert context == []


@pytest.mark.asyncio
async def test_history_limit_zero_for_channel_mode(chat_system_no_zammad):
    chat_system, memory_manager, _ = chat_system_no_zammad
    user1, channel = "user1", "test-channel"
    memory_manager.log_message(user1, "derpr", channel, 'user', user1, "A message that should be ignored",
                               datetime.now())

    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=("", {})) as mock_llm_call:
        await chat_system.generate_response(
            persona_name="derpr", user_identifier=user1, channel=channel,
            message="A new message", history_limit=0
        )
        context = mock_llm_call.call_args[0][1]['history']
        assert context == []


# --- Zammad-Dependent Tests ---

@pytest.mark.asyncio
async def test_new_ticket_creation_flow(chat_system_no_zammad, live_zammad_client, managed_zammad_user):
    """
    Tests the fundamental workflow of creating a new Zammad ticket.
    This test uses a hybrid of fixtures to get a mostly-mocked system with a live Zammad client.
    """
    chat_system, _, _ = chat_system_no_zammad
    # Inject the live Zammad client into the ChatSystem instance for this test
    chat_system.zammad_client = live_zammad_client

    user_info = managed_zammad_user
    created_ticket_id = None

    try:
        # We still mock the user search to avoid race conditions, but ticket creation is live.
        with patch.object(chat_system, '_find_active_ticket_for_user', new_callable=AsyncMock, return_value=None), \
                patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                             return_value=("Mock reply", {})):

            _, _, ticket_id = await chat_system.generate_response(
                persona_name="derpr", user_identifier=user_info["identifier"], channel="gmail",
                message="This is a new problem, please help."
            )

            assert ticket_id is not None
            created_ticket_id = ticket_id

            # Verify the ticket was actually created in Zammad
            ticket_data = live_zammad_client._make_request('get', f'tickets/{created_ticket_id}')
            assert ticket_data['customer_id'] == user_info["id"]
    finally:
        if created_ticket_id:
            live_zammad_client.delete_ticket(created_ticket_id)
