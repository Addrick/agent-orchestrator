# tests/integration/test_full_system_flow.py

import pytest
import os
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock

# Mark all tests in this file as 'integration'.
pytestmark = pytest.mark.integration

# Import components and config AFTER marking tests, to ensure pytest setup is complete
from src.chat_system import ChatSystem, ResponseType
from src.database.memory_manager import MemoryManager
from src.engine import TextEngine
from src.clients.zammad_client import ZammadClient
from src.utils.save_utils import load_personas_from_file as real_load_personas
from config.global_config import TEST_MEMORY_DATABASE_FILE, TEST_DATABASE_DIR, TEST_PERSONA_SAVE_FILE

# Use a static, persistent user for all Zammad integration tests.
PERSISTENT_TEST_USER_EMAIL = "pytest-integration-user@zammad.local"


@pytest.fixture(scope="module")
def managed_zammad_user(live_chat_system):
    """
    Finds or creates a single persistent user for all tests in this module.
    Crucially, it cleans up any of this user's tickets before the test session begins.
    It yields the user's ID and full identifier string.
    """
    _, _, zammad_client = live_chat_system

    print(f"\nSetting up persistent test user: {PERSISTENT_TEST_USER_EMAIL}")
    users = zammad_client.search_user(query=PERSISTENT_TEST_USER_EMAIL)
    if users:
        user_id = users[0]['id']
        print(f"Found existing test user with ID: {user_id}")
    else:
        print("Test user not found, creating...")
        user_data = zammad_client.create_user(
            email=PERSISTENT_TEST_USER_EMAIL, firstname="Pytest", lastname="PersistentUser"
        )
        user_id = user_data['id']
        print(f"Created new persistent user with ID: {user_id}")

    # Pre-run cleanup: Delete any orphaned tickets for this user.
    print(f"Cleaning up any pre-existing tickets for user #{user_id}...")
    orphaned_tickets = zammad_client.search_tickets(query=f"customer_id:{user_id}")
    for ticket in orphaned_tickets:
        print(f"Deleting orphaned ticket #{ticket['id']}...")
        zammad_client.delete_ticket(ticket['id'])

    yield {"id": user_id, "identifier": f"Pytest PersistentUser <{PERSISTENT_TEST_USER_EMAIL}>"}


@pytest.fixture(scope="module")
def live_chat_system():
    """
    Sets up a fully integrated ChatSystem instance for testing.
    """
    os.makedirs(TEST_DATABASE_DIR, exist_ok=True)
    if os.path.exists(TEST_MEMORY_DATABASE_FILE):
        os.remove(TEST_MEMORY_DATABASE_FILE)

    memory_manager = MemoryManager(db_path=TEST_MEMORY_DATABASE_FILE)
    memory_manager.create_schema()

    try:
        zammad_client = ZammadClient()
        zammad_client.get_self()
    except Exception as e:
        pytest.skip(f"Skipping integration tests: Zammad client setup failed. Error: {e}")

    text_engine = TextEngine()
    if not os.path.exists(TEST_PERSONA_SAVE_FILE):
        pytest.skip(f"Skipping integration tests: Test persona file not found at {TEST_PERSONA_SAVE_FILE}")

    test_personas = real_load_personas(file_path=TEST_PERSONA_SAVE_FILE)

    with patch('src.chat_system.load_personas_from_file', return_value=test_personas):
        chat_system = ChatSystem(
            memory_manager=memory_manager, text_engine=text_engine, zammad_client=zammad_client
        )

    yield chat_system, memory_manager, zammad_client

    print("\nCleaning up test database...")
    memory_manager.close()
    if os.path.exists(TEST_MEMORY_DATABASE_FILE):
        os.remove(TEST_MEMORY_DATABASE_FILE)
    print("Test database cleaned up.")


@pytest.mark.asyncio
async def test_casual_chat_flow(live_chat_system):
    """
    Tests the new flow: generate response, then log both messages.
    """
    chat_system, memory_manager, _ = live_chat_system
    user_id, persona_name, channel = "discord-user-12345", "derpr", "discord-general"
    user_message = "Hello, how are you today?"
    user_message_id, bot_message_id = "user_msg_1", "bot_msg_1"
    user_timestamp = datetime.now()

    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=("Mocked LLM response.", {})):
        response_text, response_type, ticket_id = await chat_system.generate_response(
            persona_name=persona_name, user_identifier=user_id, channel=channel, message=user_message
        )

        assert response_type == ResponseType.LLM_GENERATION
        assert ticket_id is None

        # Simulate the interface logging the interaction
        memory_manager.log_message(user_id, persona_name, channel, 'user', user_message, user_timestamp,
                                   user_message_id, ticket_id)
        memory_manager.log_message(user_id, persona_name, channel, 'assistant', response_text,
                                   user_timestamp + timedelta(microseconds=1), bot_message_id, ticket_id)

        history = memory_manager.get_personal_history(user_id, persona_name)
        assert len(history) == 2


@pytest.mark.asyncio
async def test_new_ticket_flow(live_chat_system, managed_zammad_user):
    """
    Tests that the system correctly identifies the need for a new ticket and returns its ID.
    """
    chat_system, memory_manager, zammad_client = live_chat_system
    user_info = managed_zammad_user
    persona_name = "derpr"
    created_ticket_id = None

    try:
        mock_user_return = (user_info["id"], PERSISTENT_TEST_USER_EMAIL)
        with patch.object(chat_system, '_get_or_create_zammad_user', new_callable=AsyncMock,
                          return_value=mock_user_return), \
                patch.object(chat_system, '_find_active_ticket_for_user', new_callable=AsyncMock, return_value=None), \
                patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                             return_value=("Mocked reply.", {})):

            # The interface calls generate_response
            response_text, response_type, ticket_id = await chat_system.generate_response(
                persona_name=persona_name, user_identifier=user_info["identifier"], channel="gmail",
                message="My first issue.", user_display_name="Pytest User"
            )

            assert response_type == ResponseType.LLM_GENERATION
            assert ticket_id is not None, "generate_response did not return a new ticket ID."
            created_ticket_id = ticket_id

            # The interface then logs the messages with the returned ticket_id
            memory_manager.log_message(user_info["identifier"], persona_name, "gmail", 'user', "My first issue.",
                                       datetime.now(), "user_msg_2", created_ticket_id)

            ticket_data = zammad_client._make_request('get', f'tickets/{created_ticket_id}')
            assert ticket_data['customer_id'] == user_info["id"]

    finally:
        if created_ticket_id:
            zammad_client.delete_ticket(created_ticket_id)


@pytest.mark.asyncio
async def test_existing_ticket_flow_and_suppression(live_chat_system, managed_zammad_user):
    """
    Tests finding an existing ticket, using its history, and then suppressing a message from that history.
    """
    chat_system, memory_manager, zammad_client = live_chat_system
    user_info = managed_zammad_user
    persona_name = "derpr"
    initial_ticket_id = None

    try:
        # 1. SETUP: Create an initial ticket and log a full interaction for it.
        initial_ticket_data = zammad_client.create_ticket(
            title="Initial Test Ticket", group="Users", customer_id=user_info["id"], article_body="Initial problem."
        )
        initial_ticket_id = initial_ticket_data['id']
        ts = datetime.now()
        memory_manager.log_message(user_info["identifier"], persona_name, "gmail", "user", "Initial problem.", ts,
                                   "user_msg_3", initial_ticket_id)
        memory_manager.log_message(user_info["identifier"], persona_name, "gmail", "assistant", "Initial reply.",
                                   ts + timedelta(microseconds=1), "bot_msg_3", initial_ticket_id)

        # 2. ACTION: Follow up on the ticket
        mock_user_return = (user_info["id"], PERSISTENT_TEST_USER_EMAIL)
        with patch.object(chat_system, '_get_or_create_zammad_user', new_callable=AsyncMock,
                          return_value=mock_user_return), \
                patch.object(chat_system, '_find_active_ticket_for_user', new_callable=AsyncMock,
                             return_value=initial_ticket_id), \
                patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock) as mock_llm_call:

            _, _, returned_ticket_id = await chat_system.generate_response(
                persona_name=persona_name, user_identifier=user_info["identifier"], channel="gmail",
                message="Follow-up question."
            )

            # Assert that the correct existing ticket ID was identified and returned
            assert returned_ticket_id == initial_ticket_id

            # Assert that the context sent to the LLM included the initial interaction
            mock_llm_call.assert_called_once()
            context_object = mock_llm_call.call_args[0][1]
            assert len(context_object['history']) == 3  # System prompt + 2 messages
            assert context_object['history'][1]['content'] == "Initial problem."

        # 3. SUPPRESSION: Now, suppress the user's initial message
        suppressed = memory_manager.suppress_message_by_platform_id("user_msg_3")
        assert suppressed is True

        # 4. VERIFICATION: Call generate_response again and verify the context is different
        with patch.object(chat_system, '_get_or_create_zammad_user', new_callable=AsyncMock,
                          return_value=mock_user_return), \
                patch.object(chat_system, '_find_active_ticket_for_user', new_callable=AsyncMock,
                             return_value=initial_ticket_id), \
                patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock) as mock_llm_call_2:

            await chat_system.generate_response(
                persona_name=persona_name, user_identifier=user_info["identifier"], channel="gmail",
                message="Another follow-up."
            )

            mock_llm_call_2.assert_called_once()
            context_object_2 = mock_llm_call_2.call_args[0][1]
            assert len(context_object_2['history']) == 2  # System prompt + 1 message
            assert "Initial problem." not in [msg['content'] for msg in context_object_2['history']]
            assert "Initial reply." in [msg['content'] for msg in context_object_2['history']]

    finally:
        if initial_ticket_id:
            zammad_client.delete_ticket(initial_ticket_id)
