# tests/integration/test_full_system_flow.py

import pytest
import os
import time
from unittest.mock import patch, MagicMock, AsyncMock

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
            email=PERSISTENT_TEST_USER_EMAIL,
            firstname="Pytest",
            lastname="PersistentUser"
        )
        user_id = user_data['id']
        print(f"Created new persistent user with ID: {user_id}")

    # Pre-run cleanup: Delete any orphaned tickets for this user.
    print(f"Cleaning up any pre-existing tickets for user #{user_id}...")
    orphaned_tickets = zammad_client.search_tickets(query=f"customer_id:{user_id}")
    for ticket in orphaned_tickets:
        print(f"Deleting orphaned ticket #{ticket['id']}...")
        zammad_client.delete_ticket(ticket['id'])

    yield {
        "id": user_id,
        "identifier": f"Pytest PersistentUser <{PERSISTENT_TEST_USER_EMAIL}>"
    }
    # No user cleanup is performed. The user persists.


@pytest.fixture(scope="module")
def live_chat_system():
    """
    Fixture to set up a fully integrated ChatSystem instance for testing.
    This uses a real database, real clients, and a dedicated test persona file.
    """
    # Ensure the target directory for the test database exists
    os.makedirs(TEST_DATABASE_DIR, exist_ok=True)

    # Ensure the test database file does not exist from a previous failed run
    if os.path.exists(TEST_MEMORY_DATABASE_FILE):
        os.remove(TEST_MEMORY_DATABASE_FILE)

    # 1. Initialize all live components
    memory_manager = MemoryManager(db_path=TEST_MEMORY_DATABASE_FILE)
    memory_manager.create_schema()

    # Skip tests if Zammad is not configured
    try:
        zammad_client = ZammadClient()
        zammad_client.get_self()
    except Exception as e:
        pytest.skip(f"Skipping integration tests: Zammad client setup failed. Error: {e}")

    text_engine = TextEngine()

    # 2. Initialize the main ChatSystem, ensuring it loads from the test persona file
    if not os.path.exists(TEST_PERSONA_SAVE_FILE):
        pytest.skip(f"Skipping integration tests: Test persona file not found at {TEST_PERSONA_SAVE_FILE}")

    test_personas = real_load_personas(file_path=TEST_PERSONA_SAVE_FILE)

    with patch('src.chat_system.load_personas_from_file', return_value=test_personas):
        chat_system = ChatSystem(
            memory_manager=memory_manager,
            text_engine=text_engine,
            zammad_client=zammad_client
        )

    yield chat_system, memory_manager, zammad_client

    # 3. Teardown: clean up the test database
    print("\nCleaning up test database...")
    if memory_manager:
        memory_manager.close()
    if os.path.exists(TEST_MEMORY_DATABASE_FILE):
        os.remove(TEST_MEMORY_DATABASE_FILE)
    print("Test database cleaned up.")


@pytest.mark.asyncio
async def test_casual_chat_flow(live_chat_system):
    """
    Tests a non-ticketing interaction (e.g., from Discord).
    """
    chat_system, memory_manager, _ = live_chat_system
    user_id = "discord-user-12345"
    persona_name = "derpr"

    # Mock the LLM call to make the test faster and more reliable
    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=("Mocked LLM response.", ResponseType.LLM_GENERATION)):
        response_text, response_type = await chat_system.generate_response(
            persona_name=persona_name,
            user_identifier=user_id,
            channel="discord-general",
            message="Hello, how are you today?"
        )

        assert response_type == ResponseType.LLM_GENERATION
        assert isinstance(response_text, str) and "internal error" not in response_text.lower()
        history = memory_manager.get_personal_history(user_id, persona_name)
        assert len(history) == 2


@pytest.mark.asyncio
async def test_new_ticket_lifecycle(live_chat_system, managed_zammad_user):
    """
    Tests the full end-to-end flow for creating a NEW ticket.
    """
    chat_system, memory_manager, zammad_client = live_chat_system
    user_info = managed_zammad_user
    persona_name = "derpr"  # Use a known-good persona
    created_ticket_id = None

    try:
        mock_user_return = (user_info["id"], PERSISTENT_TEST_USER_EMAIL)
        # Mock dependencies to isolate test's focus on NEW ticket creation
        with patch.object(chat_system, '_get_or_create_zammad_user', new_callable=AsyncMock,
                          return_value=mock_user_return), \
                patch.object(chat_system, '_find_active_ticket_for_user', new_callable=AsyncMock, return_value=None), \
                patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                             return_value=("Mocked new ticket reply.", ResponseType.LLM_GENERATION)):

            response_text, _ = await chat_system.generate_response(
                persona_name=persona_name,
                user_identifier=user_info["identifier"],
                channel="gmail",
                message="My first issue, please create a ticket.",
                user_display_name="Pytest PersistentUser"
            )

            assert "internal error" not in response_text.lower()
            with memory_manager._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT zammad_ticket_id FROM User_Interactions WHERE user_identifier = ? AND role = 'assistant'",
                    (user_info["identifier"],))
                row = cursor.fetchone()
                assert row is not None, "Database query did not find the logged interaction."
                created_ticket_id = row['zammad_ticket_id']
                assert created_ticket_id is not None, "A new ticket ID was not logged to the database."

            ticket_data = zammad_client._make_request('get', f'tickets/{created_ticket_id}')
            assert ticket_data['customer_id'] == user_info["id"]
            print(f"\nNew ticket lifecycle test verified ticket #{created_ticket_id} creation.")

    finally:
        if created_ticket_id:
            print(f"Cleaning up ticket from new_ticket_lifecycle: #{created_ticket_id}...")
            zammad_client.delete_ticket(created_ticket_id)


@pytest.mark.asyncio
async def test_existing_ticket_flow(live_chat_system, managed_zammad_user):
    """
    Tests that the system correctly finds and uses an existing open ticket.
    """
    chat_system, memory_manager, zammad_client = live_chat_system
    user_info = managed_zammad_user
    persona_name = "derpr"  # Use a known-good persona
    initial_ticket_id = None

    try:
        # 1. SETUP: Create an initial ticket with an article body to ensure it's in a searchable state.
        print("\nSetting up for existing_ticket_flow: Creating initial ticket...")
        initial_ticket_data = zammad_client.create_ticket(
            title="Initial Test Ticket",
            group="Users",
            customer_id=user_info["id"],
            article_body="This is the first message that creates the ticket."
        )
        initial_ticket_id = initial_ticket_data['id']
        # Log the corresponding interaction to our local DB for context history
        memory_manager.log_interaction(
            user_identifier=user_info["identifier"],
            persona_name=persona_name,
            channel="gmail",
            user_message="This is the first message.",
            bot_response="I have created a ticket for you.",
            zammad_ticket_id=initial_ticket_id
        )
        print(f"Initial ticket #{initial_ticket_id} created and history logged.")

        # 2. ACTION & ASSERTIONS
        mock_user_return = (user_info["id"], PERSISTENT_TEST_USER_EMAIL)
        with patch.object(chat_system, '_get_or_create_zammad_user', new_callable=AsyncMock,
                          return_value=mock_user_return), \
                patch.object(chat_system, '_find_active_ticket_for_user', new_callable=AsyncMock,
                             return_value=initial_ticket_id), \
                patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock) as mock_llm_call, \
                patch.object(memory_manager, 'get_ticket_history',
                             wraps=memory_manager.get_ticket_history) as mock_get_ticket_history, \
                patch.object(zammad_client, 'create_ticket', wraps=zammad_client.create_ticket) as mock_create_ticket:

            # Make the follow-up call
            await chat_system.generate_response(
                persona_name=persona_name,
                user_identifier=user_info["identifier"],
                channel="gmail",
                message="This is a follow-up on my ticket.",
                user_display_name="Pytest PersistentUser"
            )

            # Assert correct history method was used
            mock_get_ticket_history.assert_called_once()

            # Assert that the system did NOT create a new ticket
            mock_create_ticket.assert_not_called()

            # Assert correct context was passed to LLM
            mock_llm_call.assert_called_once()
            context_object = mock_llm_call.call_args[0][1]
            system_message = context_object['history'][0]
            assert system_message['role'] == 'system'
            assert f"ticket #{initial_ticket_id}" in system_message['content']

            # Assert the follow-up was logged to the SAME ticket
            with memory_manager._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM User_Interactions WHERE zammad_ticket_id = ?",
                                      (initial_ticket_id,))
                count = cursor.fetchone()[0]
                assert count == 4, "The follow-up interaction was not logged to the existing ticket."
            print("Existing ticket flow test passed.")

    finally:
        # 3. CLEANUP
        if initial_ticket_id:
            print(f"Cleaning up ticket from existing_ticket_flow: #{initial_ticket_id}...")
            zammad_client.delete_ticket(initial_ticket_id)


@pytest.mark.asyncio
async def test_dev_command_flow(live_chat_system):
    """
    Tests that a dev command is handled correctly without involving the LLM or memory.
    """
    chat_system, memory_manager, _ = live_chat_system
    user_id = "dev-user-789"
    persona_name = "derpr"
    response_text, response_type = await chat_system.generate_response(
        persona_name=persona_name, user_identifier=user_id, channel="any", message="help"
    )
    assert response_type == ResponseType.DEV_COMMAND
    assert "Talk to a specific persona" in response_text
    history = memory_manager.get_personal_history(user_id, persona_name)
    assert len(history) == 0