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

    # Mock the LLM call to make the test faster and more reliable
    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=("Mocked LLM response.", ResponseType.LLM_GENERATION)):
        response_text, response_type = await chat_system.generate_response(
            persona_name="derpr",
            user_identifier=user_id,
            channel="discord-general",
            message="Hello, how are you today?"
        )

        assert response_type == ResponseType.LLM_GENERATION
        assert isinstance(response_text, str) and "internal error" not in response_text.lower()
        history = memory_manager.get_history(user_id)
        assert len(history) == 1


def test_ticket_direct_api_lifecycle(live_chat_system, managed_zammad_user):
    """
    A direct, lower-level test to ensure the Zammad client can create and delete
    a ticket for the persistent test user.
    """
    _, _, zammad_client = live_chat_system
    user_id = managed_zammad_user["id"]
    created_ticket_id = None

    try:
        print(f"\nDirect API Test: Creating ticket for user {user_id}...")
        ticket_data = zammad_client.create_ticket(
            title="Direct API Test Ticket", group="Users", customer_id=user_id, article_body="This is a test."
        )
        assert ticket_data and 'id' in ticket_data
        created_ticket_id = ticket_data['id']
        print(f"Direct API Test: Created ticket with ID {created_ticket_id}")
    finally:
        if created_ticket_id:
            print(f"Direct API Test: Cleaning up ticket #{created_ticket_id}...")
            zammad_client.delete_ticket(created_ticket_id)
            print("Direct API Test: Zammad cleanup complete.")


@pytest.mark.asyncio
async def test_new_ticket_lifecycle(live_chat_system, managed_zammad_user):
    """
    Tests the full end-to-end flow using the persistent user.
    This test patches flaky network dependencies (LLM and user search) to be deterministic.
    """
    chat_system, memory_manager, zammad_client = live_chat_system
    user_info = managed_zammad_user
    created_ticket_id = None

    try:
        # The return_value must be a tuple (user_id, email) to match the real method's signature.
        mock_user_return = (user_info["id"], PERSISTENT_TEST_USER_EMAIL)

        # Patch flaky dependencies to isolate the test's focus
        with patch.object(chat_system, '_get_or_create_zammad_user', new_callable=AsyncMock,
                          return_value=mock_user_return), \
                patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                             return_value=("Mocked LLM reply.", ResponseType.LLM_GENERATION)):

            response_text, response_type = await chat_system.generate_response(
                persona_name="derpr",
                user_identifier=user_info["identifier"],
                channel="gmail",
                message="My computer is on fire, please help!",
                user_display_name="Pytest PersistentUser"
            )

            # Assertions must be inside the 'with' block to ensure mocks are active.
            assert response_type == ResponseType.LLM_GENERATION
            assert "internal error" not in response_text.lower(), f"Generate response returned an error: {response_text}"

            history = memory_manager.get_history(user_info["identifier"])
            assert len(history) == 1
            with memory_manager._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT zammad_ticket_id, bot_response FROM User_Interactions WHERE user_identifier = ?",
                    (user_info["identifier"],))
                row = cursor.fetchone()
                created_ticket_id = row['zammad_ticket_id']
                bot_response = row['bot_response']
                assert created_ticket_id is not None, f"Ticket ID was not logged in the database. Bot response was: '{bot_response}'"

            ticket_data = zammad_client._make_request('get', f'tickets/{created_ticket_id}')
            assert ticket_data['customer_id'] == user_info["id"]
            print(f"\nHigh-level test verified ticket #{created_ticket_id} creation.")

    finally:
        if created_ticket_id:
            print(f"High-level test: Cleaning up ticket #{created_ticket_id}...")
            zammad_client.delete_ticket(created_ticket_id)
            print("High-level test: Cleanup complete.")


@pytest.mark.asyncio
async def test_dev_command_flow(live_chat_system):
    """
    Tests that a dev command is handled correctly without involving the LLM or memory.
    """
    chat_system, memory_manager, _ = live_chat_system
    user_id = "dev-user-789"
    response_text, response_type = await chat_system.generate_response(
        persona_name="derpr", user_identifier=user_id, channel="any", message="help"
    )
    assert response_type == ResponseType.DEV_COMMAND
    assert "Talk to a specific persona" in response_text
    history = memory_manager.get_history(user_id)
    assert len(history) == 0
