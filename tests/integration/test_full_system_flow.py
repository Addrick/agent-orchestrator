# tests/integration/test_full_system_flow.py

import pytest
import os
import time
from datetime import datetime
from unittest.mock import patch, AsyncMock

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
def managed_zammad_user(live_chat_system):
    """Finds or creates a single persistent user for all tests in this module."""
    _, _, zammad_client = live_chat_system
    users = zammad_client.search_user(query=PERSISTENT_TEST_USER_EMAIL)
    if users:
        user_id = users[0]['id']
    else:
        user_data = zammad_client.create_user(
            email=PERSISTENT_TEST_USER_EMAIL, firstname="Pytest", lastname="PersistentUser"
        )
        user_id = user_data['id']

    # Pre-run cleanup for this function
    orphaned_tickets = zammad_client.search_tickets(query=f"customer_id:{user_id}")
    for ticket in orphaned_tickets:
        zammad_client.delete_ticket(ticket['id'])

    yield {"id": user_id, "identifier": f"Pytest PersistentUser <{PERSISTENT_TEST_USER_EMAIL}>"}


@pytest.fixture(scope="function")
def live_chat_system():
    """Sets up a fully integrated ChatSystem instance for each test function."""
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
        pytest.skip(f"Skipping tests: Test persona file not found at {TEST_PERSONA_SAVE_FILE}")

    test_personas = real_load_personas(file_path=TEST_PERSONA_SAVE_FILE)

    from src.persona import Persona
    if 'chatter' not in test_personas:
        test_personas['chatter'] = Persona(persona_name='chatter', model_name='mock', prompt='talk',
                                           memory_type='channel')
    if 'private' not in test_personas:
        test_personas['private'] = Persona(persona_name='private', model_name='mock', prompt='private',
                                           memory_type='personal')

    with patch('src.chat_system.load_personas_from_file', return_value=test_personas):
        chat_system = ChatSystem(
            memory_manager=memory_manager, text_engine=text_engine, zammad_client=zammad_client
        )

    yield chat_system, memory_manager, zammad_client

    memory_manager.close()
    if os.path.exists(TEST_MEMORY_DATABASE_FILE):
        os.remove(TEST_MEMORY_DATABASE_FILE)


@pytest.mark.asyncio
async def test_context_transformation_logic(live_chat_system):
    """
    Tests the core logic of transforming raw history into LLM-ready context.
    """
    chat_system, memory_manager, _ = live_chat_system
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
            persona_name="chatter", user_identifier=user_id, channel=channel, message="Final message", history_limit=10
        )

        context_chatter = mock_llm_call.call_args[0][1]['history']
        assert len(context_chatter) == 4
        assert {'role': 'assistant', 'content': 'Hi from chatter!'} in context_chatter
        assert {'role': 'user', 'content': 'private: Hi from private bot.'} in context_chatter

        mock_llm_call.reset_mock()

        await chat_system.generate_response(
            persona_name="private", user_identifier=user_id, channel=channel, message="Final message", history_limit=10
        )

        context_private = mock_llm_call.call_args[0][1]['history']
        assert len(context_private) == 2
        assert {'role': 'assistant', 'content': 'Hi from private bot.'} in context_private


@pytest.mark.asyncio
async def test_memory_type_logic(live_chat_system):
    """
    Tests the persona-based memory_type override and the channel-first default.
    """
    chat_system, memory_manager, _ = live_chat_system
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
            persona_name="derpr", user_identifier=user1, channel=channel, message="Hi", history_limit=10
        )
        context_auto = mock_llm_call.call_args[0][1]['history']
        assert len(context_auto) == 3

        mock_llm_call.reset_mock()

        await chat_system.generate_response(
            persona_name="private", user_identifier=user1, channel=channel, message="Hi", history_limit=10
        )
        context_personal = mock_llm_call.call_args[0][1]['history']
        assert len(context_personal) == 1
        assert context_personal[0]['content'] == "User1 message for private"


@pytest.mark.asyncio
async def test_ticket_priority_overrides_personal_memory_mode(live_chat_system, managed_zammad_user):
    """
    Tests that ticket history is used even if the persona is set to 'personal' mode.
    """
    chat_system, memory_manager, zammad_client = live_chat_system
    user_info = managed_zammad_user
    ticket_id = None

    try:
        ticket_data = zammad_client.create_ticket(title="Test", group="Users", customer_id=user_info['id'])
        ticket_id = ticket_data['id']

        memory_manager.log_message(user_info['identifier'], "private", "gmail", 'user', user_info['identifier'],
                                   "Ticket message", datetime.now(), zammad_ticket_id=ticket_id)
        time.sleep(0.01)
        memory_manager.log_message(user_info['identifier'], "private", "gmail", 'user', user_info['identifier'],
                                   "Personal message", datetime.now())

        with patch.object(chat_system, '_find_active_ticket_for_user', new_callable=AsyncMock, return_value=ticket_id), \
                patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                             return_value=("", {})) as mock_llm_call:

            # Use the 'private' persona, which is configured for personal history
            await chat_system.generate_response(
                persona_name="private", user_identifier=user_info['identifier'], channel="gmail", message="Follow up",
                history_limit=10
            )

            context = mock_llm_call.call_args[0][1]['history']
            assert len(context) == 2  # System prompt + 1 ticket message
            assert "Ticket message" in [m['content'] for m in context]
            assert "Personal message" not in [m['content'] for m in context]

    finally:
        if ticket_id:
            zammad_client.delete_ticket(ticket_id)


@pytest.mark.asyncio
async def test_empty_history_is_handled_gracefully(live_chat_system):
    """
    Tests that the system works correctly for a new user with no history.
    """
    chat_system, _, _ = live_chat_system

    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=("Hello!", {})) as mock_llm_call:
        response_text, response_type, ticket_id = await chat_system.generate_response(
            persona_name="derpr",
            user_identifier="new_user_123",
            channel="new-channel",
            message="First message ever"
        )

        assert response_type == ResponseType.LLM_GENERATION
        assert response_text == "Hello!"

        mock_llm_call.assert_called_once()
        context = mock_llm_call.call_args[0][1]['history']
        assert context == []  # The history passed to the LLM should be empty
