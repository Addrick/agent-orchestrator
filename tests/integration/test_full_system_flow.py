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
    """Finds or creates a single persistent user for each test that needs it."""
    _, _, zammad_client = live_chat_system
    users = zammad_client.search_user(query=PERSISTENT_TEST_USER_EMAIL)
    if users:
        user_id = users[0]['id']
    else:
        user_data = zammad_client.create_user(
            email=PERSISTENT_TEST_USER_EMAIL, firstname="Pytest", lastname="PersistentUser"
        )
        user_id = user_data['id']

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
    if 'capped_persona' not in test_personas:
        test_personas['capped_persona'] = Persona(persona_name='capped_persona', model_name='mock', prompt='talk',
                                                  context_length=100)

    with patch('src.chat_system.load_personas_from_file', return_value=test_personas):
        chat_system = ChatSystem(
            memory_manager=memory_manager, text_engine=text_engine, zammad_client=zammad_client
        )

    try:
        yield chat_system, memory_manager, zammad_client
    finally:
        memory_manager.close()
        time.sleep(0.1)
        try:
            if os.path.exists(TEST_MEMORY_DATABASE_FILE):
                os.remove(TEST_MEMORY_DATABASE_FILE)
        except PermissionError as e:
            print(f"\n[TEARDOWN WARNING] Could not remove test database file due to a lock: {e}")


@pytest.mark.asyncio
async def test_new_ticket_creation_flow(live_chat_system, managed_zammad_user):
    """
    Tests the fundamental workflow of creating a new Zammad ticket from a user message.
    """
    chat_system, _, zammad_client = live_chat_system
    user_info = managed_zammad_user
    created_ticket_id = None

    try:
        mock_user_return = (user_info["id"], PERSISTENT_TEST_USER_EMAIL)
        # Mock the text engine to return a text response, not a tool call
        mock_llm_response = ({'type': 'text', 'content': 'Mock reply'}, {})
        with patch.object(chat_system, '_get_or_create_zammad_user', new_callable=AsyncMock,
                          return_value=mock_user_return), \
                patch.object(chat_system, '_find_active_ticket_for_user', new_callable=AsyncMock, return_value=None), \
                patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                             return_value=mock_llm_response):

            _, _, ticket_id = await chat_system.generate_response(
                persona_name="derpr", user_identifier=user_info["identifier"], channel="gmail",
                message="This is a new problem, please help."
            )

            assert ticket_id is not None
            created_ticket_id = ticket_id

            # Use the direct get_ticket method for verification
            ticket_data = zammad_client.get_ticket(created_ticket_id)
            # FIX: Access the top-level customer_id key
            assert ticket_data['customer_id'] == user_info["id"]
    finally:
        if created_ticket_id:
            zammad_client.delete_ticket(created_ticket_id)


@pytest.mark.asyncio
async def test_context_transformation_and_multi_user_differentiation(live_chat_system):
    """Tests that channel history context is correctly transformed for the LLM."""
    chat_system, memory_manager, _ = live_chat_system
    channel, user1_id, user2_id = "test-channel", "user1", "user2"

    memory_manager.log_message(user1_id, "derpr", channel, 'user', "UserOne", "Hello from user one", datetime.now())
    time.sleep(0.01)
    memory_manager.log_message(user2_id, "chatter", channel, 'user', "UserTwo", "Hello from user two", datetime.now())
    time.sleep(0.01)
    memory_manager.log_message(user2_id, "chatter", channel, 'assistant', 'chatter', "Reply from chatter bot",
                               datetime.now())

    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=({'type': 'text', 'content': ''}, {})) as mock_llm_call:
        await chat_system.generate_response(
            persona_name="derpr", user_identifier=user1_id, channel=channel, message="My turn"
        )

        context = mock_llm_call.call_args[0][1]['history']
        assert len(context) == 4
        assert context[0] == {'role': 'user', 'content': 'UserOne: Hello from user one'}
        assert context[1] == {'role': 'user', 'content': 'UserTwo: Hello from user two'}
        assert context[2] == {'role': 'user', 'content': 'chatter: Reply from chatter bot'}
        assert context[3] == {'role': 'user', 'content': 'My turn'}


@pytest.mark.asyncio
async def test_ticket_history_priority_over_channel_history(live_chat_system, managed_zammad_user):
    """Tests that ticket history is used exclusively, even when channel history also exists."""
    chat_system, memory_manager, zammad_client = live_chat_system
    user_info = managed_zammad_user
    ticket_id = None
    try:
        ticket_data = zammad_client.create_ticket(title="Test", group="Users", customer_id=user_info['id'])
        ticket_id = ticket_data['id']
        memory_manager.log_message(user_info['identifier'], "derpr", "gmail", 'user', user_info['identifier'],
                                   "This is a ticket message.", datetime.now(), zammad_ticket_id=ticket_id)
        time.sleep(0.01)
        memory_manager.log_message(user_info['identifier'], "derpr", "gmail", 'user', user_info['identifier'],
                                   "This is a general channel message.", datetime.now())
        with patch.object(chat_system, '_find_active_ticket_for_user', new_callable=AsyncMock, return_value=ticket_id), \
                patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                             return_value=({'type': 'text', 'content': ''}, {})) as mock_llm_call:
            await chat_system.generate_response(
                persona_name="derpr", user_identifier=user_info['identifier'], channel="gmail", message="Follow up"
            )
            context = mock_llm_call.call_args[0][1]['history']
            assert len(context) == 3
            assert "This is a ticket message." in context[1]['content']
            assert "Follow up" in context[2]['content']
    finally:
        if ticket_id:
            zammad_client.delete_ticket(ticket_id)


@pytest.mark.asyncio
async def test_end_to_end_message_suppression(live_chat_system):
    """Tests that the ChatSystem correctly respects the suppression list from the MemoryManager."""
    chat_system, memory_manager, _ = live_chat_system
    channel, user_id = "test-channel", "user1"
    memory_manager.log_message(user_id, "derpr", channel, 'user', user_id, "Message 1", datetime.now(), "p10")
    time.sleep(0.01)
    memory_manager.log_message(user_id, "derpr", channel, 'user', user_id, "Message to suppress", datetime.now(),
                               "p11_suppress")
    time.sleep(0.01)
    memory_manager.log_message(user_id, "derpr", channel, 'user', user_id, "Message 3", datetime.now(), "p12")
    assert memory_manager.suppress_message_by_platform_id("p11_suppress") is True
    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=({'type': 'text', 'content': ''}, {})) as mock_llm_call:
        await chat_system.generate_response(
            persona_name="derpr", user_identifier=user_id, channel=channel, message="Final message"
        )
        context = mock_llm_call.call_args[0][1]['history']
        assert len(context) == 3
        assert "Message to suppress" not in [msg['content'] for msg in context]


@pytest.mark.asyncio
async def test_persona_context_length_is_capped_by_history_limit(live_chat_system):
    """Tests that the per-call history_limit acts as a cap on a persona's context_length."""
    chat_system, memory_manager, _ = live_chat_system
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
async def test_context_transformation_ignores_user_author_name(live_chat_system, managed_zammad_user):
    """Tests that the context transformation does not prepend a name for 'user' roles in ticket mode."""
    chat_system, memory_manager, zammad_client = live_chat_system
    user_info = managed_zammad_user
    ticket_id = None
    try:
        ticket_data = zammad_client.create_ticket(title="Test", group="Users", customer_id=user_info['id'])
        ticket_id = ticket_data['id']
        memory_manager.log_message(user_info['identifier'], "derpr", "gmail", 'user', "SpecificUserName", "Hello world",
                                   datetime.now(), zammad_ticket_id=ticket_id)
        with patch.object(chat_system, '_find_active_ticket_for_user', new_callable=AsyncMock, return_value=ticket_id), \
                patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                             return_value=({'type': 'text', 'content': ''}, {})) as mock_llm_call:
            await chat_system.generate_response(
                persona_name="derpr", user_identifier=user_info['identifier'], channel="gmail",
                message="Another message"
            )
            context = mock_llm_call.call_args[0][1]['history']
            assert len(context) == 3  # System prompt + 1 DB message + 1 current message
            assert context[1]['content'] == "Hello world"
            assert "SpecificUserName:" not in context[1]['content']
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
                      return_value=({'type': 'text', 'content': 'Hello!'}, {})) as mock_llm_call:
        _, _, _ = await chat_system.generate_response(
            persona_name="derpr", user_identifier="new_user_123",
            channel="new-channel", message="First message ever"
        )
        context = mock_llm_call.call_args[0][1]['history']
        assert len(context) == 1
        assert context[0] == {'role': 'user', 'content': 'First message ever'}


@pytest.mark.asyncio
async def test_history_limit_zero_for_channel_mode(live_chat_system):
    """
    Tests that history_limit=0 is respected for channel/auto mode personas.
    """
    chat_system, memory_manager, _ = live_chat_system
    user1, channel = "user1", "test-channel"
    memory_manager.log_message(user1, "derpr", channel, 'user', user1, "A message that should be ignored",
                               datetime.now())
    with patch.object(chat_system.text_engine, 'generate_response', new_callable=AsyncMock,
                      return_value=({'type': 'text', 'content': ''}, {})) as mock_llm_call:
        await chat_system.generate_response(
            persona_name="derpr", user_identifier=user1, channel=channel,
            message="A new message", history_limit=0
        )
        context = mock_llm_call.call_args[0][1]['history']
        assert len(context) == 1
        assert context[0] == {'role': 'user', 'content': 'A new message'}
