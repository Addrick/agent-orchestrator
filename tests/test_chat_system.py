# tests/test_chat_system.py

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.chat_system import ChatSystem
from src.database.memory_manager import MemoryManager
from src.engine import TextEngine
from src.clients.zammad_client import ZammadClient


@pytest.fixture
def chat_system_with_mocks():
    """Provides a ChatSystem instance with mocked dependencies for each test."""
    mock_memory_manager = MagicMock(spec=MemoryManager)
    mock_text_engine = MagicMock(spec=TextEngine)
    mock_zammad_client = MagicMock(spec=ZammadClient)
    # Mock the api_url attribute so we can parse its hostname in tests
    mock_zammad_client.api_url = "http://test.zammad.local"

    mock_text_engine.generate_response = AsyncMock(return_value=("LLM Reply", {}))
    # Use MagicMock for synchronous methods called via asyncio.to_thread
    mock_zammad_client.create_ticket = MagicMock(return_value={'id': 12345})
    mock_zammad_client.add_article_to_ticket = MagicMock()
    mock_zammad_client.search_user = MagicMock()
    mock_zammad_client.create_user = MagicMock()

    with patch('src.chat_system.load_personas_from_file') as mock_load_personas:
        mock_persona = MagicMock()
        mock_persona.get_config_for_engine.return_value = {"model_name": "test-model"}
        mock_persona.get_prompt.return_value = "You are a test bot."
        mock_persona.get_context_length.return_value = 10
        mock_load_personas.return_value = {"test_persona": mock_persona}

        system = ChatSystem(
            memory_manager=mock_memory_manager,
            text_engine=mock_text_engine,
            zammad_client=mock_zammad_client
        )
        system.bot_logic.preprocess_message = AsyncMock(return_value=None)

        # Mock the helper methods on the instance for testing the main flow
        system._should_create_ticket = MagicMock()
        system._get_or_create_zammad_user = AsyncMock(return_value=101)  # Default to returning a user ID
        system._find_ticket_id_in_message = MagicMock()

        yield system, mock_memory_manager, mock_text_engine, mock_zammad_client, mock_load_personas


# --- Tests for the main generate_response flow ---

@pytest.mark.asyncio
async def test_casual_mode_flow(chat_system_with_mocks):
    """Tests that non-ticket channels only log to memory."""
    system, mem_mock, _, zammad_mock, _ = chat_system_with_mocks
    system._should_create_ticket.return_value = False  # Simulate a non-ticket channel

    await system.generate_response(
        persona_name='test_persona',
        user_identifier='user_discord_1',
        channel='discord',
        message='Just chatting'
    )

    system._get_or_create_zammad_user.assert_not_called()
    zammad_mock.create_ticket.assert_not_called()
    zammad_mock.add_article_to_ticket.assert_not_called()

    mem_mock.log_interaction.assert_called_once_with(
        user_identifier='user_discord_1',
        channel='discord',
        user_message='Just chatting',
        bot_response='LLM Reply',
        zammad_ticket_id=None
    )


@pytest.mark.asyncio
async def test_ticket_mode_new_ticket(chat_system_with_mocks):
    """Tests that ticket channels create a new Zammad ticket."""
    system, mem_mock, _, zammad_mock, _ = chat_system_with_mocks
    system._should_create_ticket.return_value = True  # It's a ticket request
    system._find_ticket_id_in_message.return_value = None  # It's a new ticket

    await system.generate_response(
        persona_name='test_persona',
        user_identifier='user_gmail_1',
        channel='gmail',
        message='Help me!'
    )

    system._get_or_create_zammad_user.assert_awaited_once_with('user_gmail_1', 'gmail')
    zammad_mock.create_ticket.assert_called_once()
    # Bot reply is added to the newly created ticket
    zammad_mock.add_article_to_ticket.assert_called_once_with(
        ticket_id=12345,  # from the mocked create_ticket return value
        body="LLM Reply"
    )

    mem_mock.log_interaction.assert_called_once_with(
        user_identifier='user_gmail_1',
        channel='gmail',
        user_message='Help me!',
        bot_response='LLM Reply',
        zammad_ticket_id=12345
    )


@pytest.mark.asyncio
async def test_ticket_mode_existing_ticket(chat_system_with_mocks):
    """Tests that messages for existing tickets add articles instead of creating new ones."""
    system, mem_mock, _, zammad_mock, _ = chat_system_with_mocks
    system._should_create_ticket.return_value = True
    system._find_ticket_id_in_message.return_value = 999  # Existing ticket

    message_with_id = "More info for [Ticket#999]"
    await system.generate_response(
        persona_name='test_persona',
        user_identifier='user_gmail_2',
        channel='gmail',
        message=message_with_id
    )

    zammad_mock.create_ticket.assert_not_called()

    # Called twice: user message, then bot reply
    assert zammad_mock.add_article_to_ticket.call_count == 2
    zammad_mock.add_article_to_ticket.assert_any_call(ticket_id=999, body=message_with_id)
    zammad_mock.add_article_to_ticket.assert_any_call(ticket_id=999, body="LLM Reply")

    mem_mock.log_interaction.assert_called_once_with(
        user_identifier='user_gmail_2',
        channel='gmail',
        user_message=message_with_id,
        bot_response='LLM Reply',
        zammad_ticket_id=999
    )


# --- Standalone tests for helper methods ---

@patch('src.chat_system.SUPPORT_CHANNELS', ['gmail', 'tech-support'])
def test_should_create_ticket_logic(chat_system_with_mocks):
    system, _, _, _, _ = chat_system_with_mocks
    # Un-mock the method for this specific test
    system._should_create_ticket.side_effect = system.__class__._should_create_ticket.__get__(system)

    assert system._should_create_ticket('gmail', 'help') is True
    assert system._should_create_ticket('tech-support', 'help') is True
    assert system._should_create_ticket('discord', 'hello') is False
    assert system._should_create_ticket('GMAIL', 'case-insensitivity test') is True


def test_find_ticket_id_logic(chat_system_with_mocks):
    system, _, _, _, _ = chat_system_with_mocks
    system._find_ticket_id_in_message.side_effect = system.__class__._find_ticket_id_in_message.__get__(system)

    assert system._find_ticket_id_in_message("Message for [Ticket#12345]") == 12345
    assert system._find_ticket_id_in_message("message for [ticket#54321]") == 54321
    assert system._find_ticket_id_in_message("No ticket here") is None
    assert system._find_ticket_id_in_message("Malformed [Ticket#abc]") is None


@pytest.mark.asyncio
@patch('asyncio.to_thread')
async def test_get_or_create_zammad_user_existing_user(mock_to_thread, chat_system_with_mocks):
    """Tests _get_or_create_zammad_user when the user already exists in Zammad."""
    system, _, _, zammad_mock, _ = chat_system_with_mocks
    system._get_or_create_zammad_user.side_effect = system.__class__._get_or_create_zammad_user.__get__(system)

    zammad_mock.search_user.return_value = [{'id': 505}]
    mock_to_thread.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

    user_id = await system._get_or_create_zammad_user("Test User <test@example.com>", "gmail")

    assert user_id == 505
    zammad_mock.search_user.assert_called_once_with("test@example.com")
    zammad_mock.create_user.assert_not_called()


@pytest.mark.asyncio
@patch('asyncio.to_thread')
async def test_get_or_create_zammad_user_new_user(mock_to_thread, chat_system_with_mocks):
    """Tests _get_or_create_zammad_user when a new user needs to be created."""
    system, _, _, zammad_mock, _ = chat_system_with_mocks
    system._get_or_create_zammad_user.side_effect = system.__class__._get_or_create_zammad_user.__get__(system)

    zammad_mock.search_user.return_value = []  # No user found
    zammad_mock.create_user.return_value = {'id': 707}
    mock_to_thread.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

    user_id = await system._get_or_create_zammad_user("New User <new@example.com>", "gmail")

    assert user_id == 707
    zammad_mock.search_user.assert_called_once_with("new@example.com")
    zammad_mock.create_user.assert_called_once_with(
        email="new@example.com",
        firstname="New",
        lastname="User",
        note=None
    )


@pytest.mark.asyncio
@patch('asyncio.to_thread')
async def test_get_or_create_zammad_user_for_non_email_id(mock_to_thread, chat_system_with_mocks):
    """Tests that a non-email identifier creates a dummy email and a descriptive note."""
    system, _, _, zammad_mock, _ = chat_system_with_mocks
    system._get_or_create_zammad_user.side_effect = system.__class__._get_or_create_zammad_user.__get__(system)

    discord_id = "321783731146850305"
    expected_email = f"discord-{discord_id}@test.zammad.local"

    zammad_mock.search_user.return_value = []
    zammad_mock.create_user.return_value = {'id': 909}
    mock_to_thread.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

    user_id = await system._get_or_create_zammad_user(discord_id, "discord")

    assert user_id == 909
    zammad_mock.search_user.assert_called_once_with(expected_email)
    zammad_mock.create_user.assert_called_once_with(
        email=expected_email,
        firstname="Discord User",
        lastname=discord_id,
        note=f"Auto-generated user from Discord. Original identifier: {discord_id}"
    )
