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
    mock_zammad_client.api_url = "http://test.zammad.local"

    mock_text_engine.generate_response = AsyncMock(return_value=("LLM Reply", {}))
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

        system._should_create_ticket = MagicMock()
        system._get_or_create_zammad_user = AsyncMock(return_value=(101, "user@example.com"))
        system._find_ticket_id_in_message = MagicMock()
        # Also mock the new helper for active ticket search
        system._find_active_ticket_for_user = AsyncMock(return_value=None)

        yield system, mock_memory_manager, mock_text_engine, mock_zammad_client, mock_load_personas


# --- Tests for the main generate_response flow ---

@pytest.mark.asyncio
async def test_ticket_mode_updates_existing_open_ticket(chat_system_with_mocks):
    """
    Tests that if an active open ticket is found for a user, a new one is NOT created,
    and the message is added as an article to the existing ticket.
    """
    system, _, _, zammad_mock, _ = chat_system_with_mocks
    system._should_create_ticket.return_value = True
    system._find_ticket_id_in_message.return_value = None  # No explicit ID in message
    system._find_active_ticket_for_user.return_value = 555  # An active ticket IS found

    await system.generate_response(
        persona_name='test_persona',
        user_identifier='user_with_open_ticket',
        channel='gmail',
        message='I have more info on my issue.',
        user_display_name='Test User'
    )

    # Verify we searched for an active ticket for the correct user
    system._find_active_ticket_for_user.assert_awaited_once_with(101)

    # Verify a NEW ticket was NOT created
    zammad_mock.create_ticket.assert_not_called()

    # Verify the message was added to the found ticket (555)
    zammad_mock.add_article_to_ticket.assert_any_call(
        ticket_id=555,
        body='I have more info on my issue.',
        impersonate_email="user@example.com"
    )


@pytest.mark.asyncio
async def test_get_or_create_zammad_user_uses_display_name(chat_system_with_mocks):
    """
    Tests that when creating a new user for a non-email ID, the display name is used
    for the first and last name fields.
    """
    system, _, _, zammad_mock, _ = chat_system_with_mocks
    # Un-mock the method for this specific test
    system._get_or_create_zammad_user.side_effect = system.__class__._get_or_create_zammad_user.__get__(system)

    discord_id = "123456789"
    display_name = "DiscordUserName"
    expected_email = f"discord-{discord_id}@test.zammad.local"

    zammad_mock.search_user.return_value = []  # No user found
    zammad_mock.create_user.return_value = {'id': 909, 'email': expected_email}

    with patch('asyncio.to_thread', side_effect=lambda func, *args, **kwargs: func(*args, **kwargs)):
        await system._get_or_create_zammad_user(
            user_identifier=discord_id,
            channel="discord",
            user_display_name=display_name
        )

    # Verify that the display name was correctly used as the firstname
    zammad_mock.create_user.assert_called_once_with(
        email=expected_email,
        firstname="DiscordUserName",
        lastname="",  # No last name if display name is a single word
        note=f"Auto-generated user from Discord. Original identifier: {discord_id}"
    )


# --- Existing Tests (no changes needed below this line) ---

@pytest.mark.asyncio
async def test_casual_mode_flow(chat_system_with_mocks):
    """Tests that non-ticket channels only log to memory."""
    system, mem_mock, _, zammad_mock, _ = chat_system_with_mocks
    system._should_create_ticket.return_value = False

    await system.generate_response(
        persona_name='test_persona',
        user_identifier='user_discord_1',
        channel='discord',
        message='Just chatting'
    )

    system._get_or_create_zammad_user.assert_not_called()
    zammad_mock.create_ticket.assert_not_called()
    zammad_mock.add_article_to_ticket.assert_not_called()


@pytest.mark.asyncio
async def test_ticket_mode_new_ticket(chat_system_with_mocks):
    """Tests that ticket channels create a new Zammad ticket with a user-friendly title."""
    system, mem_mock, _, zammad_mock, _ = chat_system_with_mocks
    system._should_create_ticket.return_value = True
    system._find_ticket_id_in_message.return_value = None
    system._find_active_ticket_for_user.return_value = None  # No active ticket found

    await system.generate_response(
        persona_name='test_persona',
        user_identifier='user_gmail_1',
        channel='gmail',
        message='Help me!',
        user_display_name='Test User'
    )

    system._get_or_create_zammad_user.assert_awaited_once_with('user_gmail_1', 'gmail', 'Test User')
    zammad_mock.create_ticket.assert_called_once()
    call_args, call_kwargs = zammad_mock.create_ticket.call_args
    assert call_kwargs['title'] == 'New request from Test User via gmail'
    assert 'article_body' not in call_kwargs  # Verify it's an empty ticket


@pytest.mark.asyncio
async def test_ticket_mode_existing_ticket(chat_system_with_mocks):
    """Tests that messages for existing tickets add articles with correct authorship."""
    system, mem_mock, _, zammad_mock, _ = chat_system_with_mocks
    system._should_create_ticket.return_value = True
    system._find_ticket_id_in_message.return_value = 999

    message_with_id = "More info for [Ticket#999]"
    await system.generate_response(
        persona_name='test_persona',
        user_identifier='user_gmail_2',
        channel='gmail',
        message=message_with_id,
        user_display_name='Another User'
    )

    zammad_mock.create_ticket.assert_not_called()

    assert zammad_mock.add_article_to_ticket.call_count == 2
    zammad_mock.add_article_to_ticket.assert_any_call(
        ticket_id=999, body=message_with_id, impersonate_email="user@example.com"
    )
    zammad_mock.add_article_to_ticket.assert_any_call(
        ticket_id=999, body="LLM Reply"
    )
