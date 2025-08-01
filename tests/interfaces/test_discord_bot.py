# tests/interfaces/test_discord_bot.py

import pytest
import discord
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock, call
from datetime import datetime, timedelta

from src.interfaces.discord_bot import create_discord_bot
from src.chat_system import ChatSystem, ResponseType
from src.database.memory_manager import MemoryManager


@pytest.fixture
def mock_memory_manager():
    """
    Fixture for a mocked MemoryManager.
    Crucially, the methods are standard MagicMocks because the real methods are synchronous.
    """
    manager = MagicMock(spec=MemoryManager)
    manager.log_message = MagicMock()
    manager.suppress_message_by_platform_id = MagicMock(return_value=True)
    return manager


@pytest.fixture
def mock_chat_system(mock_memory_manager):
    """Fixture for a mocked ChatSystem that includes the accurately-mocked MemoryManager."""
    chat_system = MagicMock(spec=ChatSystem)
    chat_system.personas = {"derpr": MagicMock(), "test_persona": MagicMock()}
    chat_system.generate_response = AsyncMock()
    chat_system.memory_manager = mock_memory_manager
    return chat_system


@pytest.fixture
def mock_discord_client(mock_chat_system):
    """Fixture to create a bot client instance for testing."""
    client = create_discord_bot(mock_chat_system)
    # Mock the client.user to prevent self-response checks from failing
    with patch.object(type(client), 'user', new_callable=PropertyMock, return_value=MagicMock(id=999)):
        yield client


@pytest.fixture
def mock_message():
    """Fixture for a standard mock Discord message that returns a mock bot reply."""
    channel = AsyncMock(spec=discord.TextChannel, typing=MagicMock())
    channel.name = "general"
    author = MagicMock(id=123, display_name="TestAuthor")
    message = MagicMock(
        id=1001,
        author=author,
        content="derpr hello there",
        channel=channel,
        attachments=[],
        created_at=datetime.utcnow()
    )
    mock_bot_reply = AsyncMock(spec=discord.Message)
    mock_bot_reply.id = 2002
    mock_bot_reply.created_at = datetime.utcnow()
    channel.send.return_value = mock_bot_reply
    return message


@pytest.mark.asyncio
@patch('src.interfaces.discord_bot.reset_discord_status', new_callable=AsyncMock)
async def test_on_message_llm_flow(mock_reset_status, mock_discord_client, mock_chat_system, mock_message):
    """
    Tests the entire successful flow for a standard LLM response, including logging orchestration.
    """
    mock_chat_system.generate_response.return_value = ("Bot reply", ResponseType.LLM_GENERATION, 12345)
    on_message_handler = mock_discord_client.on_message

    await on_message_handler(mock_message)

    # Assert correct call to core logic
    mock_chat_system.generate_response.assert_called_once_with(
        persona_name='derpr', user_identifier='123', channel='general', message='hello there',
        image_url=None, history_limit=20, user_display_name='TestAuthor'
    )
    mock_message.channel.send.assert_called_once_with("Bot reply")

    # Assert correct logging calls were made (the test passes if asyncio.to_thread was used correctly)
    assert mock_chat_system.memory_manager.log_message.call_count == 2
    log_calls = mock_chat_system.memory_manager.log_message.call_args_list

    log_calls[0].assert_called_with(
        user_identifier='123', persona_name='derpr', channel='general', role='user',
        content='hello there', timestamp=mock_message.created_at, platform_message_id='1001', zammad_ticket_id=12345
    )
    log_calls[1].assert_called_with(
        user_identifier='123', persona_name='derpr', channel='general', role='assistant',
        content='Bot reply', timestamp=mock_message.channel.send.return_value.created_at, platform_message_id='2002',
        zammad_ticket_id=12345
    )
    mock_reset_status.assert_called_once()


@pytest.mark.asyncio
@patch('src.interfaces.discord_bot._send_dev_response', new_callable=AsyncMock)
@patch('src.interfaces.discord_bot.reset_discord_status', new_callable=AsyncMock)
async def test_on_message_dev_command_flow(mock_reset_status, mock_send_dev, mock_discord_client, mock_chat_system,
                                           mock_message):
    """Tests that dev commands are handled, not logged, and use the dev response function."""
    mock_chat_system.generate_response.return_value = ("Dev output", ResponseType.DEV_COMMAND, None)

    await mock_discord_client.on_message(mock_message)

    mock_chat_system.generate_response.assert_called_once()
    mock_send_dev.assert_called_once_with(mock_message.channel, "Dev output")
    mock_message.channel.send.assert_not_called()
    mock_chat_system.memory_manager.log_message.assert_not_called()
    mock_reset_status.assert_called_once()


@pytest.mark.asyncio
async def test_on_message_delete_flow(mock_discord_client, mock_chat_system, mock_message):
    """
    Tests that the on_message_delete event triggers suppression.
    This test will now fail if the production code uses `await` on a sync method.
    """
    mock_message.id = 5555
    on_delete_handler = mock_discord_client.on_message_delete

    await on_delete_handler(mock_message)

    mock_chat_system.memory_manager.suppress_message_by_platform_id.assert_called_once_with('5555')


@pytest.mark.asyncio
@patch('src.interfaces.discord_bot.reset_discord_status', new_callable=AsyncMock)
async def test_channel_name_persona_trigger(mock_reset_status, mock_discord_client, mock_chat_system, mock_message):
    """Tests that a persona can be triggered by the channel name."""
    mock_message.channel.name = "derpr-support"
    mock_message.content = "I need help with something."
    mock_chat_system.generate_response.return_value = ("A helpful reply", ResponseType.LLM_GENERATION, None)

    await mock_discord_client.on_message(mock_message)

    mock_chat_system.generate_response.assert_called_once()
    called_kwargs = mock_chat_system.generate_response.call_args.kwargs
    assert called_kwargs['persona_name'] == 'derpr'
    assert called_kwargs['message'] == 'I need help with something.'
    mock_reset_status.assert_called_once()


@pytest.mark.asyncio
@patch('src.interfaces.discord_bot.split_string_by_limit', return_value=["First part.", "Second part."])
@patch('src.interfaces.discord_bot.reset_discord_status', new_callable=AsyncMock)
async def test_long_message_splitting(mock_reset_status, mock_split, mock_discord_client, mock_chat_system,
                                      mock_message):
    """Tests that a long response is correctly split into multiple messages."""
    long_response = "First part. Second part."
    mock_chat_system.generate_response.return_value = (long_response, ResponseType.LLM_GENERATION, None)

    await mock_discord_client.on_message(mock_message)

    mock_split.assert_called_once_with(long_response, 2000)
    assert mock_message.channel.send.call_count == 2
    send_calls = mock_message.channel.send.call_args_list
    send_calls[0].assert_called_with("First part.")
    send_calls[1].assert_called_with("Second part.")

    # Verify that the full, unsplit message is logged for the bot's reply.
    assert mock_chat_system.memory_manager.log_message.call_count == 2
    last_log_call = mock_chat_system.memory_manager.log_message.call_args_list[1]
    _, kwargs = last_log_call
    assert kwargs['content'] == long_response
