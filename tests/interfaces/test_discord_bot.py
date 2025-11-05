# tests/interfaces/test_discord_bot.py

import pytest
import discord
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from datetime import datetime
import io
from discord import File

from src.interfaces.discord_bot import create_discord_bot
from src.chat_system import ChatSystem, ResponseType
from src.database.memory_manager import MemoryManager
from src.persona import Persona
from src.engine import LLMCommunicationError


@pytest.fixture
def mock_persona_vocal():
    """Fixture for a persona that should display its name."""
    p = MagicMock(spec=Persona)
    p.should_display_name_in_chat.return_value = True
    return p


@pytest.fixture
def mock_persona_silent():
    """Fixture for a persona that should NOT display its name."""
    p = MagicMock(spec=Persona)
    p.should_display_name_in_chat.return_value = False
    return p


@pytest.fixture
def mock_chat_system(mock_persona_vocal, mock_persona_silent):
    """Fixture for a mocked ChatSystem with different persona types."""
    chat_system = MagicMock(spec=ChatSystem)
    chat_system.personas = {
        "vocal": mock_persona_vocal,
        "silent": mock_persona_silent,
        "derpr": mock_persona_vocal
    }
    chat_system.generate_response = AsyncMock()
    chat_system.memory_manager = MagicMock(spec=MemoryManager)
    return chat_system


@pytest.fixture
def mock_discord_client(mock_chat_system):
    """Fixture to create a bot client instance for testing."""
    client = create_discord_bot(mock_chat_system)
    with patch.object(type(client), 'user', new_callable=PropertyMock, return_value=MagicMock(id=999)):
        yield client


@pytest.fixture
def mock_message():
    """Fixture for a standard mock Discord message."""
    channel = AsyncMock(spec=discord.TextChannel, typing=MagicMock())
    channel.name = "general"
    author = MagicMock(id=123, display_name="TestAuthor")
    message = MagicMock(
        id=1001, author=author, content="vocal hello there", channel=channel,
        attachments=[], created_at=datetime.utcnow()
    )
    mock_bot_reply = AsyncMock(spec=discord.Message)
    mock_bot_reply.id = 2002
    mock_bot_reply.created_at = datetime.utcnow()
    channel.send.return_value = mock_bot_reply
    return message


@pytest.mark.asyncio
@patch('src.interfaces.discord_bot.reset_discord_status', new_callable=AsyncMock)
async def test_llm_flow_with_display_name(mock_reset, mock_discord_client, mock_chat_system, mock_message):
    """Tests the full flow where the persona's name IS displayed in the chat."""
    mock_chat_system.generate_response.return_value = ("Bot reply", ResponseType.LLM_GENERATION, 12345)
    await mock_discord_client.on_message(mock_message)

    mock_message.channel.send.assert_called_once_with("**vocal:** Bot reply")
    assert mock_chat_system.memory_manager.log_message.call_count == 2
    bot_log_kwargs = mock_chat_system.memory_manager.log_message.call_args_list[1].kwargs
    assert bot_log_kwargs['content'] == "Bot reply"
    assert bot_log_kwargs['author_name'] == 'vocal'


@pytest.mark.asyncio
@patch('src.interfaces.discord_bot.reset_discord_status', new_callable=AsyncMock)
async def test_llm_flow_without_display_name(mock_reset, mock_discord_client, mock_chat_system, mock_message):
    """Tests the full flow where the persona's name is NOT displayed in the chat."""
    mock_message.content = "silent hello"
    mock_chat_system.generate_response.return_value = ("Silent reply", ResponseType.LLM_GENERATION, None)
    await mock_discord_client.on_message(mock_message)

    mock_message.channel.send.assert_called_once_with("Silent reply")


@pytest.mark.asyncio
@patch('src.interfaces.discord_bot._send_dev_response', new_callable=AsyncMock)
@patch('src.interfaces.discord_bot.reset_discord_status', new_callable=AsyncMock)
async def test_dev_command_flow(mock_reset, mock_send_dev, mock_discord_client, mock_chat_system, mock_message):
    """Tests that dev commands are handled correctly and do not trigger user message logging."""
    mock_message.content = "vocal help"
    mock_chat_system.generate_response.return_value = ("Dev output", ResponseType.DEV_COMMAND, None)
    await mock_discord_client.on_message(mock_message)

    mock_send_dev.assert_called_once_with(mock_message.channel, "Dev output")
    mock_chat_system.memory_manager.log_message.assert_not_called()
    mock_reset.assert_called_once()


@pytest.mark.asyncio
async def test_on_message_delete_flow(mock_discord_client, mock_chat_system, mock_message):
    """Tests that the on_message_delete event triggers suppression."""
    mock_message.id = 5555
    await mock_discord_client.on_message_delete(mock_message)
    mock_chat_system.memory_manager.suppress_message_by_platform_id.assert_called_once_with('5555')


@pytest.mark.asyncio
async def test_bot_ignores_unrelated_messages_in_non_ambient_channel(monkeypatch, mock_discord_client, mock_chat_system,
                                                                     mock_message):
    """Tests that the bot remains silent and does not log if not mentioned in a non-ambient channel."""
    monkeypatch.setattr('src.interfaces.discord_bot.AMBIENT_LOGGING_CHANNELS', [])
    mock_message.content = "A message not for the bot."
    await mock_discord_client.on_message(mock_message)

    mock_chat_system.generate_response.assert_not_called()
    mock_chat_system.memory_manager.log_message.assert_not_called()


@pytest.mark.asyncio
async def test_logs_ambiently_but_does_not_respond(monkeypatch, mock_discord_client, mock_chat_system, mock_message):
    """Tests that the bot logs a message in an ambient channel but does not respond if not triggered."""
    monkeypatch.setattr('src.interfaces.discord_bot.AMBIENT_LOGGING_CHANNELS', ["ambient-channel"])
    mock_message.content = "An ambient message."
    mock_message.channel.name = "ambient-channel"

    await mock_discord_client.on_message(mock_message)

    # Assert that the message WAS logged
    mock_chat_system.memory_manager.log_message.assert_called_once()
    log_kwargs = mock_chat_system.memory_manager.log_message.call_args.kwargs
    assert log_kwargs['persona_name'] == 'ambient'
    assert log_kwargs['content'] == "An ambient message."

    # Assert that the bot did NOT try to respond
    mock_chat_system.generate_response.assert_not_called()
    mock_message.channel.send.assert_not_called()


@pytest.mark.asyncio
@patch('src.interfaces.discord_bot.reset_discord_status', new_callable=AsyncMock)
async def test_graceful_failure_on_exception(mock_reset, mock_discord_client, mock_chat_system, mock_message):
    """Tests the outermost try/except block in the on_message handler for truly unexpected errors."""
    mock_chat_system.generate_response.side_effect = Exception("A critical backend error!")

    await mock_discord_client.on_message(mock_message)

    mock_message.channel.send.assert_called_once_with("A critical error occurred. Please check the logs.")
    mock_chat_system.memory_manager.log_message.assert_not_called()
    mock_reset.assert_called_once()


@pytest.mark.asyncio
@patch('src.interfaces.discord_bot.reset_discord_status', new_callable=AsyncMock)
async def test_bot_treats_empty_message_as_continuation(mock_reset, mock_discord_client, mock_chat_system,
                                                        mock_message):
    """Tests that the bot processes a message with only a persona trigger as a continuation request."""
    mock_message.content = "vocal "
    mock_chat_system.generate_response.return_value = ("Continuation response", ResponseType.LLM_GENERATION, None)

    await mock_discord_client.on_message(mock_message)

    mock_chat_system.generate_response.assert_called_once()
    called_kwargs = mock_chat_system.generate_response.call_args.kwargs
    assert called_kwargs['message'] == ''

    mock_message.channel.send.assert_called()
    assert mock_chat_system.memory_manager.log_message.call_count == 2
    mock_reset.assert_called_once()


@pytest.mark.asyncio
@patch('src.interfaces.discord_bot.reset_discord_status', new_callable=AsyncMock)
async def test_file_response_flow(mock_reset, mock_discord_client, mock_chat_system, mock_message):
    """Tests that a FILE_RESPONSE from the dev command triggers a file upload."""
    # 1. Setup
    mock_message.content = "vocal dump context"
    file_content = "This is the content of the dump file."

    # Configure generate_response to return our special string
    mock_chat_system.generate_response.return_value = (
        f"FILE_RESPONSE::dump.txt::{file_content}",
        ResponseType.DEV_COMMAND,
        None
    )

    # 2. Action
    await mock_discord_client.on_message(mock_message)  # type: ignore

    # 3. Assertions
    # Check that channel.send was called
    mock_message.channel.send.assert_called_once()

    # Check the arguments passed to channel.send
    call_args, call_kwargs = mock_message.channel.send.call_args

    # The first positional arg should be the message content
    assert call_args[0] == "Here is the context dump:"

    # The 'file' keyword argument should be a discord.File object
    sent_file = call_kwargs.get('file')
    assert isinstance(sent_file, File)
    assert sent_file.filename == "dump.txt"

    # Verify the content of the file buffer
    sent_file.fp.seek(0)
    assert sent_file.fp.read() == file_content


@pytest.mark.asyncio
@patch('src.interfaces.discord_bot.reset_discord_status', new_callable=AsyncMock)
async def test_image_url_is_extracted_and_passed(mock_reset, mock_discord_client, mock_chat_system, mock_message):
    """Tests that an image URL is correctly extracted from an attachment and passed to the chat system."""
    mock_attachment = MagicMock(spec=discord.Attachment)
    mock_attachment.content_type = 'image/png'
    mock_attachment.url = 'http://example.com/test_image.png'
    mock_message.attachments = [mock_attachment]
    mock_message.content = "vocal check out this image"

    await mock_discord_client.on_message(mock_message)

    mock_chat_system.generate_response.assert_called_once()
    called_kwargs = mock_chat_system.generate_response.call_args.kwargs
    assert called_kwargs['image_url'] == 'http://example.com/test_image.png'
