# tests/interfaces/test_discord_bot.py

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import discord

from src.interfaces.discord_bot import get_image_url, create_discord_bot, _send_dev_response
from src.chat_system import ChatSystem, ResponseType


@pytest.fixture
def mock_chat_system() -> MagicMock:
    """Fixture for a mocked ChatSystem."""
    chat_system = MagicMock(spec=ChatSystem)
    chat_system.personas = {'test_persona': MagicMock()}
    chat_system.generate_response = AsyncMock()
    return chat_system


@pytest.mark.asyncio
@pytest.mark.parametrize("attachments, content, expected_url", [
    ([MagicMock(content_type='image/png', url='https://attachment.url/image.png')], "No link here",
     "https://attachment.url/image.png"),
    ([], "Some text with a link https://content.url/image.jpg in it.", "https://content.url/image.jpg"),
    ([], "Just plain text, no images.", None),
    ([MagicMock(content_type='text/plain', url='https://attachment.url/text.txt')], "No image link.", None)
])
async def test_get_image_url(attachments, content, expected_url):
    """Test get_image_url with various message configurations."""
    mock_message = MagicMock(spec=discord.Message)
    mock_message.attachments = attachments
    mock_message.content = content

    url = await get_image_url(mock_message)
    assert url == expected_url


@pytest.mark.asyncio
@patch('src.interfaces.discord_bot.split_string_by_limit', return_value=["Formatted chunk"])
async def test_send_dev_response(mock_split, caplog):
    """Test the _send_dev_response helper function."""
    mock_channel = AsyncMock(spec=discord.TextChannel)

    await _send_dev_response(mock_channel, "This is a `dev` response.")

    mock_split.assert_called_once()
    mock_channel.send.assert_called_once_with("```\nFormatted chunk```")

    # Test error case
    mock_channel.send.side_effect = discord.HTTPException(MagicMock(), "Test error")
    await _send_dev_response(mock_channel, "This will fail.")
    assert "An error occurred sending a dev response" in caplog.text


def test_create_discord_bot(mock_chat_system):
    """Test that the bot creation function returns a client with the correct handlers."""
    client = create_discord_bot(mock_chat_system)
    assert isinstance(client, discord.Client)
    assert hasattr(client, 'on_ready')
    assert hasattr(client, 'on_message')
    assert client.chat_system == mock_chat_system


@pytest.mark.asyncio
@patch('src.interfaces.discord_bot.DISCORD_DEBUG_CHANNEL', 12345)
@patch('src.interfaces.discord_bot.reset_discord_status', new_callable=AsyncMock)
async def test_on_message_ignores_self_and_debug(mock_reset_status, mock_chat_system):
    """Test that on_message ignores messages from the bot and in the debug channel."""
    client = create_discord_bot(mock_chat_system)
    on_message = client.on_message

    with patch.object(type(client), 'user', new_callable=PropertyMock, return_value=MagicMock(id=1)):
        # Case 1: Message from self - provide a fully configured mock
        channel1 = AsyncMock(id=999, spec=discord.TextChannel)
        channel1.name = "general-channel"
        msg_from_self = MagicMock(author=MagicMock(id=1), channel=channel1, content="", attachments=[])
        await on_message(msg_from_self)
        mock_chat_system.generate_response.assert_not_called()

        # Case 2: Message in debug channel
        channel2 = AsyncMock(id=12345, spec=discord.TextChannel)
        channel2.name = "debug-channel"
        msg_in_debug = MagicMock(author=MagicMock(id=2), channel=channel2, content="", attachments=[])
        await on_message(msg_in_debug)
        mock_chat_system.generate_response.assert_not_called()


@pytest.mark.asyncio
@patch('src.interfaces.discord_bot.reset_discord_status', new_callable=AsyncMock)
async def test_on_message_no_persona_trigger(mock_reset_status, mock_chat_system):
    """Test that on_message does nothing if no persona is triggered."""
    client = create_discord_bot(mock_chat_system)
    on_message = client.on_message

    with patch.object(type(client), 'user', new_callable=PropertyMock, return_value=MagicMock(id=1)):
        mock_channel = AsyncMock(spec=discord.TextChannel)
        mock_channel.name = "general"  # Explicitly set the name attribute
        mock_message = MagicMock(
            author=MagicMock(id=2),
            content="hello world",
            channel=mock_channel,
            attachments=[]
        )
        await on_message(mock_message)
        mock_chat_system.generate_response.assert_not_called()


@pytest.mark.asyncio
@patch('src.interfaces.discord_bot.reset_discord_status', new_callable=AsyncMock)
async def test_on_message_flow_and_response_types(mock_reset_status, mock_chat_system):
    """Test the main on_message logic for LLM and DEV responses."""
    client = create_discord_bot(mock_chat_system)
    client.response_type_enum = ResponseType
    on_message = client.on_message

    mock_channel = AsyncMock(spec=discord.TextChannel, typing=MagicMock())
    mock_channel.name = "general"  # Explicitly set the name attribute
    mock_message = MagicMock(
        author=MagicMock(id=123),
        content="test_persona hello there",
        channel=mock_channel,
        attachments=[]
    )

    with patch.object(type(client), 'user', new_callable=PropertyMock, return_value=MagicMock()):
        # --- Test LLM_GENERATION path ---
        mock_chat_system.generate_response.return_value = ("LLM says hi", ResponseType.LLM_GENERATION)

        with patch('src.interfaces.discord_bot.split_string_by_limit', return_value=["LLM says hi"]) as mock_split:
            await on_message(mock_message)

        mock_chat_system.generate_response.assert_called_once_with(
            persona_name='test_persona',
            user_identifier='123',
            channel='general',
            message='hello there',
            image_url=None,
            history_limit=20
        )
        mock_split.assert_called_once_with("LLM says hi", 2000)
        mock_channel.send.assert_called_once_with("LLM says hi")
        assert mock_reset_status.call_count == 1

        # --- Reset mocks and test DEV_COMMAND path ---
        mock_chat_system.generate_response.reset_mock()
        mock_channel.send.reset_mock()
        mock_chat_system.generate_response.return_value = ("Dev command output", ResponseType.DEV_COMMAND)

        with patch('src.interfaces.discord_bot._send_dev_response', new_callable=AsyncMock) as mock_send_dev:
            await on_message(mock_message)

        mock_send_dev.assert_called_once_with(mock_channel, "Dev command output")
        mock_channel.send.assert_not_called()
        assert mock_reset_status.call_count == 2
        