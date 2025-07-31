# tests/interfaces/test_discord_bot.py

import pytest
import discord
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

from src.interfaces.discord_bot import create_discord_bot
from src.chat_system import ChatSystem, ResponseType


@pytest.fixture
def mock_chat_system():
    """Fixture for a mocked ChatSystem."""
    chat_system = MagicMock(spec=ChatSystem)
    # Set up the personas dictionary that the bot will iterate over
    chat_system.personas = {"test_persona": MagicMock()}
    chat_system.generate_response = AsyncMock()
    return chat_system


@pytest.mark.asyncio
@patch('src.interfaces.discord_bot._send_dev_response', new_callable=AsyncMock)
@patch('src.interfaces.discord_bot.reset_discord_status', new_callable=AsyncMock)
async def test_on_message_flow_and_response_types(mock_reset_status, mock_send_dev_response, mock_chat_system):
    """Test the main on_message logic for LLM and DEV responses."""
    client = create_discord_bot(mock_chat_system)
    client.response_type_enum = ResponseType
    on_message = client.on_message

    mock_channel = AsyncMock(spec=discord.TextChannel, typing=MagicMock())
    mock_channel.name = "general"
    mock_author = MagicMock(id=123, display_name="TestAuthor")
    mock_message = MagicMock(
        author=mock_author,
        content="test_persona hello there",
        channel=mock_channel,
        attachments=[]
    )

    # This mocks the client.user property to avoid errors when comparing message.author to client.user
    with patch.object(type(client), 'user', new_callable=PropertyMock, return_value=MagicMock()):
        # --- Test LLM_GENERATION path ---
        mock_chat_system.generate_response.return_value = ("LLM says hi", ResponseType.LLM_GENERATION)

        with patch('src.interfaces.discord_bot.split_string_by_limit', return_value=["LLM says hi"]) as mock_split:
            await on_message(mock_message)

        # THE FIX IS HERE: Add user_display_name to the expected call
        mock_chat_system.generate_response.assert_called_once_with(
            persona_name='test_persona',
            user_identifier='123',
            channel='general',
            message='hello there',
            image_url=None,
            history_limit=20,
            user_display_name='TestAuthor'
        )
        mock_split.assert_called_once_with("LLM says hi", 2000)
        mock_channel.send.assert_called_once_with("LLM says hi")
        mock_send_dev_response.assert_not_called()
        mock_reset_status.assert_called_once()

        # --- Reset mocks and test DEV_COMMAND path ---
        mock_chat_system.generate_response.reset_mock()
        mock_send_dev_response.reset_mock()
        mock_channel.reset_mock()
        mock_reset_status.reset_mock()

        mock_chat_system.generate_response.return_value = ("Dev command output", ResponseType.DEV_COMMAND)
        await on_message(mock_message)

        # THE FIX IS HERE AS WELL: Add user_display_name to this call too
        mock_chat_system.generate_response.assert_called_once_with(
            persona_name='test_persona',
            user_identifier='123',
            channel='general',
            message='hello there',
            image_url=None,
            history_limit=20,
            user_display_name='TestAuthor'
        )
        mock_send_dev_response.assert_called_once_with(mock_channel, "Dev command output")
        mock_channel.send.assert_not_called()
        mock_reset_status.assert_called_once()
