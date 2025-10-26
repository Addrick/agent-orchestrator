# tests/interfaces/test_gmail_bot.py

import pytest
import asyncio
import os
from unittest.mock import MagicMock, AsyncMock, patch, call
import base64
from email.mime.text import MIMEText

# Import necessary exceptions and config variables
from google.auth.exceptions import RefreshError
from config.global_config import ALLOWED_SENDER_LIST, BLOCK_EXTERNAL_SENDER_REPLIES, CREDENTIALS_DIR, GMAIL_TOKEN_FILE
from src.interfaces.gmail_bot import GmailInterface
from src.chat_system import ChatSystem, ResponseType


@pytest.fixture
def mock_chat_system() -> MagicMock:
    """Fixture for a mocked ChatSystem."""
    chat_system = MagicMock(spec=ChatSystem)
    # FIX: Update the mock to return a 3-item tuple, which is the correct signature.
    # A default ticket_id of None is a sensible default.
    chat_system.generate_response = AsyncMock(return_value=("Test reply", ResponseType.LLM_GENERATION, None))
    return chat_system


@pytest.fixture
def gmail_interface(mock_chat_system: MagicMock) -> GmailInterface:
    """Fixture to provide a GmailInterface instance."""
    return GmailInterface(mock_chat_system)


# --- Authentication Tests ---

@pytest.mark.asyncio
@patch('src.interfaces.gmail_bot.os.path.exists', return_value=True)
@patch('src.interfaces.gmail_bot.os.makedirs')
@patch('src.interfaces.gmail_bot.Credentials')
@patch('src.interfaces.gmail_bot.InstalledAppFlow')
async def test_authenticate_with_existing_token(mock_flow, mock_creds, mock_makedirs, mock_exists,
                                                gmail_interface: GmailInterface):
    """Test authentication when a valid token file exists."""
    mock_creds.from_authorized_user_file.return_value = MagicMock(valid=True, expired=False)

    await gmail_interface._authenticate()

    mock_makedirs.assert_called_once_with(CREDENTIALS_DIR, exist_ok=True)
    mock_creds.from_authorized_user_file.assert_called_once()
    mock_flow.from_client_secrets_file.assert_not_called()
    assert gmail_interface.credentials is not None


@pytest.mark.asyncio
@patch('src.interfaces.gmail_bot.os.path.exists', side_effect=[False, True])  # No token, but creds exist
@patch('src.interfaces.gmail_bot.os.makedirs')
@patch('src.interfaces.gmail_bot.InstalledAppFlow')
@patch('builtins.open')
async def test_authenticate_flow(mock_open, mock_flow, mock_makedirs, mock_exists, gmail_interface: GmailInterface):
    """Test the full OAuth2 authentication flow when no token exists."""
    mock_creds_instance = MagicMock()
    mock_creds_instance.to_json.return_value = '{"token": "test"}'
    mock_flow_instance = MagicMock()
    mock_flow_instance.run_local_server.return_value = mock_creds_instance
    mock_flow.from_client_secrets_file.return_value = mock_flow_instance

    await gmail_interface._authenticate()

    mock_makedirs.assert_called_once_with(CREDENTIALS_DIR, exist_ok=True)
    mock_flow.from_client_secrets_file.assert_called_once()
    mock_flow_instance.run_local_server.assert_called_once()
    mock_open.assert_called_once_with(gmail_interface.token_file, 'w')
    assert gmail_interface.credentials == mock_creds_instance


@pytest.mark.asyncio
@patch('src.interfaces.gmail_bot.os.remove')
@patch('src.interfaces.gmail_bot.os.path.exists',
       side_effect=[True, True, True])  # token_exists, token_exists_for_delete, creds_exist
@patch('src.interfaces.gmail_bot.os.makedirs')
@patch('src.interfaces.gmail_bot.Credentials')
@patch('src.interfaces.gmail_bot.InstalledAppFlow')
@patch('builtins.open')
async def test_authenticate_with_revoked_token(mock_open, mock_flow, mock_creds, mock_makedirs, mock_exists,
                                               mock_remove, gmail_interface: GmailInterface):
    """Test that a revoked token is deleted and re-authentication is triggered."""
    mock_expired_creds = MagicMock(valid=False, expired=True, refresh_token="revoked_token")
    mock_expired_creds.refresh.side_effect = RefreshError("Token has been expired or revoked.")
    mock_creds.from_authorized_user_file.return_value = mock_expired_creds

    # Create a fully configured mock for the *new* credentials object
    mock_new_creds = MagicMock(valid=True)
    mock_new_creds.to_json.return_value = '{"token": "new_and_valid"}'

    # Mock the re-authentication part
    mock_flow_instance = MagicMock()
    mock_flow_instance.run_local_server.return_value = mock_new_creds
    mock_flow.from_client_secrets_file.return_value = mock_flow_instance

    await gmail_interface._authenticate()

    mock_makedirs.assert_called_once_with(CREDENTIALS_DIR, exist_ok=True)
    mock_expired_creds.refresh.assert_called_once()
    mock_remove.assert_called_once_with(GMAIL_TOKEN_FILE)
    mock_flow_instance.run_local_server.assert_called_once()
    mock_open.assert_called_once_with(GMAIL_TOKEN_FILE, 'w')


@pytest.mark.asyncio
@patch('src.interfaces.gmail_bot.os.path.exists', return_value=False)
@patch('src.interfaces.gmail_bot.os.makedirs')
async def test_authenticate_fatal_no_credentials_file(mock_makedirs, mock_exists, gmail_interface: GmailInterface):
    """Test that the bot shuts down if the main credentials file is missing."""
    await gmail_interface._authenticate()
    assert gmail_interface._shutdown_event.is_set()


# --- Message Handling Tests ---

def _create_mock_email_data(from_email: str, to_email: str) -> dict:
    """Helper to create sample raw email data."""
    body = "Hello, this is a test email."
    encoded_body = base64.urlsafe_b64encode(body.encode('utf-8')).decode('utf-8')
    return {
        'id': 'test_msg_id', 'threadId': 'test_thread_id', 'payload': {
            'headers': [
                {'name': 'Subject', 'value': 'Test Subject'},
                {'name': 'From', 'value': f'Sender Name <{from_email}>'},
                {'name': 'To', 'value': to_email},
                {'name': 'Message-ID', 'value': '<test_message_id@mail.gmail.com>'}
            ],
            'body': {'data': encoded_body}}}


@pytest.mark.asyncio
@patch('src.interfaces.gmail_bot.BLOCK_EXTERNAL_SENDER_REPLIES', False)
async def test_handle_specific_message_unpacks_response_correctly(gmail_interface: GmailInterface,
                                                                  mock_chat_system: MagicMock):
    """
    Tests that the 3-item tuple from generate_response is correctly unpacked.
    This specifically catches the bug where generate_response returns 3 values but the caller expects 2,
    which would raise a ValueError.
    """
    # 1. Setup
    mock_service = MagicMock()
    msg_data = _create_mock_email_data("test@example.com", "support-persona@example.com")

    # Configure the mock to return the specific 3-item tuple that caused the bug
    mock_chat_system.generate_response.return_value = ("A detailed response.", ResponseType.LLM_GENERATION, 54321)

    with patch('src.interfaces.gmail_bot.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread, \
            patch.object(gmail_interface, '_send_reply', new_callable=AsyncMock) as mock_send_reply:
        # Simulate the async calls inside the method
        mock_to_thread.side_effect = [
            msg_data,  # First call to get the message data
            None  # Second call to mark the message as read
        ]

        # 2. Action: Call the method under test
        # If the bug exists, this line will raise a ValueError. Pytest will catch it and fail the test.
        await gmail_interface._handle_specific_message(mock_service, 'test_msg_id')

        # 3. Assertions: If the test reaches here, the unpacking was successful.
        # We also check that the subsequent logic was executed correctly.
        mock_chat_system.generate_response.assert_called_once()
        mock_send_reply.assert_called_once()
        mock_service.users().messages().modify.assert_called_once()


@pytest.mark.asyncio
@patch('src.interfaces.gmail_bot.BLOCK_EXTERNAL_SENDER_REPLIES', True)
@patch('src.interfaces.gmail_bot.ALLOWED_SENDER_LIST', ['allowed@example.com'])
async def test_handle_specific_message_sender_allowed(gmail_interface: GmailInterface, mock_chat_system: MagicMock):
    """Test that an email from an allowed sender is processed."""
    mock_service = MagicMock()
    msg_data = _create_mock_email_data("allowed@example.com", "support-testpersona@example.com")

    with patch('src.interfaces.gmail_bot.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread, \
            patch.object(gmail_interface, '_send_reply', new_callable=AsyncMock) as mock_send_reply:
        mock_to_thread.side_effect = [msg_data, None]  # get message, then mark as read

        await gmail_interface._handle_specific_message(mock_service, 'test_msg_id')

        mock_chat_system.generate_response.assert_called_once()
        mock_send_reply.assert_called_once()
        mock_service.users().messages().modify.assert_called_once()


@pytest.mark.asyncio
@patch('src.interfaces.gmail_bot.BLOCK_EXTERNAL_SENDER_REPLIES', True)
@patch('src.interfaces.gmail_bot.ALLOWED_SENDER_LIST', ['allowed@example.com'])
async def test_handle_specific_message_sender_blocked(gmail_interface: GmailInterface, mock_chat_system: MagicMock):
    """Test that an email from a blocked sender is ignored."""
    mock_service = MagicMock()
    msg_data = _create_mock_email_data("blocked@example.com", "support-testpersona@example.com")

    with patch('src.interfaces.gmail_bot.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread, \
            patch.object(gmail_interface, '_send_reply', new_callable=AsyncMock) as mock_send_reply:
        mock_to_thread.return_value = msg_data

        await gmail_interface._handle_specific_message(mock_service, 'test_msg_id')

        mock_chat_system.generate_response.assert_not_called()
        mock_send_reply.assert_not_called()
        mock_service.users().messages().modify.assert_not_called()


# --- Other Tests ---

@pytest.mark.parametrize("recipient, expected_persona", [
    ("support-derpr@example.com", "derpr"),
    ("support-testbot@example.com", "testbot"),
    ("user@example.com", "derpr"),
    ("invalid-email", "derpr"),
])
def test_get_persona_from_recipient(gmail_interface: GmailInterface, recipient: str, expected_persona: str):
    """Test persona extraction from recipient email addresses."""
    assert gmail_interface._get_persona_from_recipient(recipient) == expected_persona


@pytest.mark.asyncio
async def test_process_new_events(gmail_interface: GmailInterface):
    """Test the polling logic for new email events."""
    mock_service = MagicMock()
    gmail_interface.last_known_history_id = "100"
    gmail_interface._processed_ids.append("msg_id_old")

    history_response = {
        'historyId': "200", 'history': [
            {'messagesAdded': [{'message': {'id': 'msg_id_new', 'labelIds': ['INBOX']}}]},
            {'messagesAdded': [{'message': {'id': 'msg_id_sent', 'labelIds': ['SENT']}}]},
            {'messagesAdded': [{'message': {'id': 'msg_id_old', 'labelIds': ['INBOX']}}]}
        ]}

    with patch('src.interfaces.gmail_bot.build', return_value=mock_service), \
            patch('src.interfaces.gmail_bot.asyncio.to_thread', return_value=history_response), \
            patch.object(gmail_interface, '_handle_specific_message', new_callable=AsyncMock) as mock_handle_msg:
        await gmail_interface._process_new_events()

        mock_handle_msg.assert_called_once_with(mock_service, 'msg_id_new')
        assert gmail_interface.last_known_history_id == "200"
