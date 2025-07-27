# tests/interfaces/test_gmail_bot.py

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, call
import base64
from email.mime.text import MIMEText

from src.interfaces.gmail_bot import GmailInterface
from src.chat_system import ChatSystem, ResponseType


@pytest.fixture
def mock_chat_system() -> MagicMock:
    """Fixture for a mocked ChatSystem."""
    chat_system = MagicMock(spec=ChatSystem)
    chat_system.generate_response = AsyncMock(return_value=("Test reply", ResponseType.LLM_GENERATION))
    return chat_system


@pytest.fixture
def gmail_interface(mock_chat_system: MagicMock) -> GmailInterface:
    """Fixture to provide a GmailInterface instance."""
    return GmailInterface(mock_chat_system)


@pytest.mark.parametrize("recipient, expected_persona", [
    ("support-derpr@example.com", "derpr"),
    ("support-testbot@example.com", "testbot"),
    ("user@example.com", "derpr"),  # Default
    ("support-@example.com", "derpr"),  # Default
    ("invalid-email", "derpr"),  # Default
])
def test_get_persona_from_recipient(gmail_interface: GmailInterface, recipient: str, expected_persona: str):
    """Test persona extraction from recipient email addresses."""
    persona = gmail_interface._get_persona_from_recipient(recipient)
    assert persona == expected_persona


@pytest.mark.asyncio
@patch('os.path.exists', return_value=True)
@patch('src.interfaces.gmail_bot.Credentials')
@patch('src.interfaces.gmail_bot.InstalledAppFlow')
async def test_authenticate_with_existing_token(mock_flow, mock_creds, mock_exists, gmail_interface: GmailInterface):
    """Test authentication when a valid token file exists."""
    mock_creds.from_authorized_user_file.return_value = MagicMock(valid=True)

    await gmail_interface._authenticate()

    mock_creds.from_authorized_user_file.assert_called_once()
    mock_flow.from_client_secrets_file.assert_not_called()
    assert gmail_interface.credentials is not None


@pytest.mark.asyncio
@patch('os.path.exists', return_value=False)
@patch('src.interfaces.gmail_bot.InstalledAppFlow')
@patch('builtins.open')
async def test_authenticate_flow(mock_open, mock_flow, mock_exists, gmail_interface: GmailInterface):
    """Test the full OAuth2 authentication flow."""
    mock_creds_instance = MagicMock()
    mock_creds_instance.to_json.return_value = '{"token": "test"}'
    mock_flow_instance = MagicMock()
    mock_flow_instance.run_local_server.return_value = mock_creds_instance
    mock_flow.from_client_secrets_file.return_value = mock_flow_instance

    await gmail_interface._authenticate()

    mock_flow.from_client_secrets_file.assert_called_once()
    mock_flow_instance.run_local_server.assert_called_once()
    mock_open.assert_called_once_with(gmail_interface.token_file, 'w')
    assert gmail_interface.credentials == mock_creds_instance


def _create_mock_email_data(to_email: str) -> dict:
    """Helper to create sample raw email data."""
    body = "Hello, this is a test email."
    encoded_body = base64.urlsafe_b64encode(body.encode('utf-8')).decode('utf-8')
    return {
        'id': 'test_msg_id',
        'threadId': 'test_thread_id',
        'payload': {
            'headers': [
                {'name': 'Subject', 'value': 'Test Subject'},
                {'name': 'From', 'value': 'sender@example.com'},
                {'name': 'To', 'value': to_email},
                {'name': 'Message-ID', 'value': '<test_message_id@mail.gmail.com>'}
            ],
            'body': {'data': encoded_body}
        }
    }


@pytest.mark.asyncio
async def test_handle_specific_message(gmail_interface: GmailInterface, mock_chat_system: MagicMock):
    """Test the complete handling of a single email message."""
    mock_service = MagicMock()
    msg_data = _create_mock_email_data("support-testpersona@example.com")

    # Patch asyncio.to_thread and _send_reply
    with patch('src.interfaces.gmail_bot.asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread, \
            patch.object(gmail_interface, '_send_reply', new_callable=AsyncMock) as mock_send_reply:
        # Configure the mock to return our sample email data for the first call
        mock_to_thread.side_effect = [msg_data, None]

        await gmail_interface._handle_specific_message(mock_service, 'test_msg_id')

        # Verify chat system was called correctly
        expected_full_message = "Subject: Test Subject\n\nHello, this is a test email."
        mock_chat_system.generate_response.assert_called_once_with(
            persona_name='testpersona',
            user_identifier='sender@example.com',
            channel='gmail',
            message=expected_full_message,
            history_limit=20
        )

        # Verify a reply was sent
        mock_send_reply.assert_called_once()

        # Verify the message was marked as read
        modify_call = mock_service.users().messages().modify
        modify_call.assert_called_once_with(userId='me', id='test_msg_id', body={'removeLabelIds': ['UNREAD']})


@pytest.mark.asyncio
@patch('src.interfaces.gmail_bot.MIMEText', new_callable=MagicMock)
async def test_send_reply(mock_mime_text_class, gmail_interface: GmailInterface):
    """Test the construction and sending of an email reply."""
    mock_service = MagicMock()
    mock_mime_instance = MagicMock()
    mock_mime_instance.as_bytes.return_value = b'test bytes'  # Configure return value
    mock_mime_text_class.return_value = mock_mime_instance

    with patch('src.interfaces.gmail_bot.asyncio.to_thread', new_callable=AsyncMock):
        await gmail_interface._send_reply(
            service=mock_service,
            to="recipient@example.com",
            subject="Test Subject",
            body="This is the reply body.",
            in_reply_to="<original_id>",
            thread_id="test_thread"
        )

    mock_mime_text_class.assert_called_once_with("This is the reply body.")
    expected_calls = [
        call.__setitem__('to', 'recipient@example.com'),
        call.__setitem__('subject', 'Re: Test Subject'),
        call.__setitem__('In-Reply-To', '<original_id>'),
        call.__setitem__('References', '<original_id>')
    ]
    mock_mime_instance.assert_has_calls(expected_calls)
    mock_service.users().messages().send.assert_called_once()


@pytest.mark.asyncio
async def test_process_new_events(gmail_interface: GmailInterface):
    """Test the polling logic for new email events."""
    mock_service = MagicMock()
    gmail_interface.last_known_history_id = "100"
    gmail_interface._processed_ids.append("msg_id_old")

    history_response = {
        'historyId': "200",
        'history': [
            {'messagesAdded': [{'message': {'id': 'msg_id_new', 'labelIds': ['INBOX']}}]},
            {'messagesAdded': [{'message': {'id': 'msg_id_sent', 'labelIds': ['SENT']}}]},  # Should be skipped
            {'messagesAdded': [{'message': {'id': 'msg_id_old', 'labelIds': ['INBOX']}}]}  # Should be skipped
        ]
    }

    with patch('src.interfaces.gmail_bot.build', return_value=mock_service), \
            patch('src.interfaces.gmail_bot.asyncio.to_thread', new_callable=AsyncMock, return_value=history_response), \
            patch.object(gmail_interface, '_handle_specific_message', new_callable=AsyncMock) as mock_handle_msg:
        await gmail_interface._process_new_events()

        # Assert only the new, valid message was handled
        mock_handle_msg.assert_called_once_with(mock_service, 'msg_id_new')
        assert gmail_interface.last_known_history_id == "200"