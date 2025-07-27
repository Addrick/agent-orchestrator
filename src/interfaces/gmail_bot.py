# src/interfaces/gmail_bot.py
import asyncio
import base64
import logging
import os
import typing
from collections import deque
from email.mime.text import MIMEText
from typing import Optional, Dict, Any, List, Deque

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.cloud import pubsub_v1
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config.global_config import *

# Forward declaration for type hinting
if typing.TYPE_CHECKING:
    from src.chat_system import ChatSystem

logger = logging.getLogger(__name__)


class GmailInterface:
    """Handles all interaction with the Gmail API for receiving and responding to emails."""
    SCOPES = ["https://www.googleapis.com/auth/gmail.modify", "https://www.googleapis.com/auth/pubsub"]
    PROCESSED_ID_CACHE_SIZE = 100

    def __init__(self, chat_system: 'ChatSystem'):
        self.chat_system: 'ChatSystem' = chat_system
        self.credentials: Optional[Credentials] = None
        self.credentials_file: str = GMAIL_CREDENTIALS_FILE
        self.token_file: str = GMAIL_TOKEN_FILE
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._subscription_future = None
        self._processing_lock: asyncio.Lock = asyncio.Lock()
        self._processed_ids: Deque[str] = deque(maxlen=self.PROCESSED_ID_CACHE_SIZE)
        self.last_known_history_id: Optional[str] = None

    async def _authenticate(self) -> None:
        """Handles user authentication with Google Cloud, refreshing tokens as needed."""
        creds = None
        if os.path.exists(self.token_file):
            creds = await asyncio.to_thread(Credentials.from_authorized_user_file, self.token_file, self.SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                await asyncio.to_thread(creds.refresh, Request())
            else:
                flow = await asyncio.to_thread(InstalledAppFlow.from_client_secrets_file, self.credentials_file,
                                               self.SCOPES)
                creds = await asyncio.to_thread(flow.run_local_server, port=0)
            await asyncio.to_thread(lambda: open(self.token_file, 'w').write(creds.to_json()))
        self.credentials = creds
        logger.info("Gmail authentication successful.")

    def _get_persona_from_recipient(self, recipient_email: str) -> str:
        """Determines the target persona from the recipient email address."""
        try:
            if '@' not in recipient_email:
                raise ValueError("Invalid email format, no '@' symbol.")
            local_part = recipient_email.split('@')[0]
            if '-' in local_part:
                # Assumes format like 'support-persona@domain.com'
                parts = local_part.split('-')
                # Ensure there is a second part and it's not empty
                if len(parts) > 1 and parts[1]:
                    return parts[1]
        except (IndexError, AttributeError, ValueError):
            logger.warning(f"Could not parse persona from recipient '{recipient_email}'. Using default.")
        return "derpr"  # Default persona

    async def _handle_specific_message(self, service, msg_id: str) -> None:
        """Fetches a specific email, processes it, and orchestrates the reply."""
        try:
            msg_data = await asyncio.to_thread(
                service.users().messages().get(userId='me', id=msg_id, format='full').execute
            )
            payload = msg_data['payload']
            headers = payload['headers']
            subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), '')
            recipient = next((h['value'] for h in headers if h['name'].lower() == 'to'), '')
            message_id_header = next((h['value'] for h in headers if h['name'].lower() == 'message-id'), None)

            # Extract email body
            user_input = ""
            if "parts" in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain' and part['body'].get('data'):
                        user_input = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', 'ignore')
                        break
            elif payload['body'].get('data'):
                user_input = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', 'ignore')

            if not user_input.strip():
                logger.warning(f"Email from {sender} (ID: {msg_id}) has no text body. Skipping.")
                return

            active_persona_name = self._get_persona_from_recipient(recipient)
            full_message = f"Subject: {subject}\n\n{user_input}"

            # Use the new, stateful generate_response method
            response_text, response_type = await self.chat_system.generate_response(
                persona_name=active_persona_name,
                user_identifier=sender,  # The sender's email is the unique ID
                channel="gmail",
                message=full_message,
                history_limit=20  # Example history limit
            )

            if response_text:
                await self._send_reply(service, to=sender, subject=subject, body=response_text,
                                       in_reply_to=message_id_header, thread_id=msg_data['threadId'])
                await asyncio.to_thread(
                    service.users().messages().modify(userId='me', id=msg_id,
                                                      body={'removeLabelIds': ['UNREAD']}).execute
                )
                logger.info(f"Successfully replied to and marked as read email ID: {msg_id}")

        except Exception as e:
            logger.error(f"Error handling message ID {msg_id}: {e}", exc_info=True)

    async def _process_new_events(self) -> None:
        """Checks for new history records via the Gmail API and processes them."""
        async with self._processing_lock:
            if not self.last_known_history_id: return
            try:
                service = build('gmail', 'v1', credentials=self.credentials, cache_discovery=False)
                history_response = await asyncio.to_thread(
                    service.users().history().list(userId='me', startHistoryId=self.last_known_history_id).execute
                )
                new_history_id = history_response.get('historyId')
                if new_history_id: self.last_known_history_id = new_history_id

                if 'history' not in history_response:
                    logger.debug("History check complete. No new events.")
                    return

                for history_record in history_response['history']:
                    if 'messagesAdded' in history_record:
                        for msg_summary in history_record['messagesAdded']:
                            # Skip our own replies and duplicates
                            if 'SENT' in msg_summary['message'].get('labelIds', []) or msg_summary['message'][
                                'id'] in self._processed_ids:
                                continue

                            msg_id = msg_summary['message']['id']
                            self._processed_ids.append(msg_id)
                            await self._handle_specific_message(service, msg_id)
            except Exception as e:
                logger.error(f"Error processing new Gmail events: {e}", exc_info=True)

    async def _send_reply(self, service, to: str, subject: str, body: str, in_reply_to: Optional[str],
                          thread_id: str) -> None:
        """Constructs and sends an email reply."""
        try:
            message = MIMEText(body)
            message['to'] = to
            message['subject'] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
            if in_reply_to:
                message['In-Reply-To'] = in_reply_to
                message['References'] = in_reply_to
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            create_message = {'raw': raw_message, 'threadId': thread_id}
            await asyncio.to_thread(service.users().messages().send(userId='me', body=create_message).execute)
            logger.info(f"Reply successfully sent to {to}")
        except HttpError as error:
            logger.error(f'An error occurred while sending email reply: {error}')

    def _sync_callback_wrapper(self, message) -> None:
        """Schedules the async event processor on the main event loop from the sync Pub/Sub callback."""
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._process_new_events(), self.loop)
        message.ack()

    async def start(self) -> None:
        """Initializes authentication, sets up the watch, and starts the Pub/Sub listener."""
        await self._authenticate()
        service = build('gmail', 'v1', credentials=self.credentials, cache_discovery=False)
        watch_request = {'labelIds': ['INBOX'], 'topicName': GMAIL_PUBSUB_TOPIC}
        watch_response = await asyncio.to_thread(service.users().watch(userId='me', body=watch_request).execute)
        self.last_known_history_id = watch_response['historyId']
        logger.info(f"Gmail watch configured. Initial historyId: {self.last_known_history_id}")

        self.loop = asyncio.get_running_loop()
        subscriber = pubsub_v1.SubscriberClient(credentials=self.credentials)
        subscription_path = subscriber.subscription_path(GMAIL_PROJECT_ID, GMAIL_PUBSUB_SUBSCRIPTION_ID)
        logger.info(f"Gmail listener starting on subscription: {subscription_path}")
        self._subscription_future = subscriber.subscribe(subscription_path, self._sync_callback_wrapper)
        try:
            await self._shutdown_event.wait()
        finally:
            self._subscription_future.cancel()

    def stop(self) -> None:
        """Signals the main loop to shut down."""
        logger.info("Stopping Gmail listener...")
        self._shutdown_event.set()


def create_gmail_bot(chat_system: 'ChatSystem') -> GmailInterface:
    """Factory function to create an instance of the GmailInterface."""
    return GmailInterface(chat_system)
