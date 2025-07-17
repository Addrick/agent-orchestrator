import asyncio
import base64
import json
import logging
import os
import re
from email.mime.text import MIMEText
from collections import deque
from googleapiclient.discovery import build

from google.cloud import pubsub_v1
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError

from config.global_config import *

logger = logging.getLogger(__name__)


class GmailInterface:
    SCOPES = [
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/pubsub"
    ]
    PROCESSED_ID_CACHE_SIZE = 100

    def __init__(self, chat_system):
        self.chat_system = chat_system
        self.credentials = None
        self._subscription_future = None
        self.credentials_file = GMAIL_CREDENTIALS_FILE
        self.token_file = GMAIL_TOKEN_FILE
        self._shutdown_event = asyncio.Event()
        self.loop = None
        self._processing_lock = asyncio.Lock()
        self._processed_ids = deque(maxlen=self.PROCESSED_ID_CACHE_SIZE)
        self.last_known_history_id = None

    async def _authenticate(self):
        """Handles user authentication."""
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

    def _sync_callback_wrapper(self, message):
        """Schedules the async event processor on the main event loop."""
        if not self.loop:
            logger.error("Event loop not available for sync callback.")
            return
        asyncio.run_coroutine_threadsafe(self._process_new_events(), self.loop)
        message.ack()

    async def _process_new_events(self):
        """Checks for new history records and processes them."""
        async with self._processing_lock:
            if not self.last_known_history_id:
                logger.warning("No starting historyId found. Bot may have just started. Skipping first check.")
                return

            try:
                service = build('gmail', 'v1', credentials=self.credentials, cache_discovery=False)
                history_response = await asyncio.to_thread(
                    service.users().history().list(userId='me', startHistoryId=self.last_known_history_id).execute
                )
                new_history_id = history_response.get('historyId')
                if new_history_id: self.last_known_history_id = new_history_id

                added_messages = []
                if 'history' in history_response:
                    for history_record in history_response['history']:
                        if 'messagesAdded' in history_record:
                            added_messages.extend(history_record['messagesAdded'])

                if not added_messages:
                    logger.debug(f"History check complete. No new messages found.")
                    return

                for msg_summary in added_messages:
                    if 'SENT' in msg_summary['message'].get('labelIds', []):
                        logger.debug(f"Skipping self-sent message with ID: {msg_summary['message']['id']}")
                        continue
                    msg_id = msg_summary['message']['id']
                    if msg_id in self._processed_ids:
                        logger.warning(f"Duplicate message ID '{msg_id}' detected in history. Skipping.")
                        continue

                    self._processed_ids.append(msg_id)
                    await self._handle_specific_message(service, msg_id)

            except Exception as e:
                logger.error(f"Error processing new events: {e}", exc_info=True)

    async def _handle_specific_message(self, service, msg_id):
        """Contains the logic to process one specific email message."""
        try:
            get_response = await asyncio.to_thread(
                service.users().messages().get(userId='me', id=msg_id, format='full').execute)

            payload = get_response['payload']
            headers = payload['headers']
            subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
            sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), '')

            if BLOCK_EXTERNAL_SENDER_REPLIES:
                allowed_sender = 'adam@tech-ops.it'
                if allowed_sender not in sender:
                    logger.info(
                        f"Skipping email from '{sender}' (ID: {msg_id}) as sender blocking is enabled. Email will remain unread.")
                    return

            thread_id = get_response['threadId']
            message_id_header = next((h['value'] for h in headers if h['name'].lower() == 'message-id'), None)
            user_input = ""
            if "parts" in payload:
                for part in payload['parts']:
                    if part['mimeType'] == 'text/plain':
                        body_data = part['body'].get('data')
                        if body_data: user_input = base64.urlsafe_b64decode(body_data).decode('utf-8', 'ignore'); break
            else:
                body_data = payload['body'].get('data')
                if body_data: user_input = base64.urlsafe_b64decode(body_data).decode('utf-8', 'ignore')

            if not user_input.strip():
                logger.warning(
                    f"Email from {sender} (ID: {msg_id}) had no readable text body. Skipping. Email will remain unread.")
                return

            active_persona_name = 'testr'

            pseudo_message = type('PseudoMessage', (), {'content': user_input})()
            dev_response = self.chat_system.bot_logic.preprocess_message(active_persona_name, pseudo_message)

            response_text = None
            if dev_response is not None:
                logger.info(f"Dev command processed for email (ID: {msg_id}).")
                response_text = dev_response
            else:
                context = await self._gather_thread_history(service, thread_id)
                logger.info(f"Handing off email from {sender} (ID: {msg_id}) to '{active_persona_name}' persona.")
                response_text = await self.chat_system.generate_response(active_persona_name, user_input,
                                                                         context=context, image_url=None)

            if response_text:
                # First, send the reply
                await self._send_reply(service, to=sender, subject=subject, body=response_text,
                                       in_reply_to=message_id_header, thread_id=thread_id)

                # THEN, mark the original message as read
                await asyncio.to_thread(
                    service.users().messages().modify(userId='me', id=msg_id,
                                                      body={'removeLabelIds': ['UNREAD']}).execute)
                logger.info(f"Successfully replied to and marked email as read (ID: {msg_id}).")

        except Exception as e:
            logger.error(f"Error handling message ID {msg_id}: {e}", exc_info=True)

    async def _gather_thread_history(self, service, thread_id):
        try:
            thread_data = await asyncio.to_thread(
                service.users().threads().get(userId='me', id=thread_id, format='full').execute)
            history = []
            for msg in reversed(thread_data.get('messages', [])):
                payload = msg['payload']
                content = ""
                if "parts" in payload:
                    for part in payload['parts']:
                        if part['mimeType'] == 'text/plain':
                            body_data = part['body'].get('data')
                            if body_data: content = base64.urlsafe_b64decode(body_data).decode('utf-8', 'ignore'); break
                else:
                    body_data = payload['body'].get('data')
                    if body_data: content = base64.urlsafe_b64decode(body_data).decode('utf-8', 'ignore')
                if content:
                    sender_header = next((h['value'] for h in msg['payload']['headers'] if h['name'].lower() == 'from'),
                                         'Unknown')
                    history.append(f"From: {sender_header}\n{content}")
            return "\n---\n".join(history)
        except Exception as e:
            logger.error(f"Failed to gather email thread history: {e}")
            return ""

    async def _send_reply(self, service, to, subject, body, in_reply_to, thread_id):
        try:
            message = MIMEText(body)
            message['to'] = to
            message['subject'] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
            if in_reply_to: message['In-Reply-To'] = in_reply_to; message['References'] = in_reply_to
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            create_message = {'raw': raw_message, 'threadId': thread_id}
            await asyncio.to_thread(service.users().messages().send(userId='me', body=create_message).execute)
            logger.info(f"Reply sent successfully to {to}")
        except HttpError as error:
            logger.error(f'An error occurred while sending email: {error}')

    async def _setup_watch(self):
        """Sets up the watch request and gets the initial historyId."""
        logger.info("Setting up Gmail watch and getting initial historyId...")
        service = build('gmail', 'v1', credentials=self.credentials, cache_discovery=False)
        request = {'labelIds': ['INBOX'], 'topicName': GMAIL_PUBSUB_TOPIC}
        watch_response = await asyncio.to_thread(service.users().watch(userId='me', body=request).execute)
        self.last_known_history_id = watch_response['historyId']
        logger.info(f"Initial historyId set to: {self.last_known_history_id}")

    async def start(self):
        await self._authenticate()
        await self._setup_watch()
        self.loop = asyncio.get_running_loop()
        subscriber = pubsub_v1.SubscriberClient(credentials=self.credentials)
        subscription_path = subscriber.subscription_path(GMAIL_PROJECT_ID, GMAIL_PUBSUB_SUBSCRIPTION_ID)
        logger.info(f"Gmail listener starting on subscription: {subscription_path}")
        self._subscription_future = subscriber.subscribe(subscription_path, self._sync_callback_wrapper)
        try:
            await self._shutdown_event.wait()
        finally:
            self._subscription_future.cancel()
            logger.info("Gmail subscription future cancelled.")

    def stop(self):
        logger.info("Stopping Gmail listener...")
        self._shutdown_event.set()


def create_gmail_bot(chat_system):
    return GmailInterface(chat_system)