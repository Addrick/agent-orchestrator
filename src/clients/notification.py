# src/clients/notification.py

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Mapping, Optional, Tuple

from src.generation_events import ResponseType

logger = logging.getLogger(__name__)


def resolve_recipient(
    recipients: Optional[Mapping[str, Any]], channel: Optional[str], key: Any,
) -> Optional[str]:
    """Resolve a recipient key/id against a recipients mapping.

    Returns the mapped string ID if found in recipients, the numeric ID if key is
    digits, or None if key is empty or unresolvable.
    """
    if key is None or key == "":
        return None
    key_str = str(key).strip()
    if not key_str:
        return None
    if key_str.isdigit():
        return key_str
    if not isinstance(recipients, Mapping):
        return None
    info = recipients.get(key_str)
    if not isinstance(info, Mapping):
        return None
    if channel == "discord_channel":
        val = info.get("discord_channel_id")
    elif channel == "discord_dm":
        val = info.get("discord_user_id")
    elif channel and "email" in channel:
        val = info.get("email")
    else:
        val = None
    return str(val) if val else None


class Notifier(ABC):
    """
    Abstract base for sending notifications through various channels.

    All notifiers share a common interface so the dispatch agent can
    route messages without knowing the underlying transport.
    """

    @abstractmethod
    async def send(self, recipient: str, subject: str, body: str) -> bool:
        """
        Send a notification.

        Args:
            recipient: Channel-specific identifier (Discord user ID, email, ticket ID).
            subject: Short summary / title (ignored by channels that don't support it).
            body: Full message content.

        Returns:
            True if delivery succeeded, False otherwise.
        """
        ...

    async def conversation_key(self, recipient: str) -> Optional[Tuple[str, Optional[str]]]:
        """The (channel, server_id) chat history files a message to
        `recipient` under, or None when it is not a chat conversation."""
        return None


class DiscordNotifier(Notifier):
    """Sends a Discord DM via the bot client."""

    def __init__(self, discord_client: Any) -> None:
        self._client = discord_client

    async def send(self, recipient: str, subject: str, body: str) -> bool:
        full_content = f"**{subject}**\n{body}" if subject else body
        return bool(await self._client.send_dm(int(recipient), full_content))


class DiscordChannelNotifier(Notifier):
    """Sends a message to a Discord text channel."""

    def __init__(self, discord_client: Any) -> None:
        self._client = discord_client

    async def send(self, recipient: str, subject: str, body: str) -> bool:
        full_content = f"**{subject}**\n{body}" if subject else body
        return bool(await self._client.send_to_channel(int(recipient), full_content))

    async def conversation_key(self, recipient: str) -> Optional[Tuple[str, Optional[str]]]:
        key: Optional[Tuple[str, Optional[str]]] = await self._client.conversation_key(int(recipient))
        return key


class ZammadNotifier(Notifier):
    """Posts an internal note on a Zammad ticket."""

    def __init__(self, zammad_client: Any) -> None:
        self._client = zammad_client

    async def send(self, recipient: str, subject: str, body: str) -> bool:
        try:
            note_body = f"**{subject}**\n\n{body}" if subject else body
            await asyncio.to_thread(
                self._client.add_article_to_ticket,
                ticket_id=int(recipient),
                body=note_body,
                internal=True
            )
            logger.info(f"Zammad internal note posted to ticket {recipient}.")
            return True
        except Exception as e:
            logger.error(f"Failed to post Zammad note to ticket {recipient}: {e}")
            return False


class LogNotifier(Notifier):
    """
    Fallback notifier that logs the notification.

    Useful when no delivery channel is configured, or for testing.
    """

    async def send(self, recipient: str, subject: str, body: str) -> bool:
        logger.info(f"[LogNotifier] To={recipient} Subject={subject} Body={body[:200]}")
        return True


class NotificationRouter:
    """
    Routes notifications to the appropriate notifier by channel name.

    Usage:
        router = NotificationRouter()
        router.register("discord", DiscordNotifier(client))
        router.register("zammad", ZammadNotifier(zammad_client))
        await router.send("discord", recipient="12345", subject="Alert", body="...")
    """

    def __init__(self) -> None:
        self._notifiers: dict[str, Notifier] = {}
        self._fallback: Notifier = LogNotifier()

    def register(self, channel: str, notifier: Notifier) -> None:
        """Register a notifier for a named channel."""
        self._notifiers[channel] = notifier

    @property
    def available_channels(self) -> list[str]:
        """List of registered channel names."""
        return list(self._notifiers.keys())

    async def send(self, channel: str, recipient: str, subject: str, body: str) -> bool:
        """
        Send a notification via the named channel, falling back to LogNotifier
        if the channel is not registered.
        """
        notifier = self._notifiers.get(channel, self._fallback)
        if notifier is self._fallback and channel not in self._notifiers:
            logger.warning(f"No notifier registered for channel '{channel}'. Using fallback logger.")
        return await notifier.send(recipient, subject, body)

    async def conversation_key(self, channel: str, recipient: str) -> Optional[Tuple[str, Optional[str]]]:
        """Where chat history files a message sent to `recipient` on `channel`
        (see `Notifier.conversation_key`), so a message posted on a persona's
        behalf can be stored as its reply in that conversation."""
        notifier = self._notifiers.get(channel)
        return await notifier.conversation_key(recipient) if notifier else None


class HourlyLimiter:
    """At most `per_hour` sends in any rolling hour. In memory: a restart
    forgets the window, which errs towards sending."""

    def __init__(self, per_hour: int) -> None:
        self.per_hour = int(per_hour)
        self._sent_at: Deque[float] = deque()

    def allow(self) -> bool:
        """True, and the send is counted, when one more fits in the window."""
        now = time.monotonic()
        while self._sent_at and now - self._sent_at[0] > 3600:
            self._sent_at.popleft()
        if len(self._sent_at) >= self.per_hour:
            return False
        self._sent_at.append(now)
        return True


async def post_as_persona(
    router: NotificationRouter, persistence: Any, *, persona_name: str,
    user_identifier: str, channel: str, recipient: str, body: str, reply_text: str,
) -> bool:
    """Post `body` to a target on a persona's behalf and, once posted, store
    `reply_text` as that persona's reply in the conversation the target
    files under. The one path behind every `notify_adam` call: the native
    tool (`tools/notify_integration`) and codr's opencode relay
    (`agents/codr_agent`).

    `persistence` is the `TurnPersistence` (passed in: it sits above this
    layer). Returns whether the post went out; a target that is not a chat
    conversation (a DM, a Zammad ticket) is posted and not stored."""
    try:
        sent = await router.send(channel=channel, recipient=recipient, subject="", body=body)
    except Exception as e:  # noqa: BLE001 — a failed post must not break the caller
        logger.warning("post for persona '%s' failed: %s", persona_name, e)
        return False
    if not sent:
        return False
    key = await router.conversation_key(channel, recipient)
    if key is None:
        logger.info("no conversation for target %s:%s; message for '%s' not stored in history",
                    channel, recipient, persona_name)
        return True
    conv_channel, server_id = key
    row_id = persistence.commit_or_update_assistant(
        persona_name=persona_name, user_identifier=user_identifier,
        channel=conv_channel, server_id=server_id, final_text=reply_text,
        response_type=ResponseType.LLM_GENERATION, user_interaction_id=None,
        retry_assistant_id=None, tool_context_json=None,
    )
    if row_id is not None:
        persistence.queue_retain_safe(interaction_id=row_id, persona_name=persona_name,
                                      channel=conv_channel, untrusted=False)
    return True
