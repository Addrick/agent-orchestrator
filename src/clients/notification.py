# src/clients/notification.py

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


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
