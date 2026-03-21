# tests/clients/test_notification.py

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.clients.notification import (
    DiscordNotifier,
    ZammadNotifier,
    LogNotifier,
    NotificationRouter,
)


class TestLogNotifier:
    @pytest.mark.asyncio
    async def test_always_returns_true(self):
        notifier = LogNotifier()
        result = await notifier.send("user123", "Test Subject", "Test Body")
        assert result is True


class TestDiscordNotifier:
    @pytest.mark.asyncio
    async def test_successful_dm(self):
        mock_client = MagicMock()
        mock_user = MagicMock()
        mock_user.send = AsyncMock()
        mock_client.fetch_user = AsyncMock(return_value=mock_user)

        notifier = DiscordNotifier(mock_client)
        result = await notifier.send("12345", "Alert", "Server is down")

        assert result is True
        mock_client.fetch_user.assert_called_once_with(12345)
        mock_user.send.assert_called_once_with("**Alert**\nServer is down")

    @pytest.mark.asyncio
    async def test_dm_without_subject(self):
        mock_client = MagicMock()
        mock_user = MagicMock()
        mock_user.send = AsyncMock()
        mock_client.fetch_user = AsyncMock(return_value=mock_user)

        notifier = DiscordNotifier(mock_client)
        result = await notifier.send("12345", "", "Just a message")

        assert result is True
        mock_user.send.assert_called_once_with("Just a message")

    @pytest.mark.asyncio
    async def test_dm_failure(self):
        mock_client = MagicMock()
        mock_client.fetch_user = AsyncMock(side_effect=Exception("User not found"))

        notifier = DiscordNotifier(mock_client)
        result = await notifier.send("99999", "Alert", "Body")

        assert result is False


class TestZammadNotifier:
    @pytest.mark.asyncio
    async def test_successful_note(self):
        mock_client = MagicMock()
        mock_client.add_article_to_ticket = MagicMock(return_value={})

        notifier = ZammadNotifier(mock_client)
        result = await notifier.send("42", "Dispatch Note", "Ticket details here")

        assert result is True
        mock_client.add_article_to_ticket.assert_called_once()
        call_kwargs = mock_client.add_article_to_ticket.call_args
        assert call_kwargs[1]["ticket_id"] == 42
        assert call_kwargs[1]["internal"] is True

    @pytest.mark.asyncio
    async def test_note_failure(self):
        mock_client = MagicMock()
        mock_client.add_article_to_ticket = MagicMock(side_effect=Exception("API error"))

        notifier = ZammadNotifier(mock_client)
        result = await notifier.send("42", "Note", "Body")

        assert result is False


class TestNotificationRouter:
    @pytest.mark.asyncio
    async def test_routes_to_registered_notifier(self):
        router = NotificationRouter()
        mock_notifier = MagicMock()
        mock_notifier.send = AsyncMock(return_value=True)
        router.register("discord", mock_notifier)

        result = await router.send("discord", "user1", "Subject", "Body")

        assert result is True
        mock_notifier.send.assert_called_once_with("user1", "Subject", "Body")

    @pytest.mark.asyncio
    async def test_falls_back_to_log_notifier(self):
        router = NotificationRouter()
        result = await router.send("unknown_channel", "user1", "Subject", "Body")
        # LogNotifier always returns True
        assert result is True

    def test_available_channels(self):
        router = NotificationRouter()
        assert router.available_channels == []

        mock_notifier = MagicMock()
        router.register("discord", mock_notifier)
        router.register("zammad", mock_notifier)

        assert set(router.available_channels) == {"discord", "zammad"}

    @pytest.mark.asyncio
    async def test_multiple_channels(self):
        router = NotificationRouter()

        discord_notifier = MagicMock()
        discord_notifier.send = AsyncMock(return_value=True)
        zammad_notifier = MagicMock()
        zammad_notifier.send = AsyncMock(return_value=True)

        router.register("discord", discord_notifier)
        router.register("zammad", zammad_notifier)

        await router.send("discord", "user1", "S1", "B1")
        await router.send("zammad", "42", "S2", "B2")

        discord_notifier.send.assert_called_once_with("user1", "S1", "B1")
        zammad_notifier.send.assert_called_once_with("42", "S2", "B2")
