# tests/clients/test_zammad_service.py

import pytest
from unittest.mock import MagicMock, AsyncMock

from src.clients.zammad_service import ZammadIntegration, ZammadContext
from src.clients.zammad_client import ZammadClient


@pytest.fixture
def zammad_integration():
    """Provides a ZammadIntegration with a mocked ZammadClient."""
    mock_client = MagicMock(spec=ZammadClient)
    mock_client.api_url = "http://zammad.local"
    return ZammadIntegration(mock_client), mock_client


# --- Name ---

def test_name(zammad_integration):
    integration, _ = zammad_integration
    assert integration.name == "zammad"


# --- Ticket Number Detection ---

@pytest.mark.parametrize("message, expected", [
    ("Help with [Ticket#12345]", 12345),
    ("[ticket#54321] is the one", 54321),
    ("No ticket here", None),
    ("Invalid format [Ticket#abc]", None),
])
def test_find_ticket_number_in_message(message, expected):
    assert ZammadIntegration._find_ticket_number_in_message(message) == expected


# --- Ticket ID Lookup ---

@pytest.mark.asyncio
async def test_get_ticket_id_from_number_success(zammad_integration):
    integration, mock_client = zammad_integration
    mock_client.search_tickets.return_value = [{'id': 999}]
    result = await integration._get_ticket_id_from_number(12345)
    mock_client.search_tickets.assert_called_once_with(query="number:12345")
    assert result == 999


@pytest.mark.asyncio
async def test_get_ticket_id_from_number_not_found(zammad_integration):
    integration, mock_client = zammad_integration
    mock_client.search_tickets.return_value = []
    result = await integration._get_ticket_id_from_number(12345)
    assert result is None


# --- User Resolution ---

@pytest.mark.asyncio
async def test_get_or_create_user_existing_real_email(zammad_integration):
    integration, mock_client = zammad_integration
    mock_client.search_user.return_value = [{'id': 101, 'email': 'test@example.com'}]
    user_id, email = await integration._get_or_create_user("Test User <test@example.com>", "gmail")
    mock_client.search_user.assert_called_once_with('test@example.com')
    mock_client.create_user.assert_not_called()
    assert user_id == 101
    assert email == 'test@example.com'


@pytest.mark.asyncio
async def test_get_or_create_user_new_non_email(zammad_integration):
    integration, mock_client = zammad_integration
    mock_client.search_user.return_value = []
    mock_client.create_user.return_value = {'id': 102, 'email': 'discord-12345@zammad.local'}
    user_id, email = await integration._get_or_create_user("12345", "discord", user_display_name="DiscordUser")

    expected_email = "discord-12345@zammad.local"
    mock_client.search_user.assert_called_once_with(expected_email)
    mock_client.create_user.assert_called_once()
    assert user_id == 102
    assert email == 'discord-12345@zammad.local'


# --- resolve_context ---

@pytest.mark.asyncio
async def test_resolve_context_with_ticket_number(zammad_integration):
    """Resolves ticket from message number."""
    integration, mock_client = zammad_integration
    mock_client.search_user.return_value = [{'id': 101, 'email': 'u@test.com'}]
    mock_client.search_tickets.return_value = [{'id': 555}]

    result = await integration.resolve_context('user', 'channel', 'See [Ticket#999]', None)

    assert result['customer_id'] == 101
    assert result['ticket_id'] == 555
    assert result['user_facing_ticket_number'] == 999


@pytest.mark.asyncio
async def test_resolve_context_falls_back_to_active_ticket(zammad_integration):
    """Without ticket number in message, resolves via active ticket search."""
    integration, mock_client = zammad_integration
    mock_client.search_user.return_value = [{'id': 101, 'email': 'u@test.com'}]
    # First call for ticket by number (won't happen), second for active tickets
    mock_client.search_tickets.return_value = [{'id': 777}]

    result = await integration.resolve_context('user', 'channel', 'help me', None)

    assert result['ticket_id'] == 777
    assert result['user_facing_ticket_number'] is None


@pytest.mark.asyncio
async def test_resolve_context_no_user_found(zammad_integration):
    """Returns empty context when user resolution fails."""
    integration, mock_client = zammad_integration
    mock_client.search_user.side_effect = Exception("Connection error")

    result = await integration.resolve_context('user', 'channel', 'help', None)

    assert result['customer_id'] is None
    assert result['ticket_id'] is None


# --- on_message ---

@pytest.mark.asyncio
async def test_on_message_mirrors_to_ticket(zammad_integration):
    integration, mock_client = zammad_integration
    service_data = {"ticket_id": 42, "zammad_email": "u@test.com"}
    await integration.on_message(service_data, "Hello")
    mock_client.add_article_to_ticket.assert_called_once_with(
        ticket_id=42, body="Hello", impersonate_email="u@test.com"
    )


@pytest.mark.asyncio
async def test_on_message_skips_without_ticket(zammad_integration):
    integration, mock_client = zammad_integration
    service_data = {"ticket_id": None}
    await integration.on_message(service_data, "Hello")
    mock_client.add_article_to_ticket.assert_not_called()


# --- prepare_tool_args ---

def test_prepare_tool_args_injects_customer_id(zammad_integration):
    integration, _ = zammad_integration
    args = {"title": "Test"}
    service_data = {"customer_id": 101}
    result = integration.prepare_tool_args("create_ticket", args, service_data)
    assert result["customer_id"] == 101


def test_prepare_tool_args_preserves_explicit_customer_id(zammad_integration):
    integration, _ = zammad_integration
    args = {"title": "Test", "customer_id": 999}
    service_data = {"customer_id": 101}
    result = integration.prepare_tool_args("create_ticket", args, service_data)
    assert result["customer_id"] == 999


def test_prepare_tool_args_ignores_non_create_ticket(zammad_integration):
    integration, _ = zammad_integration
    args = {"state": "closed"}
    service_data = {"customer_id": 101}
    result = integration.prepare_tool_args("update_ticket", args, service_data)
    assert "customer_id" not in result


# --- on_tool_result ---

def test_on_tool_result_captures_ticket_id(zammad_integration):
    integration, _ = zammad_integration
    service_data = {"ticket_id": None}
    result = {"result": {"id": 50}}
    integration.on_tool_result("create_ticket", result, service_data)
    assert service_data["ticket_id"] == 50


def test_on_tool_result_ignores_non_create(zammad_integration):
    integration, _ = zammad_integration
    service_data = {"ticket_id": None}
    result = {"result": {"id": 50}}
    integration.on_tool_result("update_ticket", result, service_data)
    assert service_data["ticket_id"] is None


# --- get_system_messages ---

def test_get_system_messages_with_ticket(zammad_integration):
    integration, _ = zammad_integration
    msgs = integration.get_system_messages({"ticket_id": 42, "user_facing_ticket_number": 100})
    assert len(msgs) == 1
    assert "#100" in msgs[0]["content"]


def test_get_system_messages_without_ticket(zammad_integration):
    integration, _ = zammad_integration
    msgs = integration.get_system_messages({"ticket_id": None})
    assert len(msgs) == 0
