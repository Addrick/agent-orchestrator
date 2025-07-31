# tests/clients/test_zammad_client_unit.py

import pytest
import requests
from unittest.mock import patch, MagicMock
from src.clients.zammad_client import ZammadClient

BASE_URL = "http://test.zammad.local"


@pytest.fixture
def zammad_client(monkeypatch):
    """Provides a ZammadClient instance with a mocked environment for each test."""
    monkeypatch.setenv("ZAMMAD_URL", BASE_URL)
    monkeypatch.setenv("ZAMMAD_API_KEY", "test_api_key")
    return ZammadClient()


@patch('requests.request')
def test_make_request_success(mock_request, zammad_client):
    """Test that a successful response with JSON is parsed correctly."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'id': 1, 'title': 'Test'}
    mock_request.return_value = mock_response

    result = zammad_client._make_request('get', 'test_endpoint')

    assert result == {'id': 1, 'title': 'Test'}
    mock_response.raise_for_status.assert_called_once()


@patch('requests.request')
def test_make_request_no_content(mock_request, zammad_client):
    """Test that a response with no content returns None."""
    mock_response = MagicMock()
    mock_response.status_code = 204
    mock_response.content = b''
    mock_request.return_value = mock_response

    result = zammad_client._make_request('delete', 'test_endpoint/1')

    assert result is None


@patch('requests.request')
def test_make_request_http_error(mock_request, zammad_client):
    """Test that an HTTP error is raised correctly."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
    mock_request.return_value = mock_response

    with pytest.raises(requests.exceptions.HTTPError):
        zammad_client._make_request('get', 'invalid_endpoint')


# --- Ticket Method Tests ---

@patch('src.clients.zammad_client.ZammadClient._make_request')
def test_create_ticket(mock_make_request, zammad_client):
    """Test the payload construction for creating a ticket."""
    zammad_client.create_ticket(
        title="New Issue",
        group="Users",
        customer_id=123,
        article_body="Help me!",
        tags=["pytest", "automated"]
    )

    expected_payload = {
        "title": "New Issue",
        "group": "Users",
        "customer_id": 123,
        "article": {
            "body": "Help me!",
            "type": "note",
            "internal": False
        },
        "tags": "pytest,automated"  # Verify tags are comma-separated
    }
    mock_make_request.assert_called_once_with('post', 'tickets', json=expected_payload)


@patch('src.clients.zammad_client.ZammadClient._make_request')
def test_delete_ticket(mock_make_request, zammad_client):
    """Test the endpoint construction for deleting a ticket."""
    zammad_client.delete_ticket(999)
    mock_make_request.assert_called_once_with('delete', 'tickets/999')


@patch('src.clients.zammad_client.ZammadClient._make_request')
def test_search_tickets(mock_make_request, zammad_client):
    """Test that search_tickets calls _make_request with a params dictionary."""
    query = "customer_id:123 AND state:open"
    zammad_client.search_tickets(query=query, limit=10)

    expected_params = {'query': query, 'limit': 10}
    mock_make_request.assert_called_once_with('get', 'tickets/search', params=expected_params)


# --- User Method Tests ---

@patch('src.clients.zammad_client.ZammadClient._make_request')
def test_create_user_with_note(mock_make_request, zammad_client):
    """Test the payload construction for creating a user with a note."""
    zammad_client.create_user(
        email="test@example.com",
        firstname="Test",
        lastname="User",
        note="From automated test"
    )
    expected_payload = {
        "firstname": "Test",
        "lastname": "User",
        "email": "test@example.com",
        "roles": ["Customer"],
        "active": True,
        "note": "From automated test"
    }
    mock_make_request.assert_called_once_with('post', 'users', json=expected_payload)


@patch('src.clients.zammad_client.ZammadClient._make_request')
def test_create_user_without_note(mock_make_request, zammad_client):
    """Test the payload construction for creating a user without a note."""
    zammad_client.create_user(
        email="test@example.com",
        firstname="Test",
        lastname="User"
    )
    expected_payload = {
        "firstname": "Test",
        "lastname": "User",
        "email": "test@example.com",
        "roles": ["Customer"],
        "active": True
    }
    # Note should not be in the payload
    mock_make_request.assert_called_once_with('post', 'users', json=expected_payload)


@patch('src.clients.zammad_client.ZammadClient._make_request')
def test_delete_user(mock_make_request, zammad_client):
    """Test the endpoint construction for deleting a user."""
    zammad_client.delete_user(789)
    mock_make_request.assert_called_once_with('delete', 'users/789')


@patch('src.clients.zammad_client.ZammadClient._make_request')
def test_search_user(mock_make_request, zammad_client):
    """Test that search_user calls _make_request with a params dictionary."""
    query = "test@example.com"
    zammad_client.search_user(query=query)

    expected_params = {'query': query}
    mock_make_request.assert_called_once_with('get', 'users/search', params=expected_params)
