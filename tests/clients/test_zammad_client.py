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
def test_create_ticket_with_article(mock_make_request, zammad_client):
    """Test the payload construction for creating a ticket with an initial article."""
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
        "tags": "pytest,automated"
    }
    mock_make_request.assert_called_once_with('post', 'tickets', json=expected_payload)


@patch('src.clients.zammad_client.ZammadClient._make_request')
def test_create_ticket_without_article(mock_make_request, zammad_client):
    """Test that creating a ticket without an article body omits the 'article' key."""
    zammad_client.create_ticket(
        title="Empty Ticket",
        group="Users",
        customer_id=123
    )
    expected_payload = {
        "title": "Empty Ticket",
        "group": "Users",
        "customer_id": 123
    }
    mock_make_request.assert_called_once_with('post', 'tickets', json=expected_payload)


@patch('src.clients.zammad_client.ZammadClient._make_request')
def test_delete_ticket(mock_make_request, zammad_client):
    """Test the endpoint construction for deleting a ticket."""
    zammad_client.delete_ticket(999)
    mock_make_request.assert_called_once_with('delete', 'tickets/999')


@patch('src.clients.zammad_client.ZammadClient._make_request')
def test_add_article_with_impersonation(mock_make_request, zammad_client):
    """Test that the impersonation header is correctly added when adding an article."""
    zammad_client.add_article_to_ticket(
        ticket_id=123,
        body="User's reply",
        impersonate_email="customer@example.com"
    )
    expected_payload = {
        "ticket_id": 123,
        "body": "User's reply",
        "type": "note",
        "internal": False
    }
    mock_make_request.assert_called_once_with(
        'post', 'ticket_articles', json=expected_payload, impersonate_email="customer@example.com"
    )


@patch('src.clients.zammad_client.ZammadClient._make_request')
def test_search_tickets_with_sorting(mock_make_request, zammad_client):
    """Test that search_tickets correctly includes sorting parameters."""
    query = "customer_id:123"
    zammad_client.search_tickets(query=query, limit=10, sort_by='updated_at', order_by='asc')

    expected_params = {
        'query': query,
        'limit': 10,
        'sort_by': 'updated_at',
        'order_by': 'asc'
    }
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


@patch('src.clients.zammad_client.ZammadClient._make_request')
def test_link_tickets(mock_make_request, zammad_client):
    """Test the payload construction for linking two tickets using the new /links/add endpoint."""
    # Mock the return values for the sequence of calls
    # 1st call: get_ticket -> returns ticket dict
    # 2nd call: post links/add -> returns success dict
    mock_make_request.side_effect = [{"number": "T101"}, {"id": 1}]
    
    zammad_client.link_tickets(101, 202, link_type="parent")
    
    # 1. Verifying it fetched the ticket number
    mock_make_request.assert_any_call('get', 'tickets/101?expand=true')
    
    # 2. Verifying the link call
    expected_payload = {
        "link_type": "parent",
        "link_object_source": "Ticket",
        "link_object_source_number": "T101",
        "link_object_target": "Ticket",
        "link_object_target_value": 202
    }
    mock_make_request.assert_any_call('post', 'links/add', json=expected_payload)


@patch('src.clients.zammad_client.ZammadClient._make_request')
@patch('src.clients.zammad_client.ZammadClient.get_ticket_articles')
def test_merge_tickets(mock_get_articles, mock_make_request, zammad_client):
    """Test the orchestration of linking, moving articles, and closing the source ticket."""
    mock_get_articles.return_value = [{'id': 1, 'body': 'Note 1'}, {'id': 2, 'body': 'Note 2'}]
    # Mock the sequence of _make_request calls:
    # 1. get_ticket (inside link_tickets)
    # 2. post links/add (inside link_tickets)
    # 3. put ticket_articles/1
    # 4. put ticket_articles/2
    # 5. put tickets/101
    mock_make_request.side_effect = [
        {"number": "T101"}, # get_ticket
        {"id": 1},          # link
        {"id": 1},          # article 1
        {"id": 2},          # article 2
        {"id": 101}         # state update
    ]
    
    zammad_client.merge_tickets(source_ticket_id=101, target_ticket_id=202)

    # 1. Check link call (via links/add)
    link_payload = {
        "link_type": "normal", # default for merge
        "link_object_source": "Ticket",
        "link_object_source_number": "T101",
        "link_object_target": "Ticket",
        "link_object_target_value": 202
    }
    mock_make_request.assert_any_call('post', 'links/add', json=link_payload)

    # 2. Check article move calls
    mock_make_request.assert_any_call('put', 'ticket_articles/1', json={'ticket_id': 202})
    mock_make_request.assert_any_call('put', 'ticket_articles/2', json={'ticket_id': 202})

    # 3. Check source ticket state update (State by name)
    mock_make_request.assert_any_call('put', 'tickets/101', json={'state': 'merged'})
