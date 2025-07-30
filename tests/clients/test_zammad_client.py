import pytest
from unittest.mock import patch, MagicMock
import os

# The import path now correctly reflects the new file structure.
from src.clients.zammad_client import ZammadClient


# --- Fixtures for reusable test setup ---

@pytest.fixture
def zammad_client(monkeypatch):
    """A fixture that provides a ZammadClient instance with mocked env vars."""
    monkeypatch.setenv("ZAMMAD_URL", "http://mock-zammad.com")
    monkeypatch.setenv("ZAMMAD_API_KEY", "test-token-12345")
    return ZammadClient()


# --- Unit Tests ---

def test_zammad_client_initialization_success(zammad_client):
    """Tests that the client initializes correctly with environment variables."""
    assert zammad_client.api_url == "http://mock-zammad.com"
    assert zammad_client.api_token == "test-token-12345"
    assert zammad_client.headers['Authorization'] == "Token token=test-token-12345"


def test_zammad_client_initialization_failure_missing_env(monkeypatch):
    """Tests that the client raises a ValueError if an env var is missing."""
    # Ensure the URL is set but the token is missing
    monkeypatch.setenv("ZAMMAD_URL", "http://mock-zammad.com")
    monkeypatch.delenv("ZAMMAD_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ZAMMAD_URL and ZAMMAD_API_KEY must be set"):
        ZammadClient()


# The patch target string must match the new location of the zammad_client module
@patch('src.clients.zammad_client.requests.request')
def test_create_ticket_sends_correct_payload(mock_request, zammad_client):
    """
    Tests that the create_ticket method constructs and sends the correct
    API request payload to the Zammad API.
    """
    # 1. ARRANGE: Configure the mock to simulate a successful API response
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"id": 123, "number": "T123", "title": "Test Ticket"}
    mock_request.return_value = mock_response

    # 2. ACT: Call the method under test
    ticket_data = zammad_client.create_ticket(
        title="Test Ticket",
        group="Users",
        customer_id=5,
        article_body="This is a test article body."
    )

    # 3. ASSERT: Verify the behavior and results
    # Was the requests library called exactly once?
    mock_request.assert_called_once()

    # Inspect the arguments it was called with
    args, kwargs = mock_request.call_args

    # Assert the method and URL are correct
    assert args[0] == 'post'
    assert args[1] == "http://mock-zammad.com/api/v1/tickets"

    # Assert the headers include the correct authorization token
    assert kwargs['headers']['Authorization'] == "Token token=test-token-12345"

    # Assert the JSON payload is structured exactly as Zammad expects
    sent_payload = kwargs['json']
    assert sent_payload['title'] == "Test Ticket"
    assert sent_payload['customer_id'] == 5
    assert sent_payload['group'] == "Users"
    assert sent_payload['article']['body'] == "This is a test article body."
    assert sent_payload['article']['internal'] is False

    # Assert that the method returned the parsed JSON from the simulated response
    assert ticket_data['id'] == 123
    assert ticket_data['title'] == "Test Ticket"


@patch('src.clients.zammad_client.requests.request')
def test_add_article_sends_correct_payload(mock_request, zammad_client):
    """Tests that the add_article method sends the correct payload."""
    # ARRANGE
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"id": 987, "ticket_id": 123}
    mock_request.return_value = mock_response

    # ACT
    article_data = zammad_client.add_article_to_ticket(
        ticket_id=123,
        body="This is a follow-up message.",
        internal=True
    )

    # ASSERT
    mock_request.assert_called_once()
    args, kwargs = mock_request.call_args

    assert args[0] == 'post'
    assert args[1] == "http://mock-zammad.com/api/v1/ticket_articles"

    sent_payload = kwargs['json']
    assert sent_payload['ticket_id'] == 123
    assert sent_payload['body'] == "This is a follow-up message."
    assert sent_payload['internal'] is True  # Verify internal flag is handled

    assert article_data['id'] == 987
