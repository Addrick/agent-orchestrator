import pytest
import os
from datetime import datetime

# The import path is updated to the new location
from src.clients.zammad_client import ZammadClient

# This marker tells pytest these tests are part of the 'integration' group.
# They will be skipped unless run with `pytest -m "integration"`.
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def zammad_client():
    """
    Provides a real ZammadClient instance for the entire test module.
    It assumes .env is configured to point to the local Docker instance.
    Skips all tests in this file if the API token is not configured.
    """
    if not os.environ.get("ZAMMAD_API_KEY") or not os.environ.get("ZAMMAD_URL"):
        pytest.skip("Skipping integration tests: ZAMMAD_URL and ZAMMAD_API_TOKEN must be set in .env")

    return ZammadClient()


def test_zammad_connection_and_authentication(zammad_client):
    """
    A simple "smoke test" to verify the connection and authentication work.
    It fetches the user associated with the API token ("me").
    """
    # ARRANGE & ACT
    # Assumes you will add a `get_self()` method to your client for this test.
    # It would call the GET /api/v1/users/me endpoint.
    user_data = zammad_client.get_self()

    # ASSERT
    assert user_data is not None
    assert "id" in user_data
    assert user_data['email'] is not None  # The admin user should have an email


def test_user_and_ticket_lifecycle(zammad_client):
    """
    Tests the full lifecycle: create a user, create a ticket for them,
    and then add a follow-up article. This creates real data.
    """
    # ARRANGE: Create a unique email for the new user to avoid test collisions
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    test_email = f"test-customer-{timestamp}@example.com"

    # --- 1. Create User ---
    # ACT
    # Assumes a `create_user` method exists on your client
    new_user = zammad_client.create_user(
        firstname="Integration",
        lastname=f"Test-{timestamp}",
        email=test_email
    )
    # ASSERT
    assert "id" in new_user
    assert new_user['email'] == test_email
    customer_id = new_user['id']

    # --- 2. Create Ticket ---
    # ACT
    new_ticket = zammad_client.create_ticket(
        title=f"Integration Test Ticket {timestamp}",
        group="Users",  # Assumes 'Users' group exists
        customer_id=customer_id,
        article_body="This is the first message of the ticket."
    )
    # ASSERT
    assert "id" in new_ticket
    assert new_ticket['customer_id'] == customer_id
    ticket_id = new_ticket['id']

    # --- 3. Add Article to Ticket ---
    # ACT
    new_article = zammad_client.add_article_to_ticket(
        ticket_id=ticket_id,
        body="This is a second, follow-up message."
    )
    # ASSERT
    assert "id" in new_article
    assert new_article['ticket_id'] == ticket_id

    # Optional: A more advanced test could then fetch the ticket's articles
    # and verify that there are now two articles associated with it.
