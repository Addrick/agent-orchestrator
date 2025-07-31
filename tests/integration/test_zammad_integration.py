# tests/clients/test_zammad_client.py

import pytest
from datetime import datetime
import requests
from src.clients.zammad_client import ZammadClient

# Mark all tests in this file as 'integration'.
# To run these tests, use the command: pytest -m integration
pytestmark = pytest.mark.integration

# Use static identifiers for the test user and ticket to ensure idempotency.
TEST_USER_EMAIL = "pytest-lifecycle-user@zammad.local"
TEST_TICKET_TITLE = "[Pytest Automated Test Ticket]"


@pytest.fixture(scope="module")
def zammad_client():
    """
    Provides a ZammadClient instance for the entire test module.
    Skips tests if the client cannot be initialized or fails to connect.
    """
    try:
        client = ZammadClient()
        client.get_self()  # Verify connection and authentication
        return client
    except (ValueError, requests.exceptions.RequestException) as e:
        pytest.skip(f"Skipping Zammad integration tests: Cannot connect or authenticate. Error: {e}")


@pytest.fixture(scope="function")
def managed_test_user(zammad_client: ZammadClient) -> int:
    """
    Provides the ID of a persistent test user and cleans up their specific test ticket
    from the previous run before yielding the user ID.
    """
    # 1. Find or Create the persistent test user
    print(f"\nEnsuring test user '{TEST_USER_EMAIL}' exists...")
    users = zammad_client.search_user(query=TEST_USER_EMAIL)
    if users:
        user_id = users[0]['id']
        print(f"Found existing test user with ID: {user_id}")
    else:
        print("Test user not found, creating a new one...")
        user_data = zammad_client.create_user(
            email=TEST_USER_EMAIL,
            firstname="Pytest",
            lastname="LifecycleUser",
            note="This is a persistent user for automated integration tests."
        )
        user_id = user_data['id']
        print(f"Created new test user with ID: {user_id}")

    # 2. Pre-run Cleanup: Find and delete the specific test ticket from the *previous* run.
    print(f"Cleaning up previous test ticket for user ID {user_id}...")
    # The query is very specific to avoid deleting other tickets.
    cleanup_query = f'customer_id:{user_id} AND title:"{TEST_TICKET_TITLE}"'
    orphaned_tickets = zammad_client.search_tickets(query=cleanup_query)

    if not orphaned_tickets:
        print("No previous test ticket found to clean up.")
    else:
        for ticket in orphaned_tickets:
            ticket_id = ticket['id']
            print(f"Deleting previous test ticket #{ticket_id}...")
            zammad_client.delete_ticket(ticket_id)
            print(f"Deleted ticket #{ticket_id}.")

    yield user_id
    # No cleanup after yield; we want the newly created ticket to persist for inspection.


def test_ticket_creation_for_inspection(zammad_client: ZammadClient, managed_test_user: int):
    """
    Tests the creation of a ticket and leaves it in the system for inspection.
    The cleanup is handled by the fixture on the next test run.
    """
    customer_id = managed_test_user
    run_timestamp = datetime.now().isoformat()
    ticket_body = (
        f"This is an automated test ticket.\n"
        f"Test Run Timestamp: {run_timestamp}\n"
        f"This ticket should be automatically deleted the next time the integration test suite is run."
    )

    # CREATE the ticket for the managed user
    print(f"Attempting to create inspection ticket for user #{customer_id}")
    ticket_data = zammad_client.create_ticket(
        title=TEST_TICKET_TITLE,
        group='Users',
        customer_id=customer_id,
        article_body=ticket_body
    )

    # VERIFY creation
    assert ticket_data and 'id' in ticket_data
    created_ticket_id = ticket_data['id']
    print(f"Successfully created inspection ticket with ID: {created_ticket_id}")

    # Add an update for good measure
    print(f"Adding an update to ticket #{created_ticket_id}")
    article_data = zammad_client.add_article_to_ticket(
        ticket_id=created_ticket_id,
        body="This is a test update article."
    )
    assert article_data and 'id' in article_data
    print("Successfully added article.")
    print(f"Test finished. Ticket #{created_ticket_id} is left in Zammad for inspection.")