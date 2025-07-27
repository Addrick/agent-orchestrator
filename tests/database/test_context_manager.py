# tests/database/test_context_manager.py

import pytest
import time
import sqlite3
from datetime import datetime
from src.database.context_manager import ContextManager, UNASSOCIATED_BUSINESS_NAME


@pytest.fixture
def db_manager() -> ContextManager:
    """Fixture to set up an in-memory database for each test."""
    manager = ContextManager(db_path=":memory:")
    manager.create_schema()
    manager._initialize_db()
    return manager


def test_create_schema(db_manager: ContextManager):
    """Verify that all tables are created correctly."""
    with db_manager._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        # Filter out sqlite's internal sequence table for a more robust check
        filtered_tables = [table for table in tables if table != 'sqlite_sequence']
        expected_tables = [
            'Businesses',
            'Contacts',
            'Contact_Identifiers',
            'Tickets',
            'Interactions',
            'Contact_Placement_Guesses'
        ]
        assert all(table in filtered_tables for table in expected_tables)
        assert len(filtered_tables) == len(expected_tables)


def test_initialize_db(db_manager: ContextManager):
    """Verify the default 'Unassociated Contacts' business is created."""
    with db_manager._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT business_name FROM Businesses WHERE business_name = ?", (UNASSOCIATED_BUSINESS_NAME,))
        result = cursor.fetchone()
        assert result is not None
        assert result['business_name'] == UNASSOCIATED_BUSINESS_NAME


def test_get_or_create_contact_new(db_manager: ContextManager):
    """Test creating a new contact."""
    contact_id, is_new = db_manager._get_or_create_contact("user@example.com", "email")
    assert contact_id == 1
    assert is_new is True

    with db_manager._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Contacts WHERE contact_id = ?", (contact_id,))
        assert cursor.fetchone()[0] == 1
        cursor.execute("SELECT COUNT(*) FROM Contact_Identifiers WHERE contact_id = ?", (contact_id,))
        assert cursor.fetchone()[0] == 1


def test_get_or_create_contact_existing(db_manager: ContextManager):
    """Test retrieving an existing contact."""
    # First creation
    db_manager._get_or_create_contact("user@example.com", "email")
    # Second retrieval
    contact_id, is_new = db_manager._get_or_create_contact("user@example.com", "email")
    assert contact_id == 1
    assert is_new is False


def test_get_or_create_active_ticket_new(db_manager: ContextManager):
    """Test creating a new ticket when no active one exists."""
    contact_id, _ = db_manager._get_or_create_contact("user@example.com", "email")
    ticket_id = db_manager._get_or_create_active_ticket(contact_id)
    assert ticket_id == 1

    with db_manager._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM Tickets WHERE ticket_id = ?", (ticket_id,))
        assert cursor.fetchone()['status'] == 'open'


def test_get_or_create_active_ticket_existing(db_manager: ContextManager):
    """Test retrieving an existing active ticket."""
    contact_id, _ = db_manager._get_or_create_contact("user@example.com", "email")
    first_ticket_id = db_manager._get_or_create_active_ticket(contact_id)
    second_ticket_id = db_manager._get_or_create_active_ticket(contact_id)
    assert first_ticket_id == second_ticket_id


def test_log_interaction(db_manager: ContextManager):
    """Test logging an interaction and updating the ticket's timestamp."""
    contact_id, _ = db_manager._get_or_create_contact("user@example.com", "email")
    ticket_id = db_manager._get_or_create_active_ticket(contact_id)

    with db_manager._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT last_update_timestamp FROM Tickets WHERE ticket_id = ?", (ticket_id,))
        # The registered converter now returns a datetime object directly.
        initial_timestamp = cursor.fetchone()['last_update_timestamp']
        assert isinstance(initial_timestamp, datetime)

    # Introduce a small delay to ensure the next timestamp is different
    time.sleep(0.001)

    db_manager.log_interaction(ticket_id, 'inbound', 'Hello, world!', 'email')

    with db_manager._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT raw_content FROM Interactions WHERE ticket_id = ?", (ticket_id,))
        assert cursor.fetchone()['raw_content'] == 'Hello, world!'

        cursor.execute("SELECT last_update_timestamp FROM Tickets WHERE ticket_id = ?", (ticket_id,))
        updated_timestamp = cursor.fetchone()['last_update_timestamp']
        assert isinstance(updated_timestamp, datetime)
        assert updated_timestamp > initial_timestamp


def test_build_prompt_context_object(db_manager: ContextManager):
    """Test the construction of the context object for the LLM prompt."""
    # Corrected to unpack all three return values from get_context_for_generation
    contact_id, ticket_id, _ = db_manager.get_context_for_generation("test_user", "test_channel")

    db_manager.log_interaction(ticket_id, 'inbound', 'Message 1', 'test_channel')
    time.sleep(0.001)  # Ensure unique timestamps for stable sorting
    db_manager.log_interaction(ticket_id, 'outbound', 'Response 1', 'test_channel')
    time.sleep(0.001)  # Ensure unique timestamps for stable sorting
    db_manager.log_interaction(ticket_id, 'inbound', 'Message 2', 'test_channel')

    # Test without history limit
    context = db_manager.build_prompt_context_object(contact_id)
    assert context['user']['name'] == 'Unknown User (test_user)'
    assert context['business']['name'] == UNASSOCIATED_BUSINESS_NAME
    assert len(context['history']) == 3
    assert context['history'][0]['content'] == 'Message 1'
    assert context['history'][0]['role'] == 'user'
    assert context['history'][2]['content'] == 'Message 2'
    assert context['history'][2]['role'] == 'user'

    # Test with history limit
    limited_context = db_manager.build_prompt_context_object(contact_id, history_limit=2)
    assert len(limited_context['history']) == 2
    assert limited_context['history'][0]['content'] == 'Response 1'
    assert limited_context['history'][0]['role'] == 'assistant'
    assert limited_context['history'][1]['content'] == 'Message 2'
    assert limited_context['history'][1]['role'] == 'user'


def test_record_business_guess(db_manager: ContextManager):
    """Test that a business guess is correctly recorded."""
    contact_id, _ = db_manager._get_or_create_contact("new_contact@corp.com", "email")

    with db_manager._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Businesses (business_name) VALUES (?)", ("TestCorp",))
        business_id = cursor.lastrowid
        conn.commit()

    db_manager.record_business_guess(contact_id, business_id, "Email address matches domain.")

    with db_manager._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Contact_Placement_Guesses WHERE contact_id = ?", (contact_id,))
        guess = cursor.fetchone()
        assert guess is not None
        assert guess['guessed_business_id'] == business_id
        assert guess['reasoning'] == "Email address matches domain."
        # Verify timestamp is now a datetime object
        assert isinstance(guess['timestamp'], datetime)


def test_get_all_businesses(db_manager: ContextManager):
    """Test retrieval of all businesses, excluding the unassociated one."""
    with db_manager._get_connection() as conn:
        conn.execute("INSERT INTO Businesses (business_name) VALUES ('Client A')")
        conn.execute("INSERT INTO Businesses (business_name) VALUES ('Client B')")
        conn.commit()

    businesses = db_manager.get_all_businesses()
    assert len(businesses) == 2
    business_names = {b['business_name'] for b in businesses}
    assert 'Client A' in business_names
    assert 'Client B' in business_names
    assert UNASSOCIATED_BUSINESS_NAME not in business_names
