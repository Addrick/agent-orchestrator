# tests/database/test_memory_manager.py

import pytest
import sqlite3
import time
from src.database.memory_manager import MemoryManager


@pytest.fixture
def mem_manager():
    """Provides a MemoryManager instance with an in-memory database for each test."""
    manager = MemoryManager(db_path=':memory:')
    manager.create_schema()
    return manager


def test_create_schema(mem_manager):
    """Verify that the schema creation results in the correct table and columns."""
    conn = mem_manager._get_connection()
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='User_Interactions'")
    assert cursor.fetchone() is not None, "Table 'User_Interactions' was not created."

    # Check for columns
    cursor.execute("PRAGMA table_info(User_Interactions)")
    columns = {row['name'] for row in cursor.fetchall()}
    expected_columns = {
        'interaction_id',
        'user_identifier',
        'channel',
        'timestamp',
        'user_message',
        'bot_response',
        'zammad_ticket_id'
    }
    assert columns == expected_columns, f"Table columns are incorrect. Expected {expected_columns}, got {columns}."


def test_log_and_get_interaction(mem_manager):
    """Test logging a simple interaction and retrieving it."""
    user_id = "test_user_123"
    channel = "discord"
    user_msg = "Hello bot!"
    bot_resp = "Hello user!"

    mem_manager.log_interaction(user_id, channel, user_msg, bot_resp)

    history = mem_manager.get_history(user_id)

    assert len(history) == 1
    interaction = history[0]
    assert interaction['user_message'] == user_msg
    assert interaction['bot_response'] == bot_resp


def test_history_order(mem_manager):
    """Test that history is returned in chronological order."""
    user_id = "test_user_order"

    # Log two interactions
    mem_manager.log_interaction(user_id, "test", "First message", "First response")
    time.sleep(0.01)  # Ensure distinct timestamps
    mem_manager.log_interaction(user_id, "test", "Second message", "Second response")

    history = mem_manager.get_history(user_id)

    assert len(history) == 2
    assert history[0]['user_message'] == "First message"
    assert history[1]['user_message'] == "Second message"


def test_history_limit(mem_manager):
    """Test that the history limit is respected."""
    user_id = "test_user_limit"

    for i in range(5):
        mem_manager.log_interaction(user_id, "test", f"Message {i}", f"Response {i}")
        time.sleep(0.01)

    # Get full history
    full_history = mem_manager.get_history(user_id)
    assert len(full_history) == 5

    # Get limited history
    limited_history = mem_manager.get_history(user_id, limit=3)
    assert len(limited_history) == 3

    # Check that it returned the LATEST 3 messages
    assert limited_history[0]['user_message'] == "Message 2"
    assert limited_history[1]['user_message'] == "Message 3"
    assert limited_history[2]['user_message'] == "Message 4"


def test_zammad_ticket_id_handling(mem_manager):
    """Test that the zammad_ticket_id is stored correctly when provided and is NULL otherwise."""
    user_id_with_ticket = "user_with_ticket"
    user_id_no_ticket = "user_no_ticket"

    # Interaction with a Zammad ticket ID
    mem_manager.log_interaction(
        user_id_with_ticket, "gmail", "I have a problem.", "Creating a ticket.", zammad_ticket_id=9001
    )

    # Interaction without a Zammad ticket ID
    mem_manager.log_interaction(
        user_id_no_ticket, "discord", "Just chatting.", "Nice to chat with you."
    )

    conn = mem_manager._get_connection()

    # Verify the ticket ID was stored
    cursor_with = conn.execute("SELECT zammad_ticket_id FROM User_Interactions WHERE user_identifier = ?",
                               (user_id_with_ticket,))
    row_with = cursor_with.fetchone()
    assert row_with is not None
    assert row_with['zammad_ticket_id'] == 9001

    # Verify the ticket ID is NULL
    cursor_no = conn.execute("SELECT zammad_ticket_id FROM User_Interactions WHERE user_identifier = ?",
                             (user_id_no_ticket,))
    row_no = cursor_no.fetchone()
    assert row_no is not None
    assert row_no['zammad_ticket_id'] is None


def test_get_history_for_nonexistent_user(mem_manager):
    """Test that getting history for a user with no interactions returns an empty list."""
    history = mem_manager.get_history("nonexistent_user")
    assert history == []
