# tests/database/test_memory_manager.py

import pytest
import time
from datetime import datetime
from src.database.memory_manager import MemoryManager


@pytest.fixture
def mem_manager():
    """Provides a MemoryManager instance with an in-memory database for each test."""
    manager = MemoryManager(db_path=':memory:')
    manager.create_schema()
    yield manager
    manager.close()


def test_create_schema(mem_manager):
    """Verify that the schema creation results in the correct tables and columns."""
    conn = mem_manager._get_connection()
    cursor = conn.cursor()

    # Check User_Interactions table
    cursor.execute("PRAGMA table_info(User_Interactions)")
    columns = {row['name'] for row in cursor.fetchall()}
    expected_columns = {
        'interaction_id', 'user_identifier', 'persona_name', 'channel',
        'role', 'content', 'timestamp', 'zammad_ticket_id', 'platform_message_id'
    }
    assert columns == expected_columns

    # Check Suppressed_Interactions table
    cursor.execute("PRAGMA table_info(Suppressed_Interactions)")
    suppressed_columns = {row['name'] for row in cursor.fetchall()}
    expected_suppressed_columns = {'suppression_id', 'interaction_id', 'suppressed_at'}
    assert suppressed_columns == expected_suppressed_columns


def test_log_and_get_message(mem_manager):
    """Test logging a single message and retrieving it."""
    user_id, persona = "user1", "persona1"
    now = datetime.now()
    mem_manager.log_message(user_id, persona, "chan1", "user", "Hello", now, platform_message_id="p_id_1")

    history = mem_manager.get_personal_history(user_id, persona)
    assert len(history) == 1
    assert history[0]['content'] == "Hello"

    with mem_manager._get_connection() as conn:
        row = conn.execute("SELECT platform_message_id FROM User_Interactions").fetchone()
        assert row['platform_message_id'] == "p_id_1"


def test_suppress_message_by_platform_id(mem_manager):
    """Test that a message can be suppressed and is excluded from history."""
    user_id, persona = "user_suppress", "persona_suppress"

    # Log three messages with unique timestamps
    mem_manager.log_message(user_id, persona, "chan", "user", "Message 1", datetime.now(), platform_message_id="p1")
    time.sleep(0.01)
    mem_manager.log_message(user_id, persona, "chan", "assistant", "Message 2", datetime.now(),
                            platform_message_id="p2")
    time.sleep(0.01)
    mem_manager.log_message(user_id, persona, "chan", "user", "Message 3", datetime.now(), platform_message_id="p3")

    # History should have 3 messages
    history_before = mem_manager.get_personal_history(user_id, persona)
    assert len(history_before) == 3

    # Suppress the middle message
    success = mem_manager.suppress_message_by_platform_id("p2")
    assert success is True

    # History should now have 2 messages
    history_after = mem_manager.get_personal_history(user_id, persona)
    assert len(history_after) == 2
    contents = [msg['content'] for msg in history_after]
    assert "Message 1" in contents
    assert "Message 2" not in contents
    assert "Message 3" in contents


def test_suppress_nonexistent_message(mem_manager):
    """Test that attempting to suppress a non-existent message ID fails gracefully."""
    success = mem_manager.suppress_message_by_platform_id("nonexistent_id")
    assert success is False


def test_ticket_history_excludes_suppressed(mem_manager):
    """Test that get_ticket_history also respects suppressions."""
    user_id, persona, ticket_id = "user_ticket", "persona_ticket", 123

    mem_manager.log_message(user_id, persona, "chan", "user", "Ticket Msg 1", datetime.now(), "p_t1", ticket_id)
    time.sleep(0.01)
    mem_manager.log_message(user_id, persona, "chan", "assistant", "Ticket Msg 2", datetime.now(), "p_t2", ticket_id)

    assert len(mem_manager.get_ticket_history(ticket_id)) == 2
    mem_manager.suppress_message_by_platform_id("p_t1")
    assert len(mem_manager.get_ticket_history(ticket_id)) == 1
    assert mem_manager.get_ticket_history(ticket_id)[0]['content'] == "Ticket Msg 2"


def test_history_limit(mem_manager):
    """Test that the history limit correctly applies to non-suppressed messages."""
    user_id, persona = "user_limit", "persona_limit"

    for i in range(5):
        # Move datetime.now() inside the loop to get unique timestamps
        now = datetime.now()
        mem_manager.log_message(user_id, persona, "chan", "user", f"Msg {i}", now, f"p_{i}")
        time.sleep(0.01)

    # Suppress message with platform_id 'p_3'
    mem_manager.suppress_message_by_platform_id("p_3")

    # Ask for 3 most recent messages. Should get p_4, p_2, p_1 (since p_3 is skipped)
    limited_history = mem_manager.get_personal_history(user_id, persona, limit=3)
    assert len(limited_history) == 3

    # The final list is reversed for chronological order
    contents = [msg['content'] for msg in limited_history]
    assert contents == ["Msg 1", "Msg 2", "Msg 4"]