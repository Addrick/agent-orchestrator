# tests/database/test_memory_manager.py

import pytest
import time
from datetime import datetime
from src.database.memory_manager import MemoryManager


@pytest.fixture
def mem_manager():
    """Provides a MemoryManager instance with an in-memory database for each test."""
    manager = MemoryManager(db_path=':memory:')
    # The schema is now breaking, must be rebuilt every time.
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
        'author_role', 'author_name', 'content', 'timestamp',
        'zammad_ticket_id', 'platform_message_id'
    }
    assert columns == expected_columns

    # Check Suppressed_Interactions table
    cursor.execute("PRAGMA table_info(Suppressed_Interactions)")
    suppressed_columns = {row['name'] for row in cursor.fetchall()}
    expected_suppressed_columns = {'suppression_id', 'interaction_id', 'suppressed_at'}
    assert suppressed_columns == expected_suppressed_columns


def test_log_and_get_message(mem_manager):
    """Test logging a single message and retrieving its raw data."""
    user_id, persona = "user1", "persona1"
    now = datetime.now()
    mem_manager.log_message(
        user_identifier=user_id, persona_name=persona, channel="chan1",
        author_role="user", author_name="TestUser", content="Hello",
        timestamp=now, platform_message_id="p_id_1"
    )

    history = mem_manager.get_personal_history(user_id, persona)
    assert len(history) == 1
    expected_msg = {'author_role': 'user', 'author_name': 'TestUser', 'content': 'Hello'}
    assert history[0] == expected_msg


def test_suppress_message_by_platform_id(mem_manager):
    """Test that a message can be suppressed and is excluded from history."""
    user_id, persona = "user_suppress", "persona_suppress"

    # Log three messages with unique timestamps
    mem_manager.log_message(user_id, persona, "chan", "user", "Human", "Message 1", datetime.now(), "p1")
    time.sleep(0.01)
    mem_manager.log_message(user_id, persona, "chan", "assistant", "Bot", "Message 2", datetime.now(), "p2")
    time.sleep(0.01)
    mem_manager.log_message(user_id, persona, "chan", "user", "Human", "Message 3", datetime.now(), "p3")

    history_before = mem_manager.get_personal_history(user_id, persona)
    assert len(history_before) == 3

    success = mem_manager.suppress_message_by_platform_id("p2")
    assert success is True

    history_after = mem_manager.get_personal_history(user_id, persona)
    assert len(history_after) == 2
    contents = [msg['content'] for msg in history_after]
    assert "Message 2" not in contents


def test_double_suppression_is_handled_gracefully(mem_manager):
    """Test that attempting to suppress the same message twice fails gracefully."""
    mem_manager.log_message("user", "p", "chan", "user", "Human", "Msg", datetime.now(), "p_double")

    # First suppression should succeed
    success1 = mem_manager.suppress_message_by_platform_id("p_double")
    assert success1 is True

    # Second suppression should fail due to UNIQUE constraint, but not crash
    success2 = mem_manager.suppress_message_by_platform_id("p_double")
    assert success2 is False

    # Verify there is still only one entry in the suppression table
    with mem_manager._get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM Suppressed_Interactions")
        count = cursor.fetchone()[0]
        assert count == 1


def test_get_channel_history(mem_manager):
    """Test retrieving messages from a specific channel, ignoring other channels."""
    channel_a, channel_b = "channel-a", "channel-b"

    mem_manager.log_message("user1", "p", channel_a, "user", "Human", "Msg A1", datetime.now(), "p_a1")
    time.sleep(0.01)
    mem_manager.log_message("user2", "p", channel_b, "user", "Human", "Msg B1", datetime.now(), "p_b1")
    time.sleep(0.01)
    mem_manager.log_message("user2", "p", channel_a, "user", "Human", "Msg A2", datetime.now(), "p_a2")

    history_a = mem_manager.get_channel_history(channel_a)
    assert len(history_a) == 2
    contents_a = [msg['content'] for msg in history_a]
    assert contents_a == ["Msg A1", "Msg A2"]

    history_b = mem_manager.get_channel_history(channel_b)
    assert len(history_b) == 1
    assert history_b[0]['content'] == "Msg B1"


def test_channel_history_limit_and_suppression(mem_manager):
    """Test that get_channel_history respects both limits and suppressions."""
    channel = "test-channel"
    for i in range(5):
        mem_manager.log_message("user", "p", channel, "user", "Human", f"Msg {i}", datetime.now(), f"p_{i}")
        time.sleep(0.01)

    mem_manager.suppress_message_by_platform_id("p_2")

    # Ask for 3 messages. Should get Msg 4, Msg 3, Msg 1 (skipping p_2)
    history = mem_manager.get_channel_history(channel, limit=3)
    assert len(history) == 3
    contents = [msg['content'] for msg in history]
    assert contents == ["Msg 1", "Msg 3", "Msg 4"]
