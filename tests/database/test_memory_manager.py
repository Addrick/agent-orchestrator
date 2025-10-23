# tests/database/test_memory_manager.py

import pytest
import os
import time
from datetime import datetime

from src.database.memory_manager import MemoryManager
from config.global_config import TEST_MEMORY_DATABASE_FILE, TEST_DATABASE_DIR


@pytest.fixture
def mem_manager():
    """Provides a fresh, in-memory MemoryManager instance for each test."""
    # Ensure the directory exists, though not strictly needed for :memory:
    os.makedirs(TEST_DATABASE_DIR, exist_ok=True)

    # Use :memory: for fast, isolated tests
    manager = MemoryManager(db_path=":memory:")
    manager.create_schema()
    yield manager
    manager.close()


# --- Test Cases ---

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
        'zammad_ticket_id', 'platform_message_id', 'server_id'
    }
    assert columns == expected_columns

    # Check Suppressed_Interactions table
    cursor.execute("PRAGMA table_info(Suppressed_Interactions)")
    suppressed_columns = {row['name'] for row in cursor.fetchall()}
    expected_suppressed = {'suppression_id', 'interaction_id', 'suppressed_at'}
    assert suppressed_columns == expected_suppressed


def test_log_and_get_message(mem_manager):
    """Test basic logging and retrieval of a message."""
    user_id, persona = "user1", "persona1"
    timestamp = datetime.now()

    mem_manager.log_message(user_id, persona, "test-channel", "user", "Human", "Hello", timestamp)

    history = mem_manager.get_personal_history(user_id, persona)

    assert len(history) == 1
    assert history[0]['author_role'] == 'user'
    assert history[0]['content'] == 'Hello'


def test_suppress_message_by_platform_id(mem_manager):
    """Test that a message can be suppressed and is excluded from history."""
    user_id, persona = "user_suppress", "persona_suppress"

    mem_manager.log_message(user_id, persona, "chan", "user", "Human", "Message 1", datetime.now(),
                            platform_message_id="p1")
    time.sleep(0.01)
    mem_manager.log_message(user_id, persona, "chan", "assistant", "Bot", "Message 2", datetime.now(),
                            platform_message_id="p2")
    time.sleep(0.01)
    mem_manager.log_message(user_id, persona, "chan", "user", "Human", "Message 3", datetime.now(),
                            platform_message_id="p3")

    history_before = mem_manager.get_personal_history(user_id, persona)
    assert len(history_before) == 3

    success = mem_manager.suppress_message_by_platform_id("p2")
    assert success is True

    history_after = mem_manager.get_personal_history(user_id, persona)
    assert len(history_after) == 2
    assert "Message 2" not in [msg['content'] for msg in history_after]


def test_double_suppression_is_handled_gracefully(mem_manager):
    """Test that attempting to suppress the same message twice fails gracefully."""
    mem_manager.log_message("user", "p", "chan", "user", "Human", "Msg", datetime.now(), platform_message_id="p_double")

    success1 = mem_manager.suppress_message_by_platform_id("p_double")
    assert success1 is True

    success2 = mem_manager.suppress_message_by_platform_id("p_double")
    assert success2 is False


def test_get_channel_history(mem_manager):
    """Test retrieving messages from a specific channel, ignoring other channels."""
    channel_a, channel_b = "channel-a", "channel-b"
    persona_name = "p"

    mem_manager.log_message("user1", persona_name, channel_a, "user", "Human", "Msg A1", datetime.now())
    time.sleep(0.01)
    mem_manager.log_message("user2", persona_name, channel_b, "user", "Human", "Msg B1", datetime.now())
    time.sleep(0.01)
    mem_manager.log_message("user2", persona_name, channel_a, "user", "Human", "Msg A2", datetime.now())

    history_a = mem_manager.get_channel_history(channel_a, persona_name)
    assert len(history_a) == 2
    assert history_a[0]['content'] == "Msg A1"
    assert history_a[1]['content'] == "Msg A2"


def test_channel_history_limit_and_suppression(mem_manager):
    """Test that get_channel_history respects both limits and suppressions."""
    channel = "test-channel"
    persona_name = "p"
    for i in range(5):
        mem_manager.log_message("user", persona_name, channel, "user", "Human", f"Msg {i}", datetime.now(),
                                platform_message_id=f"p_{i}")
        time.sleep(0.01)

    mem_manager.suppress_message_by_platform_id("p_2")

    history = mem_manager.get_channel_history(channel, persona_name, limit=3)
    assert len(history) == 3
    contents = {msg['content'] for msg in history}
    assert contents == {"Msg 1", "Msg 3", "Msg 4"}
    assert "Msg 2" not in contents
    assert "Msg 0" not in contents


# --- New Expanded Coverage Tests ---

def test_get_channel_history_isolates_by_server_id(mem_manager):
    """Tests that get_channel_history separates same-named channels by server_id."""
    channel, p_name = "general", "p"
    mem_manager.log_message("u1", p_name, channel, "user", "u1", "Msg Server 1", datetime.now(), server_id="server1")
    mem_manager.log_message("u2", p_name, channel, "user", "u2", "Msg Server 2", datetime.now(), server_id="server2")

    history = mem_manager.get_channel_history(channel, p_name, server_id="server1")
    assert len(history) == 1
    assert history[0]['content'] == "Msg Server 1"


def test_get_channel_history_handles_non_server_context(mem_manager):
    """Tests that get_channel_history correctly retrieves messages where server_id is NULL."""
    channel, p_name = "dm", "p"
    mem_manager.log_message("u1", p_name, channel, "user", "u1", "DM message", datetime.now(), server_id=None)
    mem_manager.log_message("u2", p_name, channel, "user", "u2", "Server message", datetime.now(), server_id="server1")

    history = mem_manager.get_channel_history(channel, p_name, server_id=None)
    assert len(history) == 1
    assert history[0]['content'] == "DM message"


def test_get_server_history_isolates_by_persona(mem_manager):
    """Tests that get_server_history correctly filters by persona_name within a server."""
    server = "server1"
    mem_manager.log_message("u1", "persona_A", "chan", "user", "u1", "Msg A", datetime.now(), server_id=server)
    mem_manager.log_message("u2", "persona_B", "chan", "user", "u2", "Msg B", datetime.now(), server_id=server)

    history = mem_manager.get_server_history(server, "persona_A")
    assert len(history) == 1
    assert history[0]['content'] == "Msg A"
