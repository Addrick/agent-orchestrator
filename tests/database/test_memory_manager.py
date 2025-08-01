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

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='User_Interactions'")
    assert cursor.fetchone() is not None, "Table 'User_Interactions' was not created."

    cursor.execute("PRAGMA table_info(User_Interactions)")
    columns = {row['name'] for row in cursor.fetchall()}
    expected_columns = {
        'interaction_id', 'user_identifier', 'persona_name', 'channel',
        'role', 'content', 'timestamp', 'zammad_ticket_id'
    }
    assert columns == expected_columns, f"Table columns are incorrect. Expected {expected_columns}, got {columns}."


def test_log_and_get_personal_history(mem_manager):
    """Test logging an interaction and retrieving the two resulting messages."""
    user_id, persona, channel = "user1", "persona1", "chan1"
    user_msg, bot_resp = "Hello bot", "Hello user"

    mem_manager.log_interaction(user_id, persona, channel, user_msg, bot_resp)
    history = mem_manager.get_personal_history(user_id, persona)

    assert len(history) == 2
    assert history[0] == {'role': 'user', 'content': user_msg}
    assert history[1] == {'role': 'assistant', 'content': bot_resp}


def test_personal_history_order(mem_manager):
    """Test that multiple interactions are returned in chronological order."""
    user_id, persona = "user_order", "persona_order"

    mem_manager.log_interaction(user_id, persona, "test", "First message", "First response")
    time.sleep(0.01)
    mem_manager.log_interaction(user_id, persona, "test", "Second message", "Second response")

    history = mem_manager.get_personal_history(user_id, persona)

    assert len(history) == 4
    assert history[0]['content'] == "First message"
    assert history[1]['content'] == "First response"
    assert history[2]['content'] == "Second message"
    assert history[3]['content'] == "Second response"


def test_personal_history_limit(mem_manager):
    """Test that the message history limit is respected."""
    user_id, persona = "user_limit", "persona_limit"

    for i in range(5):  # This will create 10 messages
        mem_manager.log_interaction(user_id, persona, "test", f"Msg {i}", f"Resp {i}")
        time.sleep(0.01)

    full_history = mem_manager.get_personal_history(user_id, persona)
    assert len(full_history) == 10

    limited_history = mem_manager.get_personal_history(user_id, persona, limit=3)
    assert len(limited_history) == 3
    # The 3 latest messages are Resp 4, Msg 4, and Resp 3. Chronologically:
    assert limited_history[0]['content'] == "Resp 3"
    assert limited_history[1]['content'] == "Msg 4"
    assert limited_history[2]['content'] == "Resp 4"

    zero_history = mem_manager.get_personal_history(user_id, persona, limit=0)
    assert len(zero_history) == 0


def test_persona_history_isolation(mem_manager):
    """Test that history for one persona is isolated from another for the same user."""
    user_id = "multi_persona_user"
    persona_a, persona_b = "persona_a", "persona_b"

    mem_manager.log_interaction(user_id, persona_a, "test", "Message for A", "Response from A")
    time.sleep(0.01)
    mem_manager.log_interaction(user_id, persona_b, "test", "Message for B", "Response from B")

    history_a = mem_manager.get_personal_history(user_id, persona_a)
    assert len(history_a) == 2
    assert history_a[0]['content'] == "Message for A"
    assert history_a[1]['content'] == "Response from A"

    history_b = mem_manager.get_personal_history(user_id, persona_b)
    assert len(history_b) == 2
    assert history_b[0]['content'] == "Message for B"
    assert history_b[1]['content'] == "Response from B"


def test_get_ticket_history(mem_manager):
    """Test retrieving all messages for a specific Zammad ticket ID."""
    ticket_id = 12345
    mem_manager.log_interaction("user1", "p1", "c1", "First ticket msg", "First reply", ticket_id)
    time.sleep(0.01)
    mem_manager.log_interaction("user2", "p2", "c2", "Second ticket msg", "Second reply", ticket_id)
    mem_manager.log_interaction("user1", "p1", "c1", "Non-ticket msg", "Reply", 99999)

    ticket_history = mem_manager.get_ticket_history(ticket_id)
    assert len(ticket_history) == 4
    assert ticket_history[0]['content'] == "First ticket msg"
    assert ticket_history[1]['content'] == "First reply"
    assert ticket_history[2]['content'] == "Second ticket msg"
    assert ticket_history[3]['content'] == "Second reply"


def test_ticket_history_limit(mem_manager):
    """Test that the ticket history limit is respected for messages."""
    ticket_id = 54321
    for i in range(5):  # Creates 10 messages
        mem_manager.log_interaction("user", "p", "c", f"Msg {i}", f"Resp {i}", ticket_id)
        time.sleep(0.01)

    limited_history = mem_manager.get_ticket_history(ticket_id, limit=3)
    assert len(limited_history) == 3
    assert limited_history[0]['content'] == "Resp 3"
    assert limited_history[1]['content'] == "Msg 4"
    assert limited_history[2]['content'] == "Resp 4"

    zero_history = mem_manager.get_ticket_history(ticket_id, limit=0)
    assert len(zero_history) == 0

    full_history = mem_manager.get_ticket_history(ticket_id, limit=10)
    assert len(full_history) == 10


def test_zammad_ticket_id_handling(mem_manager):
    """Test that zammad_ticket_id is stored correctly for both messages in an interaction."""
    conn = mem_manager._get_connection()
    mem_manager.log_interaction("user_with_ticket", "p", "c", "msg", "resp", zammad_ticket_id=9001)
    mem_manager.log_interaction("user_no_ticket", "p", "c", "msg", "resp")

    # Verify both messages have the ticket ID
    cursor_with = conn.execute("SELECT zammad_ticket_id FROM User_Interactions WHERE user_identifier = ?", ("user_with_ticket",))
    rows_with = cursor_with.fetchall()
    assert len(rows_with) == 2
    assert rows_with[0]['zammad_ticket_id'] == 9001
    assert rows_with[1]['zammad_ticket_id'] == 9001

    # Verify both messages have NULL for ticket ID
    cursor_no = conn.execute("SELECT zammad_ticket_id FROM User_Interactions WHERE user_identifier = ?", ("user_no_ticket",))
    rows_no = cursor_no.fetchall()
    assert len(rows_no) == 2
    assert rows_no[0]['zammad_ticket_id'] is None
    assert rows_no[1]['zammad_ticket_id'] is None


def test_get_personal_history_for_nonexistent_user(mem_manager):
    """Test that getting history for a user with no interactions returns an empty list."""
    history = mem_manager.get_personal_history("nonexistent_user", "any_persona")
    assert history == []
