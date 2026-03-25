# tests/database/test_memory_manager.py

import concurrent.futures
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

    mem_manager.log_message("user1", persona_name, channel_a, "user", "Human", "Msg A1", datetime.now(), server_id=None)
    time.sleep(0.01)
    mem_manager.log_message("user2", persona_name, channel_b, "user", "Human", "Msg B1", datetime.now(), server_id=None)
    time.sleep(0.01)
    mem_manager.log_message("user2", persona_name, channel_a, "user", "Human", "Msg A2", datetime.now(), server_id=None)

    history_a = mem_manager.get_channel_history(channel_a, persona_name, server_id=None)
    assert len(history_a) == 2
    assert history_a[0]['content'] == "Msg A1"
    assert history_a[1]['content'] == "Msg A2"


def test_channel_history_limit_and_suppression(mem_manager):
    """Test that get_channel_history respects both limits and suppressions."""
    channel = "test-channel"
    persona_name = "p"
    for i in range(5):
        mem_manager.log_message("user", persona_name, channel, "user", "Human", f"Msg {i}", datetime.now(),
                                platform_message_id=f"p_{i}", server_id=None)
        time.sleep(0.01)

    mem_manager.suppress_message_by_platform_id("p_2")

    history = mem_manager.get_channel_history(channel, persona_name, server_id=None, limit=3)
    assert len(history) == 3
    contents = {msg['content'] for msg in history}
    assert contents == {"Msg 1", "Msg 3", "Msg 4"}
    assert "Msg 2" not in contents
    assert "Msg 0" not in contents


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


# --- Schema Migration Tests ---

@pytest.fixture
def legacy_mem_manager(tmp_path):
    """Provides a MemoryManager backed by a DB with the OLD schema (pre-agent-memory).

    Creates the original Agent_Actions table WITHOUT parent_id and
    WITHOUT the Agent_Action_Contexts table, then hands the manager
    back WITHOUT calling create_schema() — the test must call it to
    exercise the migration path.
    """
    import sqlite3

    db_path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE User_Interactions (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_identifier TEXT NOT NULL,
            persona_name TEXT NOT NULL,
            channel TEXT NOT NULL,
            author_role TEXT NOT NULL CHECK(author_role IN ('user', 'assistant', 'system')),
            author_name TEXT,
            content TEXT,
            timestamp TIMESTAMP NOT NULL,
            zammad_ticket_id INTEGER,
            platform_message_id TEXT,
            server_id TEXT
        );

        CREATE TABLE Suppressed_Interactions (
            suppression_id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id INTEGER NOT NULL UNIQUE,
            suppressed_at TIMESTAMP NOT NULL,
            FOREIGN KEY (interaction_id) REFERENCES User_Interactions(interaction_id) ON DELETE CASCADE
        );

        CREATE TABLE Agent_Actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            action_type TEXT NOT NULL,
            trigger_context TEXT,
            action_payload TEXT,
            outcome TEXT,
            outcome_payload TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        INSERT INTO Agent_Actions (agent_name, action_type, trigger_context, outcome)
        VALUES ('dispatch', 'dispatch', 'ticket:42', 'success');
        INSERT INTO Agent_Actions (agent_name, action_type, trigger_context, outcome)
        VALUES ('dispatch', 'dispatch', 'ticket:99', 'failed');
    """)
    conn.close()

    manager = MemoryManager(db_path=db_path)
    yield manager
    manager.close()


def test_migration_adds_parent_id_column(legacy_mem_manager):
    """create_schema() on a legacy DB adds the parent_id column via ALTER TABLE."""
    legacy_mem_manager.create_schema()

    conn = legacy_mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(Agent_Actions)")
    columns = {row['name'] for row in cursor.fetchall()}
    assert 'parent_id' in columns


def test_migration_creates_agent_action_contexts_table(legacy_mem_manager):
    """create_schema() on a legacy DB creates the Agent_Action_Contexts table."""
    legacy_mem_manager.create_schema()

    conn = legacy_mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='Agent_Action_Contexts'"
    )
    assert cursor.fetchone()[0] == 1


def test_migration_preserves_existing_data(legacy_mem_manager):
    """Existing Agent_Actions rows survive the migration unchanged."""
    legacy_mem_manager.create_schema()

    actions = legacy_mem_manager.get_agent_actions(agent_name="dispatch", limit=10)
    assert len(actions) == 2
    triggers = {a['trigger_context'] for a in actions}
    assert triggers == {'ticket:42', 'ticket:99'}


def test_migration_creates_parent_id_index(legacy_mem_manager):
    """The idx_agent_parent index is created after migration adds the column."""
    legacy_mem_manager.create_schema()

    conn = legacy_mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='index' AND name='idx_agent_parent'")
    assert cursor.fetchone()[0] == 1


def test_migration_new_features_work(legacy_mem_manager):
    """After migration, parent_id and Agent_Action_Contexts are fully usable."""
    legacy_mem_manager.create_schema()

    # Insert a parent action
    parent_id = legacy_mem_manager.log_agent_action(
        agent_name="dispatch", action_type="dispatch",
        trigger_context="ticket:200", outcome="pending",
    )

    # Insert a child step with parent_id
    child_id = legacy_mem_manager.log_agent_action(
        agent_name="dispatch", action_type="fetch_ticket",
        outcome="success", parent_id=parent_id,
    )

    # Add contexts
    legacy_mem_manager.add_action_contexts(parent_id, [
        ("ticket", "200"),
        ("customer", "jane"),
    ])

    # Verify child steps
    steps = legacy_mem_manager.get_action_steps(parent_id)
    assert len(steps) == 1
    assert steps[0]['action_type'] == 'fetch_ticket'

    # Verify context-based retrieval
    actions = legacy_mem_manager.get_relevant_agent_actions(
        agent_name="dispatch",
        match_contexts=[("ticket", "200")],
        limit=5,
    )
    assert any(a['trigger_context'] == 'ticket:200' for a in actions)


def test_migration_is_idempotent(legacy_mem_manager):
    """Running create_schema() twice on the same DB doesn't error or duplicate data."""
    legacy_mem_manager.create_schema()
    legacy_mem_manager.create_schema()  # second call should be a no-op

    actions = legacy_mem_manager.get_agent_actions(agent_name="dispatch", limit=10)
    assert len(actions) == 2  # no duplicates


# --- Thread Safety Tests ---

@pytest.fixture
def file_mem_manager(tmp_path):
    """Provides a file-backed MemoryManager (required for cross-thread access)."""
    db_path = str(tmp_path / "thread_test.db")
    manager = MemoryManager(db_path=db_path)
    manager.create_schema()
    yield manager
    manager.close()


def test_concurrent_writes_no_errors(file_mem_manager):
    """Verify that concurrent writes from multiple threads don't raise SQLite errors."""
    num_threads = 8
    writes_per_thread = 50

    def write_batch(thread_id):
        for i in range(writes_per_thread):
            file_mem_manager.log_message(
                user_identifier=f"user_{thread_id}",
                persona_name="persona",
                channel="chan",
                author_role="user",
                author_name=f"Thread-{thread_id}",
                content=f"msg-{thread_id}-{i}",
                timestamp=datetime.now(),
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(write_batch, t) for t in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()  # raises if any thread hit an exception

    history = file_mem_manager.get_global_history("persona")
    assert len(history) == num_threads * writes_per_thread


def test_concurrent_reads_and_writes(file_mem_manager):
    """Verify that simultaneous reads and writes don't corrupt data or raise errors."""
    num_writers = 4
    num_readers = 4
    writes_per_thread = 40
    errors = []

    def writer(thread_id):
        for i in range(writes_per_thread):
            file_mem_manager.log_message(
                user_identifier="shared_user",
                persona_name="persona",
                channel="chan",
                author_role="user",
                author_name=f"Writer-{thread_id}",
                content=f"w-{thread_id}-{i}",
                timestamp=datetime.now(),
            )

    def reader():
        for _ in range(writes_per_thread):
            try:
                history = file_mem_manager.get_personal_history("shared_user", "persona")
                # History should always be a valid list
                assert isinstance(history, list)
            except Exception as e:
                errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_writers + num_readers) as pool:
        futures = []
        for t in range(num_writers):
            futures.append(pool.submit(writer, t))
        for _ in range(num_readers):
            futures.append(pool.submit(reader))
        for f in concurrent.futures.as_completed(futures):
            f.result()

    assert errors == [], f"Reader threads encountered errors: {errors}"
    history = file_mem_manager.get_personal_history("shared_user", "persona")
    assert len(history) == num_writers * writes_per_thread


def test_concurrent_write_and_suppress(file_mem_manager):
    """Verify that suppression under concurrent writes doesn't corrupt state."""
    # Seed messages to suppress
    for i in range(20):
        file_mem_manager.log_message(
            "user", "persona", "chan", "user", "Human",
            f"msg-{i}", datetime.now(), platform_message_id=f"plat-{i}",
        )

    def suppress_batch(start):
        for i in range(start, start + 10):
            file_mem_manager.suppress_message_by_platform_id(f"plat-{i}")

    def write_batch():
        for i in range(20):
            file_mem_manager.log_message(
                "user", "persona", "chan", "user", "Human",
                f"new-{i}", datetime.now(), platform_message_id=f"new-plat-{i}",
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(suppress_batch, 0),
            pool.submit(suppress_batch, 10),
            pool.submit(write_batch),
        ]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    history = file_mem_manager.get_personal_history("user", "persona")
    # 20 original - 20 suppressed + 20 new = 20
    assert len(history) == 20
    # All remaining should be the new batch (none of the originals survive suppression)
    contents = {msg['content'] for msg in history}
    assert all(c.startswith("new-") for c in contents)
