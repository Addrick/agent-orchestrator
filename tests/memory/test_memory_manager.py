# tests/memory/test_memory_manager.py

import concurrent.futures
import json
import pytest
import os
import time
from datetime import datetime, timezone

from memory.memory_manager import MemoryManager
from config.global_config import TEST_DATABASE_DIR, EMBEDDING_MODEL


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
        'zammad_ticket_id', 'platform_message_id', 'server_id', 'tool_context',
        'parent_summary_id', 'reply_to_id', 'reasoning_content'
    }
    assert columns == expected_columns

    # Check Interaction_Edit_History table
    cursor.execute("PRAGMA table_info(Interaction_Edit_History)")
    columns = {row['name'] for row in cursor.fetchall()}
    expected_columns = {
        'edit_id', 'interaction_id', 'old_content', 'old_reasoning_content', 'edited_at'
    }
    assert columns == expected_columns

    # Check Suppressed_Interactions table
    cursor.execute("PRAGMA table_info(Suppressed_Interactions)")
    suppressed_columns = {row['name'] for row in cursor.fetchall()}
    expected_suppressed = {'suppression_id', 'interaction_id', 'suppressed_at'}
    assert suppressed_columns == expected_suppressed

    # Check Segment_Failures table
    cursor.execute("PRAGMA table_info(Segment_Failures)")
    sf_columns = {row['name'] for row in cursor.fetchall()}
    expected_sf = {
        'failure_id', 'channel', 'server_id', 'persona_name',
        'start_interaction_id', 'end_interaction_id', 'message_count',
        'attempts', 'last_attempt_at', 'error_reason',
    }
    assert sf_columns == expected_sf


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


# --- DP-116b: get_agent_action / get_action_contexts round-trip (real sqlite) ---

def test_get_agent_action_round_trip(mem_manager):
    """SqliteSemanticBackend.get_agent_action returns the row inserted by
    log_agent_action with all fields intact."""
    action_id = mem_manager.log_agent_action(
        agent_name="dispatch", action_type="dispatch",
        trigger_context="ticket:42",
        action_payload='{"ticket_id": 42}',
        outcome="pending",
    )
    row = mem_manager.get_agent_action(action_id)
    assert row is not None
    assert row["id"] == action_id
    assert row["agent_name"] == "dispatch"
    assert row["action_type"] == "dispatch"
    assert row["trigger_context"] == "ticket:42"
    assert row["action_payload"] == '{"ticket_id": 42}'
    assert row["outcome"] == "pending"
    assert row["parent_id"] is None


def test_get_agent_action_missing_returns_none(mem_manager):
    assert mem_manager.get_agent_action(999999) is None


def test_get_action_contexts_round_trip(mem_manager):
    """add_action_contexts rows come back via get_action_contexts as (type, value)
    tuples ordered deterministically."""
    action_id = mem_manager.log_agent_action(
        agent_name="dispatch", action_type="dispatch", outcome="pending",
    )
    mem_manager.add_action_contexts(action_id, [
        ("ticket_id", "42"),
        ("priority", "high"),
        ("channel", "zammad"),
    ])
    ctxs = mem_manager.get_action_contexts(action_id)
    assert set(ctxs) == {
        ("ticket_id", "42"), ("priority", "high"), ("channel", "zammad"),
    }
    # Ordering: SQL is ORDER BY context_type, context_value
    assert ctxs == sorted(ctxs)


def test_get_action_contexts_empty(mem_manager):
    action_id = mem_manager.log_agent_action(
        agent_name="dispatch", action_type="dispatch", outcome="pending",
    )
    assert mem_manager.get_action_contexts(action_id) == []


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


# --- Tool Context Migration Tests ---

def test_migration_adds_tool_context_column(legacy_mem_manager):
    """create_schema() on a legacy DB adds the tool_context column via ALTER TABLE."""
    legacy_mem_manager.create_schema()

    conn = legacy_mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(User_Interactions)")
    columns = {row['name'] for row in cursor.fetchall()}
    assert 'tool_context' in columns


def test_migration_preserves_existing_data_with_tool_context(legacy_mem_manager):
    """Existing User_Interactions rows survive the tool_context migration with NULL tool_context."""
    # Insert a row before migration
    conn = legacy_mem_manager._get_connection()
    conn.execute(
        "INSERT INTO User_Interactions (user_identifier, persona_name, channel, author_role, content, timestamp)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        ("user1", "persona1", "chan", "user", "pre-migration msg", datetime.now().isoformat())
    )
    conn.commit()

    legacy_mem_manager.create_schema()

    history = legacy_mem_manager.get_personal_history("user1", "persona1")
    assert len(history) == 1
    assert history[0]['content'] == "pre-migration msg"
    assert history[0]['tool_context'] is None


def test_migration_is_idempotent_with_tool_context(legacy_mem_manager):
    """Running create_schema() twice doesn't error on tool_context column."""
    legacy_mem_manager.create_schema()
    legacy_mem_manager.create_schema()  # second call should be a no-op

    conn = legacy_mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(User_Interactions)")
    columns = [row['name'] for row in cursor.fetchall()]
    assert columns.count('tool_context') == 1


def test_migration_adds_reply_to_id_column(legacy_mem_manager):
    """create_schema() on a legacy DB adds the reply_to_id column via ALTER TABLE."""
    legacy_mem_manager.create_schema()

    conn = legacy_mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(User_Interactions)")
    columns = {row['name'] for row in cursor.fetchall()}
    assert 'reply_to_id' in columns


def test_migration_preserves_existing_data_with_reply_to_id(legacy_mem_manager):
    """Existing User_Interactions rows survive the reply_to_id migration with NULL."""
    conn = legacy_mem_manager._get_connection()
    conn.execute(
        "INSERT INTO User_Interactions (user_identifier, persona_name, channel, author_role, content, timestamp)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        ("user1", "persona1", "chan", "user", "pre-migration msg", datetime.now().isoformat())
    )
    conn.commit()

    legacy_mem_manager.create_schema()

    history = legacy_mem_manager.get_personal_history("user1", "persona1")
    assert len(history) == 1
    assert history[0]['content'] == "pre-migration msg"


def test_reply_to_id_links_assistant_to_user(mem_manager):
    """Assistant message reply_to_id correctly references the user message."""
    user_id = mem_manager.log_message(
        user_identifier="u1", persona_name="p1", channel="ch",
        author_role="user", author_name="user1", content="question",
        timestamp=datetime.now(),
    )
    assistant_id = mem_manager.log_message(
        user_identifier="u1", persona_name="p1", channel="ch",
        author_role="assistant", author_name="p1", content="answer",
        timestamp=datetime.now(), reply_to_id=user_id,
    )

    conn = mem_manager._get_connection()
    row = conn.execute(
        "SELECT reply_to_id FROM User_Interactions WHERE interaction_id = ?",
        (assistant_id,)
    ).fetchone()
    assert row['reply_to_id'] == user_id

    # User msg has no reply_to_id
    user_row = conn.execute(
        "SELECT reply_to_id FROM User_Interactions WHERE interaction_id = ?",
        (user_id,)
    ).fetchone()
    assert user_row['reply_to_id'] is None


# --- Tool Context Functional Tests ---

def test_log_message_with_tool_context(mem_manager):
    """tool_context JSON round-trips correctly through log and retrieval."""
    tool_ctx = json.dumps([
        {"role": "assistant", "tool_calls": [{"id": "c1", "name": "web_search", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "web_search", "content": '{"result": "ok"}'}
    ])
    mem_manager.log_message("user1", "persona1", "chan", "assistant", "Bot",
                            "Here are results", datetime.now(), tool_context=tool_ctx)

    history = mem_manager.get_personal_history("user1", "persona1")
    assert len(history) == 1
    assert history[0]['tool_context'] == tool_ctx
    parsed = json.loads(history[0]['tool_context'])
    assert len(parsed) == 2
    assert parsed[0]['role'] == 'assistant'


def test_log_message_returns_interaction_id(mem_manager):
    """log_message returns the lastrowid (interaction_id)."""
    row_id = mem_manager.log_message("user1", "persona1", "chan", "user", "Human",
                                     "Hello", datetime.now())
    assert isinstance(row_id, int)
    assert row_id > 0

    row_id2 = mem_manager.log_message("user1", "persona1", "chan", "assistant", "Bot",
                                      "Hi", datetime.now())
    assert row_id2 == row_id + 1


def test_update_platform_message_id(mem_manager):
    """update_platform_message_id patches the row correctly."""
    row_id = mem_manager.log_message("user1", "persona1", "chan", "assistant", "Bot",
                                     "Reply", datetime.now())

    mem_manager.update_platform_message_id(row_id, "discord_msg_123")

    conn = mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT platform_message_id FROM User_Interactions WHERE interaction_id = ?", (row_id,))
    assert cursor.fetchone()['platform_message_id'] == "discord_msg_123"


def test_get_history_includes_tool_context(mem_manager):
    """All get_*_history methods return tool_context in their dicts."""
    tool_ctx = json.dumps([{"role": "tool", "content": "data"}])
    mem_manager.log_message("user1", "persona1", "chan", "assistant", "Bot",
                            "Reply", datetime.now(), server_id="srv1", tool_context=tool_ctx)

    for history in [
        mem_manager.get_personal_history("user1", "persona1"),
        mem_manager.get_channel_history("chan", "persona1", server_id="srv1"),
        mem_manager.get_server_history("srv1", "persona1"),
        mem_manager.get_global_history("persona1"),
    ]:
        assert len(history) == 1
        assert history[0]['tool_context'] == tool_ctx

    # ticket history
    mem_manager.log_message("user2", "persona2", "chan", "assistant", "Bot",
                            "Ticket reply", datetime.now(), zammad_ticket_id=42, tool_context=tool_ctx)
    ticket_history = mem_manager.get_ticket_history(42)
    assert len(ticket_history) == 1
    assert ticket_history[0]['tool_context'] == tool_ctx


# --- Long-Term Memory Schema Tests ---

def test_schema_creates_memory_tables(mem_manager):
    """Verify all three memory tables are created with correct columns."""
    conn = mem_manager._get_connection()
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(Message_Embeddings)")
    cols = {row['name'] for row in cursor.fetchall()}
    assert cols == {'interaction_id', 'embedding', 'model_name', 'created_at'}

    cursor.execute("PRAGMA table_info(Memory_Segments)")
    cols = {row['name'] for row in cursor.fetchall()}
    assert cols == {'segment_id', 'channel', 'server_id', 'persona_name',
                    'start_interaction_id', 'end_interaction_id',
                    'message_count', 'created_at',
                    'first_message_at', 'last_message_at'}

    cursor.execute("PRAGMA table_info(Memory_Summaries)")
    cols = {row['name'] for row in cursor.fetchall()}
    assert cols == {'summary_id', 'segment_id', 'content', 'embedding',
                    'model_name', 'created_at', 'summary_level', 'parent_summary_id',
                    'untrusted'}


def test_schema_creates_memory_indexes(mem_manager):
    """Verify indexes are created for memory tables."""
    conn = mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indexes = {row['name'] for row in cursor.fetchall()}
    assert 'idx_segment_channel_persona' in indexes
    assert 'idx_summary_segment' in indexes


# --- Message Embedding Tests ---

def _make_fake_embedding(dim=None) -> bytes:
    """Create a fake embedding BLOB for testing."""
    import struct
    import math
    from config.global_config import EMBEDDING_DIMENSION
    if dim is None:
        dim = EMBEDDING_DIMENSION
    # Create a normalized vector
    values = [1.0 / math.sqrt(dim)] * dim
    return struct.pack(f'{dim}f', *values)


def test_store_and_retrieve_message_embedding(mem_manager):
    """Embedding round-trips correctly through store and is queryable."""
    iid = mem_manager.log_message("u1", "p1", "chan", "user", "Alice",
                                   "Hello world", datetime.now())
    emb = _make_fake_embedding()
    mem_manager.store_message_embedding(iid, emb, EMBEDDING_MODEL, datetime.now())

    conn = mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT embedding, model_name FROM Message_Embeddings WHERE interaction_id = ?",
                   (iid,))
    row = cursor.fetchone()
    assert row['embedding'] == emb
    assert row['model_name'] == EMBEDDING_MODEL


def test_store_message_embedding_upsert(mem_manager):
    """INSERT OR REPLACE updates existing embedding."""
    iid = mem_manager.log_message("u1", "p1", "chan", "user", "Alice",
                                   "Hello", datetime.now())
    emb1 = _make_fake_embedding()
    emb2 = b'\x00' * 3072  # different blob
    mem_manager.store_message_embedding(iid, emb1, "model-a", datetime.now())
    mem_manager.store_message_embedding(iid, emb2, "model-b", datetime.now())

    conn = mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT model_name FROM Message_Embeddings WHERE interaction_id = ?", (iid,))
    assert cursor.fetchone()['model_name'] == "model-b"


# --- get_unembedded_messages Tests ---

def test_get_unembedded_messages_basic(mem_manager):
    """Returns messages without embeddings."""
    ts = datetime.now()
    id1 = mem_manager.log_message("u1", "p1", "chan", "user", "Alice", "Hello", ts)
    id2 = mem_manager.log_message("u1", "p1", "chan", "assistant", "Bot", "Hi", ts)

    msgs = mem_manager.get_unembedded_messages("p1", "chan")
    assert len(msgs) == 2
    assert msgs[0]['interaction_id'] == id1

    # Embed one, now only one is returned
    mem_manager.store_message_embedding(id1, _make_fake_embedding(), EMBEDDING_MODEL, ts)
    msgs = mem_manager.get_unembedded_messages("p1", "chan")
    assert len(msgs) == 1
    assert msgs[0]['interaction_id'] == id2


def test_get_unembedded_messages_model_name_filter(mem_manager):
    """With model_name, also returns messages with stale-model embeddings."""
    ts = datetime.now()
    id1 = mem_manager.log_message("u1", "p1", "chan", "user", "Alice", "Hello", ts)
    mem_manager.store_message_embedding(id1, _make_fake_embedding(), EMBEDDING_MODEL, ts)

    # Without model_name filter: appears embedded
    msgs = mem_manager.get_unembedded_messages("p1", "chan")
    assert len(msgs) == 0

    # With model_name filter: stale model, needs re-embedding
    msgs = mem_manager.get_unembedded_messages("p1", "chan", model_name="new-model")
    assert len(msgs) == 1
    assert msgs[0]['interaction_id'] == id1


def test_get_unembedded_messages_excludes_suppressed(mem_manager):
    """Suppressed messages are excluded."""
    ts = datetime.now()
    mem_manager.log_message("u1", "p1", "chan", "user", "Alice", "Hello", ts,
                            platform_message_id="p1")
    mem_manager.suppress_message_by_platform_id("p1")

    msgs = mem_manager.get_unembedded_messages("p1", "chan")
    assert len(msgs) == 0


def test_get_unembedded_messages_excludes_null_content(mem_manager):
    """Messages with NULL or empty content are excluded."""
    ts = datetime.now()
    mem_manager.log_message("u1", "p1", "chan", "user", "Alice", None, ts)
    mem_manager.log_message("u1", "p1", "chan", "user", "Alice", "", ts)
    mem_manager.log_message("u1", "p1", "chan", "user", "Alice", "Real content", ts)

    msgs = mem_manager.get_unembedded_messages("p1", "chan")
    assert len(msgs) == 1
    assert msgs[0]['content'] == "Real content"


def test_get_unembedded_messages_channel_persona_filter(mem_manager):
    """Only returns messages for the specified channel+persona."""
    ts = datetime.now()
    mem_manager.log_message("u1", "p1", "chan-a", "user", "Alice", "In A", ts)
    mem_manager.log_message("u1", "p2", "chan-a", "user", "Alice", "Different persona", ts)
    mem_manager.log_message("u1", "p1", "chan-b", "user", "Alice", "In B", ts)

    msgs = mem_manager.get_unembedded_messages("p1", "chan-a")
    assert len(msgs) == 1
    assert msgs[0]['content'] == "In A"


# --- Segment and Summary Tests ---

def test_store_segment_and_summary(mem_manager):
    """Segment and summary round-trip correctly."""
    ts = datetime.now()
    seg_id = mem_manager.store_segment("chan", None, "p1", 1, 5, 5, ts)
    assert isinstance(seg_id, int)

    emb = _make_fake_embedding()
    sum_id = mem_manager.store_summary(seg_id, "- Fact 1\n- Fact 2", emb, EMBEDDING_MODEL, ts)
    assert isinstance(sum_id, int)

    summaries = mem_manager.get_summaries_for_channel("chan", "p1")
    assert len(summaries) == 1
    assert summaries[0]['content'] == "- Fact 1\n- Fact 2"
    assert summaries[0]['embedding'] == emb
    assert summaries[0]['segment_id'] == seg_id


def test_get_summaries_recency_filter(mem_manager):
    """exclude_after_interaction_id filters segments starting inside the window."""
    ts = datetime.now()
    # Segment A: IDs 1-5 (outside window)
    seg_a = mem_manager.store_segment("chan", None, "p1", 1, 5, 5, ts)
    mem_manager.store_summary(seg_a, "Facts A", _make_fake_embedding(), EMBEDDING_MODEL, ts)

    # Segment B: IDs 6-10 (straddles window at 8)
    seg_b = mem_manager.store_segment("chan", None, "p1", 6, 10, 5, ts)
    mem_manager.store_summary(seg_b, "Facts B", _make_fake_embedding(), EMBEDDING_MODEL, ts)

    # Segment C: IDs 11-15 (fully inside window at 8)
    seg_c = mem_manager.store_segment("chan", None, "p1", 11, 15, 5, ts)
    mem_manager.store_summary(seg_c, "Facts C", _make_fake_embedding(), EMBEDDING_MODEL, ts)

    # Window starts at 8 — exclude segments starting at or after 8
    summaries = mem_manager.get_summaries_for_channel(
        "chan", "p1", exclude_after_interaction_id=8)
    assert len(summaries) == 2
    contents = {s['content'] for s in summaries}
    assert contents == {"Facts A", "Facts B"}


def test_get_summaries_model_name_filter(mem_manager):
    """model_name filter only returns matching summaries."""
    ts = datetime.now()
    seg = mem_manager.store_segment("chan", None, "p1", 1, 5, 5, ts)
    mem_manager.store_summary(seg, "Facts", _make_fake_embedding(), "old-model", ts)

    assert len(mem_manager.get_summaries_for_channel("chan", "p1", model_name="old-model")) == 1
    assert len(mem_manager.get_summaries_for_channel("chan", "p1", model_name="new-model")) == 0


# --- get_active_channels Tests ---

def test_get_active_channels_basic(mem_manager):
    """Returns channels with unembedded messages."""
    ts = datetime.now()
    mem_manager.log_message("u1", "p1", "chan-a", "user", "Alice", "Hello", ts)
    mem_manager.log_message("u1", "p2", "chan-b", "user", "Bob", "Hi", ts, server_id="srv1")

    channels = mem_manager.get_active_channels()
    assert len(channels) == 2
    assert ("chan-a", "p1", None) in channels
    assert ("chan-b", "p2", "srv1") in channels


def test_get_active_channels_includes_unsegmented_embedded(mem_manager):
    """Channels with embedded-but-unsegmented messages are still active."""
    ts = datetime.now()
    iid = mem_manager.log_message("u1", "p1", "chan", "user", "Alice", "Hello", ts)
    mem_manager.store_message_embedding(iid, _make_fake_embedding(), EMBEDDING_MODEL, ts)

    # Embedded but not segmented — still needs work
    assert len(mem_manager.get_active_channels()) == 1

    # Segment covers the message AND parent_summary_id is set — now fully processed
    mem_manager.store_segment("chan", None, "p1", iid, iid, 1, ts)
    conn = mem_manager._get_connection()
    conn.execute("UPDATE User_Interactions SET parent_summary_id = 1 WHERE interaction_id = ?", (iid,))
    conn.commit()
    assert len(mem_manager.get_active_channels()) == 0


def test_get_active_channels_model_name_filter(mem_manager):
    """With model_name, channels with stale-model embeddings are also returned."""
    ts = datetime.now()
    iid = mem_manager.log_message("u1", "p1", "chan", "user", "Alice", "Hello", ts)
    mem_manager.store_message_embedding(iid, _make_fake_embedding(), EMBEDDING_MODEL, ts)
    # Segment it AND mark summarized so no UNION branch matches
    mem_manager.store_segment("chan", None, "p1", iid, iid, 1, ts)
    conn = mem_manager._get_connection()
    conn.execute("UPDATE User_Interactions SET parent_summary_id = 1 WHERE interaction_id = ?", (iid,))
    conn.commit()

    assert len(mem_manager.get_active_channels()) == 0
    assert len(mem_manager.get_active_channels(model_name="new-model")) == 1


def test_get_active_channels_excludes_null_content(mem_manager):
    """Channels with only NULL/empty content messages are not returned."""
    ts = datetime.now()
    mem_manager.log_message("u1", "p1", "chan", "user", "Alice", None, ts)
    mem_manager.log_message("u1", "p1", "chan", "user", "Alice", "", ts)

    assert len(mem_manager.get_active_channels()) == 0


def test_get_active_channels_surfaces_old_unsummarized_below_segment(mem_manager):
    """Regression: embedded messages older than the last segment must still surface the channel.

    Previously q2 only returned channels with messages AFTER the segment high-water mark,
    stranding older embedded-but-unsummarized messages permanently.
    """
    ts = datetime.now()
    # Log 6 messages — IDs will be 1..6
    ids = [
        mem_manager.log_message("u1", "p1", "chan", "user", "Alice", f"msg {i}", ts)
        for i in range(6)
    ]
    # Embed all 6
    for iid in ids:
        mem_manager.store_message_embedding(iid, _make_fake_embedding(), EMBEDDING_MODEL, ts)

    # Segment covers only messages 4-6 (high-water mark = id[5])
    mem_manager.store_segment("chan", None, "p1", ids[3], ids[5], 3, ts)
    # Mark those messages as summarized
    conn = mem_manager._get_connection()
    for iid in ids[3:]:
        conn.execute("UPDATE User_Interactions SET parent_summary_id = 1 WHERE interaction_id = ?", (iid,))
    conn.commit()

    # Messages 1-3 are embedded, parent_summary_id IS NULL, but sit below the segment.
    # Channel MUST still appear as active.
    channels = mem_manager.get_active_channels()
    assert ("chan", "p1", None) in channels


# --- get_last_segment_tail_embeddings Tests ---

def test_get_last_segment_tail_embeddings_basic(mem_manager):
    """Returns tail embeddings from the most recent segment."""
    ts = datetime.now()
    # Create messages and embeddings
    ids = []
    embs = []
    for i in range(5):
        iid = mem_manager.log_message("u1", "p1", "chan", "user", "Alice",
                                       f"Message {i}", ts)
        ids.append(iid)
        emb = _make_fake_embedding()
        embs.append(emb)
        mem_manager.store_message_embedding(iid, emb, EMBEDDING_MODEL, ts)

    # Create a segment covering all messages
    mem_manager.store_segment("chan", None, "p1", ids[0], ids[-1], 5, ts)

    # Get last 3 tail embeddings
    tail = mem_manager.get_last_segment_tail_embeddings("chan", "p1", n=3)
    assert tail is not None
    assert len(tail) == 3
    # Should be in chronological order (last 3 messages)
    assert tail[0] == embs[2]
    assert tail[1] == embs[3]
    assert tail[2] == embs[4]


def test_get_last_segment_tail_embeddings_no_segment(mem_manager):
    """Returns None when no segments exist."""
    result = mem_manager.get_last_segment_tail_embeddings("chan", "p1")
    assert result is None


def test_get_last_segment_tail_embeddings_model_filter(mem_manager):
    """Returns None when previous segment used a different model."""
    ts = datetime.now()
    iid = mem_manager.log_message("u1", "p1", "chan", "user", "Alice", "Hello", ts)
    mem_manager.store_message_embedding(iid, _make_fake_embedding(), "old-model", ts)
    mem_manager.store_segment("chan", None, "p1", iid, iid, 1, ts)

    # Asking for new model — should return None (cold start)
    result = mem_manager.get_last_segment_tail_embeddings("chan", "p1", model_name="new-model")
    assert result is None

    # Asking for matching model — should return embeddings
    result = mem_manager.get_last_segment_tail_embeddings("chan", "p1", model_name="old-model")
    assert result is not None
    assert len(result) == 1


def test_get_last_segment_tail_scoped_to_channel_persona(mem_manager):
    """Tail embeddings are scoped to the correct channel+persona, not just ID range."""
    ts = datetime.now()

    # Messages in chan-a
    id_a = mem_manager.log_message("u1", "p1", "chan-a", "user", "Alice", "In A", ts)
    emb_a = _make_fake_embedding()
    mem_manager.store_message_embedding(id_a, emb_a, EMBEDDING_MODEL, ts)

    # Messages in chan-b (interleaved IDs)
    id_b = mem_manager.log_message("u1", "p1", "chan-b", "user", "Bob", "In B", ts)
    emb_b = _make_fake_embedding()
    mem_manager.store_message_embedding(id_b, emb_b, EMBEDDING_MODEL, ts)

    # Segment for chan-a that spans both IDs
    mem_manager.store_segment("chan-a", None, "p1", id_a, id_b, 1, ts)

    tail = mem_manager.get_last_segment_tail_embeddings("chan-a", "p1")
    assert tail is not None
    assert len(tail) == 1  # Only the chan-a message, not chan-b


# --- Migration Tests for Memory Tables ---

def test_migration_creates_memory_tables(legacy_mem_manager):
    """create_schema() on a legacy DB creates the memory tables."""
    legacy_mem_manager.create_schema()

    conn = legacy_mem_manager._get_connection()
    cursor = conn.cursor()
    for table in ('Message_Embeddings', 'Memory_Segments', 'Memory_Summaries'):
        cursor.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?",
            (table,))
        assert cursor.fetchone()[0] == 1, f"{table} not created"


def test_migration_memory_tables_idempotent(legacy_mem_manager):
    """Running create_schema() twice doesn't error on memory tables."""
    legacy_mem_manager.create_schema()
    legacy_mem_manager.create_schema()

    conn = legacy_mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='Memory_Segments'")
    assert cursor.fetchone()[0] == 1


def test_migration_creates_edit_history_embeddings_table(legacy_mem_manager):
    """create_schema() creates the Edit_History_Embeddings table used by portal version swap."""
    legacy_mem_manager.create_schema()

    conn = legacy_mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='Edit_History_Embeddings'")
    assert cursor.fetchone()[0] == 1

    cursor.execute("PRAGMA table_info(Edit_History_Embeddings)")
    cols = {row['name'] for row in cursor.fetchall()}
    assert {'edit_id', 'embedding', 'model_name', 'created_at'} <= cols


def test_migration_edit_history_embeddings_idempotent(legacy_mem_manager):
    """Running create_schema() twice leaves Edit_History_Embeddings intact."""
    legacy_mem_manager.create_schema()
    legacy_mem_manager.create_schema()

    conn = legacy_mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='Edit_History_Embeddings'")
    assert cursor.fetchone()[0] == 1


def test_migration_edit_history_embeddings_cascade_on_edit_delete(legacy_mem_manager):
    """Edit_History_Embeddings rows cascade-delete when their parent Interaction_Edit_History row is removed."""
    legacy_mem_manager.create_schema()

    conn = legacy_mem_manager._get_connection()
    cursor = conn.cursor()
    # Seed a User_Interactions row, then an archive row, then an embedding row.
    cursor.execute(
        "INSERT INTO User_Interactions (user_identifier, persona_name, channel, author_role, content, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("u1", "p1", "portal", "assistant", "canonical", datetime.now(timezone.utc)),
    )
    iid = cursor.lastrowid
    cursor.execute(
        "INSERT INTO Interaction_Edit_History (interaction_id, old_content, edited_at) VALUES (?, ?, ?)",
        (iid, "old", datetime.now(timezone.utc)),
    )
    edit_id = cursor.lastrowid
    cursor.execute(
        "INSERT INTO Edit_History_Embeddings (edit_id, embedding, model_name, created_at) VALUES (?, ?, ?, ?)",
        (edit_id, b"\x00" * 16, "test-model", datetime.now(timezone.utc)),
    )
    conn.commit()

    cursor.execute("DELETE FROM Interaction_Edit_History WHERE edit_id = ?", (edit_id,))
    conn.commit()

    cursor.execute("SELECT count(*) FROM Edit_History_Embeddings WHERE edit_id = ?", (edit_id,))
    assert cursor.fetchone()[0] == 0


@pytest.fixture
def legacy_mem_manager_pre_segment_timestamps(tmp_path):
    """MemoryManager with Memory_Segments table missing timestamp columns."""
    import sqlite3

    db_path = str(tmp_path / "legacy_seg.db")
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
            server_id TEXT,
            tool_context TEXT
        );

        CREATE TABLE Suppressed_Interactions (
            suppression_id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id INTEGER NOT NULL UNIQUE,
            suppressed_at TIMESTAMP NOT NULL,
            FOREIGN KEY (interaction_id) REFERENCES User_Interactions(interaction_id) ON DELETE CASCADE
        );

        CREATE TABLE Agent_Actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER,
            agent_name TEXT NOT NULL,
            action_type TEXT NOT NULL,
            trigger_context TEXT,
            action_payload TEXT,
            outcome TEXT,
            outcome_payload TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE Message_Embeddings (
            interaction_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL,
            model_name TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            FOREIGN KEY (interaction_id) REFERENCES User_Interactions(interaction_id) ON DELETE CASCADE
        );

        CREATE TABLE Memory_Segments (
            segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel TEXT NOT NULL,
            server_id TEXT,
            persona_name TEXT NOT NULL,
            start_interaction_id INTEGER NOT NULL,
            end_interaction_id INTEGER NOT NULL,
            message_count INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL
        );

        CREATE TABLE Memory_Summaries (
            summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id INTEGER NOT NULL UNIQUE,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            model_name TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            FOREIGN KEY (segment_id) REFERENCES Memory_Segments(segment_id) ON DELETE CASCADE
        );

        INSERT INTO Memory_Segments
            (channel, server_id, persona_name, start_interaction_id,
             end_interaction_id, message_count, created_at)
        VALUES ('test_channel', 'srv1', 'test_persona', 1, 10, 10,
                '2026-04-01 00:00:00');
    """)
    conn.close()

    manager = MemoryManager(db_path=db_path)
    yield manager
    manager.close()


def test_migration_adds_segment_timestamp_columns(
    legacy_mem_manager_pre_segment_timestamps,
):
    """create_schema() adds first_message_at/last_message_at to Memory_Segments."""
    mm = legacy_mem_manager_pre_segment_timestamps
    mm.create_schema()

    conn = mm._get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(Memory_Segments)")
    columns = {row['name'] for row in cursor.fetchall()}
    assert 'first_message_at' in columns
    assert 'last_message_at' in columns


def test_migration_preserves_existing_segments(
    legacy_mem_manager_pre_segment_timestamps,
):
    """Existing segments survive migration with NULL timestamp columns."""
    mm = legacy_mem_manager_pre_segment_timestamps
    mm.create_schema()

    conn = mm._get_connection()
    row = conn.execute(
        "SELECT * FROM Memory_Segments WHERE channel = 'test_channel'"
    ).fetchone()
    assert row is not None
    assert row['message_count'] == 10
    assert row['first_message_at'] is None
    assert row['last_message_at'] is None


def test_migration_segment_timestamps_idempotent(
    legacy_mem_manager_pre_segment_timestamps,
):
    """Running create_schema() twice doesn't error on segment timestamp columns."""
    mm = legacy_mem_manager_pre_segment_timestamps
    mm.create_schema()
    mm.create_schema()

    conn = mm._get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(Memory_Segments)")
    columns = [row['name'] for row in cursor.fetchall()]
    assert columns.count('first_message_at') == 1
    assert columns.count('last_message_at') == 1


# --- Transaction Context Manager Tests ---

def test_transaction_commits_on_success(mem_manager):
    """Transaction commits writes when the block completes without error."""
    now = datetime.now()
    mem_manager.log_message("u1", "p1", "chan", "user", "Alice", "msg", now)

    with mem_manager.transaction() as conn:
        conn.execute(
            """INSERT INTO Message_Embeddings (interaction_id, embedding, model_name, created_at)
               VALUES (?, ?, ?, ?)""",
            (1, b'\x00' * 12, "test-model", now)
        )

    # Verify it persisted
    rows = conn.execute("SELECT * FROM Message_Embeddings WHERE interaction_id = 1").fetchall()
    assert len(rows) == 1


def test_transaction_rolls_back_on_error(mem_manager):
    """Transaction rolls back all writes when an exception occurs."""
    now = datetime.now()
    mem_manager.log_message("u1", "p1", "chan", "user", "Alice", "msg", now)

    with pytest.raises(RuntimeError, match="deliberate"):
        with mem_manager.transaction() as conn:
            conn.execute(
                """INSERT INTO Message_Embeddings (interaction_id, embedding, model_name, created_at)
                   VALUES (?, ?, ?, ?)""",
                (1, b'\x00' * 12, "test-model", now)
            )
            raise RuntimeError("deliberate")

    # Verify nothing was persisted
    conn = mem_manager._get_connection()
    rows = conn.execute("SELECT * FROM Message_Embeddings WHERE interaction_id = 1").fetchall()
    assert len(rows) == 0


# --- Segment Failures Tests ---

def test_record_segment_failure_creates_entry(mem_manager):
    """First failure for a range creates a new record with attempts=1."""
    mem_manager.record_segment_failure(
        channel="chan", server_id=None, persona_name="p1",
        start_id=1, end_id=50, message_count=50,
        error_reason="LLM timeout",
    )
    failures = mem_manager.get_failed_segment_ranges("chan", "p1")
    assert len(failures) == 1
    assert failures[0]['start_interaction_id'] == 1
    assert failures[0]['end_interaction_id'] == 50
    assert failures[0]['attempts'] == 1
    assert failures[0]['error_reason'] == "LLM timeout"


def test_record_segment_failure_increments_attempts(mem_manager):
    """Repeated failure on same range increments attempts counter."""
    for _ in range(3):
        mem_manager.record_segment_failure(
            channel="chan", server_id=None, persona_name="p1",
            start_id=1, end_id=50, message_count=50,
        )
    failures = mem_manager.get_failed_segment_ranges("chan", "p1")
    assert len(failures) == 1
    assert failures[0]['attempts'] == 3


def test_record_segment_failure_different_ranges_separate(mem_manager):
    """Different ranges create separate failure records."""
    mem_manager.record_segment_failure("chan", None, "p1", 1, 50, 50)
    mem_manager.record_segment_failure("chan", None, "p1", 51, 100, 50)
    failures = mem_manager.get_failed_segment_ranges("chan", "p1")
    assert len(failures) == 2


def test_record_segment_failure_with_server_id(mem_manager):
    """Failures with server_id are tracked separately from NULL server_id."""
    mem_manager.record_segment_failure("chan", "srv1", "p1", 1, 50, 50)
    mem_manager.record_segment_failure("chan", None, "p1", 1, 50, 50)

    with_server = mem_manager.get_failed_segment_ranges("chan", "p1", server_id="srv1")
    without_server = mem_manager.get_failed_segment_ranges("chan", "p1", server_id=None)
    assert len(with_server) == 1
    assert len(without_server) == 1


def test_clear_segment_failure_removes_record(mem_manager):
    """Clearing a failure removes it from future queries."""
    mem_manager.record_segment_failure("chan", None, "p1", 1, 50, 50)
    assert len(mem_manager.get_failed_segment_ranges("chan", "p1")) == 1

    mem_manager.clear_segment_failure("chan", "p1", None, 1, 50)
    assert len(mem_manager.get_failed_segment_ranges("chan", "p1")) == 0


def test_get_failed_segment_ranges_respects_cooldown(mem_manager):
    """Failures with attempts < max_attempts that are old enough are not returned."""
    mem_manager.record_segment_failure("chan", None, "p1", 1, 50, 50)

    # With a very short cooldown (0 hours) and attempts < max, should not block
    failures = mem_manager.get_failed_segment_ranges(
        "chan", "p1", max_attempts=3, cooldown_hours=0.0,
    )
    assert len(failures) == 0

    # With default cooldown (24h), recent failure with 1 attempt still blocks
    failures = mem_manager.get_failed_segment_ranges(
        "chan", "p1", max_attempts=3, cooldown_hours=24.0,
    )
    assert len(failures) == 1


def test_get_failed_segment_ranges_max_attempts_always_blocks(mem_manager):
    """Failures at max_attempts block regardless of cooldown."""
    for _ in range(3):
        mem_manager.record_segment_failure("chan", None, "p1", 1, 50, 50)

    # Even with 0 cooldown, max attempts reached => still blocked
    failures = mem_manager.get_failed_segment_ranges(
        "chan", "p1", max_attempts=3, cooldown_hours=0.0,
    )
    assert len(failures) == 1


def test_migration_creates_segment_failures_table(legacy_mem_manager):
    """create_schema() on a legacy DB creates the Segment_Failures table."""
    legacy_mem_manager.create_schema()

    conn = legacy_mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='Segment_Failures'"
    )
    assert cursor.fetchone()[0] == 1


def test_migration_segment_failures_idempotent(legacy_mem_manager):
    """Running create_schema() twice doesn't error on Segment_Failures."""
    legacy_mem_manager.create_schema()
    legacy_mem_manager.create_schema()

    conn = legacy_mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='Segment_Failures'"
    )
    assert cursor.fetchone()[0] == 1


def test_migration_segment_failures_usable_after_migration(legacy_mem_manager):
    """After migration, Segment_Failures methods work on the migrated DB."""
    legacy_mem_manager.create_schema()

    legacy_mem_manager.record_segment_failure("chan", None, "p1", 1, 50, 50, "test error")
    failures = legacy_mem_manager.get_failed_segment_ranges("chan", "p1")
    assert len(failures) == 1
    assert failures[0]['error_reason'] == "test error"

    legacy_mem_manager.clear_segment_failure("chan", "p1", None, 1, 50)
    assert len(legacy_mem_manager.get_failed_segment_ranges("chan", "p1")) == 0


# --- Phase 2.3b: version list / swap / retry-embedding preservation ---

def _seed_portal_assistant(mem_manager, content: str, emb: bytes = None) -> int:
    """Log an assistant row for the portal session used in these tests, optionally with an embedding."""
    iid = mem_manager.log_message(
        "web_ui", "persona_a", "portal", "assistant", "persona_a",
        content, datetime.now(),
    )
    if emb is not None:
        mem_manager.store_message_embedding(iid, emb, EMBEDDING_MODEL, datetime.now())
        conn = mem_manager._get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO vec_Message_Embeddings (interaction_id, embedding) VALUES (?, ?)",
            (iid, emb),
        )
        conn.commit()
    return iid


def test_handle_portal_retry_moves_embedding_to_edit_history(mem_manager):
    """After retry, Message_Embeddings row is moved into Edit_History_Embeddings, not deleted."""
    emb = _make_fake_embedding()
    iid = _seed_portal_assistant(mem_manager, "first attempt", emb)

    returned = mem_manager.handle_portal_retry("persona_a", "web_ui", "portal")
    assert returned == iid

    conn = mem_manager._get_connection()
    cur = conn.cursor()

    # Canonical Message_Embeddings row gone + vec shadow gone.
    cur.execute("SELECT COUNT(*) FROM Message_Embeddings WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()[0] == 0
    cur.execute("SELECT COUNT(*) FROM vec_Message_Embeddings WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()[0] == 0

    # Exactly one archive with the original embedding preserved.
    cur.execute(
        "SELECT e.edit_id, e.old_content, eh.embedding"
        " FROM Interaction_Edit_History e LEFT JOIN Edit_History_Embeddings eh ON e.edit_id = eh.edit_id"
        " WHERE e.interaction_id = ?",
        (iid,),
    )
    rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0]['old_content'] == "first attempt"
    assert rows[0]['embedding'] == emb


def test_handle_portal_retry_noop_when_trailing_turn_is_user(mem_manager):
    """Retry on a trailing user turn must NOT archive the earlier assistant row.

    Conversation ends `assistant -> user` (a user turn with no reply yet). The
    portal's "retry" on that user turn means "generate a reply", not "regenerate
    the previous assistant". handle_portal_retry must return None so the caller
    INSERTs a fresh assistant row after the user turn instead of overwriting the
    earlier assistant turn in place (which would misroute the response and make
    the streamed block vanish on re-sync).
    """
    iid = _seed_portal_assistant(mem_manager, "prior reply", emb=_make_fake_embedding())
    # A newer user turn now terminates the conversation.
    mem_manager.log_message(
        "web_ui", "persona_a", "portal", "user", "Adam",
        "follow-up question", datetime.now(),
    )

    returned = mem_manager.handle_portal_retry("persona_a", "web_ui", "portal")
    assert returned is None

    conn = mem_manager._get_connection()
    cur = conn.cursor()
    # The earlier assistant row was left untouched: no archive, embedding intact.
    cur.execute("SELECT COUNT(*) FROM Interaction_Edit_History WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()[0] == 0
    cur.execute("SELECT COUNT(*) FROM Message_Embeddings WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()[0] == 1


def test_handle_portal_retry_noop_when_trailing_assistant_is_suppressed(mem_manager):
    """Retry must ignore a soft-deleted trailing assistant row.

    Reproduces the bespoke-UI "retry poof": the user deletes an assistant reply
    (which suppresses it but leaves the row newest in the DB), so the user turn
    becomes the trailing *visible* row. Clicking retry on that user turn must
    behave like "generate a reply" — handle_portal_retry returns None so the
    caller INSERTs a fresh *visible* assistant row. Before the suppression
    filter it would archive + overwrite the still-suppressed assistant row,
    landing the regenerated response in an invisible row (it streamed, then
    vanished on re-sync because the transcript projection filters suppressed
    rows).
    """
    # user -> assistant, then the assistant reply is deleted (suppressed).
    mem_manager.log_message(
        "web_ui", "persona_a", "portal", "user", "Adam",
        "list joy's tickets", datetime.now(),
    )
    assistant_iid = _seed_portal_assistant(
        mem_manager, "here are joy's tickets", emb=_make_fake_embedding(),
    )
    assert mem_manager.suppress_interaction(assistant_iid) is True

    # The suppressed assistant is still the newest row in the DB, but the
    # trailing *visible* row is the user turn -> retry is a no-op.
    returned = mem_manager.handle_portal_retry("persona_a", "web_ui", "portal")
    assert returned is None

    conn = mem_manager._get_connection()
    cur = conn.cursor()
    # The suppressed assistant row was left untouched: not archived, embedding intact.
    cur.execute(
        "SELECT COUNT(*) FROM Interaction_Edit_History WHERE interaction_id = ?",
        (assistant_iid,),
    )
    assert cur.fetchone()[0] == 0
    cur.execute(
        "SELECT COUNT(*) FROM Message_Embeddings WHERE interaction_id = ?",
        (assistant_iid,),
    )
    assert cur.fetchone()[0] == 1


def test_handle_portal_retry_without_embedding_is_noop_for_archive_embedding(mem_manager):
    """Retry with no prior embedding still archives content but no Edit_History_Embeddings row is inserted."""
    iid = _seed_portal_assistant(mem_manager, "no-embed content", emb=None)

    mem_manager.handle_portal_retry("persona_a", "web_ui", "portal")

    conn = mem_manager._get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT eh.edit_id FROM Interaction_Edit_History e"
        " LEFT JOIN Edit_History_Embeddings eh ON e.edit_id = eh.edit_id"
        " WHERE e.interaction_id = ?",
        (iid,),
    )
    rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0]['edit_id'] is None


def test_list_interaction_versions_ordering_and_canonical_last(mem_manager):
    """Archives are oldest-first; canonical appears last with edit_id=None."""
    iid = _seed_portal_assistant(mem_manager, "v1", emb=_make_fake_embedding())

    mem_manager.handle_portal_retry("persona_a", "web_ui", "portal")
    mem_manager.update_interaction_content(iid, "v2")
    mem_manager.store_message_embedding(iid, _make_fake_embedding(), EMBEDDING_MODEL, datetime.now())

    mem_manager.handle_portal_retry("persona_a", "web_ui", "portal")
    mem_manager.update_interaction_content(iid, "v3")

    versions = mem_manager.list_interaction_versions(iid)
    assert [v['content'] for v in versions] == ["v1", "v2", "v3"]
    assert versions[-1]['edit_id'] is None
    assert all(v['edit_id'] is not None for v in versions[:-1])


def test_swap_interaction_version_round_trip_preserves_embedding(mem_manager):
    """Select k=0 restores original content + original embedding as canonical."""
    import struct, math
    from config.global_config import EMBEDDING_DIMENSION
    dim = EMBEDDING_DIMENSION
    # Distinguishable embeddings so we can assert the correct one ends up canonical.
    emb_v1 = struct.pack(f'{dim}f', *([1.0 / math.sqrt(dim)] * dim))
    emb_v2 = struct.pack(f'{dim}f', *([-1.0 / math.sqrt(dim)] * dim))

    iid = _seed_portal_assistant(mem_manager, "v1", emb=emb_v1)

    mem_manager.handle_portal_retry("persona_a", "web_ui", "portal")
    mem_manager.update_interaction_content(iid, "v2")
    mem_manager.store_message_embedding(iid, emb_v2, EMBEDDING_MODEL, datetime.now())
    conn = mem_manager._get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO vec_Message_Embeddings (interaction_id, embedding) VALUES (?, ?)",
        (iid, emb_v2),
    )
    conn.commit()

    result = mem_manager.swap_interaction_version(iid, 0)

    assert result['current_content'] == "v1"
    assert result['interaction_id'] == iid
    assert result['total_versions'] == 2  # v2 is now archived, v1 canonical.

    cur = conn.cursor()
    cur.execute("SELECT content FROM User_Interactions WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()['content'] == "v1"

    cur.execute("SELECT embedding FROM Message_Embeddings WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()['embedding'] == emb_v1

    cur.execute("SELECT embedding FROM vec_Message_Embeddings WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()['embedding'] == emb_v1

    # Stable design keeps the target archive row, so both versions live in the
    # archive with their embeddings intact: v2 (the just-archived canonical) and
    # v1 (the kept source archive, now duplicated by canonical).
    cur.execute(
        "SELECT e.old_content, eh.embedding FROM Interaction_Edit_History e"
        " LEFT JOIN Edit_History_Embeddings eh ON e.edit_id = eh.edit_id"
        " WHERE e.interaction_id = ?",
        (iid,),
    )
    archived = {r['old_content']: r['embedding'] for r in cur.fetchall()}
    assert archived == {"v1": emb_v1, "v2": emb_v2}


def test_swap_interaction_version_out_of_bounds_raises_no_mutation(mem_manager):
    """Out-of-bounds k raises IndexError and does not mutate archive count or canonical."""
    emb = _make_fake_embedding()
    iid = _seed_portal_assistant(mem_manager, "only", emb)

    mem_manager.handle_portal_retry("persona_a", "web_ui", "portal")
    mem_manager.update_interaction_content(iid, "canonical-v2")

    conn = mem_manager._get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM Interaction_Edit_History WHERE interaction_id = ?", (iid,))
    archives_before = cur.fetchone()[0]

    with pytest.raises(IndexError):
        mem_manager.swap_interaction_version(iid, 5)
    with pytest.raises(IndexError):
        mem_manager.swap_interaction_version(iid, -1)

    cur.execute("SELECT COUNT(*) FROM Interaction_Edit_History WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()[0] == archives_before
    cur.execute("SELECT content FROM User_Interactions WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()['content'] == "canonical-v2"


def test_swap_interaction_version_unknown_id_raises(mem_manager):
    """ValueError raised when the interaction does not exist."""
    with pytest.raises(ValueError):
        mem_manager.swap_interaction_version(99999, 0)


def test_swap_total_versions_stable_across_multiple_swaps(mem_manager):
    """Swapping back and forth keeps total_versions constant (archives + canonical = N)."""
    iid = _seed_portal_assistant(mem_manager, "v1", _make_fake_embedding())

    for content in ("v2", "v3"):
        mem_manager.handle_portal_retry("persona_a", "web_ui", "portal")
        mem_manager.update_interaction_content(iid, content)
        mem_manager.store_message_embedding(iid, _make_fake_embedding(), EMBEDDING_MODEL, datetime.now())

    versions_before = mem_manager.list_interaction_versions(iid)
    total_before = len(versions_before)
    assert total_before == 3

    mem_manager.swap_interaction_version(iid, 0)
    mem_manager.swap_interaction_version(iid, 0)

    versions_after = mem_manager.list_interaction_versions(iid)
    assert len(versions_after) == total_before


def test_chevron_navigation_round_trip_matches_displayed_content(mem_manager):
    """Drive the bespoke-UI VersionChevrons pointer math against the backend and
    assert the displayed version matches at every step.

    Encodes the frontend navigation contract (VersionChevrons.tsx):
      - on load, `cur` (1-based) = position of the canonical-flagged entry;
      - clicking a chevron posts swap with k = target1 - 1 (0-indexed archive
        position pre-swap), refetches versions, and sets `cur = target1`;
      - the rendered content is `versions[cur - 1]`.

    The stable-array design must keep `total` constant, the counter tracking the
    true position, every version reachable, and rendered content == backend
    canonical at each step. This is the regression that the abandoned
    delete-on-promote redesign broke (counter froze at N/N; '>' permanently
    disabled; oldest versions unreachable).
    """
    iid = _seed_portal_assistant(mem_manager, "v1", _make_fake_embedding())
    for content in ("v2", "v3", "v4"):
        mem_manager.handle_portal_retry("persona_a", "web_ui", "portal")
        mem_manager.update_interaction_content(iid, content)
        mem_manager.store_message_embedding(iid, _make_fake_embedding(), EMBEDDING_MODEL, datetime.now())

    def load():
        versions = mem_manager.list_interaction_versions(iid)
        cidx = next((i for i, v in enumerate(versions) if v.get("canonical")), None)
        cur = (cidx + 1) if cidx is not None else len(versions)
        return versions, cur

    versions, cur = load()
    total = len(versions)
    assert total == 4
    assert cur == 4
    assert versions[cur - 1]["content"] == "v4"

    def click(direction):
        nonlocal versions, cur, total
        target1 = cur - 1 if direction == "<" else cur + 1
        result = mem_manager.swap_interaction_version(iid, target1 - 1)
        versions = mem_manager.list_interaction_versions(iid)
        cur = target1
        assert len(versions) == total, "total versions must stay constant"
        rendered = versions[cur - 1]["content"]
        assert rendered == result["current_content"], (
            f"rendered version ({rendered!r}) must match backend canonical "
            f"({result['current_content']!r})"
        )
        return rendered

    # Walk back to the oldest version, then forward, then back again.
    assert click("<") == "v3"   # 3/4
    assert click("<") == "v2"   # 2/4
    assert click("<") == "v1"   # 1/4 — oldest reachable
    assert cur == 1
    assert click(">") == "v2"   # 3/4 forward
    assert click(">") == "v3"
    assert click("<") == "v2"


# --- Phase 2.4: portal edit/delete round-trip ---

def test_suppress_interaction_inserts_row(mem_manager):
    """First call inserts a Suppressed_Interactions row and returns True."""
    iid = mem_manager.log_message("u1", "p1", "chan", "user", "Alice",
                                  "hello", datetime.now())
    assert mem_manager.suppress_interaction(iid) is True

    conn = mem_manager._get_connection()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM Suppressed_Interactions WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()[0] == 1


def test_suppress_interaction_idempotent(mem_manager):
    """Second call returns False and does not duplicate the row."""
    iid = mem_manager.log_message("u1", "p1", "chan", "user", "Alice",
                                  "hello", datetime.now())
    assert mem_manager.suppress_interaction(iid) is True
    assert mem_manager.suppress_interaction(iid) is False

    conn = mem_manager._get_connection()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM Suppressed_Interactions WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()[0] == 1


def test_suppress_interaction_excludes_from_personal_history(mem_manager):
    """A suppressed row no longer appears in get_personal_history."""
    ts = datetime.now()
    iid_a = mem_manager.log_message("u1", "p1", "chan", "user", "Alice", "keep", ts)
    iid_b = mem_manager.log_message("u1", "p1", "chan", "user", "Alice", "drop", ts)

    mem_manager.suppress_interaction(iid_b)
    history = mem_manager.get_personal_history("u1", "p1")

    contents = [row['content'] for row in history]
    assert "keep" in contents
    assert "drop" not in contents
    assert all(row['interaction_id'] != iid_b for row in history)


def test_update_interaction_content_clears_l0_embedding(mem_manager):
    """PATCH-style content update drops Message_Embeddings + vec_* so the next
    embedding batch re-encodes against the new content."""
    emb = _make_fake_embedding()
    iid = _seed_portal_assistant(mem_manager, "v1", emb=emb)

    conn = mem_manager._get_connection()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM Message_Embeddings WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()[0] == 1
    cur.execute("SELECT count(*) FROM vec_Message_Embeddings WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()[0] == 1

    assert mem_manager.update_interaction_content(iid, "v1-edited") is True

    cur.execute("SELECT content FROM User_Interactions WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()['content'] == "v1-edited"
    cur.execute("SELECT count(*) FROM Message_Embeddings WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()[0] == 0
    cur.execute("SELECT count(*) FROM vec_Message_Embeddings WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()[0] == 0


def test_update_interaction_content_rewrites_tool_context_when_passed(mem_manager):
    """Retry (review finding #8): a regenerated turn may use a different set of
    tool calls, so update_interaction_content must rewrite tool_context when it
    is explicitly passed — otherwise the row's stored tools desync from content."""
    iid = mem_manager.log_message(
        "web_ui", "persona_a", "portal", "assistant", "persona_a",
        "first attempt", datetime.now(),
        tool_context='[{"role": "tool", "name": "old_tool"}]',
    )

    assert mem_manager.update_interaction_content(
        iid, "regenerated", tool_context='[{"role": "tool", "name": "new_tool"}]'
    ) is True

    conn = mem_manager._get_connection()
    cur = conn.cursor()
    cur.execute("SELECT content, tool_context FROM User_Interactions WHERE interaction_id = ?", (iid,))
    row = cur.fetchone()
    assert row['content'] == "regenerated"
    assert row['tool_context'] == '[{"role": "tool", "name": "new_tool"}]'


def test_update_interaction_content_preserves_tool_context_when_omitted(mem_manager):
    """Manual-edit flow (no tool change) must NOT clobber tool_context: omitting
    the argument leaves the existing column value intact (sentinel default)."""
    iid = mem_manager.log_message(
        "web_ui", "persona_a", "portal", "assistant", "persona_a",
        "v1", datetime.now(),
        tool_context='[{"role": "tool", "name": "keep_me"}]',
    )

    assert mem_manager.update_interaction_content(iid, "v1-edited") is True

    conn = mem_manager._get_connection()
    cur = conn.cursor()
    cur.execute("SELECT content, tool_context FROM User_Interactions WHERE interaction_id = ?", (iid,))
    row = cur.fetchone()
    assert row['content'] == "v1-edited"
    assert row['tool_context'] == '[{"role": "tool", "name": "keep_me"}]'


def test_update_interaction_content_can_clear_tool_context(mem_manager):
    """A retry that drops all tool calls passes tool_context=None explicitly,
    which writes NULL (distinct from the omitted/sentinel case above)."""
    iid = mem_manager.log_message(
        "web_ui", "persona_a", "portal", "assistant", "persona_a",
        "v1", datetime.now(),
        tool_context='[{"role": "tool", "name": "gone"}]',
    )

    assert mem_manager.update_interaction_content(iid, "v2", tool_context=None) is True

    conn = mem_manager._get_connection()
    cur = conn.cursor()
    cur.execute("SELECT tool_context FROM User_Interactions WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()['tool_context'] is None


def test_swap_dedupe_does_not_grow_archive_count(mem_manager):
    """Toggling between two contents repeatedly keeps Interaction_Edit_History
    bounded at one row per distinct version (2) — the dup-content check skips
    redundant archive inserts so churn never grows the table.

    Stable design keeps the target archive row (so the numbered version list is
    fixed), so the bound is the distinct-version count, not 1. Index 0 is v1 and
    index 1 is v2 by archival order, so a real toggle alternates swap(1)/swap(0).
    """
    iid = _seed_portal_assistant(mem_manager, "v1", _make_fake_embedding())
    mem_manager.handle_portal_retry("persona_a", "web_ui", "portal")
    mem_manager.update_interaction_content(iid, "v2")
    mem_manager.store_message_embedding(iid, _make_fake_embedding(), EMBEDDING_MODEL, datetime.now())

    conn = mem_manager._get_connection()
    cur = conn.cursor()

    def canonical():
        cur.execute("SELECT content FROM User_Interactions WHERE interaction_id = ?", (iid,))
        return cur.fetchone()['content']

    def archive_count():
        cur.execute("SELECT count(*) FROM Interaction_Edit_History WHERE interaction_id = ?", (iid,))
        return cur.fetchone()[0]

    for _ in range(5):
        mem_manager.swap_interaction_version(iid, 0)  # -> v1 canonical
        assert canonical() == "v1"
        assert archive_count() == 2  # both distinct versions retained, no growth
        mem_manager.swap_interaction_version(iid, 1)  # -> v2 canonical
        assert canonical() == "v2"
        assert archive_count() == 2


def test_swap_dedupe_total_versions_stable_across_churn(mem_manager):
    """Across N back-and-forth swaps total_versions stays at 2 (the distinct
    version count = displayed, content-deduped list length)."""
    iid = _seed_portal_assistant(mem_manager, "v1", _make_fake_embedding())
    mem_manager.handle_portal_retry("persona_a", "web_ui", "portal")
    mem_manager.update_interaction_content(iid, "v2")
    mem_manager.store_message_embedding(iid, _make_fake_embedding(), EMBEDDING_MODEL, datetime.now())

    for _ in range(6):
        result = mem_manager.swap_interaction_version(iid, 0)
        assert result['total_versions'] == 2


def test_update_interaction_content_tool_context_sentinel(mem_manager):
    """DP-132 #3: tool_context uses a sentinel so the three intents are distinct.

      - arg omitted (manual text edit) → tool_context left UNTOUCHED;
      - tool_context=None (regen produced no tool calls) → CLEARED;
      - tool_context="..." (regen produced tool calls) → REWRITTEN.

    The None-clears branch is the bug fix: before, None meant "leave", so a regen
    from a tool-call answer to a plain-text answer kept stale tool_context and the
    transcript rendered phantom tool cards.
    """
    tool_ctx = json.dumps([{"role": "assistant", "tool_calls": [{"id": "c1"}]}])
    iid = mem_manager.log_message(
        "web_ui", "persona_a", "portal", "assistant", "persona_a",
        "ran a tool", datetime.now(), tool_context=tool_ctx,
    )

    conn = mem_manager._get_connection()
    cur = conn.cursor()

    def stored_tool_context():
        cur.execute("SELECT tool_context FROM User_Interactions WHERE interaction_id = ?", (iid,))
        return cur.fetchone()['tool_context']

    # 1. Manual edit (arg omitted) leaves tool_context untouched.
    assert mem_manager.update_interaction_content(iid, "edited text") is True
    assert stored_tool_context() == tool_ctx

    # 2. Regen with tool calls rewrites it.
    new_ctx = json.dumps([{"role": "assistant", "tool_calls": [{"id": "c2"}]}])
    assert mem_manager.update_interaction_content(iid, "ran another tool", tool_context=new_ctx) is True
    assert stored_tool_context() == new_ctx

    # 3. Regen to a plain-text answer (tool_context=None) CLEARS it.
    assert mem_manager.update_interaction_content(iid, "just text", tool_context=None) is True
    assert stored_tool_context() is None


def test_update_interaction_content_preserves_reasoning_when_omitted(mem_manager):
    """DP-141 (DATA LOSS regression): a manual edit sends only the new body and
    omits reasoning_content, so the stored `<think>` reasoning MUST survive.

    Before the fix reasoning_content defaulted to None and was written
    unconditionally, so editing an assistant turn permanently erased its
    reasoning from the DB."""
    iid = mem_manager.log_message(
        "web_ui", "persona_a", "portal", "assistant", "persona_a",
        "the answer", datetime.now(),
        reasoning_content="the chain of thought",
    )

    assert mem_manager.update_interaction_content(iid, "the edited answer") is True

    conn = mem_manager._get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT content, reasoning_content FROM User_Interactions WHERE interaction_id = ?",
        (iid,),
    )
    row = cur.fetchone()
    assert row['content'] == "the edited answer"
    assert row['reasoning_content'] == "the chain of thought"


def test_update_interaction_content_reasoning_sentinel(mem_manager):
    """DP-141: reasoning_content follows the same `_UNSET` sentinel contract as
    tool_context — omitted = leave, None = clear, value = set.

      - arg omitted (manual text edit) → reasoning left UNTOUCHED;
      - reasoning_content="..." (regen produced fresh thinking) → REWRITTEN;
      - reasoning_content=None (regen / non-thinking model) → CLEARED.
    """
    iid = mem_manager.log_message(
        "web_ui", "persona_a", "portal", "assistant", "persona_a",
        "v1", datetime.now(), reasoning_content="thought v1",
    )

    conn = mem_manager._get_connection()
    cur = conn.cursor()

    def stored_reasoning():
        cur.execute(
            "SELECT reasoning_content FROM User_Interactions WHERE interaction_id = ?",
            (iid,),
        )
        return cur.fetchone()['reasoning_content']

    # 1. Manual edit (arg omitted) leaves reasoning untouched.
    assert mem_manager.update_interaction_content(iid, "edited body") is True
    assert stored_reasoning() == "thought v1"

    # 2. Regen with fresh thinking rewrites it.
    assert mem_manager.update_interaction_content(
        iid, "v2", reasoning_content="thought v2"
    ) is True
    assert stored_reasoning() == "thought v2"

    # 3. Explicit None clears it (regen by a non-thinking model).
    assert mem_manager.update_interaction_content(iid, "v3", reasoning_content=None) is True
    assert stored_reasoning() is None


def test_handle_portal_retry_dedupes_identical_content(mem_manager):
    """DP-132 #7: archiving identical content twice must not create duplicate
    archive rows, which would flag multiple versions canonical and desync the
    chevron's findIndex(canonical) counter from the rendered version."""
    iid = _seed_portal_assistant(mem_manager, "dup", emb=_make_fake_embedding())

    # First retry archives "dup".
    mem_manager.handle_portal_retry("persona_a", "web_ui", "portal")
    # Regen reproduces identical text, then retry again — the duplicate-content
    # archive insert must be skipped.
    mem_manager.update_interaction_content(iid, "dup")
    mem_manager.handle_portal_retry("persona_a", "web_ui", "portal")

    conn = mem_manager._get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM Interaction_Edit_History"
        " WHERE interaction_id = ? AND old_content = ?",
        (iid, "dup"),
    )
    assert cur.fetchone()[0] == 1  # not 2 — dedupe held

    # Exactly one version is flagged canonical, so the chevron points correctly.
    versions = mem_manager.list_interaction_versions(iid)
    assert sum(1 for v in versions if v.get('canonical')) == 1


def test_get_distinct_channels_groups_by_channel_only(mem_manager):
    """DP-132 #8: one logical channel logged under both a NULL and a non-NULL
    server_id must collapse to a single rail entry with the combined count, not
    render twice with split counts."""
    ts = datetime.now()
    mem_manager.log_message("u1", "persona_a", "general", "user", "Adam", "m1", ts, server_id="123")
    mem_manager.log_message("u2", "persona_a", "general", "user", "Joy", "m2", ts, server_id=None)
    mem_manager.log_message("u1", "persona_a", "general", "user", "Adam", "m3", ts, server_id="123")
    mem_manager.log_message("u1", "persona_a", "random", "user", "Adam", "m4", ts, server_id="123")

    rows = mem_manager.get_distinct_channels("persona_a")
    by_channel = {r['channel']: r for r in rows}

    assert sorted(by_channel) == ["general", "random"]  # 'general' appears once
    assert by_channel["general"]["count"] == 3  # 2 server + 1 NULL, combined
    assert by_channel["random"]["count"] == 1


# --- Phase 5: Memory Taint (untrusted column) Tests ---

def test_store_summary_untrusted_roundtrip(mem_manager):
    """store_summary(untrusted=True) persists and is retrievable."""
    ts = datetime.now()
    seg_id = mem_manager.store_segment("chan", None, "p1", 1, 5, 5, ts)
    emb = _make_fake_embedding()

    sum_id_trusted = mem_manager.store_summary(seg_id, "trusted content", emb, EMBEDDING_MODEL, ts)
    sum_id_untrusted = mem_manager.store_summary(seg_id, "tainted content", emb, EMBEDDING_MODEL, ts, untrusted=True)

    conn = mem_manager._get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT untrusted FROM Memory_Summaries WHERE summary_id = ?", (sum_id_trusted,))
    assert cursor.fetchone()['untrusted'] == 0

    cursor.execute("SELECT untrusted FROM Memory_Summaries WHERE summary_id = ?", (sum_id_untrusted,))
    assert cursor.fetchone()['untrusted'] == 1


def test_store_summary_untrusted_default_false(mem_manager):
    """store_summary() without untrusted parameter defaults to 0 (trusted)."""
    ts = datetime.now()
    seg_id = mem_manager.store_segment("chan", None, "p1", 1, 5, 5, ts)
    sum_id = mem_manager.store_summary(seg_id, "default content", _make_fake_embedding(), EMBEDDING_MODEL, ts)

    conn = mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT untrusted FROM Memory_Summaries WHERE summary_id = ?", (sum_id,))
    assert cursor.fetchone()['untrusted'] == 0


def test_retrieve_relevant_summaries_includes_untrusted(mem_manager):
    """retrieve_relevant_summaries returns dicts with the untrusted key."""
    ts = datetime.now()
    seg_id = mem_manager.store_segment("chan", None, "p1", 1, 5, 5, ts)
    emb = _make_fake_embedding()
    mem_manager.store_summary(seg_id, "trusted fact", emb, EMBEDDING_MODEL, ts)
    mem_manager.store_summary(seg_id, "untrusted fact", emb, EMBEDDING_MODEL, ts, untrusted=True)

    summaries = mem_manager.retrieve_relevant_summaries(
        persona_name="p1", channel="chan", memory_mode="channel",
        model_name=EMBEDDING_MODEL,
    )
    assert len(summaries) == 2
    untrusted_vals = {s['untrusted'] for s in summaries}
    assert untrusted_vals == {0, 1}


@pytest.fixture
def legacy_mem_manager_with_summaries(tmp_path):
    """MemoryManager with Memory_Summaries v2 schema but WITHOUT the untrusted column.

    Exercises the ALTER TABLE migration path for existing DBs that already
    have summary_level but not untrusted.
    """
    import sqlite3 as _sqlite3
    import sqlite_vec as _sqlite_vec

    db_path = str(tmp_path / "legacy_summaries.db")
    conn = _sqlite3.connect(db_path)
    try:
        conn.enable_load_extension(True)
        _sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, _sqlite3.OperationalError):
        pass

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
            server_id TEXT,
            tool_context TEXT,
            parent_summary_id INTEGER,
            reply_to_id INTEGER,
            reasoning_content TEXT
        );

        CREATE TABLE Suppressed_Interactions (
            suppression_id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id INTEGER NOT NULL UNIQUE,
            suppressed_at TIMESTAMP NOT NULL,
            FOREIGN KEY (interaction_id) REFERENCES User_Interactions(interaction_id) ON DELETE CASCADE
        );

        CREATE TABLE Agent_Actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER,
            agent_name TEXT NOT NULL,
            action_type TEXT NOT NULL,
            trigger_context TEXT,
            action_payload TEXT,
            outcome TEXT,
            outcome_payload TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE Agent_Action_Contexts (
            action_id INTEGER NOT NULL,
            context_type TEXT NOT NULL,
            context_value TEXT NOT NULL,
            PRIMARY KEY (action_id, context_type, context_value)
        );

        CREATE TABLE Memory_Segments (
            segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel TEXT NOT NULL,
            server_id TEXT,
            persona_name TEXT NOT NULL,
            start_interaction_id INTEGER NOT NULL,
            end_interaction_id INTEGER NOT NULL,
            message_count INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL,
            first_message_at TIMESTAMP,
            last_message_at TIMESTAMP
        );

        -- v2 schema WITH summary_level but WITHOUT untrusted
        CREATE TABLE Memory_Summaries (
            summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id INTEGER,
            content TEXT NOT NULL,
            embedding BLOB,
            model_name TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            summary_level INTEGER NOT NULL DEFAULT 1,
            parent_summary_id INTEGER,
            FOREIGN KEY (segment_id) REFERENCES Memory_Segments(segment_id) ON DELETE CASCADE,
            FOREIGN KEY (parent_summary_id) REFERENCES Memory_Summaries(summary_id) ON DELETE CASCADE
        );

        INSERT INTO Memory_Segments (channel, server_id, persona_name, start_interaction_id, end_interaction_id, message_count, created_at)
        VALUES ('chan', NULL, 'p1', 1, 5, 5, '2026-01-01T00:00:00');

        INSERT INTO Memory_Summaries (segment_id, content, model_name, created_at, summary_level)
        VALUES (1, 'Pre-migration summary', 'test-model', '2026-01-01T00:00:00', 1);
    """)
    conn.close()

    manager = MemoryManager(db_path=db_path)
    yield manager
    manager.close()


def test_migration_adds_untrusted_column(legacy_mem_manager_with_summaries):
    """create_schema() on a DB with v2 Memory_Summaries adds the untrusted column."""
    legacy_mem_manager_with_summaries.create_schema()

    conn = legacy_mem_manager_with_summaries._get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(Memory_Summaries)")
    columns = {row['name'] for row in cursor.fetchall()}
    assert 'untrusted' in columns


def test_migration_untrusted_preserves_existing_data(legacy_mem_manager_with_summaries):
    """Existing Memory_Summaries rows survive the untrusted migration with default 0."""
    legacy_mem_manager_with_summaries.create_schema()

    conn = legacy_mem_manager_with_summaries._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT content, untrusted FROM Memory_Summaries")
    row = cursor.fetchone()
    assert row['content'] == 'Pre-migration summary'
    assert row['untrusted'] == 0


def test_migration_untrusted_is_idempotent(legacy_mem_manager_with_summaries):
    """Running create_schema() twice doesn't error on untrusted column."""
    legacy_mem_manager_with_summaries.create_schema()
    legacy_mem_manager_with_summaries.create_schema()  # second call should be a no-op

    conn = legacy_mem_manager_with_summaries._get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(Memory_Summaries)")
    columns = [row['name'] for row in cursor.fetchall()]
    assert columns.count('untrusted') == 1


def test_migration_untrusted_new_features_work(legacy_mem_manager_with_summaries):
    """After migration, store_summary with untrusted=True works on migrated DB."""
    legacy_mem_manager_with_summaries.create_schema()

    ts = datetime.now()
    seg_id = legacy_mem_manager_with_summaries.store_segment("chan", None, "p1", 10, 15, 5, ts)
    emb = _make_fake_embedding()
    sum_id = legacy_mem_manager_with_summaries.store_summary(
        seg_id, "tainted content", emb, EMBEDDING_MODEL, ts, untrusted=True
    )

    conn = legacy_mem_manager_with_summaries._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT untrusted FROM Memory_Summaries WHERE summary_id = ?", (sum_id,))
    assert cursor.fetchone()['untrusted'] == 1


# --- Proposal queue (DP-282) ---

def _queue_proposal(manager, action_type="set_priority", args=None, **kwargs):
    return manager.create_proposal(
        agent_name="managr",
        action_type=action_type,
        action_args=args or {"ticket_number": 10001, "priority": "3 high"},
        rationale="stale ticket needs attention",
        taint={"source": "zammad_board_snapshot", "cycle_action_id": 42,
               "ticket_number": 10001},
        source_action_id=42,
        **kwargs,
    )


def test_proposal_create_and_get_roundtrip(mem_manager):
    pid = _queue_proposal(mem_manager)
    row = mem_manager.get_proposal(pid)
    assert row["proposal_id"] == pid
    assert row["status"] == "pending"
    assert row["agent_name"] == "managr"
    # JSON columns decode back to dicts
    assert row["action_args"] == {"ticket_number": 10001, "priority": "3 high"}
    assert row["taint"]["source"] == "zammad_board_snapshot"
    assert row["source_action_id"] == 42
    assert mem_manager.get_proposal(9999) is None


def test_proposal_list_filters_by_status(mem_manager):
    p1 = _queue_proposal(mem_manager)
    p2 = _queue_proposal(mem_manager, action_type="add_note",
                         args={"ticket_number": 10002, "body": "note"})
    mem_manager.review_proposal(p1, "denied", "operator", "not needed")

    pending = mem_manager.list_proposals(status="pending")
    assert [p["proposal_id"] for p in pending] == [p2]
    denied = mem_manager.list_proposals(status="denied")
    assert [p["proposal_id"] for p in denied] == [p1]
    everything = mem_manager.list_proposals(status=None)
    assert len(everything) == 2


def test_proposal_review_only_moves_pending(mem_manager):
    pid = _queue_proposal(mem_manager)
    assert mem_manager.review_proposal(pid, "approved", "operator", "ok") is True
    row = mem_manager.get_proposal(pid)
    assert row["status"] == "approved"
    assert row["reviewer"] == "operator"
    assert row["review_note"] == "ok"
    assert row["reviewed_at"] is not None
    # Double-review is a no-op, not an overwrite
    assert mem_manager.review_proposal(pid, "denied", "operator", "changed mind") is False
    assert mem_manager.get_proposal(pid)["status"] == "approved"
    # Unknown id
    assert mem_manager.review_proposal(9999, "approved", "operator") is False
    with pytest.raises(ValueError):
        mem_manager.review_proposal(pid, "executed", "operator")


def test_proposal_mark_executed_requires_approval(mem_manager):
    pid = _queue_proposal(mem_manager)
    # Not approved yet: no-op
    mem_manager.mark_proposal_executed(pid, True, "done")
    assert mem_manager.get_proposal(pid)["status"] == "pending"

    mem_manager.review_proposal(pid, "approved", "operator")
    mem_manager.mark_proposal_executed(pid, True, "priority set")
    row = mem_manager.get_proposal(pid)
    assert row["status"] == "executed"
    assert row["execution_result"] == "priority set"
    assert row["executed_at"] is not None


def test_proposal_mark_executed_failure_status(mem_manager):
    pid = _queue_proposal(mem_manager)
    mem_manager.review_proposal(pid, "approved", "operator")
    mem_manager.mark_proposal_executed(pid, False, "zammad 500")
    assert mem_manager.get_proposal(pid)["status"] == "execution_failed"


def test_proposal_expiry_sweep(mem_manager):
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    stale = _queue_proposal(mem_manager, expires_at=now - timedelta(days=1))
    fresh = _queue_proposal(mem_manager, expires_at=now + timedelta(days=7))
    no_ttl = _queue_proposal(mem_manager, expires_at=None)

    assert mem_manager.expire_stale_proposals() == 1
    assert mem_manager.get_proposal(stale)["status"] == "expired"
    assert mem_manager.get_proposal(fresh)["status"] == "pending"
    assert mem_manager.get_proposal(no_ttl)["status"] == "pending"
    # Expired proposals can no longer be reviewed
    assert mem_manager.review_proposal(stale, "approved", "operator") is False
    # Sweep is idempotent
    assert mem_manager.expire_stale_proposals() == 0


def test_proposal_list_filters_by_agent_and_status_sequence(mem_manager):
    """agent_name + multi-status filtering happens in SQL, so pending rows
    from other agents can't consume the limit (managr feedback loop)."""
    mine = _queue_proposal(mem_manager)
    mem_manager.review_proposal(mine, "denied", "operator", "no")
    _queue_proposal(mem_manager)  # own, still pending
    other = mem_manager.create_proposal(
        agent_name="other_agent", action_type="add_note",
        action_args={"ticket_number": 1, "body": "x"})
    mem_manager.review_proposal(other, "approved", "operator")

    reviewed = mem_manager.list_proposals(
        status=("approved", "denied", "expired", "executed", "execution_failed"),
        agent_name="managr")
    assert [p["proposal_id"] for p in reviewed] == [mine]
    # Sequence status without agent filter spans agents
    assert len(mem_manager.list_proposals(status=("approved", "denied"))) == 2


def test_proposal_zero_microsecond_timestamps_readable(mem_manager):
    """Regression: binding tz-aware datetimes into TIMESTAMP columns made
    zero-microsecond rows unreadable under PARSE_DECLTYPES (the default
    converter chokes on isoformat offsets). Timestamps are stored as
    second-precision UTC strings instead."""
    from datetime import timedelta
    exact_second = datetime.now(timezone.utc).replace(microsecond=0)
    pid = _queue_proposal(mem_manager, expires_at=exact_second + timedelta(days=7))
    row = mem_manager.get_proposal(pid)  # crashed before the fix
    assert row["status"] == "pending"
    assert row["expires_at"] is not None
    mem_manager.review_proposal(pid, "approved", "operator")
    mem_manager.mark_proposal_executed(pid, True, "done")
    row = mem_manager.get_proposal(pid)
    assert row["reviewed_at"] is not None and row["executed_at"] is not None
    # Sweep comparisons stay format-consistent with stored expires_at
    stale = _queue_proposal(mem_manager, expires_at=exact_second - timedelta(days=1))
    assert mem_manager.expire_stale_proposals() == 1
    assert mem_manager.get_proposal(stale)["status"] == "expired"


# --- Proposals table migration (DP-282) ---

def test_migration_creates_proposals_table(legacy_mem_manager):
    """create_schema() on a legacy DB (no Proposals table) creates it."""
    legacy_mem_manager.create_schema()
    conn = legacy_mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(Proposals)")
    columns = {row['name'] for row in cursor.fetchall()}
    assert {'proposal_id', 'created_at', 'expires_at', 'agent_name', 'action_type',
            'action_args', 'rationale', 'taint', 'source_action_id', 'status',
            'reviewed_at', 'reviewer', 'review_note', 'executed_at',
            'execution_result'} == columns


def test_migration_proposals_indexes_created(legacy_mem_manager):
    legacy_mem_manager.create_schema()
    conn = legacy_mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA index_list(Proposals)")
    names = {row['name'] for row in cursor.fetchall()}
    assert 'idx_proposal_status' in names
    assert 'idx_proposal_acceptance' in names


def test_migration_proposals_usable_and_data_preserved(legacy_mem_manager):
    """Proposal CRUD works on a migrated DB and legacy rows survive."""
    legacy_mem_manager.create_schema()
    pid = _queue_proposal(legacy_mem_manager)
    assert legacy_mem_manager.get_proposal(pid)["status"] == "pending"
    # Pre-existing Agent_Actions rows are untouched
    actions = legacy_mem_manager.get_agent_actions("dispatch")
    assert len(actions) == 2


def test_migration_proposals_idempotent(legacy_mem_manager):
    legacy_mem_manager.create_schema()
    pid = _queue_proposal(legacy_mem_manager)
    legacy_mem_manager.create_schema()  # second run must not drop the table/rows
    assert legacy_mem_manager.get_proposal(pid) is not None

# --- Standing_Orders table migration (DP-281) ---

def test_migration_creates_standing_orders_table(legacy_mem_manager):
    """create_schema() on a legacy DB (no Standing_Orders table) creates it."""
    legacy_mem_manager.create_schema()
    conn = legacy_mem_manager._get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(Standing_Orders)")
    columns = {row['name'] for row in cursor.fetchall()}
    assert {'order_id', 'created_at', 'source', 'agent', 'order_text',
            'status', 'retired_at', 'retire_note'} == columns
    cursor.execute("PRAGMA index_list(Standing_Orders)")
    names = {row['name'] for row in cursor.fetchall()}
    assert 'idx_standing_order_agent_status' in names


def test_migration_standing_orders_usable_and_data_preserved(legacy_mem_manager):
    """Standing-order CRUD works on a migrated DB and legacy rows survive."""
    legacy_mem_manager.create_schema()
    order_id = legacy_mem_manager.add_standing_order(
        "client Y tickets are always low priority", source="operator")
    rows = legacy_mem_manager.list_standing_orders()
    assert [r["order_id"] for r in rows] == [order_id]
    assert legacy_mem_manager.retire_standing_order(order_id, "obsolete") is True
    assert legacy_mem_manager.retire_standing_order(order_id) is False
    # Pre-existing Agent_Actions rows are untouched
    actions = legacy_mem_manager.get_agent_actions("dispatch")
    assert len(actions) == 2


def test_migration_standing_orders_idempotent(legacy_mem_manager):
    legacy_mem_manager.create_schema()
    order_id = legacy_mem_manager.add_standing_order("keep it", source="operator")
    legacy_mem_manager.create_schema()  # second run must not drop the table/rows
    assert legacy_mem_manager.list_standing_orders()[0]["order_id"] == order_id


def test_migration_standing_orders_adds_agent_column():
    """A DB created by the first DP-281 schema (Standing_Orders without the
    agent column, old index name) is migrated in place: column added with
    the 'managr' default on existing rows, index swapped, idempotent."""
    manager = MemoryManager(db_path=":memory:")
    try:
        conn = manager._get_connection()
        conn.execute("""
            CREATE TABLE Standing_Orders (
                order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP NOT NULL,
                source TEXT NOT NULL,
                order_text TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active', 'retired')),
                retired_at TIMESTAMP,
                retire_note TEXT
            )""")
        conn.execute(
            "CREATE INDEX idx_standing_order_status ON Standing_Orders (status, created_at)")
        conn.execute(
            "INSERT INTO Standing_Orders (created_at, source, order_text) "
            "VALUES ('2026-07-07 08:00:00', 'operator', 'pre-migration order')")
        conn.commit()

        manager.create_schema()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(Standing_Orders)")
        assert 'agent' in {row['name'] for row in cursor.fetchall()}
        cursor.execute("PRAGMA index_list(Standing_Orders)")
        names = {row['name'] for row in cursor.fetchall()}
        assert 'idx_standing_order_agent_status' in names
        assert 'idx_standing_order_status' not in names
        rows = manager.list_standing_orders(agent="managr")
        assert [r["order_text"] for r in rows] == ["pre-migration order"]
        assert rows[0]["agent"] == "managr"

        manager.create_schema()  # idempotent
        assert len(manager.list_standing_orders()) == 1
    finally:
        manager.close()


def test_standing_orders_newest_first_and_limit():
    manager = MemoryManager(db_path=":memory:")
    manager.create_schema()
    try:
        ids = [manager.add_standing_order(f"order {i}", source="operator")
               for i in range(3)]
        rows = manager.list_standing_orders(limit=2)
        # Same-second timestamps: order_id DESC tiebreak keeps newest first
        assert [r["order_id"] for r in rows] == [ids[2], ids[1]]
        assert manager.list_standing_orders(status=None, limit=10)[0]["order_id"] == ids[2]
    finally:
        manager.close()


def test_add_standing_order_rejects_unlisted_source():
    """The trust boundary lives in the store: only authenticated operator
    surfaces may write orders, so model output can never become guidance."""
    manager = MemoryManager(db_path=":memory:")
    manager.create_schema()
    try:
        with pytest.raises(ValueError, match="not an authenticated"):
            manager.add_standing_order("smuggled instruction", source="model")
        assert manager.list_standing_orders(status=None) == []
    finally:
        manager.close()


def test_list_standing_orders_rejects_and_caps_limit():
    """SQLite reads LIMIT -1 as unbounded; the store refuses non-positive
    limits and caps oversized ones so the never-pruned table can't be
    dumped wholesale into a tool response."""
    manager = MemoryManager(db_path=":memory:")
    manager.create_schema()
    try:
        manager.add_standing_order("order", source="operator")
        with pytest.raises(ValueError, match="limit"):
            manager.list_standing_orders(limit=-1)
        with pytest.raises(ValueError, match="limit"):
            manager.list_standing_orders(limit=0)
        assert len(manager.list_standing_orders(limit=10 ** 9)) == 1
    finally:
        manager.close()


def test_standing_orders_scoped_by_agent():
    manager = MemoryManager(db_path=":memory:")
    manager.create_schema()
    try:
        managr_id = manager.add_standing_order("managr rule", source="operator")
        other_id = manager.add_standing_order(
            "other planner rule", source="operator", agent="planner2")
        assert [r["order_id"] for r in manager.list_standing_orders(agent="managr")] == [managr_id]
        assert [r["order_id"] for r in manager.list_standing_orders(agent="planner2")] == [other_id]
        # agent=None (the operator's list tool) still sees everything
        assert len(manager.list_standing_orders()) == 2
    finally:
        manager.close()
