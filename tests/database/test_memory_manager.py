# tests/database/test_memory_manager.py

import concurrent.futures
import json
import pytest
import os
import time
from datetime import datetime

from src.database.memory_manager import MemoryManager
from config.global_config import TEST_MEMORY_DATABASE_FILE, TEST_DATABASE_DIR, EMBEDDING_MODEL


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
        'parent_summary_id', 'reply_to_id'
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
                    'model_name', 'created_at', 'summary_level', 'parent_summary_id'}


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
