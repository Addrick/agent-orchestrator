# src/database/memory_manager.py

import sqlite3
import logging
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


# --- DATETIME <-> ISO 8601 STRING CONVERSION FOR SQLITE ---
def adapt_datetime_iso(dt_obj: datetime) -> str:
    """Adapt datetime.datetime to timezone-naive ISO 8601 format."""
    return dt_obj.isoformat()


def convert_timestamp_iso(ts_bytes: bytes) -> datetime:
    """Convert ISO 8601 format string from bytes to datetime.datetime object."""
    return datetime.fromisoformat(ts_bytes.decode('utf-8'))


sqlite3.register_adapter(datetime, adapt_datetime_iso)
sqlite3.register_converter("timestamp", convert_timestamp_iso)

# --- PATH LOGIC ---
DB_DIR: Path = Path(__file__).resolve().parent
DATABASE_FILE: Path = DB_DIR / "user_memory.db"


class MemoryManager:
    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initializes the MemoryManager.
        If db_path is None, it falls back to the DATABASE_FILE constant.
        """
        self.db_path: str = db_path if db_path is not None else str(DATABASE_FILE)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock: threading.RLock = threading.RLock()
        if self.db_path != ':memory:':
            DB_DIR.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """
        Returns a single, persistent database connection.
        Creates the connection on the first call.
        """
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                uri=True,
                check_same_thread=False
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for atomic multi-step writes.

        Acquires the lock, yields the connection, and commits on success
        or rolls back on exception.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def close(self) -> None:
        """Explicitly closes the database connection. Important for test cleanup."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info(f"Database connection to '{self.db_path}' closed.")

    def create_schema(self) -> None:
        """Creates the database schema and adds the server_id column if it doesn't exist."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Step 1: Ensure the tables exist first.
            schema_sql = """
            CREATE TABLE IF NOT EXISTS User_Interactions (
                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_identifier TEXT NOT NULL,
                persona_name TEXT NOT NULL,
                channel TEXT NOT NULL,
                author_role TEXT NOT NULL CHECK(author_role IN ('user', 'assistant', 'system')),
                author_name TEXT,
                content TEXT,
                timestamp TIMESTAMP NOT NULL,
                zammad_ticket_id INTEGER,
                platform_message_id TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_channel_timestamp
            ON User_Interactions (channel, timestamp);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_platform_message_id
            ON User_Interactions (platform_message_id);
            CREATE INDEX IF NOT EXISTS idx_zammad_ticket_id
            ON User_Interactions (zammad_ticket_id);
            CREATE INDEX IF NOT EXISTS idx_persona_timestamp
            ON User_Interactions (persona_name, timestamp);
            CREATE INDEX IF NOT EXISTS idx_user_persona
            ON User_Interactions (user_identifier, persona_name);

            CREATE TABLE IF NOT EXISTS Suppressed_Interactions (
                suppression_id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER NOT NULL UNIQUE,
                suppressed_at TIMESTAMP NOT NULL,
                FOREIGN KEY (interaction_id) REFERENCES User_Interactions(interaction_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS Message_Embeddings (
                interaction_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                model_name TEXT NOT NULL DEFAULT 'text-embedding-004',
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (interaction_id) REFERENCES User_Interactions(interaction_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS Memory_Segments (
                segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL,
                server_id TEXT,
                persona_name TEXT NOT NULL,
                start_interaction_id INTEGER NOT NULL,
                end_interaction_id INTEGER NOT NULL,
                message_count INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_segment_channel_persona
            ON Memory_Segments (channel, persona_name, server_id);

            CREATE TABLE IF NOT EXISTS Memory_Summaries (
                summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id INTEGER NOT NULL UNIQUE,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                model_name TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (segment_id) REFERENCES Memory_Segments(segment_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_summary_segment
            ON Memory_Summaries (segment_id);

            CREATE TABLE IF NOT EXISTS Agent_Actions (
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
            CREATE INDEX IF NOT EXISTS idx_agent_name_timestamp
            ON Agent_Actions (agent_name, timestamp);
            CREATE INDEX IF NOT EXISTS idx_agent_action_type
            ON Agent_Actions (agent_name, action_type);

            CREATE TABLE IF NOT EXISTS Agent_Action_Contexts (
                action_id INTEGER NOT NULL,
                context_type TEXT NOT NULL,
                context_value TEXT NOT NULL,
                PRIMARY KEY (action_id, context_type, context_value)
            );
            CREATE INDEX IF NOT EXISTS idx_action_context_lookup
            ON Agent_Action_Contexts (context_type, context_value);
            """
            conn.executescript(schema_sql)

            # Step 2: Now that the table is guaranteed to exist, check for and add the new column.
            cursor.execute("PRAGMA table_info(User_Interactions)")
            columns = [row['name'] for row in cursor.fetchall()]

            if 'server_id' not in columns:
                conn.execute("ALTER TABLE User_Interactions ADD COLUMN server_id TEXT")
                logger.info("Added 'server_id' column to User_Interactions table.")

            if 'tool_context' not in columns:
                conn.execute("ALTER TABLE User_Interactions ADD COLUMN tool_context TEXT")
                logger.info("Added 'tool_context' column to User_Interactions table.")

            # Create any new indexes that might be needed for the new column
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_server_id_timestamp
                ON User_Interactions (server_id, timestamp);
            """)

            # Step 3: Add parent_id column to Agent_Actions if it doesn't exist.
            cursor.execute("PRAGMA table_info(Agent_Actions)")
            agent_columns = [row['name'] for row in cursor.fetchall()]
            if 'parent_id' not in agent_columns:
                conn.execute("ALTER TABLE Agent_Actions ADD COLUMN parent_id INTEGER")
                logger.info("Added 'parent_id' column to Agent_Actions table.")

            # Index on parent_id — must be created AFTER migration adds the column
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_agent_parent
                ON Agent_Actions (parent_id);
            """)

            conn.commit()
            logger.info("User memory database schema created or verified successfully.")

    def log_message(self, user_identifier: str, persona_name: str, channel: str,
                    author_role: str, author_name: Optional[str], content: str,
                    timestamp: datetime, server_id: Optional[str] = None,
                    platform_message_id: Optional[str] = None,
                    zammad_ticket_id: Optional[int] = None,
                    tool_context: Optional[str] = None) -> Optional[int]:
        """Logs a single message with its author's role and name. Returns the interaction_id."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO User_Interactions
                (user_identifier, persona_name, channel, author_role, author_name, content,
                 timestamp, zammad_ticket_id, platform_message_id, server_id, tool_context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_identifier, persona_name, channel, author_role, author_name, content,
                 timestamp, zammad_ticket_id, platform_message_id, server_id, tool_context)
            )
            conn.commit()
            return cursor.lastrowid

    def update_platform_message_id(self, interaction_id: int, platform_message_id: str) -> None:
        """Patches the platform_message_id onto an existing interaction row."""
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                "UPDATE User_Interactions SET platform_message_id = ? WHERE interaction_id = ?",
                (platform_message_id, interaction_id)
            )
            conn.commit()

    def suppress_message_by_platform_id(self, platform_message_id: str) -> bool:
        """Flags a message to be ignored in future context based on its platform ID."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT interaction_id FROM User_Interactions WHERE platform_message_id = ?",
                           (platform_message_id,))
            row = cursor.fetchone()
            if not row:
                return False
            interaction_id = row['interaction_id']
            now = datetime.now()
            try:
                cursor.execute("INSERT INTO Suppressed_Interactions (interaction_id, suppressed_at) VALUES (?, ?)",
                               (interaction_id, now))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    _SUPPRESSION_SUBQUERY = (" AND interaction_id NOT IN"
                             " (SELECT interaction_id FROM Suppressed_Interactions)")

    def get_personal_history(self, user_identifier: str, persona_name: str,
                             limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = ("SELECT interaction_id, author_role, author_name, content, tool_context FROM User_Interactions"
                     " WHERE user_identifier = ? AND persona_name = ?"
                     + self._SUPPRESSION_SUBQUERY)
            params: List[Any] = [user_identifier, persona_name]

            query += " ORDER BY timestamp DESC"
            if isinstance(limit, int):
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows: List[sqlite3.Row] = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]

    def get_ticket_history(self, ticket_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = ("SELECT interaction_id, author_role, author_name, content, tool_context FROM User_Interactions"
                     " WHERE zammad_ticket_id = ?"
                     + self._SUPPRESSION_SUBQUERY)
            params: List[Any] = [ticket_id]

            query += " ORDER BY timestamp DESC"
            if isinstance(limit, int):
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows: List[sqlite3.Row] = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]

    def get_channel_history(self, channel: str, persona_name: str, server_id: Optional[str] = None,
                            limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = ("SELECT interaction_id, author_role, author_name, content, tool_context FROM User_Interactions"
                     " WHERE channel = ? AND persona_name = ?")
            params: List[Any] = [channel, persona_name]

            if server_id:
                query += " AND server_id = ?"
                params.append(server_id)
            else:
                query += " AND server_id IS NULL"

            query += self._SUPPRESSION_SUBQUERY
            query += " ORDER BY timestamp DESC"
            if isinstance(limit, int):
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows: List[sqlite3.Row] = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]

    def get_server_history(self, server_id: str, persona_name: str,
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = ("SELECT interaction_id, author_role, author_name, content, tool_context FROM User_Interactions"
                     " WHERE server_id = ? AND persona_name = ?"
                     + self._SUPPRESSION_SUBQUERY)
            params: List[Any] = [server_id, persona_name]

            query += " ORDER BY timestamp DESC"
            if isinstance(limit, int):
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows: List[sqlite3.Row] = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]

    def get_global_history(self, persona_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = ("SELECT interaction_id, author_role, author_name, content, tool_context FROM User_Interactions"
                     " WHERE persona_name = ?"
                     + self._SUPPRESSION_SUBQUERY)
            params: List[Any] = [persona_name]

            query += " ORDER BY timestamp DESC"
            if isinstance(limit, int):
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows: List[sqlite3.Row] = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]

    # --- Agent Action Methods ---

    def log_agent_action(self, agent_name: str, action_type: str,
                         trigger_context: Optional[str] = None,
                         action_payload: Optional[str] = None,
                         outcome: Optional[str] = None,
                         outcome_payload: Optional[str] = None,
                         parent_id: Optional[int] = None) -> int:
        """Logs an agent action and returns the row id."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO Agent_Actions
                   (parent_id, agent_name, action_type, trigger_context,
                    action_payload, outcome, outcome_payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (parent_id, agent_name, action_type, trigger_context,
                 action_payload, outcome, outcome_payload)
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def update_agent_action_outcome(self, action_id: int, outcome: str,
                                    outcome_payload: Optional[str] = None) -> None:
        """Updates the outcome of a previously logged agent action."""
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                "UPDATE Agent_Actions SET outcome = ?, outcome_payload = ? WHERE id = ?",
                (outcome, outcome_payload, action_id)
            )
            conn.commit()

    def get_agent_actions(self, agent_name: str, limit: int = 20,
                          action_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns recent actions for an agent, newest first."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            query = "SELECT * FROM Agent_Actions WHERE agent_name = ?"
            params: List[Any] = [agent_name]
            if action_type:
                query += " AND action_type = ?"
                params.append(action_type)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def add_action_contexts(self, action_id: int,
                            contexts: List[Tuple[str, str]]) -> None:
        """Associates context tags with an agent action for multi-dimensional retrieval."""
        with self._lock:
            conn = self._get_connection()
            for context_type, context_value in contexts:
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO Agent_Action_Contexts
                           (action_id, context_type, context_value)
                           VALUES (?, ?, ?)""",
                        (action_id, context_type, context_value)
                    )
                except sqlite3.IntegrityError:
                    pass  # Duplicate context, ignore
            conn.commit()

    _FAILURE_OUTCOMES = ('failed', 'error', 'notification_failed')

    def get_relevant_agent_actions(
        self, agent_name: str,
        match_contexts: Optional[List[Tuple[str, str]]] = None,
        match_types: Optional[List[str]] = None,
        limit: int = 15,
    ) -> List[Dict[str, Any]]:
        """Returns recent top-level actions, prioritizing entity matches and failures.

        Results are partitioned into three buckets:
        1. Entity matches — actions whose context tags match any of match_contexts
        2. Recent failures — actions with failure outcomes, not already in bucket 1
        3. Other recent actions
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Overfetch to have enough for partitioning
            fetch_limit = limit * 3

            query = ("SELECT * FROM Agent_Actions"
                     " WHERE agent_name = ? AND parent_id IS NULL")
            params: List[Any] = [agent_name]

            if match_types:
                placeholders = ",".join("?" for _ in match_types)
                query += f" AND action_type IN ({placeholders})"
                params.extend(match_types)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(fetch_limit)
            cursor.execute(query, params)
            all_actions = [dict(row) for row in cursor.fetchall()]

            # Find IDs of actions matching the provided contexts
            matched_ids: set[int] = set()
            if match_contexts:
                for ctx_type, ctx_value in match_contexts:
                    cursor.execute(
                        """SELECT action_id FROM Agent_Action_Contexts
                           WHERE context_type = ? AND context_value = ?""",
                        (ctx_type, ctx_value)
                    )
                    matched_ids.update(row['action_id'] for row in cursor.fetchall())

            # Partition into buckets
            entity_matches = []
            failures = []
            other = []
            for action in all_actions:
                if action['id'] in matched_ids:
                    entity_matches.append(action)
                elif action.get('outcome') in self._FAILURE_OUTCOMES:
                    failures.append(action)
                else:
                    other.append(action)

            merged = entity_matches + failures + other
            return merged[:limit]

    def get_action_steps(self, parent_id: int) -> List[Dict[str, Any]]:
        """Returns child steps for a task, ordered chronologically."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM Agent_Actions WHERE parent_id = ? ORDER BY timestamp ASC",
                (parent_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    # --- Long-Term Memory Methods ---

    def store_message_embedding(self, interaction_id: int, embedding: bytes,
                                model_name: str, created_at: datetime) -> None:
        """Stores or updates an embedding for a message."""
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                """INSERT OR REPLACE INTO Message_Embeddings
                   (interaction_id, embedding, model_name, created_at)
                   VALUES (?, ?, ?, ?)""",
                (interaction_id, embedding, model_name, created_at)
            )
            conn.commit()

    def get_unembedded_messages(self, persona_name: str, channel: str,
                                server_id: Optional[str] = None,
                                limit: int = 200,
                                model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns messages that need embedding (missing or stale model).

        Excludes suppressed interactions and messages with NULL/empty content.
        When model_name is provided, also returns messages whose existing embedding
        was computed by a different model (supports model migration).
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = (
                "SELECT ui.interaction_id, ui.author_role, ui.author_name, ui.content"
                " FROM User_Interactions ui"
                " LEFT JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id"
                " WHERE ui.persona_name = ? AND ui.channel = ?"
            )
            params: List[Any] = [persona_name, channel]

            if server_id is not None:
                query += " AND ui.server_id = ?"
                params.append(server_id)
            else:
                query += " AND ui.server_id IS NULL"

            # Exclude suppressed
            query += self._SUPPRESSION_SUBQUERY.replace(
                "interaction_id", "ui.interaction_id"
            )

            # Exclude non-embeddable
            query += " AND ui.content IS NOT NULL AND ui.content != ''"

            # Missing or stale-model embeddings
            if model_name:
                query += " AND (me.embedding IS NULL OR me.model_name != ?)"
                params.append(model_name)
            else:
                query += " AND me.embedding IS NULL"

            query += " ORDER BY ui.interaction_id ASC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def store_segment(self, channel: str, server_id: Optional[str],
                      persona_name: str, start_id: int, end_id: int,
                      message_count: int, created_at: datetime) -> int:
        """Stores a memory segment. Returns the segment_id."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO Memory_Segments
                   (channel, server_id, persona_name, start_interaction_id,
                    end_interaction_id, message_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (channel, server_id, persona_name, start_id, end_id,
                 message_count, created_at)
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def store_summary(self, segment_id: int, content: str, embedding: bytes,
                      model_name: str, created_at: datetime) -> int:
        """Stores a summary for a segment. Returns the summary_id."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO Memory_Summaries
                   (segment_id, content, embedding, model_name, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (segment_id, content, embedding, model_name, created_at)
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def get_summaries_for_channel(self, channel: str, persona_name: str,
                                  server_id: Optional[str] = None,
                                  exclude_after_interaction_id: Optional[int] = None,
                                  model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns summaries for a specific channel+persona.

        Low-level single-channel query used by retrieve_relevant_summaries
        and diagnostic scripts.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = (
                "SELECT ms.summary_id, ms.segment_id, ms.content, ms.embedding,"
                " ms.model_name, ms.created_at, seg.channel, seg.persona_name,"
                " seg.start_interaction_id, seg.end_interaction_id"
                " FROM Memory_Summaries ms"
                " JOIN Memory_Segments seg ON ms.segment_id = seg.segment_id"
                " WHERE seg.channel = ? AND seg.persona_name = ?"
            )
            params: List[Any] = [channel, persona_name]

            if server_id is not None:
                query += " AND seg.server_id = ?"
                params.append(server_id)
            else:
                query += " AND seg.server_id IS NULL"

            if exclude_after_interaction_id is not None:
                query += " AND seg.start_interaction_id < ?"
                params.append(exclude_after_interaction_id)

            if model_name is not None:
                query += " AND ms.model_name = ?"
                params.append(model_name)

            query += " ORDER BY seg.start_interaction_id ASC"

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_active_channels(self, model_name: Optional[str] = None
                            ) -> List[Tuple[str, str, Optional[str]]]:
        """Returns (channel, persona_name, server_id) tuples with unprocessed messages.

        When model_name is provided, also returns channels where existing embeddings
        were computed by a different model (supports model migration discovery).
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = (
                "SELECT DISTINCT ui.channel, ui.persona_name, ui.server_id"
                " FROM User_Interactions ui"
                " LEFT JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id"
                " WHERE ui.content IS NOT NULL AND ui.content != ''"
            )
            params: List[Any] = []

            # Exclude suppressed
            query += self._SUPPRESSION_SUBQUERY.replace(
                "interaction_id", "ui.interaction_id"
            )

            if model_name:
                query += " AND (me.embedding IS NULL OR me.model_name != ?)"
                params.append(model_name)
            else:
                query += " AND me.embedding IS NULL"

            cursor.execute(query, params)
            return [(row['channel'], row['persona_name'], row['server_id'])
                    for row in cursor.fetchall()]

    def get_last_segment_tail_embeddings(self, channel: str, persona_name: str,
                                         server_id: Optional[str] = None,
                                         n: int = 3,
                                         model_name: Optional[str] = None
                                         ) -> Optional[List[bytes]]:
        """Returns the last N message embeddings from the most recent segment.

        Used to seed the centroid for topic continuity across batch boundaries.
        Returns None if no previous segment or model mismatch.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Find the most recent segment for this channel+persona
            seg_query = (
                "SELECT segment_id, start_interaction_id, end_interaction_id"
                " FROM Memory_Segments"
                " WHERE channel = ? AND persona_name = ?"
            )
            seg_params: List[Any] = [channel, persona_name]

            if server_id is not None:
                seg_query += " AND server_id = ?"
                seg_params.append(server_id)
            else:
                seg_query += " AND server_id IS NULL"

            seg_query += " ORDER BY end_interaction_id DESC LIMIT 1"

            cursor.execute(seg_query, seg_params)
            seg_row = cursor.fetchone()
            if seg_row is None:
                return None

            # Get tail embeddings from this segment, scoped to channel+persona
            emb_query = (
                "SELECT me.embedding FROM Message_Embeddings me"
                " JOIN User_Interactions ui ON me.interaction_id = ui.interaction_id"
                " WHERE ui.channel = ? AND ui.persona_name = ?"
                " AND ui.interaction_id BETWEEN ? AND ?"
            )
            emb_params: List[Any] = [
                channel, persona_name,
                seg_row['start_interaction_id'],
                seg_row['end_interaction_id'],
            ]

            if server_id is not None:
                emb_query += " AND ui.server_id = ?"
                emb_params.append(server_id)
            else:
                emb_query += " AND ui.server_id IS NULL"

            if model_name is not None:
                emb_query += " AND me.model_name = ?"
                emb_params.append(model_name)

            emb_query += " ORDER BY me.interaction_id DESC LIMIT ?"
            emb_params.append(n)

            cursor.execute(emb_query, emb_params)
            rows = cursor.fetchall()

            if not rows:
                return None

            # Return in chronological order (query was DESC)
            return [row['embedding'] for row in reversed(rows)]

    @staticmethod
    def _build_summary_where(
        persona: str,
        memory_mode: str,
        channel: str,
        server_id: Optional[str],
        user_identifier: Optional[str],
        exclude_after_interaction_id: Optional[int],
        model_name: Optional[str],
    ) -> Tuple[str, List[Any]]:
        """Build a WHERE clause for summary retrieval scoped by memory mode."""
        where_parts = ["seg.persona_name = ?"]
        params: List[Any] = [persona]

        if memory_mode == "channel":
            where_parts.append("seg.channel = ?")
            params.append(channel)
            if server_id is not None:
                where_parts.append("seg.server_id = ?")
                params.append(server_id)
            else:
                where_parts.append("seg.server_id IS NULL")
        elif memory_mode == "server":
            if server_id is not None:
                where_parts.append("seg.server_id = ?")
                params.append(server_id)
            else:
                return "1=0", []  # no server_id -> no results
        elif memory_mode == "personal":
            where_parts.append(
                "seg.channel IN ("
                "SELECT DISTINCT channel FROM User_Interactions"
                " WHERE user_identifier = ? AND persona_name = ?)"
            )
            params.extend([user_identifier, persona])
        # global: no additional channel/server filter

        if exclude_after_interaction_id is not None:
            where_parts.append("seg.start_interaction_id < ?")
            params.append(exclude_after_interaction_id)

        if model_name is not None:
            where_parts.append("ms.model_name = ?")
            params.append(model_name)

        return " AND ".join(where_parts), params

    def retrieve_relevant_summaries(
        self,
        persona_name: str,
        channel: str,
        server_id: Optional[str] = None,
        user_identifier: Optional[str] = None,
        memory_mode: str = "channel",
        include_ambient: bool = True,
        exclude_after_interaction_id: Optional[int] = None,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve summaries scoped by MemoryMode for retrieval-time fan-out.

        memory_mode determines which channels' summaries to pull:
          channel  -> WHERE channel = ? AND persona_name = ?
          server   -> WHERE server_id = ? AND persona_name = ?
          personal -> WHERE persona_name = ? AND channel IN (user's channels)
          global   -> WHERE persona_name = ?
          ticket   -> returns empty (no long-term memory for tickets)

        include_ambient adds a UNION with persona_name='ambient' using same scope.
        """
        if memory_mode == "ticket":
            return []

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            base_select = (
                "SELECT ms.summary_id, ms.segment_id, ms.content, ms.embedding,"
                " ms.model_name, ms.created_at, seg.channel, seg.persona_name,"
                " seg.start_interaction_id, seg.end_interaction_id"
                " FROM Memory_Summaries ms"
                " JOIN Memory_Segments seg ON ms.segment_id = seg.segment_id"
            )

            build_args = (memory_mode, channel, server_id, user_identifier,
                          exclude_after_interaction_id, model_name)

            # Build primary query
            where_clause, params = self._build_summary_where(persona_name, *build_args)
            query = f"{base_select} WHERE {where_clause}"

            # Ambient union
            if include_ambient and persona_name != "ambient":
                amb_where, amb_params = self._build_summary_where("ambient", *build_args)
                query += f" UNION {base_select} WHERE {amb_where}"
                params.extend(amb_params)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
