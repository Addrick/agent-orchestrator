# src/database/memory_manager.py

import sqlite3
import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
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
            CREATE INDEX IF NOT EXISTS idx_agent_parent
            ON Agent_Actions (parent_id);

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

            conn.commit()
            logger.info("User memory database schema created or verified successfully.")

    def log_message(self, user_identifier: str, persona_name: str, channel: str,
                    author_role: str, author_name: Optional[str], content: str,
                    timestamp: datetime, server_id: Optional[str] = None,
                    platform_message_id: Optional[str] = None,
                    zammad_ticket_id: Optional[int] = None) -> None:
        """Logs a single message with its author's role and name."""
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT INTO User_Interactions
                (user_identifier, persona_name, channel, author_role, author_name, content,
                 timestamp, zammad_ticket_id, platform_message_id, server_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_identifier, persona_name, channel, author_role, author_name, content,
                 timestamp, zammad_ticket_id, platform_message_id, server_id)
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

            query = ("SELECT author_role, author_name, content FROM User_Interactions"
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

            query = ("SELECT author_role, author_name, content FROM User_Interactions"
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

            query = ("SELECT author_role, author_name, content FROM User_Interactions"
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

            query = ("SELECT author_role, author_name, content FROM User_Interactions"
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

            query = ("SELECT author_role, author_name, content FROM User_Interactions"
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
