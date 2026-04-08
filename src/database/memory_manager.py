# src/database/memory_manager.py

import sqlite3
import logging
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple
from pathlib import Path

# --- NEW: Import the global embedding model variable ---
from config.global_config import EMBEDDING_MODEL

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
        self.db_path: str = db_path if db_path is not None else str(DATABASE_FILE)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock: threading.RLock = threading.RLock()
        if self.db_path != ':memory:':
            DB_DIR.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
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
        with self._lock:
            conn = self._get_connection()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info(f"Database connection to '{self.db_path}' closed.")

    def create_schema(self) -> None:
        with self._lock:
            conn = self._get_connection()

            # Injecting the EMBEDDING_MODEL variable into the default schema
            schema_sql = f"""
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
                platform_message_id TEXT,
                server_id TEXT,
                tool_context TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_channel_timestamp ON User_Interactions (channel, timestamp);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_platform_message_id ON User_Interactions (platform_message_id);
            CREATE INDEX IF NOT EXISTS idx_zammad_ticket_id ON User_Interactions (zammad_ticket_id);
            CREATE INDEX IF NOT EXISTS idx_persona_timestamp ON User_Interactions (persona_name, timestamp);
            CREATE INDEX IF NOT EXISTS idx_user_persona ON User_Interactions (user_identifier, persona_name);
            CREATE INDEX IF NOT EXISTS idx_server_id_timestamp ON User_Interactions (server_id, timestamp);

            CREATE TABLE IF NOT EXISTS Suppressed_Interactions (
                suppression_id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER NOT NULL UNIQUE,
                suppressed_at TIMESTAMP NOT NULL,
                FOREIGN KEY (interaction_id) REFERENCES User_Interactions(interaction_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS Message_Embeddings (
                interaction_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                model_name TEXT NOT NULL DEFAULT '{EMBEDDING_MODEL}',
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
                created_at TIMESTAMP NOT NULL,
                first_message_at TIMESTAMP,
                last_message_at TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_segment_channel_persona ON Memory_Segments (channel, persona_name, server_id);

            CREATE TABLE IF NOT EXISTS Memory_Summaries (
                summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id INTEGER NOT NULL UNIQUE,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                model_name TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (segment_id) REFERENCES Memory_Segments(segment_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_summary_segment ON Memory_Summaries (segment_id);

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
            CREATE INDEX IF NOT EXISTS idx_agent_name_timestamp ON Agent_Actions (agent_name, timestamp);
            CREATE INDEX IF NOT EXISTS idx_agent_action_type ON Agent_Actions (agent_name, action_type);

            CREATE TABLE IF NOT EXISTS Agent_Action_Contexts (
                action_id INTEGER NOT NULL,
                context_type TEXT NOT NULL,
                context_value TEXT NOT NULL,
                PRIMARY KEY (action_id, context_type, context_value)
            );
            CREATE INDEX IF NOT EXISTS idx_action_context_lookup ON Agent_Action_Contexts (context_type, context_value);
            """
            conn.executescript(schema_sql)
            conn.commit()

            cursor = conn.cursor()
            # Agent_Actions migrations
            cursor.execute("PRAGMA table_info(Agent_Actions)")
            agent_actions_cols = {row['name'] for row in cursor.fetchall()}
            if 'parent_id' not in agent_actions_cols:
                conn.execute("ALTER TABLE Agent_Actions ADD COLUMN parent_id INTEGER")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_parent ON Agent_Actions (parent_id)")

            # Memory_Segments migrations
            cursor.execute("PRAGMA table_info(Memory_Segments)")
            memory_segments_cols = {row['name'] for row in cursor.fetchall()}
            if 'first_message_at' not in memory_segments_cols:
                conn.execute("ALTER TABLE Memory_Segments ADD COLUMN first_message_at TIMESTAMP")
            if 'last_message_at' not in memory_segments_cols:
                conn.execute("ALTER TABLE Memory_Segments ADD COLUMN last_message_at TIMESTAMP")

            # User_Interactions migrations
            cursor.execute("PRAGMA table_info(User_Interactions)")
            user_int_cols = {row['name'] for row in cursor.fetchall()}
            if 'tool_context' not in user_int_cols:
                conn.execute("ALTER TABLE User_Interactions ADD COLUMN tool_context TEXT")

            conn.commit()
            logger.info("User memory database schema created or verified successfully.")

    def log_message(self, user_identifier: str, persona_name: str, channel: str,
                    author_role: str, author_name: Optional[str], content: str,
                    timestamp: datetime, server_id: Optional[str] = None,
                    platform_message_id: Optional[str] = None,
                    zammad_ticket_id: Optional[int] = None,
                    tool_context: Optional[str] = None) -> Optional[int]:
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
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                "UPDATE User_Interactions SET platform_message_id = ? WHERE interaction_id = ?",
                (platform_message_id, interaction_id)
            )
            conn.commit()

    def suppress_message_by_platform_id(self, platform_message_id: str) -> bool:
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

    _SUPPRESSION_SUBQUERY = (" AND interaction_id NOT IN (SELECT interaction_id FROM Suppressed_Interactions)")

    @staticmethod
    def _suppression_filter(alias: str = "") -> str:
        prefix = f"{alias}." if alias else ""
        return f" AND {prefix}interaction_id NOT IN (SELECT interaction_id FROM Suppressed_Interactions)"

    def get_personal_history(self, user_identifier: str, persona_name: str, limit: Optional[int] = None) -> List[
        Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            query = ("SELECT interaction_id, author_role, author_name, content, tool_context FROM User_Interactions"
                     " WHERE user_identifier = ? AND persona_name = ?" + self._SUPPRESSION_SUBQUERY)
            params: List[Any] = [user_identifier, persona_name]
            query += " ORDER BY timestamp DESC"
            if isinstance(limit, int):
                query += " LIMIT ?"
                params.append(limit)
            cursor.execute(query, params)
            return [dict(row) for row in reversed(cursor.fetchall())]

    def get_ticket_history(self, ticket_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            query = ("SELECT interaction_id, author_role, author_name, content, tool_context FROM User_Interactions"
                     " WHERE zammad_ticket_id = ?" + self._SUPPRESSION_SUBQUERY)
            params: List[Any] = [ticket_id]
            query += " ORDER BY timestamp DESC"
            if isinstance(limit, int):
                query += " LIMIT ?"
                params.append(limit)
            cursor.execute(query, params)
            return [dict(row) for row in reversed(cursor.fetchall())]

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
            return [dict(row) for row in reversed(cursor.fetchall())]

    def get_server_history(self, server_id: str, persona_name: str, limit: Optional[int] = None) -> List[
        Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            query = ("SELECT interaction_id, author_role, author_name, content, tool_context FROM User_Interactions"
                     " WHERE server_id = ? AND persona_name = ?" + self._SUPPRESSION_SUBQUERY)
            params: List[Any] = [server_id, persona_name]
            query += " ORDER BY timestamp DESC"
            if isinstance(limit, int):
                query += " LIMIT ?"
                params.append(limit)
            cursor.execute(query, params)
            return [dict(row) for row in reversed(cursor.fetchall())]

    def get_global_history(self, persona_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            query = ("SELECT interaction_id, author_role, author_name, content, tool_context FROM User_Interactions"
                     " WHERE persona_name = ?" + self._SUPPRESSION_SUBQUERY)
            params: List[Any] = [persona_name]
            query += " ORDER BY timestamp DESC"
            if isinstance(limit, int):
                query += " LIMIT ?"
                params.append(limit)
            cursor.execute(query, params)
            return [dict(row) for row in reversed(cursor.fetchall())]

    def log_agent_action(self, agent_name: str, action_type: str, trigger_context: Optional[str] = None,
                         action_payload: Optional[str] = None, outcome: Optional[str] = None,
                         outcome_payload: Optional[str] = None, parent_id: Optional[int] = None) -> int:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO Agent_Actions (parent_id, agent_name, action_type, trigger_context, action_payload, outcome, outcome_payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (parent_id, agent_name, action_type, trigger_context, action_payload, outcome, outcome_payload)
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def update_agent_action_outcome(self, action_id: int, outcome: str, outcome_payload: Optional[str] = None) -> None:
        with self._lock:
            conn = self._get_connection()
            conn.execute("UPDATE Agent_Actions SET outcome = ?, outcome_payload = ? WHERE id = ?",
                         (outcome, outcome_payload, action_id))
            conn.commit()

    def get_agent_actions(self, agent_name: str, limit: int = 20, action_type: Optional[str] = None) -> List[
        Dict[str, Any]]:
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

    def add_action_contexts(self, action_id: int, contexts: List[Tuple[str, str]]) -> None:
        with self._lock:
            conn = self._get_connection()
            for context_type, context_value in contexts:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO Agent_Action_Contexts (action_id, context_type, context_value) VALUES (?, ?, ?)",
                        (action_id, context_type, context_value))
                except sqlite3.IntegrityError:
                    pass
            conn.commit()

    _FAILURE_OUTCOMES = ('failed', 'error', 'notification_failed')

    def get_relevant_agent_actions(self, agent_name: str, match_contexts: Optional[List[Tuple[str, str]]] = None,
                                   match_types: Optional[List[str]] = None, limit: int = 15) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            fetch_limit = limit * 3
            query = "SELECT * FROM Agent_Actions WHERE agent_name = ? AND parent_id IS NULL"
            params: List[Any] = [agent_name]

            if match_types:
                placeholders = ",".join("?" for _ in match_types)
                query += f" AND action_type IN ({placeholders})"
                params.extend(match_types)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(fetch_limit)
            cursor.execute(query, params)
            all_actions = [dict(row) for row in cursor.fetchall()]

            matched_ids: set[int] = set()
            if match_contexts:
                for ctx_type, ctx_value in match_contexts:
                    cursor.execute(
                        "SELECT action_id FROM Agent_Action_Contexts WHERE context_type = ? AND context_value = ?",
                        (ctx_type, ctx_value))
                    matched_ids.update(row['action_id'] for row in cursor.fetchall())

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

            return (entity_matches + failures + other)[:limit]

    def get_action_steps(self, parent_id: int) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Agent_Actions WHERE parent_id = ? ORDER BY timestamp ASC", (parent_id,))
            return [dict(row) for row in cursor.fetchall()]

    def store_message_embedding(self, interaction_id: int, embedding: bytes, model_name: str,
                                created_at: datetime) -> None:
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                "INSERT OR REPLACE INTO Message_Embeddings (interaction_id, embedding, model_name, created_at) VALUES (?, ?, ?, ?)",
                (interaction_id, embedding, model_name, created_at))
            conn.commit()

    def get_unembedded_messages(self, persona_name: str, channel: str, server_id: Optional[str] = None,
                                limit: int = 50, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = ("SELECT ui.interaction_id, ui.author_role, ui.author_name, ui.content"
                     " FROM User_Interactions ui LEFT JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id"
                     " WHERE ui.persona_name = ? AND ui.channel = ?")
            params: List[Any] = [persona_name, channel]

            if server_id is not None:
                query += " AND ui.server_id = ?"
                params.append(server_id)
            else:
                query += " AND ui.server_id IS NULL"

            query += self._suppression_filter("ui")
            query += " AND ui.content IS NOT NULL AND ui.content != ''"

            # Use the global model if one isn't explicitly provided
            active_model = model_name if model_name else EMBEDDING_MODEL
            query += " AND (me.embedding IS NULL OR me.model_name != ?)"
            params.append(active_model)

            query += " ORDER BY ui.interaction_id ASC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def store_segment(self, channel: str, server_id: Optional[str], persona_name: str, start_id: int, end_id: int,
                      message_count: int, created_at: datetime) -> int:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO Memory_Segments (channel, server_id, persona_name, start_interaction_id, end_interaction_id, message_count, created_at, first_message_at, last_message_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (channel, server_id, persona_name, start_id, end_id, message_count, created_at, None, None)
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def store_summary(self, segment_id: int, content: str, embedding: bytes, model_name: str,
                      created_at: datetime) -> int:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO Memory_Summaries (segment_id, content, embedding, model_name, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (segment_id, content, embedding, model_name, created_at)
            )
            conn.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    def get_summaries_for_channel(self, channel: str, persona_name: str, server_id: Optional[str] = None,
                                  exclude_after_interaction_id: Optional[int] = None,
                                  model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            query = (
                "SELECT ms.summary_id, ms.segment_id, ms.content, ms.embedding, ms.model_name, ms.created_at, seg.channel, seg.persona_name, seg.start_interaction_id, seg.end_interaction_id"
                " FROM Memory_Summaries ms JOIN Memory_Segments seg ON ms.segment_id = seg.segment_id"
                " WHERE seg.channel = ? AND seg.persona_name = ?")
            params: List[Any] = [channel, persona_name]

            if server_id is not None:
                query += " AND seg.server_id = ?"
                params.append(server_id)
            else:
                query += " AND seg.server_id IS NULL"

            if exclude_after_interaction_id is not None:
                query += " AND seg.start_interaction_id < ?"
                params.append(exclude_after_interaction_id)

            active_model = model_name if model_name else EMBEDDING_MODEL
            query += " AND ms.model_name = ?"
            params.append(active_model)

            query += " ORDER BY seg.start_interaction_id ASC"
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_unsegmented_embedded_messages(self, persona_name: str, channel: str, server_id: Optional[str] = None,
                                          model_name: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            hw_query = "SELECT MAX(end_interaction_id) AS hw FROM Memory_Segments WHERE channel = ? AND persona_name = ?"
            hw_params: List[Any] = [channel, persona_name]
            if server_id is not None:
                hw_query += " AND server_id = ?"
                hw_params.append(server_id)
            else:
                hw_query += " AND server_id IS NULL"

            hw_row = cursor.execute(hw_query, hw_params).fetchone()
            high_water = hw_row['hw'] if hw_row and hw_row['hw'] is not None else 0

            query = ("SELECT ui.interaction_id, ui.author_role, ui.author_name, ui.content, ui.timestamp, me.embedding"
                     " FROM User_Interactions ui JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id"
                     " WHERE ui.persona_name = ? AND ui.channel = ? AND ui.interaction_id > ?")
            params: List[Any] = [persona_name, channel, high_water]

            if server_id is not None:
                query += " AND ui.server_id = ?"
                params.append(server_id)
            else:
                query += " AND ui.server_id IS NULL"

            query += self._suppression_filter("ui")
            query += " AND ui.content IS NOT NULL AND ui.content != ''"

            active_model = model_name if model_name else EMBEDDING_MODEL
            query += " AND me.model_name = ?"
            params.append(active_model)

            query += " ORDER BY ui.interaction_id ASC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_active_channels(self, model_name: Optional[str] = None) -> List[Tuple[str, str, Optional[str]]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            active_model = model_name if model_name else EMBEDDING_MODEL

            q1 = ("SELECT DISTINCT ui.channel, ui.persona_name, ui.server_id FROM User_Interactions ui"
                  " LEFT JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id"
                  " WHERE ui.content IS NOT NULL AND ui.content != ''" + self._suppression_filter("ui"))
            params: List[Any] = [active_model]
            q1 += " AND (me.embedding IS NULL OR me.model_name != ?)"

            q2 = ("SELECT DISTINCT ui.channel, ui.persona_name, ui.server_id FROM User_Interactions ui"
                  " JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id"
                  " WHERE ui.content IS NOT NULL AND ui.content != '' AND ui.interaction_id > COALESCE("
                  " (SELECT MAX(seg.end_interaction_id) FROM Memory_Segments seg WHERE seg.channel = ui.channel"
                  " AND seg.persona_name = ui.persona_name AND (seg.server_id = ui.server_id OR (seg.server_id IS NULL AND ui.server_id IS NULL))), 0)"
                  + self._suppression_filter("ui"))

            query = q1 + " UNION " + q2
            cursor.execute(query, params)
            return [(row['channel'], row['persona_name'], row['server_id']) for row in cursor.fetchall()]

    def get_last_segment_tail_embeddings(self, channel: str, persona_name: str, server_id: Optional[str] = None,
                                         n: int = 3, model_name: Optional[str] = None) -> Optional[List[bytes]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            seg_query = "SELECT segment_id, start_interaction_id, end_interaction_id FROM Memory_Segments WHERE channel = ? AND persona_name = ?"
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

            emb_query = (
                "SELECT me.embedding FROM Message_Embeddings me JOIN User_Interactions ui ON me.interaction_id = ui.interaction_id"
                " WHERE ui.channel = ? AND ui.persona_name = ? AND ui.interaction_id BETWEEN ? AND ?")
            emb_params: List[Any] = [channel, persona_name, seg_row['start_interaction_id'],
                                     seg_row['end_interaction_id']]

            if server_id is not None:
                emb_query += " AND ui.server_id = ?"
                emb_params.append(server_id)
            else:
                emb_query += " AND ui.server_id IS NULL"

            active_model = model_name if model_name else EMBEDDING_MODEL
            emb_query += " AND me.model_name = ?"
            emb_params.append(active_model)

            emb_query += " ORDER BY me.interaction_id DESC LIMIT ?"
            emb_params.append(n)

            cursor.execute(emb_query, emb_params)
            rows = cursor.fetchall()
            if not rows:
                return None
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
        """Retrieve summaries scoped by MemoryMode for retrieval-time fan-out."""
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
