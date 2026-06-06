# src/memory/memory_manager.py

import sqlite3
import json
import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Coroutine, Dict, List, Callable, Optional, Set, cast, Tuple, Generator
from pathlib import Path

# --- NEW: Import the global embedding model variable ---
from config.global_config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, SEMANTIC_BACKEND, HINDSIGHT_URL
from src.memory.backend.base import MemoryHit, Experience, MentalModel, ReflectResult
import sqlite_vec

logger = logging.getLogger(__name__)

# --- Universal Summary Levels ---
# L0 conceptually refers to raw User_Interactions data.
LEVEL_UNPROCESSED = 0  # Pre-migration summaries not yet classified; still retrievable
LEVEL_EPISODIC = 1     # Summaries of raw L0 chat data
LEVEL_CORE = 2         # Meta-summaries of L1 episodes (Core Profiles)
# Level 3+ is reserved for future tertiary abstractions.


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
    def __init__(
        self,
        db_path: Optional[str] = None,
        backend: "Optional[Any]" = None,
    ) -> None:
        self.db_path: str = db_path if db_path is not None else str(DATABASE_FILE)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock: threading.RLock = threading.RLock()
        self.has_vec: bool = True
        if self.db_path != ':memory:':
            DB_DIR.mkdir(parents=True, exist_ok=True)
        # Lazy import avoids circular: backend modules import MemoryManager for
        # the static _build_summary_where helper.
        # Agent-action telemetry (Agent_Actions table) is operational state, not
        # semantic memory — it always lives in sqlite even when the semantic
        # backend is Hindsight. Pin a dedicated SqliteSemanticBackend just for
        # the action-log surface; it shares this MM's connection and lock.
        from src.memory.backend.sqlite import SqliteSemanticBackend
        self._action_log = SqliteSemanticBackend(self)

        if backend is None:
            if SEMANTIC_BACKEND == "hindsight":
                from src.memory.backend.hindsight import HindsightBackend
                backend = HindsightBackend(url=HINDSIGHT_URL)
            else:
                backend = self._action_log
        self.backend = backend

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                uri=True,
                check_same_thread=False
            )
            self._conn.row_factory = sqlite3.Row
            try:
                self._conn.enable_load_extension(True)
                sqlite_vec.load(self._conn)
                self._conn.enable_load_extension(False)
                self.has_vec = True
            except (AttributeError, sqlite3.OperationalError) as e:
                logger.warning(f"Could not load sqlite-vec extension: {e}. Vector search features will run using Python fallback.")
                self.has_vec = False
                
                # Python fallback for vec_distance_cosine
                def python_vec_distance_cosine(a: Optional[bytes], b: Optional[bytes]) -> float:
                    import struct
                    import math
                    if not a or not b:
                        return 0.0
                    try:
                        # Decode float32 vector BLOBs
                        count_a = len(a) // 4
                        count_b = len(b) // 4
                        arr_a = struct.unpack(f'{count_a}f', a)
                        arr_b = struct.unpack(f'{count_b}f', b)
                        dot_product = sum(x * y for x, y in zip(arr_a, arr_b))
                        norm_a = math.sqrt(sum(x * x for x in arr_a))
                        norm_b = math.sqrt(sum(x * x for x in arr_b))
                        if norm_a == 0 or norm_b == 0:
                            return 1.0
                        similarity = dot_product / (norm_a * norm_b)
                        return float(1.0 - similarity)
                    except Exception:
                        return 0.0

                self._conn.create_function("vec_distance_cosine", 2, python_vec_distance_cosine)
            self._conn.execute("PRAGMA foreign_keys = ON;")
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
                tool_context TEXT,
                reasoning_content TEXT
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

            CREATE TABLE IF NOT EXISTS Interaction_Edit_History (
                edit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER NOT NULL,
                old_content TEXT,
                old_reasoning_content TEXT,
                edited_at TIMESTAMP NOT NULL,
                FOREIGN KEY (interaction_id) REFERENCES User_Interactions(interaction_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_edit_history_id ON Interaction_Edit_History (interaction_id);

            CREATE TABLE IF NOT EXISTS Edit_History_Embeddings (
                edit_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                model_name TEXT NOT NULL DEFAULT '{EMBEDDING_MODEL}',
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (edit_id) REFERENCES Interaction_Edit_History(edit_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS Segment_Failures (
                failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL,
                server_id TEXT,
                persona_name TEXT NOT NULL,
                start_interaction_id INTEGER NOT NULL,
                end_interaction_id INTEGER NOT NULL,
                message_count INTEGER NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 1,
                last_attempt_at TIMESTAMP NOT NULL,
                error_reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_segment_failure_lookup
                ON Segment_Failures (channel, persona_name, server_id);

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

            CREATE TABLE IF NOT EXISTS Audit_Log (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                target_id INTEGER,
                operator_id TEXT,
                timestamp TIMESTAMP NOT NULL,
                prior_state TEXT,
                new_state TEXT,
                reason TEXT,
                metadata TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_audit_event ON Audit_Log (event_type, timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_target ON Audit_Log (target_id);
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
            if 'parent_summary_id' not in user_int_cols:
                conn.execute("ALTER TABLE User_Interactions ADD COLUMN parent_summary_id INTEGER")
            if 'reply_to_id' not in user_int_cols:
                conn.execute("ALTER TABLE User_Interactions ADD COLUMN reply_to_id INTEGER")
            if 'reasoning_content' not in user_int_cols:
                conn.execute("ALTER TABLE User_Interactions ADD COLUMN reasoning_content TEXT")

            # Interaction_Edit_History migrations
            cursor.execute("PRAGMA table_info(Interaction_Edit_History)")
            edit_hist_cols = {row['name'] for row in cursor.fetchall()}
            if 'old_reasoning_content' not in edit_hist_cols:
                conn.execute("ALTER TABLE Interaction_Edit_History ADD COLUMN old_reasoning_content TEXT")

            # Memory_Summaries migration and sqlite-vec setup
            cursor.execute("PRAGMA table_info(Memory_Summaries)")
            mem_sum_cols = {row['name'] for row in cursor.fetchall()}
            if mem_sum_cols and 'summary_level' not in mem_sum_cols:
                logger.info("Migrating Memory_Summaries to v2 schema...")
                conn.execute("ALTER TABLE Memory_Summaries RENAME TO Memory_Summaries_old")
                conn.execute("""
                    CREATE TABLE Memory_Summaries (
                        summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        segment_id INTEGER,
                        content TEXT NOT NULL,
                        embedding BLOB,
                        model_name TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        summary_level INTEGER NOT NULL DEFAULT 1,
                        parent_summary_id INTEGER,
                        untrusted INTEGER NOT NULL DEFAULT 0,
                        FOREIGN KEY (segment_id) REFERENCES Memory_Segments(segment_id) ON DELETE CASCADE,
                        FOREIGN KEY (parent_summary_id) REFERENCES Memory_Summaries(summary_id) ON DELETE CASCADE
                    )
                """)
                conn.execute("""
                    INSERT INTO Memory_Summaries 
                    (summary_id, segment_id, content, embedding, model_name, created_at, summary_level)
                    SELECT summary_id, segment_id, content, embedding, model_name, created_at, 0
                    FROM Memory_Summaries_old
                """)
                conn.execute("DROP TABLE Memory_Summaries_old")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_summary_segment ON Memory_Summaries (segment_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_summary_parent ON Memory_Summaries (parent_summary_id)")
            elif not mem_sum_cols:
                conn.execute("""
                    CREATE TABLE Memory_Summaries (
                        summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        segment_id INTEGER,
                        content TEXT NOT NULL,
                        embedding BLOB,
                        model_name TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        summary_level INTEGER NOT NULL DEFAULT 1,
                        parent_summary_id INTEGER,
                        untrusted INTEGER NOT NULL DEFAULT 0,
                        FOREIGN KEY (segment_id) REFERENCES Memory_Segments(segment_id) ON DELETE CASCADE,
                        FOREIGN KEY (parent_summary_id) REFERENCES Memory_Summaries(summary_id) ON DELETE CASCADE
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_summary_segment ON Memory_Summaries (segment_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_summary_parent ON Memory_Summaries (parent_summary_id)")

            # Memory_Summaries: add untrusted column if missing (Phase 5 tool security)
            # Re-read columns after potential v1→v2 rebuild above (which includes untrusted)
            cursor.execute("PRAGMA table_info(Memory_Summaries)")
            mem_sum_cols = {row['name'] for row in cursor.fetchall()}
            if mem_sum_cols and 'untrusted' not in mem_sum_cols:
                conn.execute("ALTER TABLE Memory_Summaries ADD COLUMN untrusted INTEGER NOT NULL DEFAULT 0")

            if self.has_vec:
                # Verify sqlite-vec virtual table dimensions match current config
                for table_name in ["vec_Message_Embeddings", "vec_Memory_Summaries"]:
                    cursor.execute(f"SELECT sql FROM sqlite_master WHERE name='{table_name}'")
                    row = cursor.fetchone()
                    if row:
                        sql = row[0]
                        # Check for "float[DIM]" in the SQL schema
                        expected = f"float[{EMBEDDING_DIMENSION}]"
                        if expected not in sql:
                            logger.warning(f"Dimension mismatch in {table_name}: schema expected {expected} but found something else. Dropping and recreating...")
                            conn.execute(f"DROP TABLE {table_name}")

                conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_Message_Embeddings USING vec0(interaction_id INTEGER PRIMARY KEY, embedding float[{EMBEDDING_DIMENSION}])")
                conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_Memory_Summaries USING vec0(summary_id INTEGER PRIMARY KEY, embedding float[{EMBEDDING_DIMENSION}])")
            else:
                conn.execute("CREATE TABLE IF NOT EXISTS vec_Message_Embeddings (interaction_id INTEGER PRIMARY KEY, embedding BLOB)")
                conn.execute("CREATE TABLE IF NOT EXISTS vec_Memory_Summaries (summary_id INTEGER PRIMARY KEY, embedding BLOB)")

            # Robustly sync embeddings from main tables to virtual vector tables
            cursor.execute("SELECT COUNT(*) FROM Message_Embeddings WHERE embedding IS NOT NULL AND interaction_id NOT IN (SELECT interaction_id FROM vec_Message_Embeddings)")
            missing_msgs = cursor.fetchone()[0]
            if missing_msgs > 0:
                logger.info(f"Syncing {missing_msgs} missing message embeddings to sqlite-vec...")
                conn.execute(f"INSERT INTO vec_Message_Embeddings(interaction_id, embedding) SELECT interaction_id, embedding FROM Message_Embeddings WHERE embedding IS NOT NULL AND length(embedding) = {EMBEDDING_DIMENSION * 4} AND interaction_id NOT IN (SELECT interaction_id FROM vec_Message_Embeddings)")

            cursor.execute("SELECT COUNT(*) FROM Memory_Summaries WHERE embedding IS NOT NULL AND summary_id NOT IN (SELECT summary_id FROM vec_Memory_Summaries)")
            missing_sums = cursor.fetchone()[0]
            if missing_sums > 0:
                logger.info(f"Syncing {missing_sums} missing memory summaries to sqlite-vec...")
                conn.execute(f"INSERT INTO vec_Memory_Summaries(summary_id, embedding) SELECT summary_id, embedding FROM Memory_Summaries WHERE embedding IS NOT NULL AND length(embedding) = {EMBEDDING_DIMENSION * 4} AND summary_id NOT IN (SELECT summary_id FROM vec_Memory_Summaries)")

            conn.commit()
            logger.info("User memory database schema created or verified successfully.")

    def log_message(self, user_identifier: str, persona_name: str, channel: str,
                    author_role: str, author_name: Optional[str], content: str,
                    timestamp: datetime, server_id: Optional[str] = None,
                    platform_message_id: Optional[str] = None,
                    zammad_ticket_id: Optional[int] = None,
                    tool_context: Optional[str] = None,
                    reply_to_id: Optional[int] = None,
                    reasoning_content: Optional[str] = None) -> Optional[int]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO User_Interactions
                (user_identifier, persona_name, channel, author_role, author_name, content,
                 timestamp, zammad_ticket_id, platform_message_id, server_id, tool_context,
                 reply_to_id, reasoning_content)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_identifier, persona_name, channel, author_role, author_name, content,
                 timestamp, zammad_ticket_id, platform_message_id, server_id, tool_context,
                 reply_to_id, reasoning_content)
            )
            conn.commit()
            return int(cursor.lastrowid) if cursor.lastrowid is not None else None

    def update_platform_message_id(self, interaction_id: int, platform_message_id: str) -> None:
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                "UPDATE User_Interactions SET platform_message_id = ? WHERE interaction_id = ?",
                (platform_message_id, interaction_id)
            )
            conn.commit()

    def invalidate_summary(self, summary_id: int) -> bool:
        """
        Public method to invalidate a summary, its segment, and reset associated messages.
        Use this to force a re-summarization of a specific timeframe.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                success = self._invalidate_summary_internal(cursor, summary_id)
                if success:
                    conn.commit()
                return bool(success)
            except sqlite3.Error as e:
                logger.error(f"Failed to invalidate summary {summary_id}: {e}")
                conn.rollback()
                return False

    def _invalidate_summary_internal(self, cursor: sqlite3.Cursor, summary_id: int) -> bool:
        """
        Internal helper to invalidate a summary. Does NOT commit or handle locks.
        """
        # 1. Find the segment owning this summary
        cursor.execute("SELECT segment_id FROM Memory_Summaries WHERE summary_id = ?", (summary_id,))
        seg_row = cursor.fetchone()
        if not seg_row:
            return False

        segment_id = seg_row['segment_id']

        # 2. Delete from vector table (Virtual tables do not support standard FK CASCADE)
        cursor.execute("DELETE FROM vec_Memory_Summaries WHERE summary_id = ?", (summary_id,))

        # 3. Delete the segment (cascades to Memory_Summaries since FKs are enabled)
        cursor.execute("DELETE FROM Memory_Segments WHERE segment_id = ?", (segment_id,))

        # 3. Reset ALL messages that were in that summary so they re-queue for the agent
        cursor.execute(
            "UPDATE User_Interactions SET parent_summary_id = NULL WHERE parent_summary_id = ?",
            (summary_id,)
        )
        return True

    def handle_message_edit(self, platform_message_id: str, new_content: str) -> bool:
        """
        Updates content of an interaction and archives the old version.
        Triggers invalidation of embeddings and L1 summaries if necessary.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 1. Fetch current version
            cursor.execute(
                "SELECT interaction_id, content, reasoning_content, parent_summary_id FROM User_Interactions WHERE platform_message_id = ?",
                (platform_message_id,)
            )
            row = cursor.fetchone()
            if not row:
                return False
            
            interaction_id = row['interaction_id']
            old_content = row['content']
            old_reasoning = row['reasoning_content']
            old_summary_id = row['parent_summary_id']
            now = datetime.now()

            try:
                # 2. Archive old version
                cursor.execute(
                    "INSERT INTO Interaction_Edit_History (interaction_id, old_content, old_reasoning_content, edited_at) VALUES (?, ?, ?, ?)",
                    (interaction_id, old_content, old_reasoning, now)
                )

                # 3. Update main interaction
                cursor.execute(
                    "UPDATE User_Interactions SET content = ?, parent_summary_id = NULL WHERE interaction_id = ?",
                    (new_content, interaction_id)
                )

                # 4. Invalidate embeddings
                cursor.execute("DELETE FROM Message_Embeddings WHERE interaction_id = ?", (interaction_id,))
                cursor.execute("DELETE FROM vec_Message_Embeddings WHERE interaction_id = ?", (interaction_id,))

                # 5. Memory Rewind: If already summarized, invalidate the segment using the helper
                if old_summary_id is not None:
                    self._invalidate_summary_internal(cursor, old_summary_id)
                
                conn.commit()
                logger.info(f"Handled edit for interaction {interaction_id} (platform_id: {platform_message_id}).")
                return True
            except sqlite3.Error as e:
                logger.error(f"Failed to handle message edit: {e}")
                conn.rollback()
                return False

    def handle_portal_retry(self, persona_name: str, user_identifier: str,
                            channel: str) -> Optional[int]:
        """Archive the most recent assistant turn for this portal session.

        Finds the latest assistant row matching (persona, user_identifier, channel),
        moves its content into Interaction_Edit_History, and invalidates its
        embedding. Returns the interaction_id so the caller can UPDATE the
        canonical row in place with the new response.

        Returns None if no prior assistant row exists (first-turn retry is a
        no-op). Also returns None when the latest *visible* interaction is a
        *user* turn with no response yet: that is a "generate a reply to this
        turn" action, not a regen, so the caller must INSERT a fresh assistant
        row rather than archive + overwrite the earlier assistant turn that
        sits before it.

        Suppressed (soft-deleted) rows are excluded from the "most recent"
        lookup so it matches the transcript projection the UI renders. Without
        this, deleting an assistant reply (which leaves its user turn trailing
        in the UI) and then retrying that user turn would archive + overwrite
        the still-suppressed assistant row — landing the regenerated response
        in an invisible row, so it appears to vanish after streaming.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            # Inspect the single most recent *visible* interaction (any role).
            # Only a trailing assistant turn is eligible for archive-in-place;
            # if a user turn is newer (or the trailing assistant was deleted),
            # there is nothing to regenerate in place.
            cursor.execute(
                "SELECT interaction_id, content, reasoning_content, author_role"
                " FROM User_Interactions"
                " WHERE persona_name = ? AND user_identifier = ? AND channel = ?"
                + self._SUPPRESSION_SUBQUERY +
                " ORDER BY timestamp DESC, interaction_id DESC LIMIT 1",
                (persona_name, user_identifier, channel),
            )
            row = cursor.fetchone()
            if not row or row['author_role'] != 'assistant':
                return None

            interaction_id = row['interaction_id']
            old_content = row['content']
            old_reasoning = row['reasoning_content']
            now = datetime.now()
            try:
                cursor.execute(
                    "INSERT INTO Interaction_Edit_History (interaction_id, old_content, old_reasoning_content, edited_at) VALUES (?, ?, ?, ?)",
                    (interaction_id, old_content, old_reasoning, now),
                )
                new_edit_id = cursor.lastrowid
                # Move L0 embedding to Edit_History_Embeddings so chevron restore can bring it back.
                # vec_Message_Embeddings is dropped — archives don't participate in retrieval k-NN.
                cursor.execute(
                    "SELECT embedding, model_name, created_at FROM Message_Embeddings WHERE interaction_id = ?",
                    (interaction_id,),
                )
                emb = cursor.fetchone()
                if emb is not None:
                    cursor.execute(
                        "INSERT INTO Edit_History_Embeddings (edit_id, embedding, model_name, created_at) VALUES (?, ?, ?, ?)",
                        (new_edit_id, emb['embedding'], emb['model_name'], emb['created_at']),
                    )
                cursor.execute("DELETE FROM Message_Embeddings WHERE interaction_id = ?", (interaction_id,))
                cursor.execute("DELETE FROM vec_Message_Embeddings WHERE interaction_id = ?", (interaction_id,))
                conn.commit()
                return int(interaction_id)
            except sqlite3.Error as e:
                logger.error(f"handle_portal_retry failed for id={interaction_id}: {e}")
                conn.rollback()
                return None

    def update_interaction_content(self, interaction_id: int, new_content: str,
                                   reasoning_content: Optional[str] = None,
                                   tool_context: Optional[str] = None) -> bool:
        """Overwrite the content of an existing interaction row in place.

        Used by portal retry and portal manual-edit flows. Clears
        `parent_summary_id` so the next summarizer pass re-groups the row, and
        drops the stale L0 embedding (`Message_Embeddings` + `vec_*`) so
        `MemoryAgent._embed_unembedded` re-encodes against the new content.

        `tool_context` is only rewritten when explicitly provided (a regen that
        produced tool calls); manual text edits pass None and leave the stored
        tool_context untouched.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                if tool_context is not None:
                    cursor.execute(
                        "UPDATE User_Interactions SET content = ?, reasoning_content = ?,"
                        " tool_context = ?, parent_summary_id = NULL"
                        " WHERE interaction_id = ?",
                        (new_content, reasoning_content, tool_context, interaction_id),
                    )
                else:
                    cursor.execute(
                        "UPDATE User_Interactions SET content = ?, reasoning_content = ?, parent_summary_id = NULL"
                        " WHERE interaction_id = ?",
                        (new_content, reasoning_content, interaction_id),
                    )
                updated = cursor.rowcount > 0
                cursor.execute("DELETE FROM Message_Embeddings WHERE interaction_id = ?", (interaction_id,))
                cursor.execute("DELETE FROM vec_Message_Embeddings WHERE interaction_id = ?", (interaction_id,))
                conn.commit()
                return bool(updated)
            except sqlite3.Error as e:
                logger.error(f"update_interaction_content failed for id={interaction_id}: {e}")
                conn.rollback()
                return False

    def list_interaction_versions(self, interaction_id: int) -> List[Dict[str, Any]]:
        """Return all versions for an interaction, oldest first, canonical last.

        Archive rows ordered by (edited_at ASC, edit_id ASC). Canonical is synthesized
        from User_Interactions with edit_id=None. Portal uses this to populate its
        retry/redo stacks after an assistant stream reveals `assistant_id`.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT edit_id, old_content, old_reasoning_content, edited_at FROM Interaction_Edit_History"
                " WHERE interaction_id = ? ORDER BY edited_at ASC, edit_id ASC",
                (interaction_id,),
            )
            results: List[Dict[str, Any]] = [
                {
                    "edit_id": r['edit_id'], 
                    "content": r['old_content'], 
                    "reasoning_content": r['old_reasoning_content'],
                    "created_at": r['edited_at']
                }
                for r in cursor.fetchall()
            ]
            cursor.execute(
                "SELECT content, reasoning_content, timestamp FROM User_Interactions WHERE interaction_id = ?",
                (interaction_id,),
            )
            canonical = cursor.fetchone()
            if canonical is not None:
                # Only append canonical if it's not already in the archives
                canonical_in_archives = any(r['content'] == canonical['content'] for r in results)
                if not canonical_in_archives:
                    results.append({
                        "edit_id": None,
                        "content": canonical['content'],
                        "reasoning_content": canonical['reasoning_content'],
                        "created_at": canonical['timestamp'],
                    })
                
                # Flag the active canonical entry
                for r in results:
                    r['canonical'] = (r['content'] == canonical['content'])

            return results

    def get_ids_with_versions(self, interaction_ids: List[int]) -> Set[int]:
        """Return the subset of `interaction_ids` that carry ≥1 edit/regen
        archive (an Interaction_Edit_History row). One query; used by the
        DP-130 transcript projection to set each chunk's `has_versions` flag
        without an N+1 `list_interaction_versions` per row.
        """
        ids = [i for i in interaction_ids if isinstance(i, int)]
        if not ids:
            return set()
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in ids)
            cursor.execute(
                "SELECT DISTINCT interaction_id FROM Interaction_Edit_History"
                f" WHERE interaction_id IN ({placeholders})",
                ids,
            )
            return {row["interaction_id"] for row in cursor.fetchall()}

    def swap_interaction_version(self, interaction_id: int, k: int) -> Dict[str, Any]:
        """Swap archive position `k` with canonical for `interaction_id`.

        `k` is 0-indexed over archives ordered ascending by archival time (pre-swap).
        Single transaction:
          1. Archive current canonical — insert new Interaction_Edit_History row;
             move Message_Embeddings row (if any) into Edit_History_Embeddings keyed
             by the new edit_id; delete from vec_Message_Embeddings.
          2. Restore target archive k — copy old_content into User_Interactions.content;
             copy Edit_History_Embeddings(target) into Message_Embeddings +
             vec_Message_Embeddings if present. The target archive row is KEPT so the
             numbered version list stays stable across navigation (the chevron `k/n`
             counter addresses a fixed list; deleting on promote would make it a
             rotating MRU and strand older versions). list_interaction_versions
             content-dedupes the now-duplicate canonical against its source archive.

        Returns `{"current_content": str, "interaction_id": int, "total_versions": int}`
        where total_versions matches the displayed (content-deduped) version count.

        Raises IndexError if k is out of bounds (no state mutation).
        Raises ValueError if interaction_id does not exist.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT content, reasoning_content FROM User_Interactions WHERE interaction_id = ?",
                (interaction_id,),
            )
            canonical_row = cursor.fetchone()
            if canonical_row is None:
                raise ValueError(f"interaction_id {interaction_id} not found")

            cursor.execute(
                "SELECT edit_id, old_content, old_reasoning_content FROM Interaction_Edit_History"
                " WHERE interaction_id = ?"
                " ORDER BY edited_at ASC, edit_id ASC",
                (interaction_id,),
            )
            archives = cursor.fetchall()
            if k < 0 or k >= len(archives):
                raise IndexError(f"version index {k} out of bounds (have {len(archives)} archives)")

            target_edit_id = archives[k]['edit_id']
            target_content = archives[k]['old_content']
            target_reasoning = archives[k]['old_reasoning_content']
            current_canonical = canonical_row['content']
            now = datetime.now()

            try:
                # 1. Archive current canonical (with content-hash dedupe).
                #    If an archive row with the same (interaction_id, old_content) already
                #    exists, skip the insert.
                cursor.execute(
                    "SELECT 1 FROM Interaction_Edit_History"
                    " WHERE interaction_id = ? AND old_content = ? LIMIT 1",
                    (interaction_id, current_canonical),
                )
                dup_archive = cursor.fetchone()
                if dup_archive is None:
                    cursor.execute(
                        "INSERT INTO Interaction_Edit_History (interaction_id, old_content, edited_at) VALUES (?, ?, ?)",
                        (interaction_id, current_canonical, now),
                    )
                    new_edit_id = cursor.lastrowid

                    cursor.execute(
                        "SELECT embedding, model_name, created_at FROM Message_Embeddings WHERE interaction_id = ?",
                        (interaction_id,),
                    )
                    canonical_emb = cursor.fetchone()
                    if canonical_emb is not None:
                        cursor.execute(
                            "INSERT INTO Edit_History_Embeddings (edit_id, embedding, model_name, created_at) VALUES (?, ?, ?, ?)",
                            (new_edit_id, canonical_emb['embedding'], canonical_emb['model_name'], canonical_emb['created_at']),
                        )
                
                cursor.execute("DELETE FROM Message_Embeddings WHERE interaction_id = ?", (interaction_id,))
                cursor.execute("DELETE FROM vec_Message_Embeddings WHERE interaction_id = ?", (interaction_id,))

                # 2. Restore target archive into canonical
                cursor.execute(
                    "UPDATE User_Interactions SET content = ?, parent_summary_id = NULL WHERE interaction_id = ?",
                    (target_content, interaction_id),
                )

                cursor.execute(
                    "SELECT embedding, model_name, created_at FROM Edit_History_Embeddings WHERE edit_id = ?",
                    (target_edit_id,),
                )
                target_emb = cursor.fetchone()
                if target_emb is not None:
                    cursor.execute(
                        "INSERT INTO Message_Embeddings (interaction_id, embedding, model_name, created_at) VALUES (?, ?, ?, ?)",
                        (interaction_id, target_emb['embedding'], target_emb['model_name'], target_emb['created_at']),
                    )
                    cursor.execute(
                        "INSERT INTO vec_Message_Embeddings (interaction_id, embedding) VALUES (?, ?)",
                        (interaction_id, target_emb['embedding']),
                    )

                # We DO NOT delete the target archive. It remains in Interaction_Edit_History
                # so that the list of versions remains perfectly stable.

                conn.commit()
            except sqlite3.Error as e:
                logger.error(f"swap_interaction_version failed for id={interaction_id} k={k}: {e}")
                conn.rollback()
                raise

            # total_versions must match what list_interaction_versions DISPLAYS:
            # all archive rows, plus canonical only if its content isn't already an
            # archive row (content-dedupe). After a swap the restored canonical
            # always duplicates its source archive row, so it is not double-counted.
            cursor.execute(
                "SELECT COUNT(*) FROM Interaction_Edit_History WHERE interaction_id = ?",
                (interaction_id,),
            )
            total_archives = int(cursor.fetchone()[0])
            cursor.execute(
                "SELECT 1 FROM Interaction_Edit_History"
                " WHERE interaction_id = ? AND old_content = ? LIMIT 1",
                (interaction_id, target_content),
            )
            canonical_in_archives = cursor.fetchone() is not None
            return {
                "current_content": target_content,
                "reasoning_content": target_reasoning,
                "interaction_id": interaction_id,
                "total_versions": total_archives + (0 if canonical_in_archives else 1),
            }

    def suppress_interaction(self, interaction_id: int) -> bool:
        """Soft-suppress a single interaction by id. Idempotent.

        Used by the portal's empty-edit (delete) flow. Suppressed rows are filtered
        out of every history / retrieval / embedding-pipeline query via
        `_suppression_filter`. Reply chains are left intact (no FK cascade, no
        nulling of `reply_to_id`).
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "INSERT INTO Suppressed_Interactions (interaction_id, suppressed_at) VALUES (?, ?)",
                    (interaction_id, datetime.now()),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def suppress_message_by_platform_id(self, platform_message_id: str) -> bool:
        """Suppresses ALL versions of messages associated with this platform ID."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT interaction_id FROM User_Interactions WHERE platform_message_id = ?",
                           (platform_message_id,))
            rows = cursor.fetchall()
            if not rows:
                return False

            now = datetime.now()
            suppressed_count = 0
            for row in rows:
                interaction_id = row['interaction_id']
                try:
                    cursor.execute("INSERT INTO Suppressed_Interactions (interaction_id, suppressed_at) VALUES (?, ?)",
                                   (interaction_id, now))
                    suppressed_count += 1
                except sqlite3.IntegrityError:
                    # Already suppressed
                    continue

            if suppressed_count > 0:
                conn.commit()
                return True
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
            query = ("SELECT interaction_id, author_role, author_name, content, tool_context, reasoning_content FROM User_Interactions"
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
            query = ("SELECT interaction_id, author_role, author_name, content, tool_context, reasoning_content FROM User_Interactions"
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
            query = ("SELECT interaction_id, author_role, author_name, content, tool_context, reasoning_content FROM User_Interactions"
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

    def get_server_history(self, server_id: Optional[str], persona_name: str, limit: Optional[int] = None) -> List[
        Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            if server_id is not None:
                query = ("SELECT interaction_id, author_role, author_name, content, tool_context, reasoning_content FROM User_Interactions"
                         " WHERE server_id = ? AND persona_name = ?" + self._SUPPRESSION_SUBQUERY)
                params: List[Any] = [server_id, persona_name]
            else:
                query = ("SELECT interaction_id, author_role, author_name, content, tool_context, reasoning_content FROM User_Interactions"
                         " WHERE server_id IS NULL AND persona_name = ?" + self._SUPPRESSION_SUBQUERY)
                params = [persona_name]
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
            query = ("SELECT interaction_id, author_role, author_name, content, tool_context, reasoning_content FROM User_Interactions"
                     " WHERE persona_name = ?" + self._SUPPRESSION_SUBQUERY)
            params: List[Any] = [persona_name]
            query += " ORDER BY timestamp DESC"
            if isinstance(limit, int):
                query += " LIMIT ?"
                params.append(limit)
            cursor.execute(query, params)
            return [dict(row) for row in reversed(cursor.fetchall())]

    def get_distinct_channels(self, persona_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List the distinct (channel, server_id) pairs seen in history.

        Drives the bespoke portal's channel list (DP-136 / handoff §10): the UI
        groups these by the channel's source prefix (`web_ui`, `discord`,
        `zammad`, `gmail`). Scoped to `persona_name` when given so the list
        reflects the channels the active persona has actually been used in.
        Each entry carries a `last_ts` (most recent activity) so the UI can sort
        and a `count` of non-suppressed rows. Suppressed rows are excluded.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            query = (
                "SELECT channel, server_id, COUNT(*) AS count,"
                " MAX(timestamp) AS last_ts FROM User_Interactions"
                " WHERE channel IS NOT NULL" + self._SUPPRESSION_SUBQUERY
            )
            params: List[Any] = []
            if persona_name is not None:
                query += " AND persona_name = ?"
                params.append(persona_name)
            query += " GROUP BY channel, server_id ORDER BY last_ts DESC"
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def log_agent_action(self, agent_name: str, action_type: str, trigger_context: Optional[str] = None,
                         action_payload: Optional[str] = None, outcome: Optional[str] = None,
                         outcome_payload: Optional[str] = None, parent_id: Optional[int] = None) -> int:
        return self._action_log.log_agent_action(
            agent_name, action_type, trigger_context, action_payload,
            outcome, outcome_payload, parent_id,
        )

    def update_agent_action_outcome(self, action_id: int, outcome: str, outcome_payload: Optional[str] = None) -> None:
        self._action_log.update_agent_action_outcome(action_id, outcome, outcome_payload)

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
        self._action_log.add_action_contexts(action_id, contexts)

    def get_relevant_agent_actions(self, agent_name: str, match_contexts: Optional[List[Tuple[str, str]]] = None,
                                   match_types: Optional[List[str]] = None, limit: int = 15) -> List[Dict[str, Any]]:
        return self._action_log.get_relevant_agent_actions(agent_name, match_contexts, match_types, limit)

    def get_action_steps(self, parent_id: int) -> List[Dict[str, Any]]:
        return self._action_log.get_action_steps(parent_id)

    def get_agent_action(self, action_id: int) -> Optional[Dict[str, Any]]:
        return self._action_log.get_agent_action(action_id)

    def get_action_contexts(self, action_id: int) -> List[Tuple[str, str]]:
        return self._action_log.get_action_contexts(action_id)

    def store_message_embedding(self, interaction_id: int, embedding: bytes, model_name: str,
                                created_at: datetime) -> None:
        self.backend.store_message_embedding(interaction_id, embedding, model_name, created_at)

    def get_unembedded_messages(self, persona_name: str, channel: str, server_id: Optional[str] = None,
                                limit: int = 50, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.backend.get_unembedded_messages(persona_name, channel, server_id, limit, model_name)

    def store_segment(self, channel: str, server_id: Optional[str], persona_name: str, start_id: int, end_id: int,
                      message_count: int, created_at: datetime) -> int:
        return self.backend.store_segment(channel, server_id, persona_name, start_id, end_id, message_count, created_at)

    def store_summary(self, segment_id: int, content: str, embedding: bytes, model_name: str,
                      created_at: datetime, summary_level: Optional[int] = None,
                      parent_summary_id: Optional[int] = None,
                      untrusted: bool = False) -> int:
        return self.backend.store_summary(segment_id, content, embedding, model_name, created_at,
                                          summary_level, parent_summary_id, untrusted)

    def get_summaries_for_channel(self, channel: str, persona_name: str, server_id: Optional[str] = None,
                                  exclude_after_interaction_id: Optional[int] = None,
                                  model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.backend.get_summaries_for_channel(channel, persona_name, server_id,
                                                     exclude_after_interaction_id, model_name)

    def get_unsegmented_embedded_messages(self, persona_name: str, channel: str, server_id: Optional[str] = None,
                                          model_name: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.backend.get_unsegmented_embedded_messages(persona_name, channel, server_id, model_name, limit)

    def record_segment_failure(
            self,
            channel: str,
            server_id: Optional[str],
            persona_name: str,
            start_id: int,
            end_id: int,
            message_count: int,
            error_reason: Optional[str] = None,
    ) -> None:
        self.backend.record_segment_failure(channel, server_id, persona_name, start_id, end_id,
                                            message_count, error_reason)

    def get_failed_segment_ranges(
            self,
            channel: str,
            persona_name: str,
            server_id: Optional[str] = None,
            max_attempts: int = 3,
            cooldown_hours: float = 24.0,
    ) -> List[Dict[str, Any]]:
        return self.backend.get_failed_segment_ranges(channel, persona_name, server_id, max_attempts, cooldown_hours)

    def clear_segment_failure(
            self,
            channel: str,
            persona_name: str,
            server_id: Optional[str],
            start_id: int,
            end_id: int,
    ) -> None:
        self.backend.clear_segment_failure(channel, persona_name, server_id, start_id, end_id)

    def get_active_channels(self, model_name: Optional[str] = None) -> List[Tuple[str, str, Optional[str]]]:
        return self.backend.get_active_channels(model_name)

    def get_last_segment_tail_embeddings(self, channel: str, persona_name: str, server_id: Optional[str] = None,
                                         n: int = 3, model_name: Optional[str] = None) -> Optional[List[bytes]]:
        return self.backend.get_last_segment_tail_embeddings(channel, persona_name, server_id, n, model_name)

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
        # We fetch LEVEL_CORE unconditionally, and LEVEL_EPISODIC / LEVEL_UNPROCESSED
        # only if they haven't been subsumed into a LEVEL_CORE profile yet
        # (parent_summary_id IS NULL). This ensures pre-migration (level 0)
        # summaries remain retrievable.
        where_parts = [
            "seg.persona_name = ?",
            f"(ms.summary_level = {LEVEL_CORE} OR (ms.summary_level <= {LEVEL_EPISODIC} AND ms.parent_summary_id IS NULL))"
        ]
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
                # [FIX]: Allow NULL server_id for Web UI (portal) support
                where_parts.append("seg.server_id IS NULL")
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
        query_embeddings: Optional[List[bytes]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return self.backend.retrieve_relevant_summaries(
            persona_name, channel, server_id, user_identifier, memory_mode,
            include_ambient, exclude_after_interaction_id, model_name,
            query_embeddings, limit,
        )

    def log_audit_event(self, event_type: str, target_id: Optional[int] = None, 
                        operator_id: Optional[str] = None, prior_state: Optional[str] = None, 
                        new_state: Optional[str] = None, reason: Optional[str] = None, 
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Public method to log security-relevant events to Audit_Log."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                self._log_audit_event(
                    cursor=cursor,
                    event_type=event_type,
                    target_id=target_id,
                    operator_id=operator_id,
                    prior_state=prior_state,
                    new_state=new_state,
                    reason=reason,
                    metadata=metadata
                )
                conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Failed to log audit event {event_type}: {e}")
                conn.rollback()

    def mark_trusted(self, summary_id: int, operator_id: str, reason: str) -> bool:
        """Mark a memory summary as trusted (untrusted=0)."""
        return self._update_summary_trust(summary_id, 0, operator_id, reason)

    def mark_untrusted(self, summary_id: int, operator_id: str, reason: str) -> bool:
        """Mark a memory summary as untrusted (untrusted=1)."""
        return self._update_summary_trust(summary_id, 1, operator_id, reason)

    def _update_summary_trust(self, summary_id: int, untrusted_value: int, operator_id: str, reason: str) -> bool:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 1. Fetch current state for audit log
            cursor.execute("SELECT untrusted FROM Memory_Summaries WHERE summary_id = ?", (summary_id,))
            row = cursor.fetchone()
            if not row:
                return False
            
            prior_val = row['untrusted']
            if prior_val == untrusted_value:
                # No change needed, but we might still log it? 
                # Let's just return True to signify success.
                return True

            try:
                # 2. Update bit
                cursor.execute("UPDATE Memory_Summaries SET untrusted = ? WHERE summary_id = ?", 
                               (untrusted_value, summary_id))
                
                # 3. Log audit event
                prior_state = "untrusted" if prior_val else "trusted"
                new_state = "untrusted" if untrusted_value else "trusted"
                
                self._log_audit_event(
                    cursor=cursor,
                    event_type="operator_override",
                    target_id=summary_id,
                    operator_id=operator_id,
                    prior_state=prior_state,
                    new_state=new_state,
                    reason=reason
                )
                
                conn.commit()
                logger.info(f"Summary {summary_id} marked as {new_state} by {operator_id}. Reason: {reason}")
                return True
            except sqlite3.Error as e:
                logger.error(f"Failed to update trust for summary {summary_id}: {e}")
                conn.rollback()
                return False

    def _log_audit_event(self, cursor: sqlite3.Cursor, event_type: str, target_id: Optional[int] = None, 
                        operator_id: Optional[str] = None, prior_state: Optional[str] = None, 
                        new_state: Optional[str] = None, reason: Optional[str] = None, 
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Internal helper to log security-relevant events to Audit_Log."""
        now = datetime.now()
        meta_json = json.dumps(metadata) if metadata else None
        
        cursor.execute(
            """INSERT INTO Audit_Log 
               (event_type, target_id, operator_id, timestamp, prior_state, new_state, reason, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_type, target_id, operator_id, now, prior_state, new_state, reason, meta_json)
        )

    # ---------- New Hindsight-shape Delegation ----------

    async def retain_turn(
        self,
        bank_id: str,
        role: str,
        content: str,
        *,
        timestamp: datetime,
        scope_tags: List[str],
        source_persona: str,
        untrusted: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return await self.backend.retain_turn(
            bank_id, role, content,
            timestamp=timestamp, scope_tags=scope_tags,
            source_persona=source_persona, untrusted=untrusted, metadata=metadata
        )

    async def retain_experience(
        self,
        bank_id: str,
        action_type: str,
        context: Dict[str, Any],
        outcome: Optional[str],
        *,
        scope_tags: List[str],
        source_persona: str,
        untrusted: bool = False,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        content_override: Optional[str] = None,
    ) -> str:
        return await self.backend.retain_experience(
            bank_id, action_type, context, outcome,
            scope_tags=scope_tags, source_persona=source_persona,
            untrusted=untrusted, timestamp=timestamp, metadata=metadata,
            document_id=document_id, content_override=content_override,
        )

    # Note: mark_trusted/mark_untrusted on MemoryManager already exist for the
    # legacy summary-level API (int summary_id). The new-shape per-hit equivalents
    # are reached via `mm.backend.mark_trusted(bank_id, hit_id, ...)` to avoid
    # name collision. Resolve when the legacy API is retired in Phase 5.

    async def recall(
        self,
        bank_id: str,
        query: str,
        *,
        k: int = 10,
        types: Optional[List[str]] = None,
        tag_filter: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        budget: Optional[str] = None,
    ) -> List[MemoryHit]:
        return await self.backend.recall(
            bank_id, query, k=k, types=types,
            tag_filter=tag_filter, max_tokens=max_tokens, budget=budget
        )

    async def recall_experiences(
        self,
        bank_id: str,
        query: str,
        *,
        match_contexts: Optional[List[Tuple[str, str]]] = None,
        k: int = 10,
    ) -> List[Experience]:
        return await self.backend.recall_experiences(
            bank_id, query, match_contexts=match_contexts, k=k
        )

    async def reflect(
        self,
        bank_id: str,
        query: str,
        *,
        tag_filter: Optional[List[str]] = None,
    ) -> ReflectResult:
        return await self.backend.reflect(bank_id, query, tag_filter=tag_filter)

    async def list_mental_models(
        self,
        bank_id: str,
        *,
        tags: Optional[List[str]] = None,
    ) -> List[MentalModel]:
        return await self.backend.list_mental_models(bank_id, tags=tags)

    async def ensure_bank(
        self,
        bank_id: str,
        *,
        retain_mission: Optional[str] = None,
        reflect_mission: Optional[str] = None,
        enable_observations: Optional[bool] = None,
        observations_mission: Optional[str] = None,
    ) -> None:
        await self.backend.ensure_bank(
            bank_id,
            retain_mission=retain_mission,
            reflect_mission=reflect_mission,
            enable_observations=enable_observations,
            observations_mission=observations_mission,
        )

    async def delete_bank(self, bank_id: str) -> None:
        await self.backend.delete_bank(bank_id)
