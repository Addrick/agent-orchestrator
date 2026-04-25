# src/memory/memory_manager.py

import sqlite3
import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple
from pathlib import Path

# --- NEW: Import the global embedding model variable ---
from config.global_config import EMBEDDING_MODEL, EMBEDDING_DIMENSION
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
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
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

            CREATE TABLE IF NOT EXISTS Interaction_Edit_History (
                edit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                interaction_id INTEGER NOT NULL,
                old_content TEXT,
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
                        FOREIGN KEY (segment_id) REFERENCES Memory_Segments(segment_id) ON DELETE CASCADE,
                        FOREIGN KEY (parent_summary_id) REFERENCES Memory_Summaries(summary_id) ON DELETE CASCADE
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_summary_segment ON Memory_Summaries (segment_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_summary_parent ON Memory_Summaries (parent_summary_id)")

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

            # Robustly sync embeddings from main tables to virtual vector tables
            cursor.execute("SELECT COUNT(*) FROM Message_Embeddings me WHERE me.embedding IS NOT NULL AND NOT EXISTS (SELECT 1 FROM vec_Message_Embeddings v WHERE v.interaction_id = me.interaction_id)")
            missing_msgs = cursor.fetchone()[0]
            if missing_msgs > 0:
                logger.info(f"Syncing {missing_msgs} missing message embeddings to sqlite-vec...")
                conn.execute(f"INSERT INTO vec_Message_Embeddings(interaction_id, embedding) SELECT interaction_id, embedding FROM Message_Embeddings me WHERE embedding IS NOT NULL AND length(embedding) = {EMBEDDING_DIMENSION * 4} AND NOT EXISTS (SELECT 1 FROM vec_Message_Embeddings v WHERE v.interaction_id = me.interaction_id)")

            cursor.execute("SELECT COUNT(*) FROM Memory_Summaries ms WHERE ms.embedding IS NOT NULL AND NOT EXISTS (SELECT 1 FROM vec_Memory_Summaries v WHERE v.summary_id = ms.summary_id)")
            missing_sums = cursor.fetchone()[0]
            if missing_sums > 0:
                logger.info(f"Syncing {missing_sums} missing memory summaries to sqlite-vec...")
                conn.execute(f"INSERT INTO vec_Memory_Summaries(summary_id, embedding) SELECT summary_id, embedding FROM Memory_Summaries ms WHERE embedding IS NOT NULL AND length(embedding) = {EMBEDDING_DIMENSION * 4} AND NOT EXISTS (SELECT 1 FROM vec_Memory_Summaries v WHERE v.summary_id = ms.summary_id)")

            conn.commit()
            logger.info("User memory database schema created or verified successfully.")

    def log_message(self, user_identifier: str, persona_name: str, channel: str,
                    author_role: str, author_name: Optional[str], content: str,
                    timestamp: datetime, server_id: Optional[str] = None,
                    platform_message_id: Optional[str] = None,
                    zammad_ticket_id: Optional[int] = None,
                    tool_context: Optional[str] = None,
                    reply_to_id: Optional[int] = None) -> Optional[int]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO User_Interactions
                (user_identifier, persona_name, channel, author_role, author_name, content,
                 timestamp, zammad_ticket_id, platform_message_id, server_id, tool_context,
                 reply_to_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_identifier, persona_name, channel, author_role, author_name, content,
                 timestamp, zammad_ticket_id, platform_message_id, server_id, tool_context,
                 reply_to_id)
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
                return success
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
                "SELECT interaction_id, content, parent_summary_id FROM User_Interactions WHERE platform_message_id = ?",
                (platform_message_id,)
            )
            row = cursor.fetchone()
            if not row:
                return False
            
            interaction_id = row['interaction_id']
            old_content = row['content']
            old_summary_id = row['parent_summary_id']
            now = datetime.now()

            try:
                # 2. Archive old version
                cursor.execute(
                    "INSERT INTO Interaction_Edit_History (interaction_id, old_content, edited_at) VALUES (?, ?, ?)",
                    (interaction_id, old_content, now)
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

        Returns None if no prior assistant row exists (first-turn retry is a no-op).
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT interaction_id, content FROM User_Interactions"
                " WHERE persona_name = ? AND user_identifier = ? AND channel = ?"
                "   AND author_role = 'assistant'"
                " ORDER BY timestamp DESC, interaction_id DESC LIMIT 1",
                (persona_name, user_identifier, channel),
            )
            row = cursor.fetchone()
            if not row:
                return None

            interaction_id = row['interaction_id']
            old_content = row['content']
            now = datetime.now()
            try:
                cursor.execute(
                    "INSERT INTO Interaction_Edit_History (interaction_id, old_content, edited_at) VALUES (?, ?, ?)",
                    (interaction_id, old_content, now),
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
                return interaction_id
            except sqlite3.Error as e:
                logger.error(f"handle_portal_retry failed for id={interaction_id}: {e}")
                conn.rollback()
                return None

    def update_interaction_content(self, interaction_id: int, new_content: str) -> bool:
        """Overwrite the content of an existing interaction row in place.

        Used by portal retry flow after handle_portal_retry has archived the
        prior content. Clears parent_summary_id so re-summarization picks up
        the updated content.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "UPDATE User_Interactions SET content = ?, parent_summary_id = NULL"
                    " WHERE interaction_id = ?",
                    (new_content, interaction_id),
                )
                conn.commit()
                return cursor.rowcount > 0
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
                "SELECT edit_id, old_content AS content, edited_at AS created_at"
                " FROM Interaction_Edit_History WHERE interaction_id = ?"
                " ORDER BY edited_at ASC, edit_id ASC",
                (interaction_id,),
            )
            versions: List[Dict[str, Any]] = [
                {"edit_id": r['edit_id'], "content": r['content'], "created_at": r['created_at']}
                for r in cursor.fetchall()
            ]
            cursor.execute(
                "SELECT content, timestamp AS created_at FROM User_Interactions WHERE interaction_id = ?",
                (interaction_id,),
            )
            canonical = cursor.fetchone()
            if canonical is not None:
                versions.append({
                    "edit_id": None,
                    "content": canonical['content'],
                    "created_at": canonical['created_at'],
                })
            return versions

    def swap_interaction_version(self, interaction_id: int, k: int) -> Dict[str, Any]:
        """Swap archive position `k` with canonical for `interaction_id`.

        `k` is 0-indexed over archives ordered ascending by archival time (pre-swap).
        Single transaction:
          1. Archive current canonical — insert new Interaction_Edit_History row;
             move Message_Embeddings row (if any) into Edit_History_Embeddings keyed
             by the new edit_id; delete from vec_Message_Embeddings.
          2. Restore target archive k — copy old_content into User_Interactions.content;
             move Edit_History_Embeddings(target) back into Message_Embeddings +
             vec_Message_Embeddings if present; delete the target archive row.

        Returns `{"current_content": str, "interaction_id": int, "total_versions": int}`.

        Raises IndexError if k is out of bounds (no state mutation).
        Raises ValueError if interaction_id does not exist.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT content FROM User_Interactions WHERE interaction_id = ?",
                (interaction_id,),
            )
            canonical_row = cursor.fetchone()
            if canonical_row is None:
                raise ValueError(f"interaction_id {interaction_id} not found")

            cursor.execute(
                "SELECT edit_id, old_content FROM Interaction_Edit_History"
                " WHERE interaction_id = ?"
                " ORDER BY edited_at ASC, edit_id ASC",
                (interaction_id,),
            )
            archives = cursor.fetchall()
            if k < 0 or k >= len(archives):
                raise IndexError(f"version index {k} out of bounds (have {len(archives)} archives)")

            target_edit_id = archives[k]['edit_id']
            target_content = archives[k]['old_content']
            current_canonical = canonical_row['content']
            now = datetime.now()

            try:
                # 1. Archive current canonical
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

                # Delete the target archive (its content is now canonical). Cascades to
                # Edit_History_Embeddings(target_edit_id).
                cursor.execute(
                    "DELETE FROM Interaction_Edit_History WHERE edit_id = ?",
                    (target_edit_id,),
                )

                conn.commit()
            except sqlite3.Error as e:
                logger.error(f"swap_interaction_version failed for id={interaction_id} k={k}: {e}")
                conn.rollback()
                raise

            cursor.execute(
                "SELECT COUNT(*) FROM Interaction_Edit_History WHERE interaction_id = ?",
                (interaction_id,),
            )
            total_archives = cursor.fetchone()[0]
            return {
                "current_content": target_content,
                "interaction_id": interaction_id,
                "total_versions": total_archives + 1,
            }

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
                      created_at: datetime, summary_level: Optional[int] = None,
                      parent_summary_id: Optional[int] = None) -> int:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # If summary_level is None, the DB default will be used (LEVEL_EPISODIC = 1)
            cols = ["segment_id", "content", "embedding", "model_name", "created_at"]
            vals = [segment_id, content, embedding, model_name, created_at]
            
            if summary_level is not None:
                cols.append("summary_level")
                vals.append(summary_level)
            if parent_summary_id is not None:
                cols.append("parent_summary_id")
                vals.append(parent_summary_id)
                
            placeholders = ", ".join("?" for _ in vals)
            col_names = ", ".join(cols)
            
            cursor.execute(
                f"INSERT INTO Memory_Summaries ({col_names}) VALUES ({placeholders})",
                vals
            )
            summary_id = cursor.lastrowid
            cursor.execute(
                """INSERT INTO vec_Memory_Summaries (summary_id, embedding)
                   VALUES (?, ?)""",
                (summary_id, embedding)
            )
            conn.commit()
            return summary_id  # type: ignore[return-value]

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
                                          model_name: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            query = ("SELECT ui.interaction_id, ui.author_role, ui.author_name, ui.content, ui.timestamp, me.embedding, ui.parent_summary_id, ui.reply_to_id"
                     " FROM User_Interactions ui JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id"
                     " WHERE ui.persona_name = ? AND ui.channel = ? AND ui.parent_summary_id IS NULL")
            params: List[Any] = [persona_name, channel]

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

            query += " ORDER BY ui.interaction_id ASC"
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

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
        now = datetime.now(timezone.utc)
        with self._lock:
            conn = self._get_connection()
            # Check for existing failure covering same range
            if server_id is not None:
                existing = conn.execute(
                    "SELECT failure_id, attempts FROM Segment_Failures"
                    " WHERE channel = ? AND persona_name = ? AND server_id = ?"
                    " AND start_interaction_id = ? AND end_interaction_id = ?",
                    (channel, persona_name, server_id, start_id, end_id),
                ).fetchone()
            else:
                existing = conn.execute(
                    "SELECT failure_id, attempts FROM Segment_Failures"
                    " WHERE channel = ? AND persona_name = ? AND server_id IS NULL"
                    " AND start_interaction_id = ? AND end_interaction_id = ?",
                    (channel, persona_name, start_id, end_id),
                ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE Segment_Failures SET attempts = ?, last_attempt_at = ?, error_reason = ? WHERE failure_id = ?",
                    (existing['attempts'] + 1, now, error_reason, existing['failure_id']),
                )
            else:
                conn.execute(
                    "INSERT INTO Segment_Failures"
                    " (channel, server_id, persona_name, start_interaction_id, end_interaction_id,"
                    "  message_count, attempts, last_attempt_at, error_reason)"
                    " VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)",
                    (channel, server_id, persona_name, start_id, end_id, message_count, now, error_reason),
                )
            conn.commit()

    def get_failed_segment_ranges(
            self,
            channel: str,
            persona_name: str,
            server_id: Optional[str] = None,
            max_attempts: int = 3,
            cooldown_hours: float = 24.0,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            cutoff = datetime.now(timezone.utc).timestamp() - (cooldown_hours * 3600)
            query = (
                "SELECT start_interaction_id, end_interaction_id, attempts, last_attempt_at, error_reason"
                " FROM Segment_Failures"
                " WHERE channel = ? AND persona_name = ?"
            )
            params: List[Any] = [channel, persona_name]
            if server_id is not None:
                query += " AND server_id = ?"
                params.append(server_id)
            else:
                query += " AND server_id IS NULL"
            # Still blocked: either under max attempts with cooldown, or at/over max attempts
            query += " AND (attempts >= ? OR last_attempt_at > ?)"
            params.extend([max_attempts, datetime.fromtimestamp(cutoff, tz=timezone.utc)])
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def clear_segment_failure(
            self,
            channel: str,
            persona_name: str,
            server_id: Optional[str],
            start_id: int,
            end_id: int,
    ) -> None:
        with self._lock:
            conn = self._get_connection()
            if server_id is not None:
                conn.execute(
                    "DELETE FROM Segment_Failures"
                    " WHERE channel = ? AND persona_name = ? AND server_id = ?"
                    " AND start_interaction_id = ? AND end_interaction_id = ?",
                    (channel, persona_name, server_id, start_id, end_id),
                )
            else:
                conn.execute(
                    "DELETE FROM Segment_Failures"
                    " WHERE channel = ? AND persona_name = ? AND server_id IS NULL"
                    " AND start_interaction_id = ? AND end_interaction_id = ?",
                    (channel, persona_name, start_id, end_id),
                )
            conn.commit()

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

            # q3: Channels with embedded messages that were never summarized (parent_summary_id IS NULL).
            # This catches historical messages that sit below existing segments' high-water mark.
            q3 = ("SELECT DISTINCT ui.channel, ui.persona_name, ui.server_id FROM User_Interactions ui"
                  " JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id"
                  " WHERE ui.content IS NOT NULL AND ui.content != ''"
                  " AND ui.parent_summary_id IS NULL AND me.model_name = ?"
                  + self._suppression_filter("ui"))
            params.append(active_model)

            query = q1 + " UNION " + q2 + " UNION " + q3
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
        """Retrieve summaries scoped by MemoryMode for retrieval-time fan-out."""
        if memory_mode == "ticket":
            return []

        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            dist_select = ""
            dist_params: List[Any] = []
            if query_embeddings:
                d_exprs = ["vec_distance_cosine(v.embedding, ?)"] * len(query_embeddings)
                if len(d_exprs) > 1:
                    dist_select = f", min({', '.join(d_exprs)}) as dist"
                else:
                    dist_select = f", {d_exprs[0]} as dist"
                dist_params.extend(query_embeddings)

            base_select = (
                f"SELECT ms.summary_id, ms.segment_id, ms.content, ms.embedding,"
                f" ms.model_name, ms.created_at, seg.channel, seg.persona_name,"
                f" seg.start_interaction_id, seg.end_interaction_id, seg.last_message_at{dist_select}"
                f" FROM Memory_Summaries ms"
                f" JOIN Memory_Segments seg ON ms.segment_id = seg.segment_id"
            )
            if query_embeddings:
                base_select += " JOIN vec_Memory_Summaries v ON ms.summary_id = v.summary_id"

            build_args = (memory_mode, channel, server_id, user_identifier,
                          exclude_after_interaction_id, model_name)

            # Build primary query
            where_clause, params = self._build_summary_where(persona_name, *build_args)
            query = f"{base_select} WHERE {where_clause}"
            final_params = dist_params + params

            # Ambient union
            if include_ambient and persona_name != "ambient":
                amb_where, amb_params = self._build_summary_where("ambient", *build_args)
                query += f" UNION {base_select} WHERE {amb_where}"
                final_params.extend(dist_params + amb_params)

            if query_embeddings:
                query += " ORDER BY dist ASC"
            
            # Log search parameters for diagnostics (exclude raw embeddings for brevity)
            if limit:
                query += " LIMIT ?"
                final_params.append(limit)

            safe_params = [f"<blob:{len(p)}b>" if isinstance(p, bytes) else p for p in final_params]
            logger.info(f"MemoryManager.retrieve: Executing query for {persona_name} (mode: {memory_mode}). Params: {safe_params}")

            cursor.execute(query, final_params)
            rows = [dict(row) for row in cursor.fetchall()]

            if query_embeddings:
                if not rows:
                    logger.info(f"MemoryManager.retrieve: 0 summaries found for {persona_name} in {channel} (mode={memory_mode})")
                else:
                    best_dist = rows[0].get('dist', 1.0)
                    logger.info(f"MemoryManager.retrieve: {len(rows)} summaries found (best similarity dist: {best_dist:.4f})")

            return rows
