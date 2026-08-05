# src/memory/memory_manager.py

import sqlite3
import hashlib
import json
import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import (Any, Coroutine, Dict, List, Callable, Optional, Sequence, Set, Union, cast,
                    Tuple, Generator)
from pathlib import Path

# --- NEW: Import the global embedding model variable ---
from config.global_config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, SEMANTIC_BACKEND, HINDSIGHT_URL
from src.memory.backend.base import MemoryHit, Experience, MentalModel, ReflectResult
from src.security.scrubber import get_scrubber
import sqlite_vec

logger = logging.getLogger(__name__)

# Sentinel for optional column updates: distinguishes "caller omitted this
# argument" (leave the column untouched) from "caller explicitly passed None"
# (write NULL). Used by update_interaction_content for tool_context and
# reasoning_content.
_UNSET: Any = object()

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

# --- Standing orders (DP-281) ---
# The store-level trust boundary: rows in Standing_Orders are injected
# verbatim into planner prompts, so writes are only accepted from
# authenticated operator surfaces (the single shared operator identity per
# DP-277 — matches OPERATOR_ID in src/proposals/service.py). Never add a
# model-facing or ticket-derived source here: that would open a prompt
# injection lane straight into the planner.
ALLOWED_STANDING_ORDER_SOURCES: "frozenset[str]" = frozenset({"operator"})
# Hard page cap for list_standing_orders; the table is never pruned.
MAX_STANDING_ORDER_PAGE = 200


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

            CREATE TABLE IF NOT EXISTS Proposals (
                proposal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP,
                agent_name TEXT NOT NULL,
                action_type TEXT NOT NULL,
                action_args TEXT NOT NULL,
                ticket_number INTEGER,
                rationale TEXT,
                taint TEXT,
                source_action_id INTEGER,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'approved', 'denied', 'expired',
                                     'executed', 'execution_failed', 'withdrawn')),
                reviewed_at TIMESTAMP,
                reviewer TEXT,
                review_note TEXT,
                executed_at TIMESTAMP,
                execution_result TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_proposal_status ON Proposals (status, created_at);
            CREATE INDEX IF NOT EXISTS idx_proposal_acceptance ON Proposals (agent_name, action_type, status);

            CREATE TABLE IF NOT EXISTS Parked_Writes (
                token TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'claimed', 'resolved',
                                     'expired', 'interrupted')),
                user_identifier TEXT NOT NULL,
                persona_name TEXT NOT NULL,
                channel TEXT NOT NULL DEFAULT '',
                server_id TEXT,
                write_call TEXT,
                call_identity TEXT NOT NULL,
                audit_info TEXT,
                confirmation_text TEXT NOT NULL DEFAULT '',
                turn_tainted INTEGER NOT NULL DEFAULT 0,
                parked_assistant_id INTEGER,
                duplicate_refs TEXT NOT NULL DEFAULT '[]',
                resolved_at REAL,
                resolution TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_parked_write_conversation
                ON Parked_Writes (user_identifier, persona_name, status, created_at);
            CREATE INDEX IF NOT EXISTS idx_parked_write_status
                ON Parked_Writes (status, created_at);

            CREATE TABLE IF NOT EXISTS Standing_Orders (
                order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP NOT NULL,
                source TEXT NOT NULL,
                agent TEXT NOT NULL DEFAULT 'managr',
                order_text TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active', 'retired')),
                retired_at TIMESTAMP,
                retire_note TEXT
            );
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

            # Standing_Orders migrations (DP-281): agent scope column so a
            # second planner never needs a schema change, and the index is
            # rebuilt to cover the per-agent injection query.
            cursor.execute("PRAGMA table_info(Standing_Orders)")
            standing_order_cols = {row['name'] for row in cursor.fetchall()}
            if 'agent' not in standing_order_cols:
                conn.execute("ALTER TABLE Standing_Orders ADD COLUMN agent TEXT NOT NULL DEFAULT 'managr'")
            conn.execute("DROP INDEX IF EXISTS idx_standing_order_status")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_standing_order_agent_status "
                "ON Standing_Orders (agent, status, created_at)"
            )

            # Proposals migrations (DP-290): self-managing queue needs the
            # dedup key column (ticket_number, extracted from action_args) and
            # the 'withdrawn' status. The status lives in a CHECK constraint,
            # which SQLite cannot ALTER — pre-DP-290 tables are rebuilt
            # (rename / copy / drop), same pattern as Memory_Summaries v2.
            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='Proposals'")
            proposals_sql = (cursor.fetchone() or {"sql": ""})["sql"] or ""
            if proposals_sql and ("ticket_number" not in proposals_sql
                                  or "'withdrawn'" not in proposals_sql):
                logger.info("Migrating Proposals to the DP-290 schema...")
                conn.execute("ALTER TABLE Proposals RENAME TO Proposals_old")
                conn.execute("""
                    CREATE TABLE Proposals (
                        proposal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TIMESTAMP NOT NULL,
                        expires_at TIMESTAMP,
                        agent_name TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        action_args TEXT NOT NULL,
                        ticket_number INTEGER,
                        rationale TEXT,
                        taint TEXT,
                        source_action_id INTEGER,
                        status TEXT NOT NULL DEFAULT 'pending'
                            CHECK(status IN ('pending', 'approved', 'denied', 'expired',
                                             'executed', 'execution_failed', 'withdrawn')),
                        reviewed_at TIMESTAMP,
                        reviewer TEXT,
                        review_note TEXT,
                        executed_at TIMESTAMP,
                        execution_result TEXT
                    )
                """)
                conn.execute("""
                    INSERT INTO Proposals
                        (proposal_id, created_at, expires_at, agent_name, action_type,
                         action_args, rationale, taint, source_action_id, status,
                         reviewed_at, reviewer, review_note, executed_at, execution_result)
                    SELECT proposal_id, created_at, expires_at, agent_name, action_type,
                           action_args, rationale, taint, source_action_id, status,
                           reviewed_at, reviewer, review_note, executed_at, execution_result
                    FROM Proposals_old
                """)
                conn.execute("DROP TABLE Proposals_old")
                # The rename dragged the indexes to Proposals_old and DROP
                # TABLE took them with it; this cycle's executescript already
                # ran, so recreate them here.
                conn.execute("CREATE INDEX IF NOT EXISTS idx_proposal_status "
                             "ON Proposals (status, created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_proposal_acceptance "
                             "ON Proposals (agent_name, action_type, status)")

            # Backfill the dedup key from action_args (deterministic pull in
            # code — same extraction create_proposal uses; no JSON1 reliance).
            cursor.execute(
                "SELECT proposal_id, action_args FROM Proposals WHERE ticket_number IS NULL")
            for row in cursor.fetchall():
                try:
                    args = json.loads(row["action_args"] or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                ticket_number = self._proposal_ticket_number(
                    args if isinstance(args, dict) else None)
                if ticket_number is not None:
                    conn.execute(
                        "UPDATE Proposals SET ticket_number = ? WHERE proposal_id = ?",
                        (ticket_number, row["proposal_id"]))

            # Collapse pre-existing duplicate pending rows before the unique
            # index can exist: keep the OLDEST row (stable id the operator may
            # already have seen) carrying the NEWEST duplicate's content —
            # exactly what upsert-supersede would have produced.
            cursor.execute("""
                SELECT agent_name, action_type, ticket_number FROM Proposals
                WHERE status = 'pending' AND ticket_number IS NOT NULL
                GROUP BY agent_name, action_type, ticket_number HAVING COUNT(*) > 1
            """)
            for dup in cursor.fetchall():
                dup_cursor = conn.execute(
                    """SELECT proposal_id, action_args, rationale, taint,
                              source_action_id, expires_at
                       FROM Proposals
                       WHERE status = 'pending' AND agent_name = ?
                         AND action_type = ? AND ticket_number = ?
                       ORDER BY proposal_id""",
                    (dup["agent_name"], dup["action_type"], dup["ticket_number"]))
                rows = [dict(r) for r in dup_cursor.fetchall()]
                keeper, newest = rows[0], rows[-1]
                conn.execute(
                    """UPDATE Proposals SET action_args = ?, rationale = ?, taint = ?,
                           source_action_id = ?, expires_at = ?
                       WHERE proposal_id = ?""",
                    (newest["action_args"], newest["rationale"], newest["taint"],
                     newest["source_action_id"], newest["expires_at"],
                     keeper["proposal_id"]))
                for stale_row in rows[1:]:
                    conn.execute(
                        """UPDATE Proposals SET status = 'withdrawn', reviewed_at = ?,
                               reviewer = ?, review_note = ?
                           WHERE proposal_id = ?""",
                        (self._utc_stamp(datetime.now(timezone.utc)),
                         dup["agent_name"],
                         f"superseded by duplicate pending proposal "
                         f"{keeper['proposal_id']} (DP-290 dedup migration)",
                         stale_row["proposal_id"]))

            # The storage-layer dedup guarantee: at most one pending proposal
            # per (agent, action_type, ticket). Partial so terminal rows keep
            # full history and keyless actions stay unconstrained.
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_proposal_pending_key "
                "ON Proposals (agent_name, action_type, ticket_number) "
                "WHERE status = 'pending' AND ticket_number IS NOT NULL")

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
            #
            # Empty-content rows are excluded too: DP-296 parks write an
            # assistant row carrying only tool_context, which renders nowhere.
            # Left in the lookup it would win "most recent", archive a blank
            # version into the chevron, and hand the retried turn the very row
            # the still-open PendingConfirmation points at — so approving that
            # park later would wipe the *retried* turn's tool context.
            cursor.execute(
                "SELECT interaction_id, content, reasoning_content, author_role"
                " FROM User_Interactions"
                " WHERE persona_name = ? AND user_identifier = ? AND channel = ?"
                + self._SUPPRESSION_SUBQUERY
                + self._NON_EMPTY_CONTENT_FILTER +
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
                # Content-hash dedupe (mirrors swap_interaction_version): if an
                # archive row with the same (interaction_id, old_content) already
                # exists, skip the insert. Duplicate-content archives would make
                # list_interaction_versions flag multiple rows canonical, so the
                # chevron's findIndex(canonical) picks the wrong index (DP-132 #7).
                cursor.execute(
                    "SELECT 1 FROM Interaction_Edit_History"
                    " WHERE interaction_id = ? AND old_content = ? LIMIT 1",
                    (interaction_id, old_content),
                )
                if cursor.fetchone() is None:
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

    def set_tool_context(self, interaction_id: int,
                         tool_context: Optional[str]) -> bool:
        """Write a row's `tool_context` column, leaving its content untouched.

        Deliberately narrower than `update_interaction_content` — no content
        rewrite, no `parent_summary_id` reset, no embedding invalidation, since
        the row's text has not changed. Two callers need exactly that:

        * a resume clearing the parked row's provisional copy (`None`), because
          the continuation re-seals the same span with real results and the
          model would otherwise see every call twice;
        * a *retried* turn that parks (DP-296), which must attach the sealed
          context to the archived assistant row without blanking the prior
          attempt's text that row still renders.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "UPDATE User_Interactions SET tool_context = ? "
                    "WHERE interaction_id = ?",
                    (tool_context, interaction_id),
                )
                conn.commit()
                return cursor.rowcount > 0
            except sqlite3.Error as e:
                logger.error(
                    f"set_tool_context failed for id={interaction_id}: {e}"
                )
                conn.rollback()
                return False

    def get_tool_context(self, interaction_id: int) -> Optional[str]:
        """Read one row's raw `tool_context` JSON, or None if it has none.

        The read half of `set_tool_context`. Needed by DP-297 to patch a single
        gated write's entry inside an already-committed row when the operator
        approves or denies it — a read-modify-write of the blob, since the
        column stores the whole sealed span rather than one row per call.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT tool_context FROM User_Interactions "
                    "WHERE interaction_id = ?",
                    (interaction_id,),
                )
                row = cursor.fetchone()
            except sqlite3.Error as e:
                logger.error(
                    f"get_tool_context failed for id={interaction_id}: {e}"
                )
                return None
        if row is None:
            return None
        value = row[0]
        return str(value) if value is not None else None

    def clear_tool_context(self, interaction_id: int) -> bool:
        """Drop a row's stored `tool_context`, leaving its content untouched."""
        return self.set_tool_context(interaction_id, None)

    def update_interaction_content(self, interaction_id: int, new_content: str,
                                   reasoning_content: Any = _UNSET,
                                   tool_context: Any = _UNSET) -> bool:
        """Overwrite the content of an existing interaction row in place.

        Used by portal retry and portal manual-edit flows. Clears
        `parent_summary_id` so the next summarizer pass re-groups the row, and
        drops the stale L0 embedding (`Message_Embeddings` + `vec_*`) so
        `MemoryAgent._embed_unembedded` re-encodes against the new content.

        `reasoning_content` and `tool_context` both follow the `_UNSET` sentinel
        contract: omitted = leave the column untouched, explicit ``None`` =
        clear it (write NULL), an explicit value = set it. A manual edit
        (DP-141) sends only the new body and must NOT erase the row's stored
        `<think>` reasoning, so it omits the arg. A regenerated (retry) turn
        passes the column explicitly — fresh reasoning, or ``None`` to clear the
        stale reasoning that no longer matches the new content. Likewise a retry
        rewrites `tool_context` to stay paired with its (possibly different) set
        of tool calls.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                set_clauses = ["content = ?"]
                params: List[Any] = [new_content]
                if reasoning_content is not _UNSET:
                    set_clauses.append("reasoning_content = ?")
                    params.append(reasoning_content)
                if tool_context is not _UNSET:
                    set_clauses.append("tool_context = ?")
                    params.append(tool_context)
                set_clauses.append("parent_summary_id = NULL")
                params.append(interaction_id)
                cursor.execute(
                    f"UPDATE User_Interactions SET {', '.join(set_clauses)} WHERE interaction_id = ?",
                    params,
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

    # Rows whose content is empty/whitespace. Only DP-296 park rows reach this
    # shape — both the user and assistant commit paths refuse empty text
    # otherwise — so this is precisely "the tool-context-only row".
    _NON_EMPTY_CONTENT_FILTER = " AND TRIM(COALESCE(content, '')) != ''"

    # A row that contributes nothing to a replayed history: no text to show the
    # model and no tool span to replay. A park whose tool_context was cleared on
    # resume is exactly that, and the history getters slice a raw `LIMIT N`, so
    # leaving it in would permanently cost one slot of the model's window per
    # resolved park (DP-296). Kept as a filter rather than a DELETE: on a
    # *retried* park the same row is the canonical archived assistant turn.
    _CONTRIBUTES_TO_HISTORY = (
        " AND (TRIM(COALESCE(content, '')) != '' OR tool_context IS NOT NULL)"
    )

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
                     " WHERE user_identifier = ? AND persona_name = ?"
                     + self._SUPPRESSION_SUBQUERY + self._CONTRIBUTES_TO_HISTORY)
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
                     " WHERE zammad_ticket_id = ?"
                     + self._SUPPRESSION_SUBQUERY + self._CONTRIBUTES_TO_HISTORY)
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
            query += self._CONTRIBUTES_TO_HISTORY
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
                         " WHERE server_id = ? AND persona_name = ?"
                         + self._SUPPRESSION_SUBQUERY + self._CONTRIBUTES_TO_HISTORY)
                params: List[Any] = [server_id, persona_name]
            else:
                query = ("SELECT interaction_id, author_role, author_name, content, tool_context, reasoning_content FROM User_Interactions"
                         " WHERE server_id IS NULL AND persona_name = ?"
                         + self._SUPPRESSION_SUBQUERY + self._CONTRIBUTES_TO_HISTORY)
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
                     " WHERE persona_name = ?"
                     + self._SUPPRESSION_SUBQUERY + self._CONTRIBUTES_TO_HISTORY)
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
            # Group by channel ONLY (not channel+server_id): one logical channel
            # logged under both a NULL and a non-NULL server_id would otherwise
            # return two rows and render the same channel twice with split counts
            # (DP-132 #8). MAX(server_id) picks a representative for the rare
            # multi-server case; the UI switches by channel name regardless.
            query = (
                "SELECT channel, MAX(server_id) AS server_id, COUNT(*) AS count,"
                " MAX(timestamp) AS last_ts FROM User_Interactions"
                " WHERE channel IS NOT NULL" + self._SUPPRESSION_SUBQUERY
            )
            params: List[Any] = []
            if persona_name is not None:
                query += " AND persona_name = ?"
                params.append(persona_name)
            query += " GROUP BY channel ORDER BY last_ts DESC"
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

    # --- Proposal queue (DP-282, managr Phase 1) ---

    @staticmethod
    def _utc_stamp(dt: datetime) -> str:
        """Format a datetime for TIMESTAMP columns (CURRENT_TIMESTAMP style,
        UTC, second precision). The connection uses PARSE_DECLTYPES, whose
        default converter can't parse isoformat offsets — a bound datetime
        object with tzinfo makes the row unreadable when microsecond == 0.
        Never bind datetime objects into TIMESTAMP columns directly."""
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _proposal_ticket_number(action_args: Optional[Dict[str, Any]]) -> Optional[int]:
        """Deterministic dedup-key extraction (DP-290): the ticket_number arg
        when it is a real int, else None (no key — the row dedups by nothing).
        bool is excluded like everywhere else in the proposal schema."""
        value = (action_args or {}).get("ticket_number")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    def create_proposal(self, agent_name: str, action_type: str, action_args: Dict[str, Any],
                        rationale: Optional[str] = None, taint: Optional[Dict[str, Any]] = None,
                        source_action_id: Optional[int] = None,
                        expires_at: Optional[datetime] = None) -> int:
        """Insert a pending proposal row. action_args/taint are stored as JSON.

        DP-290 dedup: if a pending row with the same (agent_name, action_type,
        ticket_number) already exists, that row is SUPERSEDED in place — it
        keeps its proposal_id and created_at but takes the new args, rationale,
        taint and expiry. One durable row per (ticket, action), always carrying
        the latest reasoning, regardless of how often a cycle re-fires."""
        ticket_number = self._proposal_ticket_number(action_args)
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO Proposals (created_at, expires_at, agent_name, action_type,
                                          action_args, ticket_number, rationale, taint,
                                          source_action_id, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                   ON CONFLICT (agent_name, action_type, ticket_number)
                   WHERE status = 'pending' AND ticket_number IS NOT NULL
                   DO UPDATE SET action_args = excluded.action_args,
                                 rationale = excluded.rationale,
                                 taint = excluded.taint,
                                 source_action_id = excluded.source_action_id,
                                 expires_at = excluded.expires_at""",
                (self._utc_stamp(datetime.now(timezone.utc)),
                 self._utc_stamp(expires_at) if expires_at else None, agent_name, action_type,
                 json.dumps(action_args), ticket_number, rationale,
                 json.dumps(taint) if taint is not None else None, source_action_id),
            )
            conn.commit()
            if ticket_number is None:
                return cast(int, cursor.lastrowid)
            # lastrowid is meaningless after DO UPDATE; the dedup key is
            # unique among pending rows, so this lookup is exact either way.
            cursor.execute(
                """SELECT proposal_id FROM Proposals
                   WHERE agent_name = ? AND action_type = ? AND ticket_number = ?
                     AND status = 'pending'""",
                (agent_name, action_type, ticket_number))
            return int(cursor.fetchone()["proposal_id"])

    def reaffirm_proposal(self, proposal_id: int, agent_name: str,
                          expires_at: Optional[datetime] = None) -> bool:
        """Reset a pending proposal's expiry (DP-290 'reaffirm'): the agent
        confirmed it still stands, so the TTL clock restarts. Scoped to the
        agent's own pending rows; returns False when nothing matched."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE Proposals SET expires_at = ?
                   WHERE proposal_id = ? AND status = 'pending' AND agent_name = ?""",
                (self._utc_stamp(expires_at) if expires_at else None,
                 proposal_id, agent_name),
            )
            conn.commit()
            return cursor.rowcount > 0

    def revise_proposal(self, proposal_id: int, agent_name: str,
                        action_args: Dict[str, Any], rationale: Optional[str] = None,
                        taint: Optional[Dict[str, Any]] = None,
                        expires_at: Optional[datetime] = None) -> bool:
        """Replace a pending proposal's args/rationale in place (DP-290
        'revise'), re-deriving the dedup key from the new args. Returns False
        when the row is not the agent's own pending row, or when the revision
        would collide with a DIFFERENT pending row on the dedup key — the
        caller keeps the original row untouched in that case."""
        ticket_number = self._proposal_ticket_number(action_args)
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """UPDATE Proposals SET action_args = ?, ticket_number = ?,
                           rationale = ?, taint = ?, expires_at = ?
                       WHERE proposal_id = ? AND status = 'pending' AND agent_name = ?""",
                    (json.dumps(action_args), ticket_number, rationale,
                     json.dumps(taint) if taint is not None else None,
                     self._utc_stamp(expires_at) if expires_at else None,
                     proposal_id, agent_name),
                )
            except sqlite3.IntegrityError:
                conn.rollback()
                return False
            conn.commit()
            return cursor.rowcount > 0

    def withdraw_proposal(self, proposal_id: int, agent_name: str,
                          note: Optional[str] = None) -> bool:
        """Move a PENDING proposal to 'withdrawn' (DP-290): the proposing
        agent retracting its own suggestion — deliberately distinct from
        operator 'denied' so denial-learning stays an operator-only signal.
        Scoped to the agent's own rows; returns False when nothing matched."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE Proposals SET status = 'withdrawn', reviewed_at = ?,
                       reviewer = ?, review_note = ?
                   WHERE proposal_id = ? AND status = 'pending' AND agent_name = ?""",
                (self._utc_stamp(datetime.now(timezone.utc)), agent_name, note,
                 proposal_id, agent_name),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_proposal(self, proposal_id: int) -> Optional[Dict[str, Any]]:
        """Fetch one proposal with action_args/taint decoded from JSON."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM Proposals WHERE proposal_id = ?", (proposal_id,))
            row = cursor.fetchone()
            return self._decode_proposal(dict(row)) if row else None

    def list_proposals(self, status: Optional[Union[str, Sequence[str]]] = "pending",
                       limit: int = 25,
                       agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List proposals, newest first. status accepts a single status, a
        sequence of statuses, or None for all; agent_name optionally filters
        to one proposing agent (filtered in SQL so the limit isn't consumed
        by other agents' rows)."""
        clauses: List[str] = []
        params: List[Any] = []
        if status is not None:
            statuses = [status] if isinstance(status, str) else list(status)
            clauses.append(f"status IN ({', '.join('?' * len(statuses))})")
            params.extend(statuses)
        if agent_name is not None:
            clauses.append("agent_name = ?")
            params.append(agent_name)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM Proposals{where} ORDER BY created_at DESC LIMIT ?",
                (*params, limit))
            return [self._decode_proposal(dict(row)) for row in cursor.fetchall()]

    def review_proposal(self, proposal_id: int, status: str, reviewer: str,
                        review_note: Optional[str] = None) -> bool:
        """Move a PENDING proposal to approved/denied. Returns False if the
        proposal is missing or not pending (already reviewed/expired) — the
        WHERE clause makes double-review a no-op rather than an overwrite."""
        if status not in ("approved", "denied"):
            raise ValueError(f"Invalid review status: {status}")
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE Proposals SET status = ?, reviewed_at = ?, reviewer = ?, review_note = ?
                   WHERE proposal_id = ? AND status = 'pending'""",
                (status, self._utc_stamp(datetime.now(timezone.utc)), reviewer, review_note,
                 proposal_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def mark_proposal_executed(self, proposal_id: int, success: bool, result: str) -> None:
        """Record execution outcome of an approved proposal."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE Proposals SET status = ?, executed_at = ?, execution_result = ?
                   WHERE proposal_id = ? AND status = 'approved'""",
                ("executed" if success else "execution_failed",
                 self._utc_stamp(datetime.now(timezone.utc)), result[:2000], proposal_id),
            )
            conn.commit()

    def expire_stale_proposals(self) -> int:
        """Mark pending proposals past their expires_at as expired. Returns count."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE Proposals SET status = 'expired'
                   WHERE status = 'pending' AND expires_at IS NOT NULL AND expires_at < ?""",
                (self._utc_stamp(datetime.now(timezone.utc)),),
            )
            conn.commit()
            return cursor.rowcount

    @staticmethod
    def _decode_proposal(row: Dict[str, Any]) -> Dict[str, Any]:
        for key in ("action_args", "taint"):
            if row.get(key):
                try:
                    row[key] = json.loads(row[key])
                except (TypeError, json.JSONDecodeError):
                    pass
        return row

    # ---- parked writes (DP-319) -----------------------------------------
    #
    # Durable backing for `ConfirmationManager`. Before DP-319 the pending set
    # was a process-lifetime dict, so the effective TTL was
    # `min(PENDING_ACTION_TTL, uptime)` — a 24h promise on a store that dies
    # with the process. These rows are what make the promise keepable.
    #
    # Deliberately NOT the `Proposals` table. `ProposalExecutor` dispatches a
    # whitelist of `action_type`s and that whitelist IS the ADR-2026-07-04
    # privilege separation; a chat park is an *arbitrary* tool call, so sharing
    # storage would put rows shaped like "call any tool" one executor bug away
    # from that whitelist. Separate table, separate executor, same DB.
    #
    # Secret handling: `write_call` holds the real argument values, because the
    # approved call must execute with them and not with a literal "[REDACTED]"
    # (same reason `Audit_Log` scrubs at its sink instead of at its callers).
    # That is why `finalize_parked_write` NULLs the payload columns the moment
    # the park reaches a terminal state — a decided park keeps only its hashed
    # identity, so the duplicate guard still works and the args stop living on
    # disk for the rest of the row's retention.

    @staticmethod
    def parked_write_identity(name: str, canonical_args: str) -> str:
        """Hash of `write_call_identity` output, for storage.

        Hashed rather than stored raw because the canonical args are the same
        secret-bearing payload `finalize_parked_write` exists to erase — a
        plaintext identity column would quietly preserve what that erasure
        removes. Equality is all the duplicate guard needs.
        """
        digest = hashlib.sha256()
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(canonical_args.encode("utf-8"))
        return digest.hexdigest()

    def insert_parked_write(self, *, token: str, created_at: float,
                            user_identifier: str, persona_name: str,
                            channel: str, server_id: Optional[str],
                            write_call: Dict[str, Any], call_identity: str,
                            audit_info: Dict[str, Any],
                            confirmation_text: str, turn_tainted: bool,
                            parked_assistant_id: Optional[int],
                            duplicate_refs: List[Any]) -> bool:
        """Persist a newly parked write as `pending`. Returns False on failure.

        `INSERT OR REPLACE` so a token re-registered after a restore is not an
        error; tokens are uuid4 hex, so a genuine collision is not the case
        being handled.

        `write_call` is serialized STRICTLY — no `default=str` fallback. It is
        the payload an approved write executes with after a restart, so a
        lossy encoding would mean running the tool with the repr of an argument
        instead of the argument. A call that cannot round-trip is therefore not
        persisted at all: it stays live in memory for this process (exactly the
        pre-DP-319 behaviour) rather than being stored in a form that could
        execute something other than what the operator approved. `audit_info`
        is advisory, so it degrades instead of blocking.
        """
        try:
            write_call_json = json.dumps(write_call)
        except (TypeError, ValueError) as e:
            logger.error(
                "Parked write %s has unserializable arguments (%s); it will "
                "not survive a restart. Refusing to store a lossy copy.",
                token, e,
            )
            return False
        try:
            audit_json = json.dumps(audit_info)
        except (TypeError, ValueError):
            audit_json = json.dumps(audit_info, default=str)
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO Parked_Writes
                       (token, created_at, status, user_identifier, persona_name,
                        channel, server_id, write_call, call_identity, audit_info,
                        confirmation_text, turn_tainted, parked_assistant_id,
                        duplicate_refs)
                       VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (token, created_at, user_identifier, persona_name, channel,
                     server_id, write_call_json, call_identity,
                     audit_json, confirmation_text,
                     1 if turn_tainted else 0, parked_assistant_id,
                     json.dumps(duplicate_refs)),
                )
                conn.commit()
                return True
            except sqlite3.Error as e:
                logger.error(f"Failed to persist parked write {token}: {e}")
                conn.rollback()
                return False

    def update_parked_write_refs(self, token: str, *,
                                 parked_assistant_id: Optional[int] = None,
                                 duplicate_refs: Optional[List[Any]] = None) -> bool:
        """Patch the history-pointer columns of a live park.

        Both are learned after the row exists: `parked_assistant_id` when the
        turn commits, `duplicate_refs` whenever a later turn re-proposes the
        same action. Scoped to non-terminal rows so a late duplicate cannot
        resurrect pointers on a park that was already decided.
        """
        sets: List[str] = []
        values: List[Any] = []
        if parked_assistant_id is not None:
            sets.append("parked_assistant_id = ?")
            values.append(parked_assistant_id)
        if duplicate_refs is not None:
            sets.append("duplicate_refs = ?")
            values.append(json.dumps(duplicate_refs))
        if not sets:
            return False
        values.append(token)
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                f"""UPDATE Parked_Writes SET {', '.join(sets)}
                    WHERE token = ? AND status IN ('pending', 'claimed')""",
                tuple(values),
            )
            conn.commit()
            return cursor.rowcount > 0

    def claim_parked_write(self, token: str) -> bool:
        """`pending` -> `claimed`. False when the row is missing or not pending.

        The conditional UPDATE is the durable half of `ConfirmationManager.take`
        — it is what stops a park surviving a restart from being resolvable
        twice, in the window where the in-memory index has been rebuilt but a
        stale surface still holds the old affordance.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE Parked_Writes SET status = 'claimed'
                   WHERE token = ? AND status = 'pending'""",
                (token,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def release_parked_write(self, token: str) -> bool:
        """`claimed` -> `pending`, for a claim that turned out to be invalid."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE Parked_Writes SET status = 'pending'
                   WHERE token = ? AND status = 'claimed'""",
                (token,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def finalize_parked_write(self, token: str, status: str,
                              resolution: Optional[str] = None,
                              now: Optional[float] = None) -> bool:
        """Move a park to a terminal state and erase its payload columns.

        The row is kept rather than deleted: a *resolved* park is what lets the
        duplicate guard recognize a re-proposal of an action that already ran,
        which a deleted row cannot do (the guard would see nothing pending, park
        a fresh copy, and an approval would execute the write a second time).

        What is kept is only `call_identity` — a hash — plus the outcome. The
        arguments, the audit blob and the operator-facing confirmation text all
        go to NULL here, so the decided park stops holding anything sensitive.
        """
        if status not in ("resolved", "expired", "interrupted"):
            raise ValueError(f"Not a terminal park status: {status}")
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE Parked_Writes
                   SET status = ?, resolved_at = ?, resolution = ?,
                       write_call = NULL, audit_info = NULL,
                       confirmation_text = ''
                   WHERE token = ? AND status IN ('pending', 'claimed')""",
                (status, time.time() if now is None else now,
                 resolution, token),
            )
            conn.commit()
            return cursor.rowcount > 0

    def load_parked_writes(self, statuses: Sequence[str]) -> List[Dict[str, Any]]:
        """Every park in the given states, oldest first, payloads decoded."""
        if not statuses:
            return []
        placeholders = ", ".join("?" for _ in statuses)
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                f"""SELECT * FROM Parked_Writes WHERE status IN ({placeholders})
                    ORDER BY created_at ASC""",
                tuple(statuses),
            )
            rows = [dict(r) for r in cursor.fetchall()]
        for row in rows:
            for key in ("write_call", "audit_info", "duplicate_refs"):
                if row.get(key):
                    try:
                        row[key] = json.loads(row[key])
                    except (TypeError, json.JSONDecodeError):
                        row[key] = None
        return rows

    def find_resolved_parked_write(self, user_identifier: str,
                                   persona_name: str, call_identity: str,
                                   since: float,
                                   resolutions: Sequence[str]) -> Optional[Dict[str, Any]]:
        """Most recent decided park matching this call identity.

        Both filters are load-bearing. `since` because "this action was decided
        at some point in history" is no reason to suppress a fresh proposal — a
        user who asks for the same write again next week means it. `resolutions`
        because only an outcome where the tool actually RAN makes a second park
        a double execution; a denied one executed nothing, so re-proposing it is
        a new request (DP-297 supports that case explicitly).
        """
        if not resolutions:
            return None
        placeholders = ", ".join("?" for _ in resolutions)
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                f"""SELECT token, status, resolution, resolved_at
                    FROM Parked_Writes
                    WHERE user_identifier = ? AND persona_name = ?
                      AND call_identity = ? AND status = 'resolved'
                      AND resolution IN ({placeholders})
                      AND resolved_at IS NOT NULL AND resolved_at >= ?
                    ORDER BY resolved_at DESC LIMIT 1""",
                (user_identifier, persona_name, call_identity,
                 *resolutions, since),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def purge_parked_writes(self, before: float) -> int:
        """Drop terminal park rows older than `before`. Returns how many.

        Terminal rows are retained only to answer the duplicate guard, whose
        window is bounded, so nothing needs them forever. Without this the
        table is the one park structure that grows without limit.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """DELETE FROM Parked_Writes
                   WHERE status IN ('resolved', 'expired', 'interrupted')
                     AND resolved_at IS NOT NULL AND resolved_at < ?""",
                (before,),
            )
            conn.commit()
            return cursor.rowcount

    def add_standing_order(self, order_text: str, source: str,
                           agent: str = "managr") -> int:
        """Insert an active standing order (DP-281). `source` records the
        operator surface it entered through — orders must only ever originate
        from authenticated operator surfaces, never from model output. That
        trust boundary is enforced here at the store, not just at the tool
        handler: an unrecognized source is rejected so no future caller can
        silently turn model output into planner guidance."""
        if source not in ALLOWED_STANDING_ORDER_SOURCES:
            raise ValueError(
                f"Standing order source '{source}' is not an authenticated "
                f"operator surface (allowed: {sorted(ALLOWED_STANDING_ORDER_SOURCES)})."
            )
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO Standing_Orders (created_at, source, agent, order_text, status)
                   VALUES (?, ?, ?, ?, 'active')""",
                (self._utc_stamp(datetime.now(timezone.utc)), source, agent, order_text),
            )
            conn.commit()
            return cast(int, cursor.lastrowid)

    def list_standing_orders(self, status: Optional[str] = "active",
                             limit: int = 50,
                             agent: Optional[str] = None) -> List[Dict[str, Any]]:
        """List standing orders, newest first. status=None returns all
        statuses; agent=None returns all agents' orders. Rejects a
        non-positive limit (SQLite reads LIMIT -1 as unbounded) and caps it,
        since orders are never deleted and the table grows monotonically."""
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")
        limit = min(limit, MAX_STANDING_ORDER_PAGE)
        clauses: List[str] = []
        params: List[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if agent is not None:
            clauses.append("agent = ?")
            params.append(agent)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM Standing_Orders{where} ORDER BY created_at DESC, order_id DESC LIMIT ?",
                (*params, limit))
            return [dict(row) for row in cursor.fetchall()]

    def retire_standing_order(self, order_id: int, note: Optional[str] = None) -> bool:
        """Retire an ACTIVE standing order. Returns False when the order is
        missing or already retired. Orders are never deleted — the retired
        row (with note) keeps the operator-guidance history auditable."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE Standing_Orders SET status = 'retired', retired_at = ?, retire_note = ?
                   WHERE order_id = ? AND status = 'active'""",
                (self._utc_stamp(datetime.now(timezone.utc)), note, order_id),
            )
            conn.commit()
            return cursor.rowcount > 0

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
        """Internal helper to log security-relevant events to Audit_Log.

        Egress scrub (DP-225, audit sink): `metadata` and `reason` are the two
        free-text fields here, and unlike every other scrub boundary this one
        writes to disk *permanently* — an Audit_Log row outlives the process,
        the conversation, and any TTL.

        Redaction happens at this sink rather than at each call site because
        callers legitimately hold raw secrets: `ConfirmationManager` carries the
        approved write's real argument values (it must, or the tool would
        execute with a literal "[REDACTED]"), and `proposals/service` carries
        `action_args`. Scrubbing here means a caller cannot leak by forgetting,
        and a future caller inherits the protection instead of re-introducing
        the hole.

        Recursive dict scrub rather than scrubbing the serialized JSON: the
        pattern fallback is skipped on strings past MAX_PATTERN_SCAN_LEN, so
        scrubbing field-by-field keeps unregistered-shape detection alive on a
        large metadata blob that would exceed the limit once flattened.
        """
        now = datetime.now()
        scrubber = get_scrubber()
        safe_metadata = (
            cast(Dict[str, Any], scrubber.scrub(metadata)) if metadata else None
        )
        safe_reason = cast(Optional[str], scrubber.scrub(reason)) if reason else reason
        meta_json = json.dumps(safe_metadata) if safe_metadata else None

        cursor.execute(
            """INSERT INTO Audit_Log
               (event_type, target_id, operator_id, timestamp, prior_state, new_state, reason, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_type, target_id, operator_id, now, prior_state, new_state, safe_reason, meta_json)
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
