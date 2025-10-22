# src/database/memory_manager.py

import sqlite3
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
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
            platform_message_id TEXT,
            server_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_channel_timestamp
        ON User_Interactions (channel, timestamp);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_platform_message_id
        ON User_Interactions (platform_message_id);

        CREATE TABLE IF NOT EXISTS Suppressed_Interactions (
            suppression_id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id INTEGER NOT NULL UNIQUE,
            suppressed_at TIMESTAMP NOT NULL,
            FOREIGN KEY (interaction_id) REFERENCES User_Interactions(interaction_id) ON DELETE CASCADE
        );
        """
        conn.executescript(schema_sql)

        # Step 2: Now that the table is guaranteed to exist, check for and add the new column.
        cursor.execute("PRAGMA table_info(User_Interactions)")
        columns = [row['name'] for row in cursor.fetchall()]

        if 'server_id' not in columns:
            conn.execute("ALTER TABLE User_Interactions ADD COLUMN server_id TEXT")
            logging.info("Added 'server_id' column to User_Interactions table.")

        # Create any new indexes that might be needed for the new column
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_server_id_timestamp
            ON User_Interactions (server_id, timestamp);
        """)

        conn.commit()
        logging.info("User memory database schema created or verified successfully.")

    def log_message(self, user_identifier: str, persona_name: str, channel: str,
                    author_role: str, author_name: Optional[str], content: str,
                    timestamp: datetime, server_id: Optional[str] = None,
                    platform_message_id: Optional[str] = None,
                    zammad_ticket_id: Optional[int] = None) -> None:
        """Logs a single message with its author's role and name."""
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
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT interaction_id FROM User_Interactions WHERE platform_message_id = ?",
                       (platform_message_id,))
        row = cursor.fetchone()
        if not row: return False
        interaction_id = row['interaction_id']
        now = datetime.now()
        try:
            cursor.execute("INSERT INTO Suppressed_Interactions (interaction_id, suppressed_at) VALUES (?, ?)",
                           (interaction_id, now))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_personal_history(self, user_identifier: str, persona_name: str, limit: Optional[int] = None) -> List[
        Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT interaction_id FROM Suppressed_Interactions")
        suppressed_ids: List[int] = [row['interaction_id'] for row in cursor.fetchall()]

        query = "SELECT author_role, author_name, content FROM User_Interactions WHERE user_identifier = ? AND persona_name = ?"
        params: List[Any] = [user_identifier, persona_name]

        if suppressed_ids:
            placeholders = ', '.join('?' for _ in suppressed_ids)
            query += f" AND interaction_id NOT IN ({placeholders})"
            params.extend(suppressed_ids)

        query += " ORDER BY timestamp DESC"
        if isinstance(limit, int):
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows: List[sqlite3.Row] = cursor.fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_ticket_history(self, ticket_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT interaction_id FROM Suppressed_Interactions")
        suppressed_ids: List[int] = [row['interaction_id'] for row in cursor.fetchall()]

        query = "SELECT author_role, author_name, content FROM User_Interactions WHERE zammad_ticket_id = ?"
        params: List[Any] = [ticket_id]

        if suppressed_ids:
            placeholders = ', '.join('?' for _ in suppressed_ids)
            query += f" AND interaction_id NOT IN ({placeholders})"
            params.extend(suppressed_ids)

        query += " ORDER BY timestamp DESC"
        if isinstance(limit, int):
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows: List[sqlite3.Row] = cursor.fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_channel_history(self, channel: str, persona_name: str, server_id: Optional[str] = None,
                            limit: Optional[int] = None) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT interaction_id FROM Suppressed_Interactions")
        suppressed_ids: List[int] = [row['interaction_id'] for row in cursor.fetchall()]

        query = "SELECT author_role, author_name, content FROM User_Interactions WHERE channel = ? AND persona_name = ?"
        params: List[Any] = [channel, persona_name]

        # This if/else block is the corrected logic that fixes the bug.
        if server_id is not None:
            query += " AND server_id = ?"
            params.append(server_id)
        else:
            query += " AND server_id IS NULL"

        if suppressed_ids:
            placeholders = ', '.join('?' for _ in suppressed_ids)
            query += f" AND interaction_id NOT IN ({placeholders})"
            params.extend(suppressed_ids)

        query += " ORDER BY timestamp DESC"
        if isinstance(limit, int):
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows: List[sqlite3.Row] = cursor.fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_server_history(self, server_id: str, persona_name: str, limit: Optional[int] = None) -> List[
        Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT interaction_id FROM Suppressed_Interactions")
        suppressed_ids: List[int] = [row['interaction_id'] for row in cursor.fetchall()]

        query = "SELECT author_role, author_name, content FROM User_Interactions WHERE server_id = ? AND persona_name = ?"
        params: List[Any] = [server_id, persona_name]

        if suppressed_ids:
            placeholders = ', '.join('?' for _ in suppressed_ids)
            query += f" AND interaction_id NOT IN ({placeholders})"
            params.extend(suppressed_ids)

        query += " ORDER BY timestamp DESC"
        if isinstance(limit, int):
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows: List[sqlite3.Row] = cursor.fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_global_history(self, persona_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT interaction_id FROM Suppressed_Interactions")
        suppressed_ids: List[int] = [row['interaction_id'] for row in cursor.fetchall()]

        query = "SELECT author_role, author_name, content FROM User_Interactions WHERE persona_name = ?"
        params: List[Any] = [persona_name]

        if suppressed_ids:
            placeholders = ', '.join('?' for _ in suppressed_ids)
            query += f" AND interaction_id NOT IN ({placeholders})"
            params.extend(suppressed_ids)

        query += " ORDER BY timestamp DESC"
        if isinstance(limit, int):
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows: List[sqlite3.Row] = cursor.fetchall()
        return [dict(row) for row in reversed(rows)]
