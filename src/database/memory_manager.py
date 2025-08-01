# src/database/memory_manager.py

import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# --- DATETIME <-> ISO 8601 STRING CONVERSION FOR SQLITE ---
def adapt_datetime_iso(dt_obj):
    """Adapt datetime.datetime to timezone-naive ISO 8601 format."""
    return dt_obj.isoformat()


def convert_timestamp_iso(ts_bytes):
    """Convert ISO 8601 format string from bytes to datetime.datetime object."""
    return datetime.fromisoformat(ts_bytes.decode('utf-8'))


sqlite3.register_adapter(datetime, adapt_datetime_iso)
sqlite3.register_converter("timestamp", convert_timestamp_iso)

# --- PATH LOGIC ---
DB_DIR = Path(__file__).resolve().parent
DATABASE_FILE = DB_DIR / "user_memory.db"


class MemoryManager:
    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initializes the MemoryManager.
        If db_path is None, it falls back to the DATABASE_FILE constant.
        """
        self.db_path = db_path if db_path is not None else str(DATABASE_FILE)
        self._conn: Optional[sqlite3.Connection] = None
        if self.db_path != ':memory:':
            DB_DIR.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """
        Returns a single, persistent database connection.
        Creates the connection on the first call.
        """
        if self._conn is None:
            # check_same_thread=False is required for this connection to be used
            # across different threads, which is what `asyncio.to_thread` does and
            # what happens in our pytest integration tests.
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
        """Creates the database schema, including tables for interactions and suppressions."""
        # NOTE: This is a breaking schema change. Old databases must be deleted.
        schema_sql = """
        CREATE TABLE IF NOT EXISTS User_Interactions (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_identifier TEXT NOT NULL,
            persona_name TEXT NOT NULL,
            channel TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
            content TEXT,
            timestamp TIMESTAMP NOT NULL,
            zammad_ticket_id INTEGER,
            platform_message_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_user_persona_ticket_timestamp
        ON User_Interactions (user_identifier, persona_name, zammad_ticket_id, timestamp);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_platform_message_id
        ON User_Interactions (platform_message_id);

        CREATE TABLE IF NOT EXISTS Suppressed_Interactions (
            suppression_id INTEGER PRIMARY KEY AUTOINCREMENT,
            interaction_id INTEGER NOT NULL UNIQUE,
            suppressed_at TIMESTAMP NOT NULL,
            FOREIGN KEY (interaction_id) REFERENCES User_Interactions(interaction_id) ON DELETE CASCADE
        );
        """
        conn = self._get_connection()
        conn.executescript(schema_sql)
        conn.commit()
        logging.info("User memory database schema created or verified successfully.")

    def log_message(self, user_identifier: str, persona_name: str, channel: str, role: str, content: str,
                    timestamp: datetime, platform_message_id: Optional[str] = None,
                    zammad_ticket_id: Optional[int] = None) -> None:
        """Logs a single message to the database."""
        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO User_Interactions 
            (user_identifier, persona_name, channel, role, content, timestamp, zammad_ticket_id, platform_message_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_identifier, persona_name, channel, role, content, timestamp, zammad_ticket_id, platform_message_id)
        )
        conn.commit()

    def suppress_message_by_platform_id(self, platform_message_id: str) -> bool:
        """Flags a message to be ignored in future context based on its platform ID."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT interaction_id FROM User_Interactions WHERE platform_message_id = ?",
                       (platform_message_id,))
        row = cursor.fetchone()

        if not row:
            logger.warning(f"Could not find message with platform_id '{platform_message_id}' to suppress.")
            return False

        interaction_id = row['interaction_id']
        now = datetime.now()

        try:
            cursor.execute("INSERT INTO Suppressed_Interactions (interaction_id, suppressed_at) VALUES (?, ?)",
                           (interaction_id, now))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Message with interaction_id {interaction_id} is already suppressed.")
            return False

    def get_personal_history(self, user_identifier: str, persona_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves the most recent non-suppressed messages for a given user and persona."""
        query = """
            SELECT T1.role, T1.content FROM User_Interactions AS T1
            LEFT JOIN Suppressed_Interactions AS T2 ON T1.interaction_id = T2.interaction_id
            WHERE T1.user_identifier = :user_identifier 
              AND T1.persona_name = :persona_name
              AND T2.suppression_id IS NULL
            ORDER BY T1.timestamp DESC
        """
        params = {'user_identifier': user_identifier, 'persona_name': persona_name}

        if isinstance(limit, int):
            query += " LIMIT :limit"
            params['limit'] = limit

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_ticket_history(self, ticket_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves all non-suppressed messages for a given Zammad ticket ID."""
        query = """
            SELECT T1.role, T1.content FROM User_Interactions AS T1
            LEFT JOIN Suppressed_Interactions AS T2 ON T1.interaction_id = T2.interaction_id
            WHERE T1.zammad_ticket_id = :ticket_id
              AND T2.suppression_id IS NULL
            ORDER BY T1.timestamp DESC
        """
        params = {'ticket_id': ticket_id}

        if isinstance(limit, int):
            query += " LIMIT :limit"
            params['limit'] = limit

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in reversed(rows)]
