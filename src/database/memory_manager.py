# src/database/memory_manager.py

import sqlite3
import logging
from datetime import datetime
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
        """Creates the user interactions table if it doesn't exist."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS User_Interactions (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_identifier TEXT NOT NULL,
            channel TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            user_message TEXT,
            bot_response TEXT,
            zammad_ticket_id INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_user_identifier_timestamp
        ON User_Interactions (user_identifier, timestamp);
        """
        conn = self._get_connection()
        conn.executescript(schema_sql)
        conn.commit()
        logging.info("User memory database schema created or verified successfully.")

    def log_interaction(self, user_identifier: str, channel: str, user_message: str, bot_response: str,
                        zammad_ticket_id: Optional[int] = None) -> None:
        """Logs a complete user-bot interaction."""
        now = datetime.now()
        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO User_Interactions 
            (user_identifier, channel, timestamp, user_message, bot_response, zammad_ticket_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_identifier, channel, now, user_message, bot_response, zammad_ticket_id)
        )
        conn.commit()

    def get_history(self, user_identifier: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves the most recent interactions for a given user."""
        query = """
            SELECT user_message, bot_response FROM User_Interactions
            WHERE user_identifier = :user_identifier
            ORDER BY timestamp DESC
        """
        params = {'user_identifier': user_identifier}

        if limit is not None and limit > 0:
            query += " LIMIT :limit"
            params['limit'] = limit

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        # We want chronological order for the prompt, so we reverse the DESC query result
        rows = cursor.fetchall()
        return [dict(row) for row in reversed(rows)]
