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
        """Creates the user interactions table with a message-centric schema."""
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
            zammad_ticket_id INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_user_persona_ticket_timestamp
        ON User_Interactions (user_identifier, persona_name, zammad_ticket_id, timestamp);
        """
        conn = self._get_connection()
        conn.executescript(schema_sql)
        conn.commit()
        logging.info("User memory database schema created or verified successfully.")

    def log_interaction(self, user_identifier: str, persona_name: str, channel: str, user_message: str, bot_response: str,
                        zammad_ticket_id: Optional[int] = None) -> None:
        """Logs a user message and a bot response as two separate entries."""
        conn = self._get_connection()
        user_timestamp = datetime.now()
        # Ensure the assistant's timestamp is slightly later for deterministic sorting
        assistant_timestamp = user_timestamp + timedelta(microseconds=1)

        # Insert user message
        conn.execute(
            """
            INSERT INTO User_Interactions 
            (user_identifier, persona_name, channel, role, content, timestamp, zammad_ticket_id)
            VALUES (?, ?, ?, 'user', ?, ?, ?)
            """,
            (user_identifier, persona_name, channel, user_message, user_timestamp, zammad_ticket_id)
        )

        # Insert bot response
        conn.execute(
            """
            INSERT INTO User_Interactions 
            (user_identifier, persona_name, channel, role, content, timestamp, zammad_ticket_id)
            VALUES (?, ?, ?, 'assistant', ?, ?, ?)
            """,
            (user_identifier, persona_name, channel, bot_response, assistant_timestamp, zammad_ticket_id)
        )
        conn.commit()

    def get_personal_history(self, user_identifier: str, persona_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves the most recent messages for a given user and persona."""
        query = """
            SELECT role, content FROM User_Interactions
            WHERE user_identifier = :user_identifier AND persona_name = :persona_name
            ORDER BY timestamp DESC
        """
        params = {'user_identifier': user_identifier, 'persona_name': persona_name}

        if isinstance(limit, int):
            query += " LIMIT :limit"
            params['limit'] = limit

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        # Return in chronological order, already formatted for the LLM API.
        return [dict(row) for row in reversed(rows)]

    def get_ticket_history(self, ticket_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves all messages for a given Zammad ticket ID."""
        query = """
            SELECT role, content FROM User_Interactions
            WHERE zammad_ticket_id = :ticket_id
            ORDER BY timestamp DESC
        """
        params = {'ticket_id': ticket_id}

        if isinstance(limit, int):
            query += " LIMIT :limit"
            params['limit'] = limit

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        # Return in chronological order, already formatted for the LLM API.
        return [dict(row) for row in reversed(rows)]
