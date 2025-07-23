import sqlite3
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional, NoReturn
from pathlib import Path

# --- PATH LOGIC ---
DB_DIR = Path(__file__).resolve().parent
DATABASE_FILE = DB_DIR / "it_support_memory.db"

UNASSOCIATED_BUSINESS_NAME = "Unassociated Contacts"

class ContextManager:
    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initializes the ContextManager.
        If db_path is None, it falls back to the DATABASE_FILE constant.
        """
        self.db_path = db_path if db_path is not None else str(DATABASE_FILE)

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a configured database connection."""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES, uri=True)
        conn.row_factory = sqlite3.Row
        return conn


    def create_schema(self) -> None:
        """Creates the full database schema if it doesn't exist. Ideal for setup."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS Businesses (
            business_id INTEGER PRIMARY KEY AUTOINCREMENT,
            business_name TEXT NOT NULL UNIQUE,
            client_infra_overview TEXT,
            common_issues TEXT
        );
        CREATE TABLE IF NOT EXISTS Contacts (
            contact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            business_id INTEGER,
            full_name TEXT NOT NULL,
            role_in_company TEXT,
            communication_style_summary TEXT,
            FOREIGN KEY (business_id) REFERENCES Businesses(business_id)
        );
        CREATE TABLE IF NOT EXISTS Contact_Identifiers (
            identifier_id INTEGER PRIMARY KEY AUTOINCREMENT,
            contact_id INTEGER NOT NULL,
            channel TEXT NOT NULL,
            identifier_value TEXT NOT NULL,
            UNIQUE(channel, identifier_value),
            FOREIGN KEY (contact_id) REFERENCES Contacts(contact_id)
        );
        CREATE TABLE IF NOT EXISTS Tickets (
            ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
            contact_id INTEGER NOT NULL,
            business_id INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('open', 'in_progress', 'resolved', 'closed')),
            creation_timestamp TIMESTAMP NOT NULL,
            last_update_timestamp TIMESTAMP NOT NULL,
            ticket_summary TEXT,
            FOREIGN KEY (contact_id) REFERENCES Contacts(contact_id),
            FOREIGN KEY (business_id) REFERENCES Businesses(business_id)
        );
        CREATE TABLE IF NOT EXISTS Interactions (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            channel TEXT NOT NULL,
            direction TEXT NOT NULL CHECK(direction IN ('inbound', 'outbound')),
            raw_content TEXT NOT NULL,
            image_url TEXT,
            FOREIGN KEY (ticket_id) REFERENCES Tickets(ticket_id)
        );
        CREATE TABLE IF NOT EXISTS Contact_Placement_Guesses (
            guess_id INTEGER PRIMARY KEY AUTOINCREMENT,
            contact_id INTEGER NOT NULL,
            guessed_business_id INTEGER,
            reasoning TEXT,
            timestamp TIMESTAMP NOT NULL,
            FOREIGN KEY (contact_id) REFERENCES Contacts(contact_id),
            FOREIGN KEY (guessed_business_id) REFERENCES Businesses(business_id)
        );
        """
        with self._get_connection() as conn:
            conn.executescript(schema_sql)
            logging.info("Database schema created or verified successfully.")

    def _initialize_db(self) -> NoReturn:
        """Ensures the default 'Unassociated' business exists."""
        DB_DIR.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT business_id FROM Businesses WHERE business_name = ?", (UNASSOCIATED_BUSINESS_NAME,))
            if cursor.fetchone() is None:
                cursor.execute(
                    "INSERT INTO Businesses (business_name, client_infra_overview) VALUES (?, ?)",
                    (UNASSOCIATED_BUSINESS_NAME, "No infrastructure details. Awaiting manual association.")
                )
                conn.commit()
                logging.info(f"Created default '{UNASSOCIATED_BUSINESS_NAME}' business.")

    def _get_or_create_contact(self, user_identifier: str, channel: str) -> Tuple[int, bool]:
        """Finds a contact or creates a new, unassociated one. Returns contact_id and a boolean indicating if new."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT c.contact_id FROM Contacts c JOIN Contact_Identifiers ci ON c.contact_id = ci.contact_id WHERE ci.identifier_value = ? AND ci.channel = ?",
                (user_identifier, channel)
            )
            contact_row = cursor.fetchone()

            if contact_row:
                return contact_row['contact_id'], False
            else:
                logging.info(f"Creating new contact for identifier '{user_identifier}' on channel '{channel}'.")
                cursor.execute("SELECT business_id FROM Businesses WHERE business_name = ?", (UNASSOCIATED_BUSINESS_NAME,))
                unassociated_id = cursor.fetchone()['business_id']
                cursor.execute(
                    "INSERT INTO Contacts (business_id, full_name, role_in_company) VALUES (?, ?, ?)",
                    (unassociated_id, f"Unknown User ({user_identifier})", "Awaiting Review")
                )
                contact_id = cursor.lastrowid
                cursor.execute(
                    "INSERT INTO Contact_Identifiers (contact_id, channel, identifier_value) VALUES (?, ?, ?)",
                    (contact_id, channel, user_identifier)
                )
                conn.commit()
                return contact_id, True

    def _get_or_create_active_ticket(self, contact_id: int) -> int:
        """Finds the most recent open ticket or creates a new one."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT ticket_id FROM Tickets WHERE contact_id = ? AND status = 'open' ORDER BY last_update_timestamp DESC LIMIT 1", (contact_id,))
            ticket_row = cursor.fetchone()
            if ticket_row:
                return ticket_row['ticket_id']
            else:
                cursor.execute("SELECT business_id FROM Contacts WHERE contact_id = ?", (contact_id,))
                business_id = cursor.fetchone()['business_id']
                now = datetime.now()
                # Use isoformat() to explicitly convert datetime to string
                cursor.execute("INSERT INTO Tickets (contact_id, business_id, status, creation_timestamp, last_update_timestamp, ticket_summary) VALUES (?, ?, 'open', ?, ?, ?)",
                               (contact_id, business_id, now.isoformat(), now.isoformat(), "New ticket initiated."))
                conn.commit()
                return cursor.lastrowid

    def log_interaction(self, ticket_id: int, direction: str, content: str, channel: str, image_url: Optional[str] = None) -> NoReturn:
        """Logs a single message to the database and updates the parent ticket's timestamp."""
        now = datetime.now()
        with self._get_connection() as conn:
            # Use isoformat() to explicitly convert datetime to string
            conn.execute("INSERT INTO Interactions (ticket_id, timestamp, channel, direction, raw_content, image_url) VALUES (?, ?, ?, ?, ?, ?)",
                         (ticket_id, now.isoformat(), channel, direction, content, image_url))
            conn.execute("UPDATE Tickets SET last_update_timestamp = ? WHERE ticket_id = ?", (now.isoformat(), ticket_id))
            conn.commit()

    def get_all_businesses(self) -> List[Dict[str, Any]]:
        """Retrieves a list of all businesses for the guessing prompt."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT business_id, business_name FROM Businesses WHERE business_name != ?", (UNASSOCIATED_BUSINESS_NAME,))
            return [dict(row) for row in cursor.fetchall()]

    def record_business_guess(self, contact_id: int, guessed_business_id: Optional[int], reasoning: str) -> NoReturn:
        """Records the LLM's guess for a new contact's business affiliation."""
        with self._get_connection() as conn:
            # Use isoformat() here as well for consistency
            conn.execute("INSERT INTO Contact_Placement_Guesses (contact_id, guessed_business_id, reasoning, timestamp) VALUES (?, ?, ?, ?)",
                         (contact_id, guessed_business_id, reasoning, datetime.now().isoformat()))
            conn.commit()

    def get_context_for_generation(self, user_identifier: str, channel: str) -> Tuple[int, int, bool]:
        """A consolidated method to get/create contact and ticket IDs."""
        contact_id, is_new = self._get_or_create_contact(user_identifier, channel)
        ticket_id = self._get_or_create_active_ticket(contact_id)
        return contact_id, ticket_id, is_new

    def build_prompt_context_object(self, contact_id: int, history_limit: Optional[int] = None) -> Dict[str, Any]:
        """Builds the context object, with an optional limit on interaction history length."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT c.full_name, c.role_in_company, b.business_name, b.client_infra_overview FROM Contacts c JOIN Businesses b ON c.business_id = b.business_id WHERE c.contact_id = ?",
                           (contact_id,))
            user_data = cursor.fetchone()

            query = """
                SELECT direction, raw_content FROM (
                    SELECT i.direction, i.raw_content, i.timestamp FROM Interactions i
                    JOIN Tickets t ON i.ticket_id = t.ticket_id
                    WHERE t.contact_id = :contact_id
                    ORDER BY i.timestamp DESC
                    LIMIT :limit
                ) sub
                ORDER BY sub.timestamp ASC;
            """
            params = {'contact_id': contact_id, 'limit': -1 if history_limit is None else history_limit}
            cursor.execute(query, params)
            history_rows = cursor.fetchall()

            interaction_history = [{"role": "user" if row['direction'] == 'inbound' else 'assistant', "content": row['raw_content']} for row in history_rows]

            return {
                "user": {"name": user_data['full_name'], "role": user_data['role_in_company']},
                "business": {"name": user_data['business_name'], "infra_overview": user_data['client_infra_overview']},
                "history": interaction_history
            }