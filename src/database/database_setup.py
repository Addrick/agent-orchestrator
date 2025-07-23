# src/database_setup.py
import sqlite3
import logging
from typing import NoReturn

DATABASE_FILE = "it_support_memory.db"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_database() -> NoReturn:
    """Creates and initializes the database tables if they do not exist."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        # Table for client businesses
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Businesses (
            business_id INTEGER PRIMARY KEY AUTOINCREMENT,
            business_name TEXT NOT NULL UNIQUE,
            client_infra_overview TEXT,
            common_issues TEXT
        );
        """)

        # Table for individual contacts at businesses
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Contacts (
            contact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            business_id INTEGER,
            full_name TEXT NOT NULL,
            role_in_company TEXT,
            communication_style_summary TEXT,
            FOREIGN KEY (business_id) REFERENCES Businesses (business_id)
        );
        """)

        # Table to link contacts to their various platform identifiers
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Contact_Identifiers (
            identifier_id INTEGER PRIMARY KEY AUTOINCREMENT,
            contact_id INTEGER NOT NULL,
            channel TEXT NOT NULL,
            identifier_value TEXT NOT NULL,
            UNIQUE(channel, identifier_value),
            FOREIGN KEY (contact_id) REFERENCES Contacts (contact_id)
        );
        """)

        # Table for support tickets
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Tickets (
            ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
            contact_id INTEGER NOT NULL,
            business_id INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('open', 'in_progress', 'resolved', 'closed')),
            creation_timestamp DATETIME NOT NULL,
            last_update_timestamp DATETIME NOT NULL,
            ticket_summary TEXT,
            FOREIGN KEY (contact_id) REFERENCES Contacts (contact_id),
            FOREIGN KEY (business_id) REFERENCES Businesses (business_id)
        );
        """)

        # Table for individual messages/interactions within a ticket
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Interactions (
            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id INTEGER NOT NULL,
            timestamp DATETIME NOT NULL,
            channel TEXT NOT NULL,
            direction TEXT NOT NULL CHECK(direction IN ('inbound', 'outbound')),
            raw_content TEXT NOT NULL,
            image_url TEXT,
            FOREIGN KEY (ticket_id) REFERENCES Tickets (ticket_id)
        );
        """)

        # New Table for storing LLM guesses about new contact affiliations
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Contact_Placement_Guesses (
            guess_id INTEGER PRIMARY KEY AUTOINCREMENT,
            contact_id INTEGER NOT NULL,
            guessed_business_id INTEGER,
            reasoning TEXT,
            timestamp DATETIME NOT NULL,
            FOREIGN KEY (contact_id) REFERENCES Contacts (contact_id),
            FOREIGN KEY (guessed_business_id) REFERENCES Businesses (business_id)
        );
        """)

        conn.commit()
        logging.info("Database and tables, including Contact_Placement_Guesses, verified/created successfully.")
    except sqlite3.Error as e:
        logging.error(f"Database error during setup: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    setup_database()