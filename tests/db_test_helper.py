# tests/db_test_helper.py
import sqlite3

def create_test_db_schema(conn: sqlite3.Connection):
    """
    Creates the full database schema in the provided database connection.
    Used for setting up in-memory databases for isolated testing.
    """
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Businesses (
        business_id INTEGER PRIMARY KEY AUTOINCREMENT,
        business_name TEXT NOT NULL UNIQUE,
        client_infra_overview TEXT,
        common_issues TEXT
    );
    """)
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