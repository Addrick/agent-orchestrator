import unittest
import os
import json
import sqlite3
from unittest.mock import patch, AsyncMock

# Ensure the test can find the src modules
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chat_system import ChatSystem, ResponseType
from src.engine import TextEngine
from src.database.context_manager import ContextManager
from src.utils.save_utils import load_personas_from_file

# Assuming the ResponseType Enum exists and can be imported.
# If this path is wrong, you can adjust it.


class TestIntegration(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        """Set up a clean environment for each test."""
        self.test_persona_file = "test_personas"

        test_persona_data = {
            "personas": [
                {
                    "name": "tester",
                    "model_name": "mock-model",
                    "prompt": "You are a test persona.",
                    "context_limit": 10,
                    "token_limit": 100
                }
            ],
            "models": {
                "mock-model": ["mock-model"]
            }
        }

        with open(self.test_persona_file, 'w') as f:
            json.dump(test_persona_data, f, indent=4)

        # Create one single connection for the entire test
        self.connection = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
        self.connection.row_factory = sqlite3.Row

        self.cm = ContextManager(db_path=":memory:")
        # Monkey-patch the instance to use our single, persistent connection
        self.cm._get_connection = lambda: self.connection

        # Perform setup using the instance. All calls will use the same connection.
        self.cm.create_schema()
        self.cm._initialize_db()

        self.text_engine = TextEngine()
        self.chat_system = ChatSystem(context_manager=self.cm, text_engine=self.text_engine)
        self.chat_system.personas = load_personas_from_file(file_path=self.test_persona_file)

    def tearDown(self):
        """Clean up resources after each test."""
        if self.connection:
            self.connection.close()
        if os.path.exists(self.test_persona_file):
            os.remove(self.test_persona_file)

    @patch('src.engine.TextEngine.generate_response', new_callable=AsyncMock)
    async def test_full_interaction_flow(self, mock_generate_response):
        """
        Tests the entire flow from receiving a message to storing the result.
        Mocks the TextEngine to isolate the test from external LLM APIs.
        """
        # --- 1. Arrange ---
        mock_response_text = "This is a mock response from the LLM."
        mock_api_payload = {"model": "mock-model", "prompt": "You are a test persona.", "mocked": True}
        mock_generate_response.return_value = (mock_response_text, mock_api_payload)

        user_identifier = "test_user_123"
        persona_name = "tester"
        channel = "test_channel"
        message = "Hello, this is a test message."

        # --- 2. Act ---
        final_response_tuple = await self.chat_system.generate_response(
            persona_name=persona_name,
            user_identifier=user_identifier,
            channel=channel,
            message=message
        )

        # --- 3. Assert ---
        response_text, response_type = final_response_tuple

        self.assertEqual(response_text, mock_response_text)
        self.assertEqual(response_type, ResponseType.LLM_GENERATION)

        mock_generate_response.assert_called_once()

        cursor = self.connection.cursor()

        cursor.execute("SELECT contact_id FROM Contact_Identifiers WHERE identifier_value = ?", (user_identifier,))
        contact_row = cursor.fetchone()
        self.assertIsNotNone(contact_row)
        contact_id = contact_row['contact_id']

        cursor.execute("SELECT ticket_id FROM Tickets WHERE contact_id = ?", (contact_id,))
        ticket_row = cursor.fetchone()
        self.assertIsNotNone(ticket_row)
        ticket_id = ticket_row['ticket_id']

        cursor.execute("SELECT direction, raw_content FROM Interactions WHERE ticket_id = ?", (ticket_id,))
        interactions = cursor.fetchall()
        self.assertEqual(len(interactions), 2)

        inbound_msg = [row['raw_content'] for row in interactions if row['direction'] == 'inbound']
        outbound_msg = [row['raw_content'] for row in interactions if row['direction'] == 'outbound']

        self.assertEqual(inbound_msg[0], message)
        self.assertEqual(outbound_msg[0], mock_response_text)

        stored_payload = self.chat_system.last_api_requests[user_identifier][persona_name]
        self.assertEqual(stored_payload, mock_api_payload)


if __name__ == '__main__':
    unittest.main()