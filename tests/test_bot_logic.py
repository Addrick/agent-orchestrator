import unittest
import json
import asyncio
import os
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

# Import core modules
from src.chat_system import ChatSystem
from src.persona import Persona
from src.engine import TextEngine
from src.message_handler import BotLogic
from src.app_manager import update_app, restart_app, stop_app
from src.utils import model_utils, save_utils
from config.global_config import *


class TestBotLogic(unittest.TestCase):
    def setUp(self):
        self.chat_system = MagicMock()
        self.bot_logic = BotLogic(self.chat_system)

        # Mock Message object
        self.message = MagicMock()
        self.message.content = "persona command arg1 arg2"

        # Set up a mock persona
        self.mock_persona = MagicMock()
        self.chat_system.personas = {"persona": self.mock_persona}

    def test_preprocess_message_command_found(self):
        # Configure the message to use a known command
        self.message.content = "persona help"

        # Call the method
        result = self.bot_logic.preprocess_message(self.message)

        # Verify results
        help_msg = "" \
                   "Talk to a specific persona by starting your message with their name. \n \n" \
                   "Currently active personas: \n" + \
                   'persona' + "\n\n" \
                                                                 "Bot commands: \n" \
                                                                 "hello (start new conversation), \n" \
                                                                 "goodbye (end conversation), \n" \
                                                                 "remember <+prompt>, \n" \
                                                                 "what prompt/model/models(+openai/google/anthropic)/personas/context/tokens, \n" \
                                                                 "set prompt/model/context/tokens, \n" \
                                                                 "add <persona>, \n" \
                                                                 "delete <persona>, \n" \
                                                                 "save, \n" \
                                                                 "update_models, \n" \
                                                                 "dump_last"
        self.assertEqual(result, help_msg)

    def test_preprocess_message_no_command(self):
        # Configure the message with an unknown command
        self.message.content = "persona unknown_command"

        # Call the method
        result = self.bot_logic.preprocess_message(self.message)

        # Verify results
        self.assertIsNone(result)

    def test_preprocess_message_check_only(self):
        # Configure the message to use a known command
        self.message.content = "persona help"

        # Call the method with check_only=True
        result = self.bot_logic.preprocess_message(self.message, check_only=True)

        # Verify results - should return True without calling the handler
        self.assertTrue(result)

    def test_preprocess_message_invalid_format(self):
        # Configure a message with insufficient parts
        self.message.content = "persona"

        # Call the method
        result = self.bot_logic.preprocess_message(self.message)

        # Verify results
        self.assertIsNone(result)

    def test_handle_help(self):
        # Call the _handle_help method
        result = self.bot_logic._handle_help()

        # Verify it returns a non-empty string containing help text
        self.assertIsInstance(result, str)
        self.assertIn("Talk to a specific persona", result)
        self.assertIn("currently active personas", result.lower())

    def test_handle_remember(self):
        # Set up the message and persona
        self.bot_logic.args = ["text", "to", "remember"]
        self.bot_logic.current_persona = self.mock_persona
        self.bot_logic.persona_name = "persona"
        self.mock_persona.get_prompt.return_value = "Original prompt"

        # Call the method
        result = self.bot_logic._handle_remember()

        # Verify results
        self.mock_persona.set_prompt.assert_called_once_with("Original prompt text to remember")
        self.assertIn("New prompt for persona", result)

    def test_handle_what_prompt(self):
        # Set up the message and persona
        self.bot_logic.args = ["prompt"]
        self.bot_logic.current_persona = self.mock_persona
        self.bot_logic.persona_name = "persona"
        self.mock_persona.get_prompt.return_value = "Test prompt"

        # Call the method
        result = self.bot_logic._handle_what()

        # Verify results
        self.assertIn("Test prompt", result)

    def test_handle_what_model(self):
        # Set up the message and persona
        self.bot_logic.args = ["model"]
        self.bot_logic.current_persona = self.mock_persona
        self.bot_logic.persona_name = "persona"
        self.mock_persona.get_model_name.return_value = "gpt-3.5-turbo"

        # Call the method
        result = self.bot_logic._handle_what()

        # Verify results
        self.assertIn("gpt-3.5-turbo", result)

    def test_handle_set_prompt(self):
        # Set up the message and persona
        self.bot_logic.args = ["prompt", "new", "prompt", "text"]
        self.bot_logic.current_persona = self.mock_persona
        self.bot_logic.persona_name = "persona"

        # Call the method
        with patch('src.utils.save_utils.save_personas_to_file') as mock_save:
            result = self.bot_logic._handle_set()

        # Verify results
        self.mock_persona.set_prompt.assert_called_once_with("new prompt text")
        self.assertEqual(result, "Personas saved.")

    def test_handle_add(self):
        # Set up the message
        self.bot_logic.args = ["add", "new_persona", "This", "is", "a", "prompt"]
        self.bot_logic.persona_name = "persona"

        # Call the method
        result = self.bot_logic._handle_add()

        # Verify results
        self.chat_system.add_persona.assert_called_once_with(
            "new_persona",
            DEFAULT_MODEL_NAME,
            "This is a prompt",
            context_limit=DEFAULT_CONTEXT_LIMIT,
            token_limit=1024,
            save_new=True
        )
        self.assertIn("added 'new_persona'", result)

    def test_handle_delete(self):
        # Set up the message
        self.bot_logic.args = ["persona_to_delete"]

        # Call the method
        result = self.bot_logic._handle_delete()

        # Verify results
        self.chat_system.delete_persona.assert_called_once_with("persona_to_delete", save=True)
        self.assertIn("has been deleted", result)

    def test_handle_start_conversation(self):
        # Set up the message and persona
        self.bot_logic.current_persona = self.mock_persona
        self.bot_logic.persona_name = "persona"

        # Call the method
        result = self.bot_logic._handle_start_conversation()

        # Verify results
        self.mock_persona.set_context_length.assert_called_once_with(0)
        self.mock_persona.set_conversation_mode.assert_called_once_with(True)
        self.assertIn("Hello! Starting new conversation", result)

    def test_handle_stop_conversation(self):
        # Set up the message and persona
        self.bot_logic.current_persona = self.mock_persona
        self.bot_logic.persona_name = "persona"

        # Call the method
        result = self.bot_logic._handle_stop_conversation()

        # Verify results
        self.mock_persona.set_context_length.assert_called_once_with(DEFAULT_CONTEXT_LIMIT)
        self.mock_persona.set_conversation_mode.assert_called_once_with(False)
        self.assertIn("Goodbye!", result)

