import unittest
from unittest.mock import MagicMock, patch

from src.message_handler import BotLogic
from config.global_config import DEFAULT_MODEL_NAME, DEFAULT_CONTEXT_LIMIT


class TestBotLogic(unittest.TestCase):
    def setUp(self):
        """Set up a fresh environment for each test."""
        self.chat_system = MagicMock()
        self.bot_logic = BotLogic(self.chat_system)
        self.message = MagicMock()
        self.mock_persona = MagicMock()
        self.mock_persona.name = "persona"
        self.chat_system.personas = {"persona": self.mock_persona}

    def test_preprocess_message_command_found(self):
        """Verify preprocess_message correctly dispatches a valid command using patch.dict."""
        self.message.content = "help"
        mock_handler = MagicMock(return_value="Help dispatched")

        with patch.dict(self.bot_logic.command_handlers, {'help': mock_handler}):
            result = self.bot_logic.preprocess_message('persona', self.message)

        self.assertEqual(result, "Help dispatched")
        mock_handler.assert_called_once_with([], self.mock_persona)

    def test_preprocess_message_no_command(self):
        """Verify preprocess_message returns None for an unknown command."""
        self.message.content = "unknown_command"
        result = self.bot_logic.preprocess_message('persona', self.message)
        self.assertIsNone(result)

    def test_preprocess_message_check_only(self):
        """Verify check_only=True returns True for a valid command format without executing it."""
        # CHANGE: We now simulate the interface's behavior. The interface identifies the persona
        # and provides the cleaned message content to the pre-processor.
        persona_name = 'persona'  # The interface would determine this.
        self.message.content = 'help'  # This is now the cleaned input, not the raw message.

        # The call now explicitly passes the persona name.
        result = self.bot_logic.preprocess_message(persona_name, self.message, check_only=True)
        self.assertTrue(result)

    def test_preprocess_message_invalid_format(self):
        """Verify preprocess_message returns None for messages with an empty command."""
        # CHANGE: The concept of "too few parts" is now handled by the interface (which strips the
        # persona name). The test now checks the resulting behavior: what happens when the
        # command part of the message is empty.
        persona_name = 'persona'
        self.message.content = ''  # The interface would pass an empty string after stripping "persona" from the input.

        result = self.bot_logic.preprocess_message(persona_name, self.message)
        self.assertIsNone(result)

    def test_preprocess_message_unknown_persona(self):
        """Verify preprocess_message returns None if passed a persona that doesn't exist."""
        # CHANGE: The responsibility for checking if a persona exists is now in the interface.
        # However, this test is still valuable to ensure the function itself is robust
        # and doesn't crash if an interface somehow passes an invalid persona name.
        unknown_persona_name = 'unknown_persona'
        self.message.content = 'help'  # The content of the message is less important here.

        result = self.bot_logic.preprocess_message(unknown_persona_name, self.message)
        self.assertIsNone(result)
    def test_handle_help_no_args(self):
        """Verify _handle_help returns the help string when called with no arguments."""
        result = self.bot_logic._handle_help(args=[], persona=self.mock_persona)
        self.assertIsInstance(result, str)
        self.assertIn("Talk to a specific persona", result)

    def test_handle_help_with_args(self):
        """Verify _handle_help returns None to avoid ambiguity with conversational 'help'."""
        result = self.bot_logic._handle_help(args=["me"], persona=self.mock_persona)
        self.assertIsNone(result)

    @patch('src.message_handler.save_utils.save_personas_to_file')
    def test_handle_remember(self, mock_save):
        """Verify _handle_remember correctly modifies the persona's prompt."""
        self.mock_persona.get_prompt.return_value = "Original prompt"
        args = ["text", "to", "remember"]
        result = self.bot_logic._handle_remember(args, self.mock_persona)
        self.mock_persona.set_prompt.assert_called_once_with("Original prompt text to remember")
        self.assertIn("New prompt for persona", result)
        mock_save.assert_called_once()

    def test_handle_what_dispatches_correctly(self):
        """Verify _handle_what calls the correct sub-handler using patch.dict."""
        mock_sub_handler = MagicMock(return_value="Prompt Response")

        with patch.dict(self.bot_logic.what_handlers, {'prompt': mock_sub_handler}):
            result = self.bot_logic._handle_what(['prompt'], self.mock_persona)

        self.assertEqual(result, "Prompt Response")
        mock_sub_handler.assert_called_once_with(['prompt'], self.mock_persona)

    def test_handle_what_returns_none_for_invalid_subcommand(self):
        """Verify _handle_what returns None for an unknown subcommand to avoid ambiguity."""
        result = self.bot_logic._handle_what(['invalid_sub'], self.mock_persona)
        self.assertIsNone(result)

    def test_what_model(self):
        """Verify the _what_model sub-handler returns the correct model name."""
        self.mock_persona.get_model_name.return_value = "gpt-4"
        result = self.bot_logic._what_model([], self.mock_persona)
        self.assertIn("persona is using gpt-4", result)

    @patch('src.message_handler.save_utils.save_personas_to_file')
    def test_set_prompt_success(self, mock_save):
        """Verify _set_prompt correctly calls set_prompt and the save decorator."""
        args = ['prompt', 'new', 'prompt', 'text']
        mock_sub_handler = MagicMock(return_value="Prompt saved.")

        with patch.dict(self.bot_logic.set_handlers, {'prompt': mock_sub_handler}):
            result = self.bot_logic._handle_set(args, self.mock_persona)

        self.assertEqual(result, "Prompt saved.")
        mock_sub_handler.assert_called_once_with(args, self.mock_persona)

    @patch('src.message_handler.save_utils.save_personas_to_file')
    def test_set_model_failure_no_save(self, mock_save):
        """Verify a failed set command does not trigger a save."""
        args = ['model']
        result = self.bot_logic._handle_set(args, self.mock_persona)

        self.assertTrue(result.lower().startswith('error'))
        mock_save.assert_not_called()

    @patch('src.message_handler.save_utils.save_personas_to_file')
    def test_handle_add(self, mock_save):
        """Verify _handle_add calls add_persona and the save decorator."""
        args = ["new_persona", "This", "is", "a", "prompt"]
        result = self.bot_logic._handle_add(args, self.mock_persona)
        self.chat_system.add_persona.assert_called_once_with(
            "new_persona",
            DEFAULT_MODEL_NAME,
            "This is a prompt",
            context_limit=DEFAULT_CONTEXT_LIMIT,
            token_limit=1024,
            save_new=False
        )
        self.assertIn("added 'new_persona'", result)
        mock_save.assert_called_once()

    @patch('src.message_handler.save_utils.save_personas_to_file')
    def test_handle_delete(self, mock_save):
        """Verify _handle_delete calls delete_persona and the save decorator."""
        args = ["persona_to_delete"]
        result = self.bot_logic._handle_delete(args, self.mock_persona)
        self.chat_system.delete_persona.assert_called_once_with("persona_to_delete", save=False)
        self.assertIn("has been deleted", result)
        mock_save.assert_called_once()

    def test_handle_start_conversation(self):
        """Verify _handle_start_conversation correctly configures the persona."""
        result = self.bot_logic._handle_start_conversation([], self.mock_persona)
        self.mock_persona.set_context_length.assert_called_once_with(0)
        self.mock_persona.set_conversation_mode.assert_called_once_with(True)
        self.assertIn("Hello! Starting new conversation", result)

    def test_handle_stop_conversation(self):
        """Verify _handle_stop_conversation correctly resets the persona."""
        result = self.bot_logic._handle_stop_conversation([], self.mock_persona)
        self.mock_persona.set_context_length.assert_called_once_with(DEFAULT_CONTEXT_LIMIT)
        self.mock_persona.set_conversation_mode.assert_called_once_with(False)
        self.assertIn("Goodbye!", result)


if __name__ == '__main__':
    unittest.main()