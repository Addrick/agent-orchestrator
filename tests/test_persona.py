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



class TestPersona(IsolatedAsyncioTestCase):
    def setUp(self):
        self.persona = Persona("test_persona", "gpt-3.5-turbo", "You are a helpful assistant.", 10, 100)

    def test_init(self):
        self.assertEqual(self.persona.persona_name, "test_persona")
        self.assertEqual(self.persona.prompt, "You are a helpful assistant.")
        self.assertEqual(self.persona.context_length, 10)
        self.assertEqual(self.persona.response_token_limit, 100)
        self.assertEqual(self.persona.get_model_name(), "gpt-3.5-turbo")
        self.assertFalse(self.persona.conversation_mode)

    def test_set_prompt(self):
        new_prompt = "You are a friendly chatbot."
        self.persona.set_prompt(new_prompt)
        self.assertEqual(self.persona.get_prompt(), new_prompt)

    def test_update_prompt(self):
        original_prompt = self.persona.get_prompt()
        self.persona.update_prompt(" Be concise.")
        self.assertEqual(self.persona.get_prompt(), original_prompt + " Be concise.")

    def test_set_context_length(self):
        new_context_length = 20
        self.persona.set_context_length(new_context_length)
        self.assertEqual(self.persona.get_context_length(), new_context_length)

        # Test with string input (should convert to int)
        self.persona.set_context_length("30")
        self.assertEqual(self.persona.get_context_length(), 30)

    def test_get_response_token_limit(self):
        self.assertEqual(self.persona.get_response_token_limit(), 100)

    def test_set_response_token_limit(self):
        # Valid input
        result = self.persona.set_response_token_limit(200)
        self.assertTrue(result)
        self.assertEqual(self.persona.get_response_token_limit(), 200)

        # Invalid input
        result = self.persona.set_response_token_limit("not an integer")
        self.assertFalse(result)
        self.assertEqual(self.persona.get_response_token_limit(), 200)  # Should not change

    def test_set_temperature(self):
        with patch.object(self.persona.model, 'set_temperature') as mock_set_temperature:
            self.persona.set_temperature(0.8)
            mock_set_temperature.assert_called_once_with(0.8)

    def test_set_top_p(self):
        with patch.object(self.persona.model, 'set_top_p') as mock_set_top_p:
            self.persona.set_top_p(0.95)
            mock_set_top_p.assert_called_once_with(0.95)

    def test_set_top_k(self):
        with patch.object(self.persona.model, 'set_top_k') as mock_set_top_k:
            self.persona.set_top_k(40)
            mock_set_top_k.assert_called_once_with(40)

    def test_set_model(self):
        new_model = self.persona.set_model("gpt-4")
        self.assertEqual(self.persona.get_model_name(), "gpt-4")
        self.assertEqual(new_model.model_name, "gpt-4")

    def test_set_conversation_mode(self):
        self.assertFalse(self.persona.conversation_mode)
        self.persona.set_conversation_mode(True)
        self.assertTrue(self.persona.conversation_mode)

    def test_last_json_handling(self):
        test_json = {"test": "json_data"}
        self.persona.set_last_json(test_json)
        self.assertEqual(self.persona.get_last_json(), test_json)

    async def test_generate_response(self):
        with patch.object(self.persona.model, 'generate_response') as mock_generate_response:
            mock_generate_response.return_value = "This is a test response."
            response = await self.persona.generate_response("Hello", ["Message 1", "Message 2", "Message 3"])
            self.assertEqual(response, "This is a test response.")

            # Check context handling (should reverse and join with newlines)
            expected_context = 'recent chat history: \nMessage 3 \nMessage 2 \nMessage 1'
            mock_generate_response.assert_called_once()
            args, kwargs = mock_generate_response.call_args
            self.assertEqual(args[2], expected_context)

    async def test_generate_response_with_conversation_mode(self):
        with patch.object(self.persona.model, 'generate_response') as mock_generate_response:
            mock_generate_response.return_value = "Conversation mode response."

            # Enable conversation mode
            self.persona.set_conversation_mode(True)
            self.persona.set_context_length(0)

            # Test that context length increases after each response
            await self.persona.generate_response("Hello", ["Message 1"])
            self.assertEqual(self.persona.context_length, 2)

            await self.persona.generate_response("Hi again", ["Message 2", "Message 1"])
            self.assertEqual(self.persona.context_length, 4)

            # Test that it doesn't exceed GLOBAL_CONTEXT_LIMIT
            self.persona.set_context_length(GLOBAL_CONTEXT_LIMIT - 1)
            await self.persona.generate_response("Last message", ["Message 3", "Message 2", "Message 1"])
            self.assertEqual(self.persona.context_length, GLOBAL_CONTEXT_LIMIT - 1)  # Should not increase

