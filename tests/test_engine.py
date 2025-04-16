import unittest
import json
import asyncio
import os
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from dotenv import load_dotenv

# Import core modules
from src.chat_system import ChatSystem
from src.persona import Persona
from src.engine import TextEngine
from src.message_handler import BotLogic
from src.app_manager import update_app, restart_app, stop_app
from src.utils import model_utils, save_utils
from config.global_config import *

load_dotenv()


class TestTextEngine(IsolatedAsyncioTestCase):
    def setUp(self):
        self.text_engine = TextEngine("gpt-3.5-turbo",
                                      token_limit=100,
                                      temperature=0.7,
                                      top_p=0.9,
                                      top_k=40)

    def test_init(self):
        self.assertEqual(self.text_engine.model_name, "gpt-3.5-turbo")
        self.assertEqual(self.text_engine.max_tokens, 100)
        self.assertEqual(self.text_engine.temperature, 0.7)
        self.assertEqual(self.text_engine.top_p, 0.9)
        self.assertEqual(self.text_engine.top_k, 40)
        self.assertIsNone(self.text_engine.openai_client)

    def test_get_max_tokens(self):
        self.assertEqual(self.text_engine.get_max_tokens(), 100)

    def test_set_response_token_limit(self):
        # Valid input
        result = self.text_engine.set_response_token_limit(200)
        self.assertTrue(result)
        self.assertEqual(self.text_engine.get_max_tokens(), 200)

        # Invalid input
        result = self.text_engine.set_response_token_limit("not an integer")
        self.assertFalse(result)
        self.assertEqual(self.text_engine.get_max_tokens(), 200)  # Should not change

    def test_set_temperature(self):
        self.text_engine.set_temperature(0.8)
        self.assertEqual(self.text_engine.temperature, 0.8)

    def test_set_top_p(self):
        self.text_engine.set_top_p(0.95)
        self.assertEqual(self.text_engine.top_p, 0.95)

    def test_set_top_k(self):
        self.text_engine.set_top_k(50)
        self.assertEqual(self.text_engine.top_k, 50)

    def test_parse_request_json(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        result = self.text_engine.parse_request_json(messages)

        self.assertEqual(result["model"], "gpt-3.5-turbo")
        self.assertEqual(result["messages"], messages)
        self.assertEqual(result["options"]["temperature"], 0.7)
        self.assertEqual(result["options"]["max_completion_tokens"], 100)
        self.assertEqual(result["options"]["top_p"], 0.9)

    def test_get_raw_json_request(self):
        self.assertIsNone(self.text_engine.get_raw_json_request())

        messages = [{"role": "user", "content": "Test message"}]
        self.text_engine.json_request = self.text_engine.parse_request_json(messages)

        request = self.text_engine.get_raw_json_request()
        self.assertEqual(request["messages"], messages)

    async def test_initialize_openai_client(self):
        with patch('openai.AsyncOpenAI') as mock_openai:
            await self.text_engine.initialize_openai_client()
            mock_openai.assert_called_once()
            self.assertIsNotNone(self.text_engine.openai_client)

    async def test_generate_openai_response(self):
        with patch('openai.AsyncOpenAI') as mock_openai_class:
            # Mock the AsyncOpenAI client and its methods
            mock_client = MagicMock()
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock()]
            mock_completion.choices[0].message.content = "This is a test response."
            mock_completion.usage.total_tokens = 10

            mock_create = AsyncMock(return_value=mock_completion)
            mock_client.chat.completions.create = mock_create
            mock_openai_class.return_value = mock_client

            # Set up the test
            await self.text_engine.initialize_openai_client()
            response = await self.text_engine._generate_openai_response(
                "You are a helpful assistant.", "Hello, how are you?", None)

            # Verify results
            self.assertIn("This is a test response.", response)
            self.assertIn("10 tokens", response)
            self.assertIn("gpt-3.5-turbo", response)

            # Check the messages were correctly formatted
            expected_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ]
            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            self.assertEqual(kwargs["messages"], expected_messages)

    async def test_generate_openai_response_with_context(self):
        with patch('openai.AsyncOpenAI') as mock_openai_class:
            # Mock the AsyncOpenAI client and its methods
            mock_client = MagicMock()
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock()]
            mock_completion.choices[0].message.content = "Response with context."
            mock_completion.usage.total_tokens = 15

            mock_create = AsyncMock(return_value=mock_completion)
            mock_client.chat.completions.create = mock_create
            mock_openai_class.return_value = mock_client

            # Set up the test
            await self.text_engine.initialize_openai_client()
            response = await self.text_engine._generate_openai_response(
                "You are a helpful assistant.",
                "What did I ask before?",
                "Previous message context.")

            # Verify results
            self.assertIn("Response with context.", response)

            # Check the messages were correctly formatted with context
            expected_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Previous message context."},
                {"role": "user", "content": "What did I ask before?"}
            ]
            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            self.assertEqual(kwargs["messages"], expected_messages)

    async def test_generate_response_routing(self):
        # We'll test the routing logic by checking which specific generate method is called

        # Mock all possible generate methods
        with patch.object(self.text_engine, '_generate_openai_response') as mock_openai, \
                patch.object(self.text_engine, '_generate_openai_reasoning_response') as mock_reasoning, \
                patch.object(self.text_engine, '_generate_openai_search_response') as mock_search, \
                patch.object(self.text_engine, '_generate_anthropic_response') as mock_anthropic, \
                patch.object(self.text_engine, '_generate_google_response_vertex') as mock_google_vertex, \
                patch.object(self.text_engine, '_generate_google_response_ai_studio_async') as mock_google_ai_studio, \
                patch.object(self.text_engine, '_generate_local_response') as mock_local:
            # Set return values
            mock_openai.return_value = "OpenAI response"
            mock_reasoning.return_value = "Reasoning response"
            mock_search.return_value = "Search response"
            mock_anthropic.return_value = "Anthropic response"
            mock_google_vertex.return_value = "Google vertex response"
            mock_google_ai_studio.return_value = "Google ai studio response"
            mock_local.return_value = "Local response"

            # Test OpenAI chat models
            with patch.object(self.text_engine, 'openai_models_available', ["gpt-3.5-turbo", "gpt-4"]):
                self.text_engine.model_name = "gpt-3.5-turbo"
                response = await self.text_engine.generate_response("prompt", "message", "context", None, 100)
                mock_openai.assert_called_once()
                self.assertEqual(response, "OpenAI response")
                mock_openai.reset_mock()

            # Test OpenAI reasoning models
            with patch.object(self.text_engine, 'openai_models_available', ["o1-preview"]):
                self.text_engine.model_name = "o1-preview"
                response = await self.text_engine.generate_response("prompt", "message", "context", None, 100)
                mock_reasoning.assert_called_once()
                self.assertEqual(response, "Reasoning response")
                mock_reasoning.reset_mock()

            # Test OpenAI search models
            with patch.object(self.text_engine, 'openai_models_available', ["gpt-4-search"]):
                self.text_engine.model_name = "gpt-4-search"
                response = await self.text_engine.generate_response("prompt", "message", "context", None, 100)
                mock_search.assert_called_once()
                self.assertEqual(response, "Search response")
                mock_search.reset_mock()

            # Test Anthropic models
            with patch.object(self.text_engine, 'anthropic_models_available', ["claude-3-opus"]):
                self.text_engine.model_name = "claude-3-opus"
                response = await self.text_engine.generate_response("prompt", "message", "context", None, 100)
                mock_anthropic.assert_called_once()
                self.assertEqual(response, "Anthropic response")
                mock_anthropic.reset_mock()

            # Test Google AI Studio models
            with patch.object(self.text_engine, 'google_models_available', ["gemini-pro"]):
                self.text_engine.model_name = "gemini-pro"
                response = await self.text_engine.generate_response("prompt", "message", "context", None, 100)
                mock_google_ai_studio.assert_called_once()
                self.assertEqual(response, "Google ai studio response")
                mock_google_ai_studio.reset_mock()

            # Test Google Vertex AI models
            ## not currently routed for response for any models
            # with patch.object(self.text_engine, 'google_models_available', ["gemini-pro"]):
            #     self.text_engine.model_name = "gemini-pro"
            #     response = await self.text_engine.generate_response("prompt", "message", "context", None, 100)
            #     mock_google_vertex.assert_called_once()
            #     self.assertEqual(response, "Google ai studio response")
            #     mock_google_vertex.reset_mock()

            # Test Local model
            self.text_engine.model_name = "local"
            response = await self.text_engine.generate_response("prompt", "message", "context", None, 100)
            mock_local.assert_called_once()
            self.assertEqual(response, "Local response")

