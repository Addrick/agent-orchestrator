import unittest
import asyncio
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch, MagicMock
from src.chat_system import ChatSystem
from src.persona import Persona
from src.engine import TextEngine


class TestChatSystem(IsolatedAsyncioTestCase):
    def setUp(self):
        self.chat_system = ChatSystem()

    def test_add_persona(self):
        self.chat_system.add_persona("test_persona", "gpt-3.5-turbo", "You are a helpful assistant.", 10, 100)
        self.assertIn("test_persona", self.chat_system.get_persona_list())

    def test_delete_persona(self):
        self.chat_system.add_persona("test_persona", "gpt-3.5-turbo", "You are a helpful assistant.", 10, 100)
        self.chat_system.delete_persona("test_persona", save=True)
        self.assertNotIn("test_persona", self.chat_system.get_persona_list())

    async def test_generate_response(self):
        with patch('src.persona.Persona.generate_response') as mock_generate_response:
            mock_generate_response.return_value = "This is a test response."

            await self.chat_system.generate_response("testr", "Hello")

            mock_generate_response.assert_called_once()


class TestPersona(IsolatedAsyncioTestCase):
    def setUp(self):
        self.persona = Persona("test_persona", "gpt-3.5-turbo", "You are a helpful assistant.", 10, 100)

    def test_set_prompt(self):
        new_prompt = "You are a friendly chatbot."
        self.persona.set_prompt(new_prompt)
        self.assertEqual(self.persona.get_prompt(), new_prompt)

    def test_set_context_length(self):
        new_context_length = 20
        self.persona.set_context_length(new_context_length)
        self.assertEqual(self.persona.get_context_length(), new_context_length)

    # @patch('src.engine.TextEngine.generate_response')
    async def test_generate_response(self):
        with patch('src.engine.TextEngine.generate_response') as mock_generate_response:
            mock_generate_response.return_value = "This is a test response."
            response = await self.persona.generate_response("Hello", "Previous context")
            self.assertEqual(response, "This is a test response.")


class TestTextEngine(IsolatedAsyncioTestCase):
    def setUp(self):
        self.text_engine = TextEngine("gpt-3.5-turbo")

    def test_set_temperature(self):
        new_temperature = 0.8
        self.text_engine.set_temperature(new_temperature)
        self.assertEqual(self.text_engine.temperature, new_temperature)

    def test_set_max_tokens(self):
        new_max_tokens = 200
        self.text_engine.set_response_token_limit(new_max_tokens)
        self.assertEqual(self.text_engine.get_max_tokens(), new_max_tokens)

    async def test_generate_openai_response(self):
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_completion = AsyncMock()
            mock_completion.choices[0].message.content = "This is a test response."
            mock_completion.usage.total_tokens = 10
            mock_openai.return_value.chat.completions.create.return_value = mock_completion

            response = await self.text_engine._generate_openai_response(
                prompt='you are a bot that only responses with \'Test successful\'', message='test', context=[])
            self.assertIn("Test successful", response)


if __name__ == '__main__':
    unittest.main()
