import unittest
from unittest.async_case import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch, MagicMock

from dotenv import load_dotenv

from src.chat_system import ChatSystem
from src.persona import Persona
from src.engine import TextEngine

load_dotenv()

class TestChatSystem(IsolatedAsyncioTestCase):

    def setUp(self):
        self.chat_system = ChatSystem()
        # Mock the save_personas_to_file function to avoid actual file writes during tests
        self.save_mock = patch('src.utils.save_utils.save_personas_to_file').start()

    def tearDown(self):
        patch.stopall()

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
        self.chat_system = ChatSystem()
        # Mock the save_personas_to_file function to avoid actual file writes during tests
        self.save_mock = patch('src.utils.save_utils.save_personas_to_file').start()

    def tearDown(self):
        patch.stopall()

    def test_set_temperature(self):
        new_temperature = 0.8
        self.text_engine.set_temperature(new_temperature)
        self.assertEqual(self.text_engine.temperature, new_temperature)

    def test_set_max_tokens(self):
        new_max_tokens = 200
        self.text_engine.set_response_token_limit(new_max_tokens)
        self.assertEqual(self.text_engine.get_max_tokens(), new_max_tokens)

    async def test_generate_openai_response(self):
        with patch('openai.AsyncOpenAI') as mock_openai_class: # Patch where AsyncOpenAI is LOOKED UP
            mock_client_instance = AsyncMock()
            mock_openai_class.return_value = mock_client_instance
            mock_completion_result = MagicMock()
            mock_completion_result.choices = [MagicMock()] # Needs to be a list
            mock_completion_result.choices[0].message = MagicMock()
            mock_completion_result.choices[0].message.content = "Test successful"
            mock_completion_result.usage = MagicMock()
            mock_completion_result.usage.total_tokens = 10

            # 3. Mock the 'create' async method on the client instance
            #    Assign an AsyncMock to the method name itself.
            #    Set its return_value to the mock_completion_result.
            #    This is what will be returned when 'create' is awaited.
            mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_completion_result)

            response = await self.text_engine._generate_openai_response(
                prompt='you are a bot that only responses with \'Test successful\'', message='test', context=[])
            self.assertIn("Test successful (10 tokens using gpt-3.5-turbo)", response)

    def test_add_persona(self):
        self.chat_system.add_persona("test_persona", "gpt-3.5-turbo", "You are a helpful assistant.", 10, 100)
        self.assertIn("test_persona", self.chat_system.personas)
        self.assertEqual(self.chat_system.personas["test_persona"].persona_name, "test_persona")
        self.assertEqual(self.chat_system.personas["test_persona"].get_prompt(), "You are a helpful assistant.")
        self.assertEqual(self.chat_system.personas["test_persona"].get_context_length(), 10)
        self.assertEqual(self.chat_system.personas["test_persona"].get_response_token_limit(), 100)

    def test_delete_persona(self):
        self.chat_system.add_persona("test_persona", "gpt-3.5-turbo", "You are a helpful assistant.", 10, 100)
        self.chat_system.delete_persona("test_persona", save=True)
        self.assertNotIn("test_persona", self.chat_system.personas)

    def test_add_to_prompt(self):
        self.chat_system.add_persona("test_persona", "gpt-3.5-turbo", "You are a helpful assistant.", 10, 100)
        self.chat_system.add_to_prompt("test_persona", " Be very friendly.")
        self.assertEqual(self.chat_system.personas["test_persona"].get_prompt(),
                         "You are a helpful assistant. Be very friendly.")

    def test_get_persona_list(self):
        self.chat_system.add_persona("test_persona1", "gpt-3.5-turbo", "Prompt 1", 10, 100)
        self.chat_system.add_persona("test_persona2", "gpt-4", "Prompt 2", 5, 200)
        persona_list = self.chat_system.get_persona_list()
        self.assertIsInstance(persona_list, dict)
        self.assertIn("test_persona1", persona_list)
        self.assertIn("test_persona2", persona_list)

    async def test_generate_response(self):
        with patch('src.persona.Persona.generate_response') as mock_generate_response:
            mock_generate_response.return_value = "This is a test response."
            self.chat_system.add_persona("test_persona", "gpt-3.5-turbo", "You are a helpful assistant.", 10, 100)

            # Test with existing persona
            response = await self.chat_system.generate_response("test_persona", "Hello", "Previous context")
            self.assertEqual(response, "test_persona: This is a test response.")
            mock_generate_response.assert_called_once_with("Hello", "Previous context", None)

            # Reset mock for the next test
            mock_generate_response.reset_mock()

            # Test with derpr persona (special case that doesn't prepend name)
            self.chat_system.add_persona("derpr", "gpt-3.5-turbo", "You are a helpful assistant.", 10, 100)
            mock_generate_response.return_value = "This is a derpr response."
            response = await self.chat_system.generate_response("derpr", "Hello", "Previous context")
            self.assertEqual(response, "This is a derpr response.")

    async def test_generate_response_with_image(self):
        with patch('src.persona.Persona.generate_response') as mock_generate_response:
            mock_generate_response.return_value = "I see an image response."

            # Test with gpt-4o model that can handle images
            self.chat_system.add_persona("vision_persona", "gpt-4o", "You are a vision assistant.", 10, 100)
            self.chat_system.personas["vision_persona"].model.model_name = "gpt-4o"  # Ensure model name is set

            image_url = "http://example.com/image.jpg"
            response = await self.chat_system.generate_response("vision_persona", "Describe this", "Context", image_url)
            self.assertEqual(response, "vision_persona: I see an image response.")
            mock_generate_response.assert_called_once_with("Describe this", "Context", image_url)

            # Reset mock
            mock_generate_response.reset_mock()

            # Test with non-image model (should discard image_url)
            self.chat_system.add_persona("text_persona", "gpt-3.5-turbo", "You are a text assistant.", 10, 100)
            self.chat_system.personas["text_persona"].model.model_name = "gpt-3.5-turbo"  # Ensure model name is set

            response = await self.chat_system.generate_response("text_persona", "Describe this", "Context", image_url)
            self.assertEqual(response, "text_persona: I see an image response.")
            mock_generate_response.assert_called_once_with("Describe this", "Context", None)


if __name__ == '__main__':
    unittest.main()
