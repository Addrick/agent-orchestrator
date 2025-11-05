# tests/test_engine.py

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import base64
from openai import APIStatusError, APIConnectionError
import anthropic
import aiohttp
import json

from src.engine import TextEngine, LLMCommunicationError
from config.global_config import EMPTY_RESPONSE_RETRIES
from google.genai.types import Tool, GoogleSearch

pytestmark = pytest.mark.slow


@pytest.fixture
def text_engine():
    """
    Provides a fresh, isolated TextEngine instance for each test function.
    This prevents state from bleeding between tests.
    """
    return TextEngine()


@pytest.fixture
def base_context():
    return {
        "persona_prompt": "You are a test bot.", "history": [],
        "current_message": {"text": "Hello"}
    }


@pytest.fixture
def openai_config():
    return {"model_name": "gpt-4"}


@pytest.fixture
def anthropic_config():
    return {"model_name": "claude-3-opus-20240229", "max_output_tokens": 100}


@pytest.fixture
def google_config():
    return {"model_name": "gemini-pro"}


@pytest.fixture
def local_config():
    return {"model_name": "local"}


class TestGenerateResponseLogic:
    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    @patch('src.engine.TextEngine._generate_openai_response', new_callable=AsyncMock)
    async def test_retry_on_empty_response_succeeds(self, mock_provider_call, text_engine, openai_config, base_context):
        mock_provider_call.side_effect = [
            ({}, {"payload": 1}),
            ({"type": "text", "content": "Valid response"}, {"payload": 2})
        ]
        response, _ = await text_engine.generate_response(openai_config, base_context)
        assert response == {"type": "text", "content": "Valid response"}
        assert mock_provider_call.call_count == 2

    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    @patch('src.engine.TextEngine._generate_openai_response', new_callable=AsyncMock)
    async def test_retry_on_empty_response_fails(self, mock_provider_call, text_engine, openai_config, base_context):
        mock_provider_call.return_value = ({}, {"payload": 1})
        with pytest.raises(LLMCommunicationError, match="LLM provider returned an empty or invalid response after all retries."):
            await text_engine.generate_response(openai_config, base_context)
        assert mock_provider_call.call_count == EMPTY_RESPONSE_RETRIES + 1


@patch('src.engine.AsyncOpenAI')
class TestOpenAI:
    @pytest.mark.asyncio
    async def test_success_text_response(self, mock_openai_class, text_engine, openai_config, base_context, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_openai_class.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="Success", tool_calls=None))])
        )
        response, _ = await text_engine.generate_response(openai_config, base_context)
        assert response == {"type": "text", "content": "Success"}

    @pytest.mark.asyncio
    async def test_success_tool_call_response(self, mock_openai_class, text_engine, openai_config, base_context, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_openai_class.return_value
        mock_function = MagicMock()
        mock_function.name = "get_weather"
        mock_function.arguments = '{"location": "Boston"}'
        mock_tool_call = MagicMock(id="call_123", function=mock_function)
        mock_instance.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=None, tool_calls=[mock_tool_call]))])
        )
        # FIX: Pass a non-empty 'tools' list to trigger the tool-call logic path.
        response, _ = await text_engine.generate_response(openai_config, base_context, tools=[{"type": "function", "function": {"name": "get_weather"}}])
        assert response['type'] == 'tool_calls'
        assert response['calls'][0]['name'] == 'get_weather'

    @pytest.mark.asyncio
    async def test_api_error_raises_llm_error(self, mock_openai_class, text_engine, openai_config, base_context, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_openai_class.return_value
        error = APIStatusError("Server error", response=MagicMock(status_code=500), body=None)
        mock_instance.chat.completions.create.side_effect = error
        with pytest.raises(LLMCommunicationError, match="OpenAI API returned an error"):
            await text_engine.generate_response(openai_config, base_context)


@patch('src.engine.anthropic.Anthropic')
class TestAnthropic:
    @pytest.mark.asyncio
    async def test_success_text_response(self, mock_anthropic_class, text_engine, anthropic_config, base_context, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_anthropic_class.return_value
        mock_instance.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Claude success")], stop_reason="end_turn"
        )
        response, _ = await text_engine.generate_response(anthropic_config, base_context)
        assert response == {"type": "text", "content": "Claude success"}

    @pytest.mark.asyncio
    async def test_success_tool_call_response(self, mock_anthropic_class, text_engine, anthropic_config, base_context, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_anthropic_class.return_value
        mock_tool_use = MagicMock(type='tool_use', id='tool_123', input={'ticker': 'GOOG'})
        mock_tool_use.name = 'get_stock_price'
        mock_instance.messages.create.return_value = MagicMock(content=[mock_tool_use], stop_reason="tool_use")
        response, _ = await text_engine.generate_response(anthropic_config, base_context, tools=[{"name": "get_stock_price"}])
        assert response['type'] == 'tool_calls'
        assert response['calls'][0]['name'] == 'get_stock_price'

    @pytest.mark.asyncio
    async def test_api_error_raises_llm_error(self, mock_anthropic_class, text_engine, anthropic_config, base_context, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_anthropic_class.return_value
        error = anthropic.APIStatusError("Server error", response=MagicMock(status_code=500), body=None)
        mock_instance.messages.create.side_effect = error
        with pytest.raises(LLMCommunicationError, match="Anthropic API returned an error"):
            await text_engine.generate_response(anthropic_config, base_context)


    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_image_url_passed_to_anthropic(self, mock_get, mock_anthropic_class, text_engine, anthropic_config,
                                                 base_context, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")

        # Mock the image download
        mock_response = AsyncMock()
        mock_response.read.return_value = b'imagedata'
        mock_response.content_type = 'image/png'
        mock_get.return_value.__aenter__.return_value = mock_response

        # Mock the Claude API response
        mock_instance = mock_anthropic_class.return_value
        mock_instance.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Image received")], stop_reason="end_turn"
        )

        base_context["current_message"]["image_url"] = "http://example.com/image.png"
        base_context["history"] = [{"role": "user", "content": "Check this out"}]

        await text_engine.generate_response(anthropic_config, base_context)

        # Verify that the image was included in the API call
        call_args = mock_instance.messages.create.call_args[1]
        assert call_args['messages'][-1]['content'][-1]['type'] == 'image'
        assert call_args['messages'][-1]['content'][-1]['source']['data'] == base64.b64encode(b'imagedata').decode('utf-8')


@patch('src.engine.genai.client.AsyncClient')
class TestGoogle:
    @pytest.mark.asyncio
    async def test_success_text_response(self, mock_google_client_class, text_engine, google_config, base_context, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value
        mock_part = MagicMock(text="Google success", function_call=None)
        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]), grounding_metadata=None)
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
        )
        response, _ = await text_engine.generate_response(google_config, base_context)
        assert response == {"type": "text", "content": "Google success"}

    @pytest.mark.asyncio
    async def test_success_tool_call_response(self, mock_google_client_class, text_engine, google_config, base_context,
                                              monkeypatch):
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value

        # Mock the specific Google API structure for a function call
        mock_function_call = MagicMock()
        mock_function_call.name = "search_web"
        # Note: Google's 'args' attribute is already a dict-like object, not a JSON string
        mock_function_call.args = {'query': 'python testing'}

        mock_part = MagicMock(text=None, function_call=mock_function_call)
        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]), grounding_metadata=None)
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
        )

        # Pass a non-empty 'tools' list to trigger the tool-call logic path
        response, _ = await text_engine.generate_response(google_config, base_context, tools=[
            {"type": "function", "function": {"name": "search_web"}}])

        assert response['type'] == 'tool_calls'
        assert len(response['calls']) == 1
        assert response['calls'][0]['name'] == 'search_web'
        assert response['calls'][0]['arguments'] == {'query': 'python testing'}

    @pytest.mark.asyncio
    async def test_api_error_raises_llm_error(self, mock_google_client_class, text_engine, google_config, base_context, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value
        mock_instance.models.generate_content.side_effect = Exception("API failure")
        with pytest.raises(LLMCommunicationError, match="An error occurred with Google API"):
            await text_engine.generate_response(google_config, base_context)


    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_image_url_passed_to_google(self, mock_get, mock_google_client_class, text_engine, google_config,
                                              base_context, monkeypatch):
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")

        # Mock the image download
        mock_response = AsyncMock()
        mock_response.read.return_value = b'imagedata'
        mock_response.content_type = 'image/jpeg'
        mock_get.return_value.__aenter__.return_value = mock_response

        # Mock the Gemini API response
        mock_instance = mock_google_client_class.return_value
        mock_part = MagicMock()
        mock_part.function_call = None
        mock_part.text = "Image received"
        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]))
        mock_candidate.grounding_metadata = None
        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
        )

        base_context["current_message"]["image_url"] = "http://example.com/image.jpg"
        base_context["history"] = [{"role": "user", "content": "Check this out"}]

        await text_engine.generate_response(google_config, base_context)

        # Verify that the image was included in the API call
        call_args = mock_instance.models.generate_content.call_args[1]
        assert len(call_args['contents'][-1]['parts']) == 2
        assert call_args['contents'][-1]['parts'][-1].inline_data.data == b'imagedata'


class TestLocalModel:
    @pytest.mark.asyncio
    @patch('src.engine.AsyncOpenAI')
    async def test_success_text_response(self, mock_async_openai, text_engine, local_config, base_context):
        mock_client_instance = mock_async_openai.return_value
        mock_client_instance.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="Local success", tool_calls=None))])
        )
        response, _ = await text_engine.generate_response(local_config, base_context)
        assert response == {"type": "text", "content": "Local success"}
        mock_client_instance.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    @patch('src.engine.AsyncOpenAI')
    async def test_success_tool_call_response(self, mock_async_openai, text_engine, local_config, base_context):
        """
        Tests that a successful local model tool call is parsed correctly.
        """
        mock_client_instance = mock_async_openai.return_value

        # Mock the OpenAI-compatible response for a tool call
        mock_function = MagicMock()
        mock_function.name = "run_code"
        mock_function.arguments = '{"code": "print(\'hello from local\')"}'

        mock_tool_call = MagicMock(id="call_local_123", function=mock_function)
        mock_client_instance.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=None, tool_calls=[mock_tool_call]))])
        )

        # Pass a non-empty 'tools' list to trigger the tool-call logic path
        response, _ = await text_engine.generate_response(local_config, base_context, tools=[
            {"type": "function", "function": {"name": "run_code"}}])

        assert response['type'] == 'tool_calls'
        assert len(response['calls']) == 1
        assert response['calls'][0]['name'] == 'run_code'
        assert response['calls'][0]['arguments'] == {'code': "print('hello from local')"}

    @pytest.mark.asyncio
    @patch('src.engine.AsyncOpenAI')
    async def test_connection_error_raises_llm_error(self, mock_async_openai, text_engine, local_config, base_context):
        mock_client_instance = mock_async_openai.return_value
        mock_client_instance.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())
        with pytest.raises(LLMCommunicationError, match="An unexpected error occurred with the Local API."):
            await text_engine.generate_response(local_config, base_context)
