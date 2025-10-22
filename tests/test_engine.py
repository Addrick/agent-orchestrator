# tests/test_engine.py

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from openai import APIStatusError
import anthropic
import aiohttp
import json

from src.engine import TextEngine, LLMCommunicationError
from config.global_config import EMPTY_RESPONSE_RETRIES
from google.genai.types import Tool, GoogleSearch


@pytest.fixture
def text_engine():
    """Provides a fresh TextEngine instance for each test."""
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
    return {"model_name": "local", "max_output_tokens": 100}


class TestGenerateResponseLogic:
    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    async def test_retry_on_empty_response_succeeds(self, text_engine, openai_config, base_context):
        with patch.object(text_engine, '_generate_openai_response', new_callable=AsyncMock) as mock_provider_call:
            mock_provider_call.side_effect = [
                ({}, {"payload": 1}),
                ({"type": "text", "content": "Valid response"}, {"payload": 2})
            ]
            response, _ = await text_engine.generate_response(openai_config, base_context)
            assert response == {"type": "text", "content": "Valid response"}
            assert mock_provider_call.call_count == 2

    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    async def test_retry_on_empty_response_fails(self, text_engine, openai_config, base_context):
        with patch.object(text_engine, '_generate_openai_response', new_callable=AsyncMock) as mock_provider_call:
            mock_provider_call.return_value = ({}, {"payload": 1})
            with pytest.raises(LLMCommunicationError,
                               match="LLM provider returned an empty or invalid response after all retries."):
                await text_engine.generate_response(openai_config, base_context)
            assert mock_provider_call.call_count == EMPTY_RESPONSE_RETRIES + 1


@patch('src.engine.AsyncOpenAI')
class TestOpenAI:
    @pytest.mark.asyncio
    async def test_success_text_response(self, mock_openai_class, text_engine, openai_config, base_context,
                                         monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_openai_class.return_value
        mock_instance.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="Success", tool_calls=None))])
        )
        response, _ = await text_engine.generate_response(openai_config, base_context)
        assert response == {"type": "text", "content": "Success"}
        mock_instance.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_success_tool_call_response(self, mock_openai_class, text_engine, openai_config, base_context,
                                              monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_openai_class.return_value
        mock_function = MagicMock()
        mock_function.name = "get_weather"
        mock_function.arguments = '{"location": "Boston"}'
        mock_tool_call = MagicMock(id="call_123", function=mock_function)
        mock_instance.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=None, tool_calls=[mock_tool_call]))])
        )
        response, _ = await text_engine.generate_response(openai_config, base_context)
        assert response['type'] == 'tool_calls'
        assert response['calls'][0]['name'] == 'get_weather'

    @pytest.mark.asyncio
    async def test_api_error_raises_llm_error(self, mock_openai_class, text_engine, openai_config, base_context,
                                              monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_openai_class.return_value
        error = APIStatusError("Server error", response=MagicMock(status_code=500), body=None)
        mock_instance.chat.completions.create = AsyncMock(side_effect=error)
        with pytest.raises(LLMCommunicationError, match="OpenAI API returned an error"):
            await text_engine.generate_response(openai_config, base_context)


@patch('src.engine.anthropic.Anthropic')
class TestAnthropic:
    @pytest.mark.asyncio
    async def test_success_text_response(self, mock_anthropic_class, text_engine, anthropic_config, base_context,
                                         monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_anthropic_class.return_value
        mock_instance.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Claude success")], stop_reason="end_turn"
        )
        response, _ = await text_engine.generate_response(anthropic_config, base_context)
        assert response == {"type": "text", "content": "Claude success"}

    @pytest.mark.asyncio
    async def test_success_tool_call_response(self, mock_anthropic_class, text_engine, anthropic_config, base_context,
                                              monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_anthropic_class.return_value
        mock_tool_use = MagicMock(type='tool_use', id='tool_123', input={'ticker': 'GOOG'})
        mock_tool_use.name = 'get_stock_price'
        mock_instance.messages.create.return_value = MagicMock(content=[mock_tool_use], stop_reason="tool_use")
        response, _ = await text_engine.generate_response(anthropic_config, base_context)
        assert response['type'] == 'tool_calls'
        assert response['calls'][0]['name'] == 'get_stock_price'

    @pytest.mark.asyncio
    async def test_api_error_raises_llm_error(self, mock_anthropic_class, text_engine, anthropic_config, base_context,
                                              monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_anthropic_class.return_value
        error = anthropic.APIStatusError("Server error", response=MagicMock(status_code=500), body=None)
        mock_instance.messages.create.side_effect = error
        with pytest.raises(LLMCommunicationError, match="Anthropic API returned an error"):
            await text_engine.generate_response(anthropic_config, base_context)


@patch('src.engine.genai.client.AsyncClient')
class TestGoogle:
    @pytest.mark.asyncio
    async def test_success_text_response(self, mock_google_client_class, text_engine, google_config, base_context,
                                         monkeypatch):
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value
        mock_part = MagicMock()
        mock_part.function_call = None
        mock_part.text = "Google success"

        mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]))
        mock_candidate.grounding_metadata = None

        mock_instance.models.generate_content = AsyncMock(
            return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
        )
        response, _ = await text_engine.generate_response(google_config, base_context)
        assert response == {"type": "text", "content": "Google success"}

    @pytest.mark.asyncio
    async def test_api_error_raises_llm_error(self, mock_google_client_class, text_engine, google_config, base_context,
                                              monkeypatch):
        monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
        mock_instance = mock_google_client_class.return_value
        mock_instance.models.generate_content = AsyncMock(side_effect=Exception("API failure"))
        with pytest.raises(LLMCommunicationError, match="An error occurred with Google API"):
            await text_engine.generate_response(google_config, base_context)


class TestLocalModel:
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_success_text_response(self, mock_post, text_engine, local_config, base_context):
        mock_response = AsyncMock(
            json=AsyncMock(return_value={'results': [{'text': 'Local success'}]}),
            raise_for_status=MagicMock()
        )
        mock_post.return_value.__aenter__.return_value = mock_response
        response, _ = await text_engine.generate_response(local_config, base_context)
        assert response == {"type": "text", "content": "Local success"}

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_connection_error_raises_llm_error(self, mock_post, text_engine, local_config, base_context):
        mock_post.side_effect = aiohttp.ClientError("Connection failed")
        with pytest.raises(LLMCommunicationError, match="Could not connect to local model."):
            await text_engine.generate_response(local_config, base_context)