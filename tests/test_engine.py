# tests/engine/test_engine.py

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from openai import APIStatusError, APITimeoutError
import anthropic
import aiohttp

from src.engine import TextEngine, LLMCommunicationError
from config.global_config import EMPTY_RESPONSE_RETRIES
from google.genai.types import Tool, GoogleSearch


# A common set of configs and contexts for all tests
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
                ("", {"payload": 1}),
                ("Valid response", {"payload": 2})
            ]

            response, _ = await text_engine.generate_response(openai_config, base_context)

            assert response == "Valid response"
            assert mock_provider_call.call_count == 2

    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    async def test_retry_on_empty_response_fails(self, text_engine, openai_config, base_context):
        with patch.object(text_engine, '_generate_openai_response', new_callable=AsyncMock) as mock_provider_call:
            mock_provider_call.return_value = ("", {"payload": 1})

            with pytest.raises(LLMCommunicationError,
                               match="LLM provider returned an empty response after all retries."):
                await text_engine.generate_response(openai_config, base_context)

            assert mock_provider_call.call_count == EMPTY_RESPONSE_RETRIES + 1


class TestOpenAI:
    @pytest.mark.asyncio
    @patch('src.engine.TextEngine._get_openai_client', new_callable=AsyncMock)
    async def test_success_first_try(self, mock_get_client, text_engine, openai_config, base_context):
        mock_client_instance = AsyncMock()
        mock_get_client.return_value = mock_client_instance
        mock_create = mock_client_instance.chat.completions.create
        mock_success = MagicMock(choices=[MagicMock(message=MagicMock(content="Success"))])
        mock_create.return_value = mock_success

        response, _ = await text_engine._generate_openai_response(openai_config, base_context)
        assert response == "Success"
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    @patch('src.engine.TextEngine._get_openai_client', new_callable=AsyncMock)
    async def test_retry_on_500_error_succeeds(self, mock_get_client, text_engine, openai_config, base_context):
        mock_client_instance = AsyncMock()
        mock_get_client.return_value = mock_client_instance
        mock_create = mock_client_instance.chat.completions.create
        mock_response_500 = MagicMock(status_code=500)
        error_500 = APIStatusError("Server error", response=mock_response_500, body=None)
        mock_success = MagicMock(choices=[MagicMock(message=MagicMock(content="Success on retry"))])
        mock_create.side_effect = [error_500, mock_success]

        response, _ = await text_engine._generate_openai_response(openai_config, base_context)
        assert response == "Success on retry"
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    @patch('src.engine.TextEngine._get_openai_client', new_callable=AsyncMock)
    async def test_retry_fails_raises_custom_error(self, mock_get_client, text_engine, openai_config, base_context):
        mock_client_instance = AsyncMock()
        mock_get_client.return_value = mock_client_instance
        mock_create = mock_client_instance.chat.completions.create
        mock_response_500 = MagicMock(status_code=500)
        error_500 = APIStatusError("Server error", response=mock_response_500, body=None)
        mock_create.side_effect = [error_500, error_500]

        with pytest.raises(LLMCommunicationError, match="OpenAI API is unavailable after retry."):
            await text_engine._generate_openai_response(openai_config, base_context)
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    @patch('src.engine.TextEngine._get_openai_client', new_callable=AsyncMock)
    async def test_4xx_error_does_not_retry(self, mock_get_client, text_engine, openai_config, base_context):
        mock_client_instance = AsyncMock()
        mock_get_client.return_value = mock_client_instance
        mock_create = mock_client_instance.chat.completions.create
        mock_response_400 = MagicMock(status_code=400)
        error_400 = APIStatusError("Bad request", response=mock_response_400, body=None)
        mock_create.side_effect = error_400

        with pytest.raises(LLMCommunicationError, match="OpenAI API returned a client error"):
            await text_engine._generate_openai_response(openai_config, base_context)
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.engine.TextEngine._get_openai_client', new_callable=AsyncMock)
    async def test_malformed_response_raises_error(self, mock_get_client, text_engine, openai_config, base_context):
        mock_client_instance = AsyncMock()
        mock_get_client.return_value = mock_client_instance
        mock_create = mock_client_instance.chat.completions.create
        malformed_response = MagicMock(choices=[])  # Empty choices list
        mock_create.return_value = malformed_response

        with pytest.raises(LLMCommunicationError, match="An unexpected error occurred with the OpenAI API."):
            await text_engine._generate_openai_response(openai_config, base_context)


class TestAnthropic:
    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    @patch('src.engine.TextEngine._get_anthropic_client')
    async def test_retry_on_5xx_error_succeeds(self, mock_get_client, text_engine, anthropic_config, base_context):
        mock_client_instance = MagicMock()
        mock_get_client.return_value = mock_client_instance
        mock_create = mock_client_instance.messages.create
        mock_response_503 = MagicMock(status_code=503)
        error_503 = anthropic.APIStatusError("Service unavailable", response=mock_response_503, body=None)
        mock_success = MagicMock(content=[MagicMock(text="Claude success")])
        mock_create.side_effect = [error_503, mock_success]

        response, _ = await text_engine._generate_anthropic_response(anthropic_config, base_context)
        assert response == "Claude success"
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    @patch('src.engine.TextEngine._get_anthropic_client')
    async def test_retry_fails_raises_custom_error(self, mock_get_client, text_engine, anthropic_config, base_context):
        mock_client_instance = MagicMock()
        mock_get_client.return_value = mock_client_instance
        mock_create = mock_client_instance.messages.create
        mock_response_500 = MagicMock(status_code=500)
        error_500 = anthropic.APIStatusError("Server error", response=mock_response_500, body=None)
        mock_create.side_effect = [error_500, error_500]

        with pytest.raises(LLMCommunicationError, match="Anthropic API is unavailable after retry."):
            await text_engine._generate_anthropic_response(anthropic_config, base_context)
        assert mock_create.call_count == 2


class TestGoogle:
    @pytest.mark.asyncio
    @patch('src.engine.TextEngine._initialize_google_client')
    async def test_empty_response_is_handled(self, mock_init_google, text_engine, google_config, base_context):
        mock_init_google.return_value = None
        text_engine.google_search_tool = Tool(google_search=GoogleSearch())
        mock_generate_content = AsyncMock()
        text_engine.google_client = MagicMock(models=MagicMock(generate_content=mock_generate_content))
        empty_response = MagicMock(prompt_feedback=None, candidates=[MagicMock(content=MagicMock(parts=[]))])
        mock_generate_content.return_value = empty_response

        response, _ = await text_engine._generate_google_response(google_config, base_context)
        assert response == ""

    @pytest.mark.asyncio
    @patch('src.engine.TextEngine._initialize_google_client')
    async def test_blocked_prompt_raises_error(self, mock_init_google, text_engine, google_config, base_context):
        mock_init_google.return_value = None
        text_engine.google_search_tool = Tool(google_search=GoogleSearch())
        mock_generate_content = AsyncMock()
        text_engine.google_client = MagicMock(models=MagicMock(generate_content=mock_generate_content))

        # THE FIX: Mock the .name attribute and the __str__ representation
        mock_block_reason = MagicMock()
        mock_block_reason.name = "SAFETY"

        blocked_response = MagicMock(prompt_feedback=MagicMock(block_reason=mock_block_reason))
        mock_generate_content.return_value = blocked_response

        with pytest.raises(LLMCommunicationError, match="Response blocked by Google due to SAFETY"):
            await text_engine._generate_google_response(google_config, base_context)


class TestLocalModel:
    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    @patch('aiohttp.ClientSession')
    async def test_retry_on_client_error_succeeds(self, mock_session, text_engine, local_config, base_context):
        mock_session_cm = AsyncMock(__aenter__=AsyncMock(return_value=AsyncMock(post=MagicMock())))
        mock_session.return_value = mock_session_cm
        session_instance = mock_session_cm.__aenter__.return_value

        mock_response = AsyncMock(json=AsyncMock(return_value={'results': [{'text': 'Local success'}]}),
                                  raise_for_status=MagicMock())
        mock_response_cm = AsyncMock(__aenter__=AsyncMock(return_value=mock_response))
        error = aiohttp.ClientError("Connection failed")
        session_instance.post.side_effect = [error, mock_response_cm]

        response, _ = await text_engine._generate_local_response(local_config, base_context)
        assert response == "Local success"
        assert session_instance.post.call_count == 2
