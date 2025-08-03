# tests/engine/test_engine.py

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from openai import APIStatusError, APITimeoutError
import anthropic
import aiohttp

from src.engine import TextEngine, LLMCommunicationError


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


class TestOpenAI:
    @pytest.mark.asyncio
    @patch('src.engine.TextEngine._get_openai_client', new_callable=AsyncMock)
    async def test_success_first_try(self, mock_get_client, text_engine, openai_config, base_context):
        mock_client_instance = AsyncMock()
        mock_get_client.return_value = mock_client_instance

        mock_create = mock_client_instance.chat.completions.create
        mock_success = MagicMock()
        mock_success.choices = [MagicMock(message=MagicMock(content="Success"))]
        mock_create.return_value = mock_success

        response, _ = await text_engine.generate_response(openai_config, base_context)

        assert response == "Success"
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    @patch('src.engine.TextEngine._get_openai_client', new_callable=AsyncMock)
    async def test_retry_on_500_error_succeeds(self, mock_get_client, text_engine, openai_config, base_context):
        mock_client_instance = AsyncMock()
        mock_get_client.return_value = mock_client_instance
        mock_create = mock_client_instance.chat.completions.create

        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500
        error_500 = APIStatusError("Server error", response=mock_response_500, body=None)

        mock_success = MagicMock()
        mock_success.choices = [MagicMock(message=MagicMock(content="Success on retry"))]
        mock_create.side_effect = [error_500, mock_success]

        response, _ = await text_engine.generate_response(openai_config, base_context)

        assert response == "Success on retry"
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    @patch('src.engine.TextEngine._get_openai_client', new_callable=AsyncMock)
    async def test_retry_on_timeout_succeeds(self, mock_get_client, text_engine, openai_config, base_context):
        mock_client_instance = AsyncMock()
        mock_get_client.return_value = mock_client_instance
        mock_create = mock_client_instance.chat.completions.create

        error_timeout = APITimeoutError(request=MagicMock())
        mock_success = MagicMock()
        mock_success.choices = [MagicMock(message=MagicMock(content="Success after timeout"))]
        mock_create.side_effect = [error_timeout, mock_success]

        response, _ = await text_engine.generate_response(openai_config, base_context)

        assert response == "Success after timeout"
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    @patch('src.engine.TextEngine._get_openai_client', new_callable=AsyncMock)
    async def test_retry_fails_raises_custom_error(self, mock_get_client, text_engine, openai_config, base_context):
        mock_client_instance = AsyncMock()
        mock_get_client.return_value = mock_client_instance
        mock_create = mock_client_instance.chat.completions.create

        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500
        error_500 = APIStatusError("Server error", response=mock_response_500, body=None)
        mock_create.side_effect = [error_500, error_500]

        with pytest.raises(LLMCommunicationError, match="OpenAI API is unavailable after retry."):
            await text_engine.generate_response(openai_config, base_context)
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    @patch('src.engine.TextEngine._get_openai_client', new_callable=AsyncMock)
    async def test_4xx_error_does_not_retry(self, mock_get_client, text_engine, openai_config, base_context):
        mock_client_instance = AsyncMock()
        mock_get_client.return_value = mock_client_instance
        mock_create = mock_client_instance.chat.completions.create

        mock_response_400 = MagicMock()
        mock_response_400.status_code = 400
        error_400 = APIStatusError("Bad request", response=mock_response_400, body=None)
        mock_create.side_effect = error_400

        with pytest.raises(LLMCommunicationError, match="OpenAI API returned a client error"):
            await text_engine.generate_response(openai_config, base_context)
        mock_create.assert_called_once()


class TestAnthropic:
    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    @patch('anthropic.Anthropic')
    async def test_retry_on_5xx_error_succeeds(self, mock_anthropic_client, text_engine, anthropic_config,
                                               base_context):
        mock_client = mock_anthropic_client.return_value.messages
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        error_503 = anthropic.APIStatusError("Service unavailable", response=mock_response_503, body=None)

        mock_success = MagicMock()
        mock_success.content = [MagicMock(text="Claude success")]
        mock_client.create.side_effect = [error_503, mock_success]

        response, _ = await text_engine.generate_response(anthropic_config, base_context)

        assert response == "Claude success"
        assert mock_client.create.call_count == 2

    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    @patch('anthropic.Anthropic')
    async def test_retry_fails_raises_custom_error(self, mock_anthropic_client, text_engine, anthropic_config,
                                                   base_context):
        mock_client = mock_anthropic_client.return_value.messages
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        error_503 = anthropic.APIStatusError("Service unavailable", response=mock_response_503, body=None)
        mock_client.create.side_effect = [error_503, error_503]

        with pytest.raises(LLMCommunicationError, match="Anthropic API is unavailable after retry."):
            await text_engine.generate_response(anthropic_config, base_context)
        assert mock_client.create.call_count == 2

    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    @patch('anthropic.Anthropic')
    async def test_4xx_error_does_not_retry(self, mock_anthropic_client, text_engine, anthropic_config, base_context):
        mock_client = mock_anthropic_client.return_value.messages
        mock_response_401 = MagicMock()
        mock_response_401.status_code = 401
        error_401 = anthropic.APIStatusError("Unauthorized", response=mock_response_401, body=None)
        mock_client.create.side_effect = error_401

        with pytest.raises(LLMCommunicationError, match="Anthropic API returned a client error"):
            await text_engine.generate_response(anthropic_config, base_context)
        mock_client.create.assert_called_once()


class TestLocalModel:
    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    @patch('aiohttp.ClientSession')
    async def test_retry_on_client_error_succeeds(self, mock_session, text_engine, local_config, base_context):
        # This mock structure correctly simulates the nested async context managers
        mock_session_cm = AsyncMock()
        mock_session_instance = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session_instance
        mock_session.return_value = mock_session_cm

        # Make session.post a regular mock that returns an async context manager
        mock_session_instance.post = MagicMock()

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={'results': [{'text': 'Local success'}]})
        mock_response.raise_for_status = MagicMock()
        mock_response_cm = AsyncMock()
        mock_response_cm.__aenter__.return_value = mock_response

        error = aiohttp.ClientError("Connection failed")
        mock_session_instance.post.side_effect = [error, mock_response_cm]

        response, _ = await text_engine.generate_response(local_config, base_context)

        assert response == "Local success"
        assert mock_session_instance.post.call_count == 2

    @pytest.mark.asyncio
    @patch('src.engine.time.sleep', MagicMock())
    @patch('aiohttp.ClientSession')
    async def test_retry_fails_raises_custom_error(self, mock_session, text_engine, local_config, base_context):
        mock_session_cm = AsyncMock()
        mock_session_instance = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session_instance
        mock_session.return_value = mock_session_cm

        error = aiohttp.ClientError("Connection failed")
        mock_session_instance.post = MagicMock(side_effect=[error, error])

        with pytest.raises(LLMCommunicationError, match="Could not connect to local model after retry."):
            await text_engine.generate_response(local_config, base_context)
        assert mock_session_instance.post.call_count == 2
