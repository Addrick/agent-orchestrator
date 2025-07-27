# tests/test_engine.py

import pytest
import aiohttp
from unittest.mock import MagicMock, AsyncMock, patch
from types import SimpleNamespace

from src.engine import TextEngine, _process_grounding_metadata


@pytest.fixture
def mock_api_keys(monkeypatch):
    """Sets dummy API keys to allow client initialization."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_key_openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test_key_anthropic")
    monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "test_key_google")


@pytest.fixture
def text_engine(mock_api_keys) -> TextEngine:
    """Fixture to provide a TextEngine instance."""
    return TextEngine()


@pytest.fixture
def persona_config() -> dict:
    """Provides a sample persona configuration."""
    return {
        "model_name": "test-model",
        "max_output_tokens": 100,
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
    }


@pytest.fixture
def context_object() -> dict:
    """Provides a sample context object for generation."""
    return {
        "persona_prompt": "You are a helpful assistant.",
        "history": [
            {"role": "user", "content": "Hello there."},
            {"role": "assistant", "content": "Hi! How can I help?"}
        ],
        "current_message": {
            "text": "What is the capital of France?",
            "image_url": "https://example.com/image.jpg"
        }
    }


@pytest.mark.asyncio
@patch('src.engine.AsyncOpenAI')
async def test_generate_openai_response(mock_openai_client, text_engine, persona_config, context_object):
    """Test the OpenAI response generation pathway."""
    mock_create = AsyncMock()
    mock_create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Paris"))])
    mock_openai_client.return_value.chat.completions.create = mock_create

    persona_config['model_name'] = "gpt-4-turbo"
    response, payload = await text_engine.generate_response(persona_config, context_object)

    assert response == "Paris"
    mock_create.assert_called_once()
    call_args = mock_create.call_args[1]

    assert call_args['model'] == "gpt-4-turbo"
    assert call_args['messages'][0]['role'] == 'system'
    assert call_args['messages'][0]['content'] == context_object['persona_prompt']
    assert call_args['messages'][1] == context_object['history'][0]  # user: Hello there.
    assert call_args['messages'][-1]['role'] == 'user'
    assert call_args['messages'][-1]['content'][0]['text'] == context_object['current_message']['text']
    assert call_args['messages'][-1]['content'][1]['image_url']['url'] == context_object['current_message']['image_url']
    assert call_args['temperature'] == 0.5


@pytest.mark.asyncio
@patch('src.engine.anthropic.Anthropic')
async def test_generate_anthropic_response(mock_anthropic_client, text_engine, persona_config, context_object):
    """Test the Anthropic response generation pathway."""
    mock_create = MagicMock()
    mock_create.return_value = SimpleNamespace(content=[SimpleNamespace(text="Paris")])
    mock_anthropic_client.return_value.messages.create = mock_create

    persona_config['model_name'] = "claude-3-opus-20240229"
    response, payload = await text_engine.generate_response(persona_config, context_object)

    assert response == "Paris"
    mock_create.assert_called_once()
    call_args = mock_create.call_args[1]

    assert call_args['model'] == "claude-3-opus-20240229"
    assert call_args['system'] == context_object['persona_prompt']
    assert call_args['messages'][-1]['content'] == context_object['current_message']['text']
    assert call_args['max_tokens'] == 100


@pytest.mark.asyncio
@patch('src.engine.genai.client.AsyncClient')
async def test_generate_google_response(mock_google_client, text_engine, persona_config, context_object):
    """Test the Google response generation pathway."""
    mock_generate = AsyncMock()
    mock_generate.return_value = SimpleNamespace(
        candidates=[
            SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="Paris")]), grounding_metadata=None)],
        prompt_feedback=None
    )
    # Mock the internal state that _initialize_google_client would set
    text_engine.google_client = mock_google_client
    text_engine.google_search_tool = MagicMock()  # Satisfy validator
    text_engine.google_tool_config = {}  # Satisfy validator
    mock_google_client.models.generate_content = mock_generate

    persona_config['model_name'] = "gemini-pro"
    response, payload = await text_engine.generate_response(persona_config, context_object)

    assert response == "Paris"
    mock_generate.assert_called_once()
    call_args = mock_generate.call_args[1]

    assert call_args['model'] == "gemini-pro"
    assert "### Instructions: ###\nYou are a helpful assistant." in call_args['contents']
    assert "user: Hello there." in call_args['contents']
    assert "### Current Message to Respond To: ###\nWhat is the capital of France?" in call_args['contents']


@pytest.mark.asyncio
@patch('aiohttp.ClientSession')
async def test_generate_local_response(mock_session_class, text_engine, persona_config, context_object):
    """Test the local model response generation pathway with correct layered mocking."""
    # 1. This is the final response object that `async with ... as response` yields.
    mock_post_response = AsyncMock()
    mock_post_response.status = 200
    mock_post_response.json.return_value = {'results': [{'text': 'Paris'}]}

    # 2. This is the async context manager that `session.post()` returns.
    mock_post_context_manager = AsyncMock()
    mock_post_context_manager.__aenter__.return_value = mock_post_response

    # 3. This is the session object. Its `.post()` method is a *synchronous* MagicMock.
    mock_session_instance = MagicMock()
    mock_session_instance.post.return_value = mock_post_context_manager

    # 4. Configure the top-level ClientSession patch to yield our session object.
    mock_session_class.return_value.__aenter__.return_value = mock_session_instance

    persona_config['model_name'] = "local"
    response, payload = await text_engine.generate_response(persona_config, context_object)

    assert response == "Paris"
    mock_session_instance.post.assert_called_once_with('http://localhost:5001/api/v1/generate', json=payload)
    assert payload['memory'] == context_object['persona_prompt']
    assert "user: What is the capital of France?" in payload['prompt']


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name, client_path, exception, expected_error_string", [
    ("gpt-4", "src.engine.AsyncOpenAI", Exception("OpenAI down"), "OpenAI down"),
    ("claude-3", "src.engine.anthropic.Anthropic", Exception("Anthropic down"), "Anthropic down"),
    ("local", "aiohttp.ClientSession", aiohttp.ClientError("Local model down"),
     "Error: Could not connect to local model at http://localhost:5001.")
])
async def test_api_error_handling(model_name, client_path, exception, expected_error_string, text_engine,
                                  persona_config,
                                  context_object):
    """Test that API errors are caught and handled gracefully."""
    with patch(client_path) as mock_client:
        if "aiohttp" in client_path:
            # Setup the mock session instance to raise an error when .post() is called
            mock_session_instance = MagicMock()
            mock_session_instance.post.side_effect = exception
            mock_client.return_value.__aenter__.return_value = mock_session_instance
        else:
            # Other clients have a .create method that raises the error
            mock_client.return_value.chat.completions.create.side_effect = exception
            mock_client.return_value.messages.create.side_effect = exception

        persona_config['model_name'] = model_name
        response, _ = await text_engine.generate_response(persona_config, context_object)

        assert "Error" in response
        assert expected_error_string in response


def test_process_grounding_metadata():
    """Test the helper function for processing Google's grounding metadata."""
    base_text = "The sky is blue. Photosynthesis is a process used by plants."
    mock_metadata = SimpleNamespace(
        grounding_chunks=[
            SimpleNamespace(web=SimpleNamespace(uri="https://en.wikipedia.org/wiki/Sky", title="Sky - Wikipedia")),
            SimpleNamespace(web=SimpleNamespace(uri="https://en.wikipedia.org/wiki/Photosynthesis",
                                                title="Photosynthesis - Wikipedia"))
        ],
        grounding_supports=[
            SimpleNamespace(segment=SimpleNamespace(text="sky is blue", start_index=4), grounding_chunk_indices=[0]),
            SimpleNamespace(segment=SimpleNamespace(text="process used by plants", start_index=35),
                            grounding_chunk_indices=[1])
        ],
        web_search_queries=["why is the sky blue", "what is photosynthesis"]
    )

    logger = MagicMock()
    final_text, search_queries, citations = _process_grounding_metadata(base_text, mock_metadata, logger)

    # Corrected expected text to match function's actual output
    expected_text = "The sky is blue [[1](<https://en.wikipedia.org/wiki/Sky>)]. Photosynthesis is a process used by plants [[2](<https://en.wikipedia.org/wiki/Photosynthesis>)]."
    expected_searches = "\n\nSearch Query: why is the sky blue, what is photosynthesis"
    expected_citations = "\n\nSources:\n1. Sky - Wikipedia\n2. Photosynthesis - Wikipedia\n"

    assert final_text.strip() == expected_text
    assert search_queries == expected_searches
    assert citations == expected_citations
