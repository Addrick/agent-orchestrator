# tests/interfaces/test_kobold_api.py

import pytest
from unittest.mock import patch
from src.interfaces.kobold_api import LocalModel


@pytest.fixture
def local_model() -> LocalModel:
    """Provides an instance of the LocalModel class for testing."""
    return LocalModel()


@patch('src.interfaces.kobold_api.requests')
@pytest.mark.parametrize(
    "method_name, http_verb, endpoint, method_args, expected_payload",
    [
        # --- /api/v1 endpoints ---
        ('get_max_context_length', 'get', '/api/v1/config/max_context_length', (), None),
        ('get_max_length', 'get', '/api/v1/config/max_length', (), None),
        ('generate_text', 'post', '/api/v1/generate', ('test prompt',), {'prompt': 'test prompt'}),
        ('get_api_version', 'get', '/api/v1/info/version', (), None),
        ('get_model_string', 'get', '/api/v1/model', (), None),
        # --- /api/extra endpoints ---
        ('get_true_max_context_length', 'get', '/api/extra/true_max_context_length', (), None),
        ('get_backend_version', 'get', '/api/extra/version', (), None),
        ('get_preloaded_story', 'get', '/api/extra/preloadstory', (), None),
        ('get_performance_info', 'get', '/api/extra/perf', (), None),
        ('generate_text_stream', 'post', '/api/extra/generate/stream', ('stream prompt',), {'prompt': 'stream prompt'}),
        ('poll_generation_results', 'get', '/api/extra/generate/check', (), None),
        ('token_count', 'post', '/api/extra/tokencount', ('count this text',), {'text': 'count this text'}),
        ('abort_generation', 'post', '/api/extra/abort', (), None),
        # --- /sdapi/v1 endpoints ---
        ('get_image_generation_models', 'get', '/sdapi/v1/sd-models', (), None),
        ('get_image_generation_config', 'get', '/sdapi/v1/options', (), None),
        ('get_supported_samplers', 'get', '/sdapi/v1/samplers', (), None),
        ('generate_image_from_text', 'post', '/sdapi/v1/txt2img', ('image prompt',), {'prompt': 'image prompt'}),
        ('generate_image_caption', 'post', '/sdapi/v1/interrogate', ('/path/to/image.png',),
         {'image_path': '/path/to/image.png'}),
        # --- OpenAI-compatible /v1 endpoints ---
        ('generate_text_completions', 'post', '/v1/completions', ('completion prompt',),
         {'prompt': 'completion prompt'}),
        ('generate_chat_completions', 'post', '/v1/chat/completions', ([{'role': 'user', 'content': 'hi'}],),
         {'messages': [{'role': 'user', 'content': 'hi'}]}),
        ('get_available_models', 'get', '/v1/models', (), None),
    ]
)
def test_api_methods(mock_requests, local_model, method_name, http_verb, endpoint, method_args, expected_payload):
    """
    Tests that each method in LocalModel calls the correct requests verb
    with the correct endpoint and payload.
    """
    method_to_call = getattr(local_model, method_name)
    method_to_call(*method_args)

    mock_http_method = getattr(mock_requests, http_verb)

    # Assert that the correct HTTP verb was called once
    mock_http_method.assert_called_once()

    # Extract arguments from the call
    call_args, call_kwargs = mock_http_method.call_args

    # Assert the URL is correct
    expected_url = local_model.BASE_URL + endpoint
    assert call_args[0] == expected_url

    # Assert the payload is correct for POST requests
    if http_verb == 'post':
        assert call_kwargs.get('json') == expected_payload
