import pytest
from unittest.mock import patch
from src.utils import model_utils


@patch('src.utils.model_utils.persona_store.load_models_from_file')
def test_get_model_list_no_update(mock_load_models):
    """update=False reads the cache and merges the static (non-API) models in."""
    mock_load_models.return_value = {"From OpenAI": ["gpt-4"]}

    result = model_utils.get_model_list(update=False)

    mock_load_models.assert_called_once()
    # API group from the cache is preserved...
    assert result["From OpenAI"] == ["gpt-4"]
    # ...and the static groups are always present.
    assert result["Antigravity (OAuth tier)"] == ["agy-flash"]
    assert result["Local"] == ["local"]


@patch('src.utils.model_utils.persona_store.load_models_from_file')
def test_get_model_list_no_update_stale_cache_gets_static_models(mock_load_models):
    """A cache written before agy existed still exposes agy/local on read.

    This is the bug the fix targets: the web UI dropdown and `what models`
    both read the cache, so static models must not depend on update_models
    having been run after they were introduced.
    """
    mock_load_models.return_value = {
        "From OpenAI": ["gpt-4"],
        "From Anthropic": ["claude-3"],
        "Local": ["local"],
        # NOTE: no 'Antigravity (OAuth tier)' group — pre-agy cache shape.
    }

    result = model_utils.get_model_list(update=False)

    assert "Antigravity (OAuth tier)" in result
    assert result["Antigravity (OAuth tier)"] == ["agy-flash"]
    # The pre-existing API/local groups are untouched.
    assert result["From OpenAI"] == ["gpt-4"]


@patch('src.utils.model_utils.persona_store.load_models_from_file')
def test_get_model_list_no_update_empty_cache(mock_load_models):
    """No cache at all (fresh install) still surfaces the static models."""
    mock_load_models.return_value = None

    result = model_utils.get_model_list(update=False)

    assert result == model_utils.STATIC_MODELS
    assert result["Antigravity (OAuth tier)"] == ["agy-flash"]
    assert result["Local"] == ["local"]


@patch('src.utils.model_utils.persona_store.save_models_to_file')
@patch('src.utils.model_utils.refresh_available_anthropic_models')
@patch('src.utils.model_utils.refresh_available_google_models')
@patch('src.utils.model_utils.refresh_available_openai_models')
def test_get_model_list_with_update(mock_openai, mock_google, mock_anthropic, mock_save):
    """Tests get_model_list when update=True, ensuring it calls refresh and save utilities."""
    mock_openai.return_value = ["gpt-4"]
    mock_google.return_value = ["gemini-pro"]
    mock_anthropic.return_value = ["claude-3"]

    expected_combined_dict = {
        'From OpenAI': ["gpt-4"],
        'From Google': ["gemini-pro"],
        'From Anthropic': ["claude-3"],
        'Antigravity (OAuth tier)': ['agy-flash'],
        'Claude Code (sandboxed)': ['cc-sonnet', 'cc-opus', 'cc-haiku'],
        'Local': ['local']
    }

    result = model_utils.get_model_list(update=True)

    mock_openai.assert_called_once()
    mock_google.assert_called_once()
    mock_anthropic.assert_called_once()
    mock_save.assert_called_once_with(expected_combined_dict)
    assert result == expected_combined_dict


@patch('src.utils.model_utils.get_model_list')
def test_check_model_available(mock_get_list):
    """Tests the check_model_available utility."""
    mock_get_list.return_value = {
        "ProviderA": ["model-a1", "model-a2"],
        "ProviderB": ["model-b1"],
        "Local": ["local-model"]
    }

    assert model_utils.check_model_available("model-b1") is True
    assert model_utils.check_model_available("local-model") is True
    assert model_utils.check_model_available("non-existent-model") is False


@pytest.mark.parametrize("model_name,expected", [
    ("agy-flash", "agy"),
    ("AGY-Flash", "agy"),
    ("gpt-4", "gpt"),
    ("claude-3-opus", "claude"),
    ("gemini-3.1-pro", "gemini-3.1"),
    ("gemini-pro", "gemini"),
    ("gemma-4-31b-it", "gemma"),
    ("local", "local"),
    ("something-else", "unknown"),
])
def test_get_model_prefix(model_name, expected):
    """The agy family must resolve so routing/registration recognise it."""
    assert model_utils.get_model_prefix(model_name) == expected
