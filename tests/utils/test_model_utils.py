import pytest
from unittest.mock import patch, MagicMock
from src.utils import model_utils

@patch('src.utils.model_utils.save_utils.load_models_from_file')
def test_get_model_list_no_update(mock_load_models):
    """Tests get_model_list when update=False, ensuring it calls the load utility."""
    expected_models = {"Local": ["local-model"]}
    mock_load_models.return_value = expected_models

    result = model_utils.get_model_list(update=False)

    mock_load_models.assert_called_once()
    assert result == expected_models

@patch('src.utils.model_utils.save_utils.save_models_to_file')
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
    