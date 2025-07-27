import json
import os
import pytest
from src.utils import save_utils
from src.persona import Persona


@pytest.fixture
def mock_personas():
    """Provides a dictionary of mock Persona objects for testing."""
    return {
        "p1": Persona("p1", "model1", "prompt1", 100, 10),
        "p2": Persona("p2", "model2", "prompt2", 200, 20),
    }


@pytest.fixture
def temp_save_file(tmp_path):
    """Creates a temporary save file with a default structure."""
    d = tmp_path / "data"
    d.mkdir()
    p = d / "test_save.json"
    # The save functions read before writing, so the file must exist with the base keys.
    with open(p, 'w') as f:
        json.dump({"personas": [], "models": {}}, f)
    return p


def test_save_and_load_personas(temp_save_file, mock_personas):
    """Tests saving persona data to a file and loading it back."""
    # Convert mock objects to a dict format for comparison, as the save function does.
    expected_personas_dict = save_utils.to_dict(mock_personas)

    # Save the personas
    save_utils.save_personas_to_file(mock_personas, file_path_override=temp_save_file)

    # Load the personas
    loaded_personas = save_utils.load_personas_from_file(file_path=temp_save_file)
    loaded_personas_dict = save_utils.to_dict(loaded_personas)

    assert loaded_personas_dict == expected_personas_dict
    assert loaded_personas["p1"].get_prompt() == "prompt1"
    assert loaded_personas["p2"].get_response_token_limit() == 200


def test_save_and_load_models(temp_save_file):
    """Tests saving model data to a file and loading it back."""
    mock_models_dict = {
        "From OpenAI": ["gpt-4", "gpt-3.5-turbo"],
        "Local": ["local"]
    }

    # Save the models
    save_utils.save_models_to_file(mock_models_dict, file_path_override=temp_save_file)

    # Load the models
    loaded_models = save_utils.load_models_from_file(file_path=temp_save_file)

    assert loaded_models == mock_models_dict


def test_load_personas_file_not_found(tmp_path):
    """Tests that loading from a non-existent file is handled gracefully."""
    non_existent_file = tmp_path / "non_existent.json"
    result = save_utils.load_personas_from_file(file_path=non_existent_file)
    assert result is None
