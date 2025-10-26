# tests/utils/test_save_utils.py

import pytest
import os
import json
from pathlib import Path
from src.utils import save_utils
from src.persona import Persona, MemoryMode, ExecutionMode
from config.global_config import TEST_PERSONA_SAVE_FILE, PERSONA_SAVE_FILE


@pytest.fixture
def mock_personas():
    """Provides a dictionary of mock Persona objects."""
    p1 = Persona(
        persona_name="p1",
        model_name="test_model",
        prompt="prompt1",
        memory_mode=MemoryMode.PERSONAL,
        execution_mode=ExecutionMode.ASSISTED_DISPATCH
    )
    p2 = Persona(
        persona_name="p2",
        model_name="another_model",
        prompt="prompt2",
        context_length=500
    )
    return {"p1": p1, "p2": p2}


@pytest.fixture
def temp_save_file(tmp_path: Path) -> Path:
    """Creates a temporary directory and returns a path to a non-existent file in it."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir / "test_save.json"


def test_save_and_load_personas(temp_save_file: Path, mock_personas: dict):
    """Tests saving persona data to a file and loading it back."""
    # Save the personas
    save_utils.save_personas_to_file(mock_personas, file_path_override=str(temp_save_file))

    # Load the personas
    loaded_personas = save_utils.load_personas_from_file(file_path_override=str(temp_save_file))

    assert loaded_personas is not None
    assert "p1" in loaded_personas
    assert "p2" in loaded_personas
    assert loaded_personas["p1"].get_name() == "p1"
    assert loaded_personas["p2"].get_prompt() == "prompt2"
    assert loaded_personas["p1"].get_memory_mode() == MemoryMode.PERSONAL


def test_save_and_load_models(temp_save_file: Path):
    """Tests saving model data to a file and loading it back."""
    mock_models_dict = {
        "From OpenAI": ["gpt-4", "gpt-3.5-turbo"],
        "Local": ["local"]
    }

    # Save the models
    save_utils.save_models_to_file(mock_models_dict, file_path_override=str(temp_save_file))

    # Load the models
    loaded_models = save_utils.load_models_from_file(file_path_override=str(temp_save_file))

    assert loaded_models is not None
    assert loaded_models == mock_models_dict


def test_load_personas_file_not_found(tmp_path: Path):
    """Tests that loading from a non-existent file is handled gracefully."""
    non_existent_file = tmp_path / "non_existent.json"
    result = save_utils.load_personas_from_file(file_path_override=str(non_existent_file))
    assert result is None


def test_save_uses_test_file_in_pytest_env(mock_personas: dict):
    """
    Tests that when running under pytest, the save function defaults to the
    TEST_PERSONA_SAVE_FILE, not the production one.
    """
    # Ensure the test file does not exist before the test
    if os.path.exists(TEST_PERSONA_SAVE_FILE):
        os.remove(TEST_PERSONA_SAVE_FILE)
    # Ensure production file is not touched
    if os.path.exists(PERSONA_SAVE_FILE):
        os.remove(PERSONA_SAVE_FILE)

    try:
        # Call the function with NO override path
        save_utils.save_personas_to_file(mock_personas)

        # Assert that the test file was created and the production file was not
        assert os.path.exists(TEST_PERSONA_SAVE_FILE)
        assert not os.path.exists(PERSONA_SAVE_FILE)

        # Verify content
        with open(TEST_PERSONA_SAVE_FILE, 'r') as f:
            data = json.load(f)
        assert len(data['personas']) == 2
        assert data['personas'][0]['name'] == 'p1'

    finally:
        # Clean up the created test file
        if os.path.exists(TEST_PERSONA_SAVE_FILE):
            os.remove(TEST_PERSONA_SAVE_FILE)
        if os.path.exists(PERSONA_SAVE_FILE):
            os.remove(PERSONA_SAVE_FILE)
