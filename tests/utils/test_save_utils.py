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
        execution_mode=ExecutionMode.CONFIRM
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
    # Save the personas.json
    save_utils.save_personas_to_file(mock_personas, file_path_override=str(temp_save_file))

    # Load the personas.json
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



def test_load_persona_attributes_integrity(tmp_path):
    """
    Verifies that a known-good JSON structure is correctly parsed into
    a Persona object with all attributes preserved.
    """
    from src.utils.save_utils import load_personas_from_file
    from src.persona import ExecutionMode, MemoryMode
    import json

    # 1. Create a temporary JSON file with specific test values
    test_file = tmp_path / "integrity_test.json"
    test_data = {
        "personas": [
            {
                "name": "integrity_bot",
                "model_name": "gpt-4-test-variant",
                "prompt": "You are a test.",
                "context_length": 99,
                "token_limit": 500,
                "temperature": 0.1,
                "top_p": 0.9,
                "execution_mode": "CONFIRM",
                "memory_mode": "TICKET_ISOLATED",
                "enabled_tools": ["create_ticket"]
            }
        ]
    }
    test_file.write_text(json.dumps(test_data))

    # 2. Load the file using your utility
    loaded_personas = load_personas_from_file(str(test_file))

    # 3. Verify every attribute
    assert "integrity_bot" in loaded_personas
    p = loaded_personas["integrity_bot"]

    assert p.get_name() == "integrity_bot"
    assert p.get_model_name() == "gpt-4-test-variant"
    assert p.get_base_context_length() == 99
    assert p.get_response_token_limit() == 500
    assert p.get_temperature() == 0.1
    assert p.get_top_p() == 0.9
    assert p.get_execution_mode() == ExecutionMode.CONFIRM
    assert p.get_memory_mode() == MemoryMode.TICKET_ISOLATED
    assert p.get_enabled_tools() == ["create_ticket"]


def test_max_context_tokens_round_trip(temp_save_file: Path):
    """Phase 3: max_context_tokens survives save/load."""
    p = Persona(
        persona_name="ctx_persona",
        model_name="m",
        prompt="p",
        max_context_tokens=8192,
    )
    save_utils.save_personas_to_file({"ctx_persona": p}, file_path_override=str(temp_save_file))
    loaded = save_utils.load_personas_from_file(file_path_override=str(temp_save_file))
    assert loaded["ctx_persona"].get_max_context_tokens() == 8192


def test_max_context_tokens_missing_uses_default(tmp_path: Path):
    """Old config without max_context_tokens loads with the default."""
    from config import global_config

    test_file = tmp_path / "old_config.json"
    test_file.write_text(json.dumps({
        "personas": [{
            "name": "legacy",
            "model_name": "m",
            "prompt": "p",
        }]
    }))
    loaded = save_utils.load_personas_from_file(str(test_file))
    assert loaded["legacy"].get_max_context_tokens() == global_config.DEFAULT_MAX_CONTEXT_TOKENS
