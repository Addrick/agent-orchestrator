# tests/personas/test_store.py

import pytest
import os
import json
from pathlib import Path
from src.personas import store as save_utils
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
        history_messages=500
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
    save_utils.save_personas_to_file(mock_personas, set(), file_path_override=str(temp_save_file))

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
    from src.personas.store import load_personas_from_file
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
                "history_messages": 99,
                "params": {"temperature": 0.1, "top_p": 0.9, "max_tokens": 500},
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
    assert p.get_base_history_messages() == 99
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
    save_utils.save_personas_to_file({"ctx_persona": p}, set(), file_path_override=str(temp_save_file))
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


# --- Phase A: GenerationParams nested shape on disk ---


def test_save_writes_nested_params_block(temp_save_file: Path):
    """to_dict writes a `params` block, no legacy flat sampler keys."""
    p = Persona(
        persona_name="phase_a",
        model_name="m",
        prompt="p",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        token_limit=2048,
    )
    save_utils.save_personas_to_file({"phase_a": p}, set(), file_path_override=str(temp_save_file))
    with open(temp_save_file) as f:
        on_disk = json.load(f)
    entry = on_disk["personas"][0]
    assert "params" in entry
    assert entry["params"]["temperature"] == 0.7
    assert entry["params"]["top_p"] == 0.9
    assert entry["params"]["top_k"] == 40
    assert entry["params"]["max_tokens"] == 2048
    assert entry["params"]["stop_sequences"] == []
    assert entry["params"]["seed"] is None
    assert entry["params"]["provider_extras"] == {}
    # Legacy flat sampler keys are no longer written.
    for legacy_key in ("temperature", "top_p", "top_k", "token_limit"):
        assert legacy_key not in entry, f"unexpected legacy key {legacy_key!r} in saved entry"


def test_save_round_trip_preserves_provider_extras(temp_save_file: Path):
    """Provider extras + new universal fields survive save/load."""
    p = Persona(
        persona_name="kbd",
        model_name="local",
        prompt="p",
        params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 1024,
            "stop_sequences": ["</s>"],
            "seed": 1234,
            "provider_extras": {"kobold": {"rep_pen": 1.05, "min_p": 0.05}},
        },
    )
    save_utils.save_personas_to_file({"kbd": p}, set(), file_path_override=str(temp_save_file))
    loaded = save_utils.load_personas_from_file(file_path_override=str(temp_save_file))
    gp = loaded["kbd"].get_generation_params()
    assert gp.stop_sequences == ["</s>"]
    assert gp.seed == 1234
    assert gp.provider_extras == {"kobold": {"rep_pen": 1.05, "min_p": 0.05}}


def test_load_legacy_flat_shape_ignored(tmp_path: Path):
    """DP-221: flat sampler keys are no longer honored on load — a file that
    only sets them yields default generation params, not the flat values."""
    test_file = tmp_path / "legacy.json"
    test_file.write_text(json.dumps({
        "personas": [{
            "name": "legacy_bot",
            "model_name": "m",
            "prompt": "p",
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 25,
            "token_limit": 777,
        }]
    }))
    loaded = save_utils.load_personas_from_file(str(test_file))
    p = loaded["legacy_bot"]
    assert p.get_temperature() is None
    assert p.get_top_p() is None
    assert p.get_top_k() is None
    assert p.get_response_token_limit() != 777


def test_save_personas_excludes_system_personas(temp_save_file: Path):
    """System personas in the in-memory dict must not be persisted to the user file.

    Regression: prod chatbot's data/personas.json got contaminated with
    triage_*, model_selector, etc. after edits. Combined with load-time
    validation rejections, the next save-on-edit silently erased user
    personas (e.g. joy). The save path is the single chokepoint for the fix.
    """
    user = Persona(persona_name="joy", model_name="m", prompt="p")
    sys_a = Persona(persona_name="triage_scout", model_name="m", prompt="p")
    sys_b = Persona(persona_name="model_selector", model_name="m", prompt="p")
    save_utils.save_personas_to_file(
        {"joy": user, "triage_scout": sys_a, "model_selector": sys_b},
        {"triage_scout", "model_selector"},
        file_path_override=str(temp_save_file),
    )
    with open(temp_save_file) as f:
        data = json.load(f)
    assert [entry["name"] for entry in data["personas"]] == ["joy"]


# --- DP-227: self_edit field load/save ---

def test_self_edit_absent_in_config_defaults_false(tmp_path: Path):
    """Old config without the self_edit key loads with self_edit False."""
    test_file = tmp_path / "no_self_edit.json"
    test_file.write_text(json.dumps({
        "personas": [{
            "name": "legacy_bot",
            "model_name": "gemini-2.5-flash",
            "prompt": "hi",
        }]
    }))
    loaded = save_utils.load_personas_from_file(str(test_file))
    assert loaded["legacy_bot"].get_self_edit() is False


def test_self_edit_present_in_config(tmp_path: Path):
    """A config with self_edit:true loads the persona in self-edit mode."""
    test_file = tmp_path / "with_self_edit.json"
    test_file.write_text(json.dumps({
        "personas": [{
            "name": "fixr_like",
            "model_name": "cc-sonnet",
            "prompt": "fix bugs",
            "self_edit": True,
        }]
    }))
    loaded = save_utils.load_personas_from_file(str(test_file))
    assert loaded["fixr_like"].get_self_edit() is True


def test_self_edit_round_trip(temp_save_file: Path):
    """self_edit survives save -> load."""
    p = Persona(persona_name="se", model_name="cc-sonnet", prompt="p", self_edit=True)
    save_utils.save_personas_to_file({"se": p}, set(), file_path_override=str(temp_save_file))
    loaded = save_utils.load_personas_from_file(file_path_override=str(temp_save_file))
    assert loaded["se"].get_self_edit() is True


def test_default_personas_json_loads_fixr():
    """The shipped default_personas.json defines a working fixr persona routed
    to the cc-* engine with self_edit enabled (DP-227)."""
    from config import global_config
    default_file = os.path.join(global_config.CONFIG_DIR, "default_personas.json")
    loaded = save_utils.load_personas_from_file(default_file)
    assert loaded is not None
    assert "fixr" in loaded
    fixr = loaded["fixr"]
    assert fixr.get_self_edit() is True
    assert fixr.get_model_name() == "cc-sonnet"
    assert fixr.get_execution_mode() == ExecutionMode.AUTONOMOUS
    # config exposes self_edit to the engine
    assert fixr.get_config_for_engine().get("self_edit") is True
