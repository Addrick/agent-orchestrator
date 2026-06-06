# tests/test_save_utils_edge_cases.py
"""
DP-199 Batch 7 — save_utils edge cases.

Covers persona/model file save/load round-trip + auto-seed behavior, malformed
input handling, and the legacy flat-key fallback path through _resolve_params_kwargs.

NOTE: test_save_utils_no_tool_imports and test_save_utils_no_policy_import are
deferred — they are slice-4 contract tests that will be added once the tool
imports have been removed from save_utils.py.
"""

import json
import os
import shutil

import pytest

from config import global_config
from src.persona import Persona, ExecutionMode, MemoryMode
from src.utils.save_utils import (
    _resolve_params_kwargs,
    to_dict,
    save_personas_to_file,
    load_personas_from_file,
    load_system_personas_from_file,
    save_models_to_file,
    load_models_from_file,
)


@pytest.fixture
def base_args():
    return {
        "persona_name": "tester",
        "model_name": "test-model",
        "prompt": "You are a test persona.",
    }


# ---------------------------------------------------------------------------
# _resolve_params_kwargs
# ---------------------------------------------------------------------------

def test_resolve_params_kwargs_nested_priority():
    """When a `params` dict is present, _resolve_params_kwargs returns it (and flat keys are ignored)."""
    entry = {
        "params": {"temperature": 0.4, "top_p": 0.9},
        "temperature": 0.9,  # legacy flat key — must be ignored
        "top_p": 0.1,
    }
    resolved = _resolve_params_kwargs(entry)
    assert resolved["params"] == {"temperature": 0.4, "top_p": 0.9}
    # Legacy flat keys not echoed (only token_limit is)
    assert "temperature" not in resolved
    assert "top_p" not in resolved


def test_resolve_params_kwargs_legacy_fallback():
    """No `params` block → legacy flat keys are surfaced one-for-one."""
    entry = {
        "temperature": 0.5,
        "top_p": 0.85,
        "top_k": 40,
        "token_limit": 1024,
    }
    resolved = _resolve_params_kwargs(entry)
    assert resolved == {
        "temperature": 0.5,
        "top_p": 0.85,
        "top_k": 40,
        "token_limit": 1024,
    }


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------

def test_to_dict_nested_params_write(base_args):
    """to_dict writes generation params under a nested `params` block."""
    p = Persona(**base_args, temperature=0.4, top_p=0.7, top_k=20, token_limit=512)
    [out] = to_dict({"tester": p})
    assert "params" in out
    assert out["params"]["temperature"] == 0.4
    assert out["params"]["top_p"] == 0.7
    assert out["params"]["top_k"] == 20
    assert out["params"]["max_tokens"] == 512


def test_to_dict_no_flat_keys(base_args):
    """Flat keys (temperature/top_p/top_k/token_limit) are NOT written at top level."""
    p = Persona(**base_args, temperature=0.4)
    [out] = to_dict({"tester": p})
    for legacy_key in ("temperature", "top_p", "top_k", "token_limit"):
        assert legacy_key not in out, f"legacy key '{legacy_key}' must not be written"


# ---------------------------------------------------------------------------
# load_personas_from_file — auto-seed + corner cases
# ---------------------------------------------------------------------------

def test_load_personas_auto_seed(tmp_path, monkeypatch):
    """If the default save path is missing, load_personas_from_file copies default_personas.json."""
    fake_data = tmp_path / "data"
    fake_data.mkdir()
    fake_config = tmp_path / "config"
    fake_config.mkdir()
    save_path = fake_data / "personas.json"

    # Build a minimal default_personas.json next to a fake CONFIG_DIR
    default_file = fake_config / "default_personas.json"
    default_file.write_text(json.dumps({
        "personas": [{"name": "seeded", "model_name": "m", "prompt": "p"}]
    }))

    monkeypatch.setattr(global_config, "PERSONA_SAVE_FILE", str(save_path))
    monkeypatch.setattr(global_config, "CONFIG_DIR", str(fake_config))

    loaded = load_personas_from_file()
    assert loaded is not None
    assert "seeded" in loaded
    # Save file must now exist on disk.
    assert save_path.exists()


def test_load_personas_override_missing(tmp_path):
    """An explicit override path that doesn't exist returns None (no auto-seed)."""
    missing = tmp_path / "nothing.json"
    assert load_personas_from_file(file_path_override=str(missing)) is None


def test_load_personas_empty_file(tmp_path):
    """An empty personas file returns an empty dict."""
    save_file = tmp_path / "personas.json"
    save_file.write_text("")
    loaded = load_personas_from_file(file_path_override=str(save_file))
    assert loaded == {}


def test_load_personas_missing_name(tmp_path):
    """Persona entries lacking 'name' are skipped silently."""
    save_file = tmp_path / "personas.json"
    save_file.write_text(json.dumps({
        "personas": [
            {"name": "good", "model_name": "m", "prompt": "p"},
            {"model_name": "m2", "prompt": "p2"},  # no name
        ]
    }))
    loaded = load_personas_from_file(file_path_override=str(save_file))
    assert loaded is not None
    assert "good" in loaded
    assert len(loaded) == 1


def test_load_personas_security_violation(tmp_path):
    """A persona whose tool composition violates security invariants is refused."""
    save_file = tmp_path / "personas.json"
    save_file.write_text(json.dumps({
        "personas": [
            {
                "name": "good",
                "model_name": "m",
                "prompt": "p",
                "enabled_tools": [],
            },
            {
                "name": "insecure",
                "model_name": "m",
                "prompt": "p",
                # web_search (network:read, untrusted) + manage_agent (local:write)
                # triggers Rule 1 (network_read_local_write) without an override.
                "tool_policy": {
                    "default": "deny",
                    "allow": ["web_search", "manage_agent"],
                },
            },
        ]
    }))
    loaded = load_personas_from_file(file_path_override=str(save_file))
    assert loaded is not None
    assert "good" in loaded
    assert "insecure" not in loaded


def test_load_personas_context_length_legacy(tmp_path):
    """Legacy field `context_length` falls through to history_messages."""
    save_file = tmp_path / "personas.json"
    save_file.write_text(json.dumps({
        "personas": [{
            "name": "legacy",
            "model_name": "m",
            "prompt": "p",
            "context_length": 7,
        }]
    }))
    loaded = load_personas_from_file(file_path_override=str(save_file))
    assert loaded["legacy"].get_base_history_messages() == 7


def test_load_personas_params_merge(tmp_path):
    """A persona file with a nested params block round-trips through load."""
    save_file = tmp_path / "personas.json"
    save_file.write_text(json.dumps({
        "personas": [{
            "name": "p",
            "model_name": "m",
            "prompt": "x",
            "params": {
                "temperature": 0.42,
                "top_p": 0.91,
                "top_k": 30,
                "max_tokens": 4096,
                "provider_extras": {"kobold": {"rep_pen": 1.1}},
            },
        }]
    }))
    loaded = load_personas_from_file(file_path_override=str(save_file))
    p = loaded["p"]
    assert p.get_temperature() == 0.42
    assert p.get_top_p() == 0.91
    assert p.get_top_k() == 30
    assert p.get_response_token_limit() == 4096
    assert p.get_provider_extra("kobold", "rep_pen") == 1.1


# ---------------------------------------------------------------------------
# load_system_personas_from_file
# ---------------------------------------------------------------------------

def test_load_system_personas_structure(tmp_path, monkeypatch):
    """A system persona file loads and yields Persona instances."""
    sys_file = tmp_path / "system_personas.json"
    sys_file.write_text(json.dumps({
        "personas": [{
            "name": "sys_one",
            "model_name": "m",
            "prompt": "p",
            "execution_mode": "AUTONOMOUS",
            "memory_mode": "global",
        }]
    }))
    monkeypatch.setattr(global_config, "SYSTEM_PERSONA_FILE", str(sys_file))
    loaded = load_system_personas_from_file()
    assert "sys_one" in loaded
    assert loaded["sys_one"].get_memory_mode() is MemoryMode.GLOBAL


def test_load_system_personas_response_token_limit(tmp_path, monkeypatch):
    """response_token_limit (system-persona legacy key) is mapped to token_limit."""
    sys_file = tmp_path / "system_personas.json"
    sys_file.write_text(json.dumps({
        "personas": [{
            "name": "rtl",
            "model_name": "m",
            "prompt": "p",
            "response_token_limit": 256,
        }]
    }))
    monkeypatch.setattr(global_config, "SYSTEM_PERSONA_FILE", str(sys_file))
    loaded = load_system_personas_from_file()
    assert loaded["rtl"].get_response_token_limit() == 256


# ---------------------------------------------------------------------------
# save_personas_to_file
# ---------------------------------------------------------------------------

def test_save_personas_create_file(tmp_path, base_args):
    """save_personas_to_file creates the file (and parent dir) when missing."""
    save_file = tmp_path / "sub" / "personas.json"
    p = Persona(**base_args)
    save_personas_to_file({"tester": p}, set(), file_path_override=str(save_file))
    assert save_file.exists()
    with open(save_file) as f:
        data = json.load(f)
    assert "personas" in data
    assert any(entry["name"] == "tester" for entry in data["personas"])


def test_save_personas_corrupt_existing(tmp_path, base_args):
    """An unparseable existing file is replaced with a clean structure."""
    save_file = tmp_path / "personas.json"
    save_file.write_text("{ not valid json ::: ")
    p = Persona(**base_args)
    save_personas_to_file({"tester": p}, set(), file_path_override=str(save_file))
    with open(save_file) as f:
        data = json.load(f)  # must now parse
    assert "personas" in data
    assert any(entry["name"] == "tester" for entry in data["personas"])


def test_save_personas_preserves_models(tmp_path, base_args):
    """save_personas_to_file does not clobber the `models` section."""
    save_file = tmp_path / "personas.json"
    save_file.write_text(json.dumps({
        "personas": [],
        "models": {"From OpenAI": ["gpt-4"], "Local": ["local"]},
    }))
    p = Persona(**base_args)
    save_personas_to_file({"tester": p}, set(), file_path_override=str(save_file))
    with open(save_file) as f:
        data = json.load(f)
    assert data["models"] == {"From OpenAI": ["gpt-4"], "Local": ["local"]}


# ---------------------------------------------------------------------------
# save_models_to_file / load_models_from_file
# ---------------------------------------------------------------------------

def test_save_models_preserves_personas(tmp_path):
    """save_models_to_file leaves the personas section untouched."""
    save_file = tmp_path / "personas.json"
    save_file.write_text(json.dumps({
        "personas": [{"name": "keepme", "model_name": "m", "prompt": "p"}],
        "models": {"Local": ["local"]},
    }))
    save_models_to_file({"From OpenAI": ["gpt-4"]}, file_path_override=str(save_file))
    with open(save_file) as f:
        data = json.load(f)
    assert data["models"] == {"From OpenAI": ["gpt-4"]}
    assert data["personas"] == [{"name": "keepme", "model_name": "m", "prompt": "p"}]


def test_save_models_corrupt_existing(tmp_path):
    """Corrupt existing file → fresh default structure with our models written in."""
    save_file = tmp_path / "personas.json"
    save_file.write_text("garbage @!#")
    save_models_to_file({"Local": ["x"]}, file_path_override=str(save_file))
    with open(save_file) as f:
        data = json.load(f)
    assert data["models"] == {"Local": ["x"]}
    assert data["personas"] == []


def test_load_models_missing_file(tmp_path):
    """load_models_from_file returns None when the file is missing."""
    assert load_models_from_file(file_path_override=str(tmp_path / "no.json")) is None


def test_load_models_missing_key(tmp_path):
    """A file that exists but lacks the `models` key returns an empty dict."""
    save_file = tmp_path / "personas.json"
    save_file.write_text(json.dumps({"personas": []}))
    assert load_models_from_file(file_path_override=str(save_file)) == {}


# ---------------------------------------------------------------------------
# Slice-4 contract (deferred)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="DP-199 slice-4 contract — defer until save_utils tool imports removed")
def test_save_utils_no_tool_imports():
    """save_utils must not import src.tools.definitions (slice-4 contract)."""
    pass


@pytest.mark.skip(reason="DP-199 slice-4 contract — defer until save_utils policy import removed")
def test_save_utils_no_policy_import():
    """save_utils must not import src.tools.policy (slice-4 contract)."""
    pass
