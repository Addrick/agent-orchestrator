# tests/test_persona_edge_cases.py
"""
DP-199 Batch 7 — Persona mutation + serialization edge cases.

Covers:
  * GenerationParams override priority and resolution
  * _resolve_enum invalid inputs
  * Setter clamp / coerce / fallback behavior
  * provider_extras dotted-path semantics
  * Persona <-> ToolPolicy convergence
  * Dynamic history state machine
  * Round-trip serialization through save_utils (slice 4 hard-blocker)
"""

import json
import pytest

from config import global_config
from src.persona import Persona, ExecutionMode, MemoryMode
from src.generation_params import GenerationParams
from src.tools.policy import ToolPolicy
from src.personas.store import (
    save_personas_to_file,
    load_personas_from_file,
)


TEST_DEFAULT_HISTORY_MESSAGES = 15
TEST_DEFAULT_TOKEN_LIMIT = 4096
TEST_DEFAULT_MAX_CONTEXT_TOKENS = 131072


@pytest.fixture(autouse=True)
def _patch_defaults(monkeypatch):
    monkeypatch.setattr(global_config, "DEFAULT_HISTORY_MESSAGES", TEST_DEFAULT_HISTORY_MESSAGES)
    monkeypatch.setattr(global_config, "DEFAULT_TOKEN_LIMIT", TEST_DEFAULT_TOKEN_LIMIT)
    monkeypatch.setattr(global_config, "DEFAULT_MAX_CONTEXT_TOKENS", TEST_DEFAULT_MAX_CONTEXT_TOKENS)


@pytest.fixture
def base_args():
    return {
        "persona_name": "tester",
        "model_name": "test-model",
        "prompt": "You are a test persona.",
    }


@pytest.fixture
def persona(base_args):
    return Persona(**base_args)


# ---------------------------------------------------------------------------
# GenerationParams override priority
# ---------------------------------------------------------------------------

def test_persona_params_override_priority(base_args):
    """Flat kwargs override values supplied through the nested params block."""
    p = Persona(
        **base_args,
        params={"temperature": 0.1, "top_p": 0.2, "top_k": 5, "max_tokens": 256},
        temperature=0.99,
        top_p=0.88,
        top_k=42,
    )
    assert p.get_temperature() == 0.99
    assert p.get_top_p() == 0.88
    assert p.get_top_k() == 42
    # token_limit kwarg wasn't given, so max_tokens from params survives
    assert p.get_response_token_limit() == 256


def test_persona_params_dict_plus_flat_override(base_args):
    """A dict params block hydrates a fresh GenerationParams; flat kwargs override on top."""
    p = Persona(
        **base_args,
        params={"temperature": 0.5, "provider_extras": {"kobold": {"rep_pen": 1.1}}},
        temperature=0.7,
    )
    assert p.get_temperature() == 0.7
    # provider_extras from params is preserved
    assert p.get_generation_params().provider_extras == {"kobold": {"rep_pen": 1.1}}


def test_persona_params_none_creates_fresh(base_args):
    """params=None yields a fresh GenerationParams with defaults."""
    p = Persona(**base_args)
    params = p.get_generation_params()
    assert isinstance(params, GenerationParams)
    assert params.temperature is None
    assert params.top_p is None
    assert params.top_k is None
    assert params.provider_extras == {}


# ---------------------------------------------------------------------------
# _resolve_enum
# ---------------------------------------------------------------------------

def test_resolve_enum_invalid_string(base_args):
    """Invalid string for execution_mode falls back to default (AUTONOMOUS)."""
    p = Persona(**base_args, execution_mode="not_a_mode")
    assert p.get_execution_mode() is ExecutionMode.AUTONOMOUS


def test_resolve_enum_wrong_type(base_args):
    """Wrong-type value (int) falls back to default."""
    p = Persona(**base_args, memory_mode=12345)
    assert p.get_memory_mode() is MemoryMode.CHANNEL_ISOLATED


# ---------------------------------------------------------------------------
# Numeric setter coercion
# ---------------------------------------------------------------------------

def test_set_response_token_limit_clamp(persona):
    """Values below 100 clamp up to 100."""
    assert persona.set_response_token_limit(50) == 100
    assert persona.get_response_token_limit() == 100


def test_set_response_token_limit_non_numeric(persona):
    """Non-numeric input reverts to global default."""
    assert persona.set_response_token_limit("abc") == TEST_DEFAULT_TOKEN_LIMIT


def test_set_temperature_non_numeric(persona):
    assert persona.set_temperature("warm") is None
    assert persona.get_temperature() is None


def test_set_top_p_non_numeric(persona):
    assert persona.set_top_p("nope") is None
    assert persona.get_top_p() is None


def test_set_top_k_non_integer(persona):
    assert persona.set_top_k(1.5) == 1  # int() coerces 1.5 -> 1
    assert persona.set_top_k("ten") is None
    assert persona.get_top_k() is None


# ---------------------------------------------------------------------------
# set_execution_mode / set_memory_mode
# ---------------------------------------------------------------------------

def test_set_execution_mode_invalid_string(persona):
    """Invalid string leaves the mode untouched (no change)."""
    persona.set_execution_mode("CONFIRM")
    assert persona.get_execution_mode() is ExecutionMode.CONFIRM
    persona.set_execution_mode("not_a_mode")
    assert persona.get_execution_mode() is ExecutionMode.CONFIRM


def test_set_execution_mode_wrong_type(persona):
    """Non-string, non-enum input leaves the mode untouched."""
    persona.set_execution_mode(ExecutionMode.CONFIRM)
    persona.set_execution_mode(42)
    assert persona.get_execution_mode() is ExecutionMode.CONFIRM


def test_set_memory_mode_invalid_string(persona):
    persona.set_memory_mode("PERSONAL")
    assert persona.get_memory_mode() is MemoryMode.PERSONAL
    persona.set_memory_mode("not_a_mode")
    assert persona.get_memory_mode() is MemoryMode.PERSONAL


def test_set_memory_mode_wrong_type(persona):
    persona.set_memory_mode(MemoryMode.PERSONAL)
    persona.set_memory_mode(3.14)
    assert persona.get_memory_mode() is MemoryMode.PERSONAL


# ---------------------------------------------------------------------------
# max_context_tokens
# ---------------------------------------------------------------------------

def test_set_max_context_tokens_clamp_low(persona):
    persona.set_max_context_tokens(10)
    assert persona.get_max_context_tokens() == 100


def test_set_max_context_tokens_invalid(persona):
    persona.set_max_context_tokens(8192)
    persona.set_max_context_tokens("not-a-number")
    assert persona.get_max_context_tokens() == TEST_DEFAULT_MAX_CONTEXT_TOKENS


# ---------------------------------------------------------------------------
# provider_extras
# ---------------------------------------------------------------------------

def test_set_provider_extra_only_one_level(persona):
    """set_provider_extra writes a single key under provider_extras[provider]."""
    persona.set_provider_extra("kobold", "min_p", 0.05)
    extras = persona.get_generation_params().provider_extras
    assert extras == {"kobold": {"min_p": 0.05}}


def test_set_provider_extra_overwrite(persona):
    persona.set_provider_extra("kobold", "min_p", 0.05)
    persona.set_provider_extra("kobold", "min_p", 0.1)
    assert persona.get_provider_extra("kobold", "min_p") == 0.1


def test_clear_provider_extra_prunes_block(persona):
    persona.set_provider_extra("kobold", "lone_key", 1)
    assert persona.clear_provider_extra("kobold", "lone_key") is True
    assert "kobold" not in persona.get_generation_params().provider_extras


def test_clear_provider_extra_missing_block(persona):
    """Clearing a key from a provider block that was never set returns False."""
    assert persona.clear_provider_extra("nonexistent_provider", "any_key") is False


# ---------------------------------------------------------------------------
# enabled_tools / ToolPolicy convergence
# ---------------------------------------------------------------------------

def test_get_enabled_tools_wildcard(base_args):
    """When policy default=allow with '*', get_enabled_tools returns ['*']."""
    p = Persona(**base_args, tool_policy={"default": "allow", "allow": ["*"]})
    assert p.get_enabled_tools() == ["*"]


def test_get_enabled_tools_empty(base_args):
    """A persona with no enabled tools and no policy returns []."""
    p = Persona(**base_args)
    assert p.get_enabled_tools() == []


def test_set_enabled_tools_syncs_both_fields(persona):
    """set_enabled_tools updates legacy _enabled_tools AND _tool_policy."""
    persona.set_enabled_tools(["tool_a", "tool_b"])
    assert sorted(persona.get_enabled_tools()) == ["tool_a", "tool_b"]
    policy = persona.get_tool_policy()
    assert policy.default == "deny"
    assert sorted(policy.allow) == ["tool_a", "tool_b"]


def test_set_tool_policy_syncs_legacy_field(persona):
    """set_tool_policy(dict) updates _tool_policy and refreshes legacy _enabled_tools."""
    persona.set_tool_policy({"default": "deny", "allow": ["foo", "bar"]})
    policy = persona.get_tool_policy()
    assert sorted(policy.allow) == ["bar", "foo"]
    # legacy field should mirror policy.allow
    assert sorted(persona.get_enabled_tools()) == ["bar", "foo"]


def test_get_enabled_tools_prefers_policy(base_args):
    """When legacy enabled_tools and policy disagree, get_enabled_tools follows policy."""
    p = Persona(
        **base_args,
        enabled_tools=["legacy_only"],
        tool_policy={"default": "deny", "allow": ["policy_only"], "ask": ["ask_tool"]},
    )
    result = p.get_enabled_tools()
    assert "policy_only" in result
    assert "ask_tool" in result
    assert "legacy_only" not in result


def test_set_enabled_tools_updates_policy(persona):
    """A second set_enabled_tools replaces — does not append to — the policy."""
    persona.set_enabled_tools(["a"])
    persona.set_enabled_tools(["b", "c"])
    assert sorted(persona.get_tool_policy().allow) == ["b", "c"]


def test_set_tool_policy_dict_syncs_legacy(persona):
    persona.set_tool_policy({"default": "deny", "allow": ["x", "y"]})
    assert sorted(persona.get_enabled_tools()) == ["x", "y"]


def test_set_tool_policy_object_direct(persona):
    """Passing a ToolPolicy instance directly is accepted as-is."""
    pol = ToolPolicy(default="allow", allow=["*"])
    persona.set_tool_policy(pol)
    assert persona.get_tool_policy() is pol
    assert persona.get_enabled_tools() == ["*"]


# ---------------------------------------------------------------------------
# Dynamic history
# ---------------------------------------------------------------------------

def test_start_new_conversation_dynamic(persona):
    persona.start_new_conversation(0)
    assert persona.is_in_dynamic_history() is True
    assert persona.get_current_effective_history_messages() == 0


def test_get_history_messages_dynamic_increment(persona):
    """get_history_messages increments the override by 2 per call."""
    persona.start_new_conversation(0)
    assert persona.get_history_messages() == 0
    assert persona.get_history_messages() == 2
    assert persona.get_history_messages() == 4


def test_end_new_conversation_clear(persona):
    persona.set_history_messages(20)
    persona.start_new_conversation(0)
    persona.end_new_conversation()
    assert persona.is_in_dynamic_history() is False
    assert persona.get_history_messages() == 20


def test_is_in_dynamic_history_state(persona):
    assert persona.is_in_dynamic_history() is False
    persona.start_new_conversation(0)
    assert persona.is_in_dynamic_history() is True
    # set_history_messages should drop the dynamic override.
    persona.set_history_messages(10)
    assert persona.is_in_dynamic_history() is False


# ---------------------------------------------------------------------------
# Round-trip serialization (slice 4 hard-blocker)
# ---------------------------------------------------------------------------

def _roundtrip(personas, tmp_path):
    save_file = str(tmp_path / "personas.json")
    save_personas_to_file(personas, set(), file_path_override=save_file)
    loaded = load_personas_from_file(file_path_override=save_file)
    assert loaded is not None
    return loaded


def test_persona_meta_visible_round_trip(base_args, tmp_path):
    p_on = Persona(**{**base_args, "persona_name": "visible"}, meta_visible=True)
    p_off = Persona(**{**base_args, "persona_name": "hidden"}, meta_visible=False)
    loaded = _roundtrip({"visible": p_on, "hidden": p_off}, tmp_path)
    assert loaded["visible"].get_meta_visible() is True
    assert loaded["hidden"].get_meta_visible() is False


def test_persona_generation_params_round_trip(base_args, tmp_path):
    p = Persona(
        **base_args,
        temperature=0.42,
        top_p=0.91,
        top_k=37,
        token_limit=2048,
    )
    p.set_provider_extra("kobold", "rep_pen", 1.15)
    p.set_provider_extra("kobold", "min_p", 0.05)
    loaded = _roundtrip({"tester": p}, tmp_path)
    rp = loaded["tester"]
    assert rp.get_temperature() == 0.42
    assert rp.get_top_p() == 0.91
    assert rp.get_top_k() == 37
    assert rp.get_response_token_limit() == 2048
    extras = rp.get_generation_params().provider_extras
    assert extras == {"kobold": {"rep_pen": 1.15, "min_p": 0.05}}


def test_persona_max_context_tokens_round_trip(base_args, tmp_path):
    p = Persona(**base_args, max_context_tokens=8192)
    loaded = _roundtrip({"tester": p}, tmp_path)
    assert loaded["tester"].get_max_context_tokens() == 8192


def test_persona_provider_extras_serialized(base_args, tmp_path):
    """Nested-of-nested provider_extras survive JSON serialization."""
    p = Persona(**base_args)
    p.set_provider_extra("kobold", "min_p", 0.05)
    p.set_provider_extra("anthropic", "thinking", {"type": "enabled", "budget": 1024})
    save_file = str(tmp_path / "personas.json")
    save_personas_to_file({"tester": p}, set(), file_path_override=save_file)

    # Inspect raw JSON to confirm nested structure is preserved literally.
    with open(save_file, "r") as f:
        raw = json.load(f)
    [entry] = raw["personas"]
    assert entry["params"]["provider_extras"]["kobold"] == {"min_p": 0.05}
    assert entry["params"]["provider_extras"]["anthropic"] == {
        "thinking": {"type": "enabled", "budget": 1024}
    }

    loaded = load_personas_from_file(file_path_override=save_file)
    rp = loaded["tester"]
    assert rp.get_provider_extra("kobold", "min_p") == 0.05
    assert rp.get_provider_extra("anthropic", "thinking") == {
        "type": "enabled",
        "budget": 1024,
    }


def test_persona_tool_policy_round_trip(base_args, tmp_path):
    policy_dict = {
        "default": "deny",
        "allow": ["search", "summarize"],
        "ask": ["delete_message"],
        "capabilities_required": [],
        "explicit_overrides": ["network_read_local_write"],
    }
    p = Persona(**base_args, tool_policy=policy_dict)
    loaded = _roundtrip({"tester": p}, tmp_path)
    pol = loaded["tester"].get_tool_policy()
    assert pol.default == "deny"
    assert sorted(pol.allow) == ["search", "summarize"]
    assert pol.ask == ["delete_message"]
    assert pol.explicit_overrides == ["network_read_local_write"]


def test_persona_modes_round_trip(base_args, tmp_path):
    p = Persona(
        **base_args,
        execution_mode=ExecutionMode.CONFIRM,
        memory_mode=MemoryMode.PERSONAL,
    )
    loaded = _roundtrip({"tester": p}, tmp_path)
    rp = loaded["tester"]
    assert rp.get_execution_mode() is ExecutionMode.CONFIRM
    assert rp.get_memory_mode() is MemoryMode.PERSONAL


def test_persona_tool_policy_legacy_migration(base_args, tmp_path):
    """A legacy file with only `enabled_tools` (no `tool_policy`) loads as a deny policy."""
    file_path = tmp_path / "personas.json"
    file_path.write_text(json.dumps({
        "personas": [{
            "name": "legacy",
            "model_name": "m",
            "prompt": "p",
            "enabled_tools": ["tool_a", "tool_b"],
        }],
    }))
    loaded = load_personas_from_file(file_path_override=str(file_path))
    assert loaded is not None and "legacy" in loaded
    pol = loaded["legacy"].get_tool_policy()
    assert pol.default == "deny"
    assert sorted(pol.allow) == ["tool_a", "tool_b"]
