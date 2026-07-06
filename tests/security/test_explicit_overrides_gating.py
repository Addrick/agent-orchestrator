# tests/security/test_explicit_overrides_gating.py
"""DP-277 Phase 1 — `explicit_overrides` is a gated persona field.

The override list is the kill switch for the tool-composition invariants
(`ToolPolicy.validate_composition` skips a rule when its name is present), so
it must be unreachable through any caller-supplied policy dict:

  door #1: PATCH/POST persona body -> apply_persona_patch_body (tool_policy
           was never a patch key, but the create route accepts arbitrary keys)
  door #2: `set tool_policy <json>` -> Persona.set_tool_policy ->
           ToolPolicy.from_dict

Both converge on ToolPolicy.from_dict / Persona.set_tool_policy — closed here.
The only mutation path is Persona.set_explicit_overrides (the dedicated
operator command), and saved legacy values are grandfathered by the store.
"""

import json

import pytest
from unittest.mock import MagicMock

from src.persona import Persona
from src.personas.store import load_personas_from_file, save_personas_to_file
from src.tool_policy import KNOWN_OVERRIDES, ToolPolicy
from tests.helpers import make_bot_logic, OPERATOR_ORIGIN


ALL_OVERRIDES = sorted(KNOWN_OVERRIDES)


def make_persona(**kwargs):
    return Persona(
        persona_name="tester",
        model_name="test-model",
        prompt="You are a test persona.",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# ToolPolicy layer — the shared chokepoint below both doors
# ---------------------------------------------------------------------------

def test_from_dict_ignores_explicit_overrides():
    """The DP-277 curl attack body: full grant + all three overrides."""
    policy = ToolPolicy.from_dict({
        "default": "allow",
        "allow": ["*"],
        "explicit_overrides": ALL_OVERRIDES,
    })
    assert policy.explicit_overrides == []


def test_to_dict_omits_explicit_overrides():
    policy = ToolPolicy(explicit_overrides=["network_read_local_write"])
    assert "explicit_overrides" not in policy.to_dict()


def test_from_dict_to_dict_round_trip_has_no_override_channel():
    """A dict that went through from_dict can never smuggle overrides back."""
    original = {"default": "deny", "allow": ["a"], "explicit_overrides": ALL_OVERRIDES}
    rt = ToolPolicy.from_dict(original).to_dict()
    assert "explicit_overrides" not in rt


# ---------------------------------------------------------------------------
# Persona layer — gated setter + preservation across policy rebuilds
# ---------------------------------------------------------------------------

def test_ctor_policy_dict_cannot_set_overrides():
    p = make_persona(tool_policy={"default": "deny", "allow": [], "explicit_overrides": ALL_OVERRIDES})
    assert p.get_explicit_overrides() == []


def test_set_tool_policy_dict_cannot_set_overrides():
    p = make_persona()
    p.set_tool_policy({"default": "allow", "allow": ["*"], "explicit_overrides": ALL_OVERRIDES})
    assert p.get_explicit_overrides() == []


def test_set_tool_policy_preserves_existing_overrides():
    """A benign policy edit must not drop a grandfathered override."""
    p = make_persona(explicit_overrides=["network_read_local_write"])
    p.set_tool_policy({"default": "deny", "allow": ["web_search"]})
    assert p.get_explicit_overrides() == ["network_read_local_write"]


def test_set_enabled_tools_preserves_existing_overrides():
    p = make_persona(explicit_overrides=["network_read_local_write"])
    p.set_enabled_tools(["web_search"])
    assert p.get_explicit_overrides() == ["network_read_local_write"]


def test_set_explicit_overrides_applies_and_returns_prior():
    p = make_persona()
    prior = p.set_explicit_overrides(["network_read_local_write"])
    assert prior == []
    assert p.get_explicit_overrides() == ["network_read_local_write"]
    prior = p.set_explicit_overrides([])
    assert prior == ["network_read_local_write"]
    assert p.get_explicit_overrides() == []


def test_set_explicit_overrides_rejects_unknown_names():
    p = make_persona()
    with pytest.raises(ValueError):
        p.set_explicit_overrides(["disable_all_security"])
    assert p.get_explicit_overrides() == []


def test_ctor_kwarg_grandfathers_unknown_names_inert(caplog):
    """Load path is lenient (a saved unknown name must not crash the load)
    but the name is inert — composition rules only honor KNOWN_OVERRIDES."""
    p = make_persona(explicit_overrides=["bogus_override"])
    assert p.get_explicit_overrides() == ["bogus_override"]


# ---------------------------------------------------------------------------
# Store layer — legacy migration + round-trip (CLAUDE.md config-schema rule:
# old shape and new shape both load; behavior tested with realistic config)
# ---------------------------------------------------------------------------

def _write_personas(tmp_path, personas_json):
    f = tmp_path / "personas.json"
    f.write_text(json.dumps({"personas": personas_json, "models": {}}))
    return str(f)

# A composition that trips Rule 1 (network:read + local:write) without the
# override — the realistic grandfather case.
_RULE1_TOOLS = ["web_search", "update_core_memory"]


def test_legacy_in_policy_override_is_migrated_and_honored(tmp_path):
    """OLD save shape: overrides INSIDE tool_policy. Must load into the gated
    field and still suppress the matching composition rule (no quarantine)."""
    save_file = _write_personas(tmp_path, [{
        "name": "legacy",
        "model_name": "m",
        "prompt": "p",
        "tool_policy": {
            "default": "deny",
            "allow": _RULE1_TOOLS,
            "ask": [],
            "capabilities_required": [],
            "explicit_overrides": ["network_read_local_write"],
        },
    }])
    loaded = load_personas_from_file(file_path_override=save_file)
    p = loaded["legacy"]
    assert p.get_explicit_overrides() == ["network_read_local_write"]
    assert not p.is_security_blocked()


def test_legacy_override_round_trips_into_top_level_field(tmp_path):
    """Load legacy -> save -> reload: override survives, lives at the top
    level, and is gone from the serialized tool_policy dict."""
    save_file = _write_personas(tmp_path, [{
        "name": "legacy",
        "model_name": "m",
        "prompt": "p",
        "tool_policy": {
            "default": "deny",
            "allow": _RULE1_TOOLS,
            "explicit_overrides": ["network_read_local_write"],
        },
    }])
    loaded = load_personas_from_file(file_path_override=save_file)
    save_personas_to_file(loaded, exclude_names=[], file_path_override=save_file)

    raw = json.loads(open(save_file).read())
    [entry] = raw["personas"]
    assert entry["explicit_overrides"] == ["network_read_local_write"]
    assert "explicit_overrides" not in entry["tool_policy"]

    reloaded = load_personas_from_file(file_path_override=save_file)
    p = reloaded["legacy"]
    assert p.get_explicit_overrides() == ["network_read_local_write"]
    assert not p.is_security_blocked()


def test_new_shape_top_level_override_loads(tmp_path):
    save_file = _write_personas(tmp_path, [{
        "name": "modern",
        "model_name": "m",
        "prompt": "p",
        "explicit_overrides": ["network_read_local_write"],
        "tool_policy": {"default": "deny", "allow": _RULE1_TOOLS},
    }])
    loaded = load_personas_from_file(file_path_override=save_file)
    p = loaded["modern"]
    assert p.get_explicit_overrides() == ["network_read_local_write"]
    assert not p.is_security_blocked()


def test_absent_override_key_loads_and_quarantines_bad_composition(tmp_path):
    """No override anywhere (old + new files alike): rules stay armed."""
    save_file = _write_personas(tmp_path, [{
        "name": "plain",
        "model_name": "m",
        "prompt": "p",
        "tool_policy": {"default": "deny", "allow": _RULE1_TOOLS},
    }])
    loaded = load_personas_from_file(file_path_override=save_file)
    p = loaded["plain"]
    assert p.get_explicit_overrides() == []
    assert p.is_security_blocked()


def test_empty_override_not_serialized(tmp_path):
    p = make_persona()
    save_file = str(tmp_path / "personas.json")
    save_personas_to_file({"tester": p}, exclude_names=[], file_path_override=save_file)
    [entry] = json.loads(open(save_file).read())["personas"]
    assert "explicit_overrides" not in entry
    assert "explicit_overrides" not in entry["tool_policy"]


# ---------------------------------------------------------------------------
# Command layer — `set tool_policy` door closed; `set explicit_overrides`
# is the dedicated path (revalidates + audits)
# ---------------------------------------------------------------------------

@pytest.fixture
def chat_state():
    chat_system = MagicMock()
    chat_system.personas = {
        "derpr": Persona("derpr", "gpt-4", "You are derpr."),
    }
    chat_system.last_api_iterations = {}
    return chat_system


@pytest.fixture
def bot_logic(chat_state):
    return make_bot_logic(chat_state)


@pytest.mark.asyncio
async def test_set_tool_policy_command_cannot_grant_overrides(bot_logic, chat_state):
    """The exact escalation payload from the DP-277 task file: full grant with
    all rules overridden. Even from an OPERATOR origin (the strongest caller),
    the overrides are dropped and the wildcard composition trips quarantine
    instead of validating clean."""
    payload = json.dumps({"default": "allow", "allow": ["*"], "explicit_overrides": ALL_OVERRIDES})
    result = await bot_logic.preprocess_message(OPERATOR_ORIGIN, "derpr", "operator", f"set tool_policy {payload}")
    persona = chat_state.personas["derpr"]
    assert persona.get_explicit_overrides() == []
    assert persona.is_security_blocked()
    assert result is not None and "QUARANTINED" in result["response"]


@pytest.mark.asyncio
async def test_set_explicit_overrides_command_applies_and_audits(bot_logic, chat_state):
    result = await bot_logic.preprocess_message(OPERATOR_ORIGIN, 
        "derpr", "operator", "set explicit_overrides network_read_local_write"
    )
    persona = chat_state.personas["derpr"]
    assert persona.get_explicit_overrides() == ["network_read_local_write"]
    assert result is not None and result["mutated"] is True

    audit = bot_logic.memory_manager.log_audit_event
    audit.assert_called_once()
    kwargs = audit.call_args.kwargs
    assert kwargs["event_type"] == "explicit_overrides_change"
    assert kwargs["operator_id"] == "operator"
    assert json.loads(kwargs["prior_state"]) == []
    assert json.loads(kwargs["new_state"]) == ["network_read_local_write"]


@pytest.mark.asyncio
async def test_set_explicit_overrides_rejects_unknown_name(bot_logic, chat_state):
    result = await bot_logic.preprocess_message(OPERATOR_ORIGIN, 
        "derpr", "operator", "set explicit_overrides disable_everything"
    )
    persona = chat_state.personas["derpr"]
    assert persona.get_explicit_overrides() == []
    assert result is not None and result["mutated"] is False
    assert "Unknown override" in result["response"]
    bot_logic.memory_manager.log_audit_event.assert_not_called()


@pytest.mark.asyncio
async def test_set_explicit_overrides_clears_quarantine(bot_logic, chat_state):
    """Adding the matching override re-validates and lifts the quarantine."""
    persona = chat_state.personas["derpr"]
    persona.set_tool_policy({"default": "deny", "allow": _RULE1_TOOLS})
    await bot_logic.preprocess_message(OPERATOR_ORIGIN, "derpr", "op", "set tool_policy " + json.dumps(
        {"default": "deny", "allow": _RULE1_TOOLS}
    ))
    assert persona.is_security_blocked()

    result = await bot_logic.preprocess_message(OPERATOR_ORIGIN, 
        "derpr", "op", "set explicit_overrides network_read_local_write"
    )
    assert result is not None and result["mutated"] is True
    assert not persona.is_security_blocked()


@pytest.mark.asyncio
async def test_set_explicit_overrides_json_list_and_clear(bot_logic, chat_state):
    persona = chat_state.personas["derpr"]
    await bot_logic.preprocess_message(OPERATOR_ORIGIN, 
        "derpr", "op", 'set explicit_overrides ["network_read_local_write","pii_read_network_any"]'
    )
    assert sorted(persona.get_explicit_overrides()) == [
        "network_read_local_write", "pii_read_network_any",
    ]
    await bot_logic.preprocess_message(OPERATOR_ORIGIN, "derpr", "op", "set explicit_overrides none")
    assert persona.get_explicit_overrides() == []
