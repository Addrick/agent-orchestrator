# tests/persona/test_persona.py

import pytest
from src.persona import Persona
from config import global_config

# Use known values for the default limits in tests
TEST_DEFAULT_HISTORY_MESSAGES = 15
TEST_DEFAULT_TOKEN_LIMIT = 4096


@pytest.fixture(autouse=True)
def patch_global_config(monkeypatch):
    """Fixture to ensure consistent default limits for all tests."""
    monkeypatch.setattr(global_config, 'DEFAULT_HISTORY_MESSAGES', TEST_DEFAULT_HISTORY_MESSAGES)
    monkeypatch.setattr(global_config, 'DEFAULT_TOKEN_LIMIT', TEST_DEFAULT_TOKEN_LIMIT)


@pytest.fixture
def base_persona_args():
    """Provides a dictionary of basic arguments to create a Persona."""
    return {
        "persona_name": "tester",
        "model_name": "test_model",
        "prompt": "You are a test persona."
    }


@pytest.fixture
def persona(base_persona_args):
    """Provides a standard Persona instance for tests."""
    return Persona(**base_persona_args)


# --- Initialization Tests ---

def test_persona_initialization_with_all_values(base_persona_args):
    """Tests that a Persona is created correctly when all values are provided."""
    p = Persona(
        **base_persona_args,
        history_messages=10,
        display_name_in_chat=True
    )
    assert p.get_name() == "tester"
    assert p.get_history_messages() == 10
    assert p.should_display_name_in_chat() is True


def test_persona_initialization_defaults_context_length(base_persona_args):
    """
    Tests that context_length defaults to the global config if the argument is None.
    This is a critical test for our new logic.
    """
    p = Persona(**base_persona_args, history_messages=None)
    assert p.get_history_messages() == TEST_DEFAULT_HISTORY_MESSAGES


def test_persona_initialization_uses_provided_zero_context(base_persona_args):
    """Tests that an explicit context_length of 0 is respected during initialization."""
    p = Persona(**base_persona_args, history_messages=0)
    assert p.get_history_messages() == 0


def test_persona_initialization_sanitizes_token_limit(base_persona_args):
    """Tests that the constructor sanitizes invalid or missing token_limit values."""
    # Test initialization with an invalid string like "none"
    p_invalid_str = Persona(**base_persona_args, token_limit="none")
    assert p_invalid_str.get_response_token_limit() == TEST_DEFAULT_TOKEN_LIMIT

    # Test initialization with None
    p_none = Persona(**base_persona_args, token_limit=None)
    assert p_none.get_response_token_limit() == TEST_DEFAULT_TOKEN_LIMIT


# --- Setter Tests for context_length ---

def test_set_context_length_valid(persona):
    """Tests setting a valid, non-zero context length."""
    result = persona.set_history_messages(5)
    assert result == 5
    assert persona.get_history_messages() == 5


def test_set_context_length_zero(persona):
    """Tests that setting context length to 0 is a valid operation."""
    result = persona.set_history_messages(0)
    assert result == 0
    assert persona.get_history_messages() == 0


def test_set_context_length_invalid(persona):
    """
    Tests that setting an invalid context length (e.g., a string) causes it to
    revert to the global default.
    """
    persona.set_history_messages(99)  # Start with a known value
    result = persona.set_history_messages("invalid_string")
    assert result == TEST_DEFAULT_HISTORY_MESSAGES
    assert persona.get_history_messages() == TEST_DEFAULT_HISTORY_MESSAGES


# --- Setter Tests for Other Attributes ---


def test_set_display_name(persona):
    """Tests the setter for display_name_in_chat."""
    assert persona.should_display_name_in_chat() is False
    persona.set_display_name_in_chat(True)
    assert persona.should_display_name_in_chat() is True


def test_set_prompt(persona):
    """Tests the setter for the prompt."""
    new_prompt = "This is a new prompt."
    persona.set_prompt(new_prompt)
    assert persona.get_prompt() == new_prompt


def test_set_model_name(persona):
    """Tests the setter for the model name."""
    new_model = "gpt-5"
    persona.set_model_name(new_model)
    assert persona.get_model_name() == new_model


def test_default_model_sentinel_resolves(base_persona_args, monkeypatch):
    """A persona authored with model_name="default" inherits DEFAULT_MODEL_NAME
    at runtime, while the raw value (for save + UI) stays the sentinel."""
    monkeypatch.setattr(global_config, "DEFAULT_MODEL_NAME", "gemini-3.1-flash-lite")
    args = {**base_persona_args, "model_name": "default"}
    p = Persona(**args)
    assert p.get_raw_model_name() == "default"
    assert p.get_model_name() == "gemini-3.1-flash-lite"
    # The engine config carries the resolved id (never the sentinel).
    assert p.get_config_for_engine()["model_name"] == "gemini-3.1-flash-lite"
    # Tracks the global default if it changes.
    monkeypatch.setattr(global_config, "DEFAULT_MODEL_NAME", "gemini-2.5-flash")
    assert p.get_model_name() == "gemini-2.5-flash"


def test_concrete_model_name_not_resolved(base_persona_args):
    """A concrete id passes through untouched — only the "default" literal is a
    sentinel."""
    p = Persona(**{**base_persona_args, "model_name": "gemini-2.5-flash"})
    assert p.get_raw_model_name() == "gemini-2.5-flash"
    assert p.get_model_name() == "gemini-2.5-flash"


def test_set_response_token_limit(persona):
    """Tests valid, invalid, and edge cases for setting token limit."""
    assert persona.set_response_token_limit(500) == 500
    assert persona.get_response_token_limit() == 500
    # Test the minimum value enforcement
    assert persona.set_response_token_limit(50) == 100
    assert persona.get_response_token_limit() == 100
    # Test invalid value falls back to default, not None
    assert persona.set_response_token_limit("invalid") == TEST_DEFAULT_TOKEN_LIMIT
    assert persona.get_response_token_limit() == TEST_DEFAULT_TOKEN_LIMIT


def test_set_temperature(persona):
    """Tests valid and invalid values for temperature."""
    assert persona.set_temperature(0.8) == 0.8
    assert persona.get_temperature() == 0.8
    assert persona.set_temperature("invalid") is None
    assert persona.get_temperature() is None


def test_set_top_p(persona):
    """Tests valid and invalid values for top_p."""
    assert persona.set_top_p(0.9) == 0.9
    assert persona.get_top_p() == 0.9
    assert persona.set_top_p("invalid") is None
    assert persona.get_top_p() is None


def test_set_top_k(persona):
    """Tests valid and invalid values for top_k."""
    assert persona.set_top_k(40) == 40
    assert persona.get_top_k() == 40
    assert persona.set_top_k("invalid") is None
    assert persona.get_top_k() is None


# --- Utility Method Tests ---

def test_append_to_prompt(persona):
    """Tests appending text to the existing prompt."""
    initial_prompt = persona.get_prompt()
    persona.append_to_prompt(" More text.")
    assert persona.get_prompt() == initial_prompt + " More text."


def test_get_config_for_engine(base_persona_args):
    """Tests that the engine config dictionary is assembled correctly."""
    p = Persona(
        **base_persona_args,
        token_limit=1024,
        temperature=0.8,
        top_p=0.9,
        top_k=40
    )
    expected_config = {
        "persona_name": "tester",
        "model_name": "test_model",
        "max_output_tokens": 1024,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "max_context_tokens": p.get_max_context_tokens(),
    }
    assert p.get_config_for_engine() == expected_config


# --- Ambient Memory Tests ---

def test_include_ambient_memory_default(base_persona_args):
    """include_ambient_memory defaults to True."""
    p = Persona(**base_persona_args)
    assert p.get_include_ambient_memory() is True


def test_include_ambient_memory_explicit_false(base_persona_args):
    """include_ambient_memory can be set to False."""
    p = Persona(**base_persona_args, include_ambient_memory=False)
    assert p.get_include_ambient_memory() is False


def test_include_ambient_memory_absent_in_config(base_persona_args):
    """When loading from config without the field, defaults to True (backward compat)."""
    # Simulates old config without include_ambient_memory key
    p = Persona(**base_persona_args)
    assert p.get_include_ambient_memory() is True


# --- DP-118: ingest_bank ---

def test_ingest_bank_default_is_none(base_persona_args):
    p = Persona(**base_persona_args)
    assert p.get_ingest_bank() is None


def test_ingest_bank_explicit(base_persona_args):
    p = Persona(**base_persona_args, ingest_bank="custom_bank")
    assert p.get_ingest_bank() == "custom_bank"


def test_ingest_bank_empty_string_normalized_to_none(base_persona_args):
    p = Persona(**base_persona_args, ingest_bank="")
    assert p.get_ingest_bank() is None


# --- Phase 3: max_context_tokens ---

def test_max_context_tokens_default(base_persona_args, monkeypatch):
    monkeypatch.setattr(global_config, 'DEFAULT_MAX_CONTEXT_TOKENS', 131072)
    p = Persona(**base_persona_args)
    assert p.get_max_context_tokens() == 131072


def test_max_context_tokens_explicit(base_persona_args):
    p = Persona(**base_persona_args, max_context_tokens=8192)
    assert p.get_max_context_tokens() == 8192


def test_max_context_tokens_invalid_uses_default(base_persona_args, monkeypatch):
    monkeypatch.setattr(global_config, 'DEFAULT_MAX_CONTEXT_TOKENS', 131072)
    p = Persona(**base_persona_args, max_context_tokens="not-a-number")
    assert p.get_max_context_tokens() == 131072


def test_set_max_context_tokens_clamps_low(base_persona_args):
    p = Persona(**base_persona_args, max_context_tokens=8192)
    p.set_max_context_tokens(50)
    assert p.get_max_context_tokens() == 100


def test_set_max_context_tokens_invalid_resets(base_persona_args, monkeypatch):
    monkeypatch.setattr(global_config, 'DEFAULT_MAX_CONTEXT_TOKENS', 131072)
    p = Persona(**base_persona_args, max_context_tokens=8192)
    p.set_max_context_tokens("garbage")
    assert p.get_max_context_tokens() == 131072


# --- provider_extras helpers (Phase E) ---

def test_provider_extra_set_and_get(persona):
    persona.set_provider_extra("kobold", "mirostat", 2)
    assert persona.get_provider_extra("kobold", "mirostat") == 2
    assert persona.get_generation_params().provider_extras == {"kobold": {"mirostat": 2}}


def test_provider_extra_get_unset_returns_none(persona):
    assert persona.get_provider_extra("kobold", "missing") is None
    assert persona.get_provider_extra("nonexistent", "anything") is None


def test_provider_extra_clear_existing_and_prunes_empty_block(persona):
    persona.set_provider_extra("kobold", "rep_pen", 1.1)
    assert persona.clear_provider_extra("kobold", "rep_pen") is True
    assert persona.get_provider_extra("kobold", "rep_pen") is None
    # block pruned when empty
    assert "kobold" not in persona.get_generation_params().provider_extras


def test_provider_extra_clear_missing_returns_false(persona):
    assert persona.clear_provider_extra("kobold", "never_set") is False


def test_provider_extra_clear_keeps_other_keys(persona):
    persona.set_provider_extra("kobold", "a", 1)
    persona.set_provider_extra("kobold", "b", 2)
    persona.clear_provider_extra("kobold", "a")
    assert persona.get_provider_extra("kobold", "b") == 2
    assert persona.get_generation_params().provider_extras == {"kobold": {"b": 2}}


# --- meta_visible (DP-111 / Sprint 4 fan-out) ---

def test_meta_visible_default_false(persona):
    assert persona.get_meta_visible() is False


def test_meta_visible_explicit_true(base_persona_args):
    p = Persona(**base_persona_args, meta_visible=True)
    assert p.get_meta_visible() is True


def test_meta_visible_setter(persona):
    persona.set_meta_visible(True)
    assert persona.get_meta_visible() is True
    persona.set_meta_visible(False)
    assert persona.get_meta_visible() is False


def test_meta_visible_round_trip_save_load(base_persona_args, tmp_path):
    """Round-trip meta_visible through save_personas_to_file → load_personas_from_file."""
    from src.personas.store import save_personas_to_file, load_personas_from_file

    p_visible = Persona(**{**base_persona_args, "persona_name": "visible"}, meta_visible=True)
    p_hidden = Persona(**{**base_persona_args, "persona_name": "hidden"}, meta_visible=False)
    save_file = str(tmp_path / "personas.json")
    save_personas_to_file({"visible": p_visible, "hidden": p_hidden}, set(), file_path_override=save_file)

    loaded = load_personas_from_file(file_path_override=save_file)
    assert loaded["visible"].get_meta_visible() is True
    assert loaded["hidden"].get_meta_visible() is False


def test_meta_visible_absent_in_legacy_config_defaults_false(base_persona_args, tmp_path):
    """A persona JSON without meta_visible (old config file) loads as False."""
    import json
    from src.personas.store import load_personas_from_file

    save_file = tmp_path / "personas.json"
    save_file.write_text(json.dumps({
        "personas": [{
            "name": "legacy",
            "model_name": "m",
            "prompt": "p",
        }],
    }))
    loaded = load_personas_from_file(file_path_override=str(save_file))
    assert loaded["legacy"].get_meta_visible() is False


# --- DP-128: security quarantine (load-but-block) ---

# An insecure tool composition: allow-all exposes web/agent/zammad tools at once,
# tripping the network:read+local:write and pii:read+foreign-egress rules.
_INSECURE_POLICY = {"default": "allow", "allow": ["*"], "ask": []}
# Zammad-only is the documented same-origin clean set.
_SECURE_ZAMMAD_TOOLS = [
    "get_ticket_details", "create_ticket", "update_ticket", "search_tickets",
    "add_note_to_ticket", "search_user", "create_user", "update_user",
    "delete_user", "merge_tickets",
]


def test_persona_not_security_blocked_by_default(persona):
    """A persona built without block reasons is not quarantined."""
    assert persona.is_security_blocked() is False
    assert persona.get_security_block_reasons() == []


def test_persona_security_blocked_via_constructor(base_persona_args):
    """security_block_reasons passed at construction quarantines the persona."""
    reasons = ["Insecure composition: network:read + local:write"]
    p = Persona(**base_persona_args, security_block_reasons=reasons)
    assert p.is_security_blocked() is True
    assert p.get_security_block_reasons() == reasons
    # Returned list is a copy — callers can't mutate internal state.
    p.get_security_block_reasons().append("x")
    assert p.get_security_block_reasons() == reasons


def test_revalidate_security_trips_block_on_insecure_policy(base_persona_args):
    """revalidate_persona_security flags an insecure composition even if loaded clean."""
    from src.tools.composition import revalidate_persona_security
    p = Persona(**base_persona_args, service_bindings=["zammad", "agents"],
                tool_policy=_INSECURE_POLICY)
    assert p.is_security_blocked() is False  # constructor does not auto-validate
    assert revalidate_persona_security(p) is True
    assert p.is_security_blocked() is True
    assert p.get_security_block_reasons()


def test_set_enabled_tools_is_a_pure_mutator(base_persona_args):
    """The raw setter does NOT re-validate (internal callers use it freely);
    re-validation is an explicit, operator-edit-boundary concern (DP-128)."""
    p = Persona(**base_persona_args, service_bindings=["zammad", "agents"])
    p.set_enabled_tools(["*"])  # insecure composition, but setter stays quiet
    assert p.is_security_blocked() is False


def test_revalidate_clears_quarantine_after_scoping_tools(base_persona_args):
    """A quarantined persona is repaired by scoping tools then re-validating —
    the sequence the `set tools` dev command runs (no restart)."""
    p = Persona(**base_persona_args, service_bindings=["zammad", "agents"],
                tool_policy=_INSECURE_POLICY,
                security_block_reasons=["Insecure composition: network:read + local:write"])
    from src.tools.composition import revalidate_persona_security
    assert p.is_security_blocked() is True
    p.set_enabled_tools(_SECURE_ZAMMAD_TOOLS)
    assert revalidate_persona_security(p) is False
    assert p.is_security_blocked() is False
    assert p.get_security_block_reasons() == []


def test_revalidate_trips_block_on_insecure_edit(base_persona_args):
    """Editing a clean persona into an insecure policy then re-validating
    quarantines it."""
    from src.tools.composition import revalidate_persona_security
    p = Persona(**base_persona_args, service_bindings=["zammad"],
                enabled_tools=_SECURE_ZAMMAD_TOOLS)
    assert p.is_security_blocked() is False
    p.set_tool_policy(_INSECURE_POLICY)
    assert revalidate_persona_security(p) is True
    assert p.is_security_blocked() is True


def test_insecure_persona_loads_quarantined_not_dropped(tmp_path):
    """DP-128: an insecure persona LOADS (quarantined) instead of being dropped."""
    import json
    from src.personas.store import load_personas_from_file

    save_file = tmp_path / "personas.json"
    save_file.write_text(json.dumps({"personas": [
        {"name": "bad", "model_name": "m", "prompt": "p",
         "enabled_tools": ["*"], "service_bindings": ["zammad", "agents"]},
        {"name": "good", "model_name": "m", "prompt": "p",
         "enabled_tools": _SECURE_ZAMMAD_TOOLS, "service_bindings": ["zammad"]},
    ]}))
    loaded = load_personas_from_file(file_path_override=str(save_file))
    assert "bad" in loaded, "insecure persona must still load (quarantined), not be dropped"
    assert loaded["bad"].is_security_blocked() is True
    assert loaded["bad"].get_security_block_reasons()
    assert loaded["good"].is_security_blocked() is False


# --- inject_timestamp tests ---

def test_inject_timestamp_default_true(base_persona_args):
    """inject_timestamp defaults to True for user/chat personas."""
    p = Persona(**base_persona_args)
    assert p.get_inject_timestamp() is True


def test_inject_timestamp_explicit_false(base_persona_args):
    """inject_timestamp can be explicitly initialized to False."""
    p = Persona(**base_persona_args, inject_timestamp=False)
    assert p.get_inject_timestamp() is False


def test_inject_timestamp_setter(persona):
    """inject_timestamp has a working setter."""
    assert persona.get_inject_timestamp() is True
    persona.set_inject_timestamp(False)
    assert persona.get_inject_timestamp() is False
    persona.set_inject_timestamp(True)
    assert persona.get_inject_timestamp() is True


def test_inject_timestamp_round_trip_save_load(base_persona_args, tmp_path):
    """Round-trip inject_timestamp through save_personas_to_file → load_personas_from_file."""
    from src.personas.store import save_personas_to_file, load_personas_from_file

    p_inject = Persona(**{**base_persona_args, "persona_name": "inject"}, inject_timestamp=True)
    p_no_inject = Persona(**{**base_persona_args, "persona_name": "no_inject"}, inject_timestamp=False)
    save_file = str(tmp_path / "personas.json")
    save_personas_to_file({"inject": p_inject, "no_inject": p_no_inject}, set(), file_path_override=save_file)

    loaded = load_personas_from_file(file_path_override=save_file)
    assert loaded["inject"].get_inject_timestamp() is True
    assert loaded["no_inject"].get_inject_timestamp() is False


def test_inject_timestamp_absent_in_legacy_config_defaults_true(base_persona_args, tmp_path):
    """A persona JSON without inject_timestamp (old config file) loads as True."""
    import json
    from src.personas.store import load_personas_from_file

    save_file = tmp_path / "personas.json"
    save_file.write_text(json.dumps({
        "personas": [{
            "name": "legacy",
            "model_name": "m",
            "prompt": "p",
        }],
    }))
    loaded = load_personas_from_file(file_path_override=str(save_file))
    assert loaded["legacy"].get_inject_timestamp() is True


def test_build_wire_messages_inject_timestamp(base_persona_args):
    from src.tools.tool_loop import build_wire_messages
    p = Persona(**base_persona_args, inject_timestamp=True)
    history = [{"role": "user", "content": "hello"}]
    messages = build_wire_messages(p, history)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "[Current Time:" in messages[0]["content"]
    assert "You are a test persona." in messages[0]["content"]


def test_build_wire_messages_no_inject_timestamp(base_persona_args):
    from src.tools.tool_loop import build_wire_messages
    p = Persona(**base_persona_args, inject_timestamp=False)
    history = [{"role": "user", "content": "hello"}]
    messages = build_wire_messages(p, history)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "[Current Time:" not in messages[0]["content"]
    assert messages[0]["content"] == "You are a test persona."




# --- DP-255: Hindsight retain-tuning fields ---

def test_persona_retain_fields_default_to_none(persona):
    """Old persona JSON has none of these fields; they must default to None/neutral."""
    assert persona.get_retain_mission() is None
    assert persona.get_reflect_mission() is None
    assert persona.get_observations_mission() is None
    assert persona.get_enable_observations() is None
    assert persona.get_disposition() is None


def test_persona_retain_fields_present(base_persona_args):
    """When supplied (new config), the fields are stored and readable."""
    p = Persona(
        **base_persona_args,
        retain_mission="Extract durable signal only.",
        reflect_mission="Reason over durable preferences.",
        observations_mission="Consolidate beliefs.",
        enable_observations=True,
        disposition={"skepticism": 4, "literalism": 2, "empathy": 3},
    )
    assert p.get_retain_mission() == "Extract durable signal only."
    assert p.get_reflect_mission() == "Reason over durable preferences."
    assert p.get_observations_mission() == "Consolidate beliefs."
    assert p.get_enable_observations() is True
    assert p.get_disposition() == {"skepticism": 4, "literalism": 2, "empathy": 3}


def test_persona_disposition_sanitizes_and_clamps(base_persona_args):
    """Out-of-range ints clamp to 1-5; unknown keys and bad values are dropped."""
    p = Persona(
        **base_persona_args,
        disposition={"skepticism": 9, "literalism": 0, "empathy": "x", "bogus": 5},
    )
    assert p.get_disposition() == {"skepticism": 5, "literalism": 1}


def test_persona_disposition_empty_is_none(base_persona_args):
    p = Persona(**base_persona_args, disposition={})
    assert p.get_disposition() is None


def test_persona_retain_field_setters(persona):
    persona.set_retain_mission("new mission")
    assert persona.get_retain_mission() == "new mission"
    persona.set_retain_mission(None)
    assert persona.get_retain_mission() is None
    persona.set_enable_observations(False)
    assert persona.get_enable_observations() is False
    persona.set_disposition({"empathy": 7})
    assert persona.get_disposition() == {"empathy": 5}
    persona.set_disposition(None)
    assert persona.get_disposition() is None


def test_persona_retain_fields_roundtrip_through_store(tmp_path, base_persona_args):
    """to_dict -> load preserves the fields; a persona without them omits the keys."""
    from src.personas import store

    tuned = Persona(
        **base_persona_args,
        retain_mission="Tuned retain.",
        disposition={"skepticism": 4},
        enable_observations=True,
    )
    plain = Persona(persona_name="plain", model_name="m", prompt="p")

    serialized = store.to_dict({"tester": tuned, "plain": plain})
    by_name = {e["name"]: e for e in serialized}
    # tuned persona carries the keys
    assert by_name["tester"]["retain_mission"] == "Tuned retain."
    assert by_name["tester"]["disposition"] == {"skepticism": 4}
    assert by_name["tester"]["enable_observations"] is True
    # plain persona (old-style) omits all of them
    assert "retain_mission" not in by_name["plain"]
    assert "disposition" not in by_name["plain"]
    assert "enable_observations" not in by_name["plain"]

    # Round-trip back through the loader.
    save_file = tmp_path / "personas.json"
    save_file.write_text(__import__("json").dumps({"personas": serialized, "models": {}}))
    loaded = store.load_personas_from_file(str(save_file))
    assert loaded["tester"].get_retain_mission() == "Tuned retain."
    assert loaded["tester"].get_disposition() == {"skepticism": 4}
    assert loaded["plain"].get_retain_mission() is None
    assert loaded["plain"].get_disposition() is None


def test_load_persona_json_without_retain_fields(tmp_path):
    """An old persona JSON entry (no retain fields at all) still loads cleanly."""
    from src.personas import store

    old_entry = {
        "personas": [
            {"name": "legacy", "model_name": "m", "prompt": "hello"}
        ],
        "models": {},
    }
    save_file = tmp_path / "old.json"
    save_file.write_text(__import__("json").dumps(old_entry))
    loaded = store.load_personas_from_file(str(save_file))
    assert loaded is not None
    p = loaded["legacy"]
    assert p.get_retain_mission() is None
    assert p.get_enable_observations() is None
    assert p.get_disposition() is None


# --- DP-327: the hypr infra-operator persona (opt-in, not seeded) -----------

HYPR_FILE = "optional_personas/hypr.json"


def _hypr_entry():
    """The hypr definition as it ships — deliberately NOT in the seed file."""
    import json
    import os

    path = os.path.join(global_config.CONFIG_DIR, HYPR_FILE)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["personas"][0]


def test_hypr_is_not_seeded_into_fresh_deployments():
    """Opt-in on purpose: seeding it would put node power ops on every deploy.

    hypr is the only persona holding the `proxmox` binding, and a park is
    approved by whoever raised it — so persona reachability *is* the authz
    boundary. An operator adds it to a live instance by hand.
    """
    import json
    import os

    path = os.path.join(global_config.CONFIG_DIR, "default_personas.json")
    with open(path, "r", encoding="utf-8") as fh:
        seed = json.load(fh)
    assert not [p for p in seed["personas"] if p.get("name") == "hypr"]


def test_hypr_declares_the_proxmox_binding_and_matching_policy():
    hypr = _hypr_entry()
    assert hypr["name"] == "hypr"
    # DP-265 added the second binding: hf_search/hf_files/install_model/
    # install_status provision a model onto the same node. Ordered pin, because
    # both are deliberate and a third would be a widening nobody reviewed.
    assert hypr["service_bindings"] == ["proxmox", "huggingface"]
    # Binding and tool list are two independent gates (request_builder filters on
    # the binding, ToolPolicy on the names) — a persona with only one of them
    # loads fine and then silently has no tools, so pin both.
    assert hypr["tool_policy"]["default"] == "deny"
    assert set(hypr["tool_policy"]["allow"]) == set(hypr["enabled_tools"])


def test_hypr_enabled_tools_are_all_real_tools_of_its_bindings():
    """Guards the rename/typo case: a tool name that no longer exists is inert.

    Both binding's toolsets, exactly — not a subset. hypr is the only persona
    holding either, so a tool added to one of those modules and not listed here
    is a tool nothing can reach, and one listed here but deleted from the module
    is a silently dead name in the allow list.
    """
    from src.tools.tool_defs.huggingface import HUGGINGFACE_TOOLS
    from src.tools.tool_defs.proxmox import PROXMOX_TOOLS

    hypr = _hypr_entry()
    real = {t["function"]["name"] for t in PROXMOX_TOOLS + HUGGINGFACE_TOOLS}
    assert set(hypr["enabled_tools"]) == real


def test_hypr_holds_no_binding_beyond_the_box_it_operates():
    """Blast radius: the box operator must not also reach zammad/fixr/agents.

    DP-265 widened this by one binding on purpose — `huggingface` provisions
    models onto the very node `proxmox` operates, so it is the same blast
    radius, not a new one. Everything that would cross into another subsystem
    (tickets, code edits, subagents, arbitrary MCP servers) stays out.
    """
    hypr = _hypr_entry()
    assert set(hypr["service_bindings"]) == {"proxmox", "huggingface"}
    assert not set(hypr["service_bindings"]) & {
        "zammad", "fixr", "agents", "voice", "mcp", "proposals",
    }
    assert "*" not in hypr["enabled_tools"]


def test_hypr_prompt_warns_that_it_operates_its_own_host():
    """It operates the node it lives on: stopping the guest derpr runs in kills
    derpr *and* the approval gate that parks its own writes. The prompt has to
    say so, because nothing in the toolset refuses a self-directed power-off."""
    prompt = _hypr_entry()["prompt"]
    assert "YOU RUN ON THE MACHINE YOU OPERATE" in prompt
    assert "approval gate" in prompt
    assert "reboot_node" in prompt


def test_hypr_template_names_no_real_guest_and_keeps_the_operator_slots():
    """The template ships in a public repo, so it must describe roles, not any
    deployment's actual topology — the operator fills the two `<unset>` slots in
    their own instance. A concrete name creeping back in is the regression."""
    prompt = _hypr_entry()["prompt"]
    assert prompt.count("<unset>") == 2
    assert "OPERATOR NOTE" in prompt
    # The lookup is live-state driven, so the template needs no example names at
    # all — and every name that appeared here previously was a real guest.
    assert "pve_status" in prompt


def test_hypr_loads_unquarantined():
    """DP-128 loads a composition-violating persona quarantined rather than
    dropping it — so 'it loaded' is not evidence it works. Assert the empty
    block-reason list explicitly."""
    import os

    from src.personas import store

    loaded = store.load_personas_from_file(
        os.path.join(global_config.CONFIG_DIR, HYPR_FILE)
    )
    assert loaded is not None and "hypr" in loaded
    assert loaded["hypr"].get_security_block_reasons() == []
    assert loaded["hypr"].get_service_bindings() == ["proxmox", "huggingface"]


# --- DP-330: persona origin allowlist --------------------------------------
#
# Config-schema change, so per CLAUDE.md both states are pinned: the key ABSENT
# (every persona file that predates this field) and the key PRESENT.

def _origin(transport="discord", server=None, channel=None, author=None,
            operator=False):
    from src.origin import Origin

    return Origin(transport=transport, server_id=server, channel_id=channel,
                  author_id=author, operator=operator)


def test_origin_allowlist_absent_is_unrestricted(persona):
    """The field's default must leave every existing persona reachable — this
    is what makes the gate a no-op for personas that never opt in."""
    from src.origin import ANONYMOUS

    assert persona.get_origin_allowlist() == []
    assert persona.is_addressable_from(ANONYMOUS) is True
    assert persona.is_addressable_from(_origin(server="999")) is True
    assert persona.is_addressable_from(_origin(transport="portal")) is True


def test_origin_allowlist_present_gates_by_guild(base_persona_args):
    p = Persona(**base_persona_args, origin_allowlist=["12345"])
    assert p.get_origin_allowlist() == ["12345"]
    # bare guild id = whole server, any channel, any author
    assert p.is_addressable_from(
        _origin(server="12345", channel="c1", author="a1")) is True
    assert p.is_addressable_from(
        _origin(server="99999", channel="c1", author="a1")) is False


def test_origin_allowlist_refuses_dms_and_non_discord_transports(base_persona_args):
    """Everything without a gateway-asserted guild fails closed: DMs, the
    portal, gmail/zammad bodies, and internal agent-initiated turns."""
    from src.origin import ANONYMOUS

    p = Persona(**base_persona_args, origin_allowlist=["12345"])
    assert p.is_addressable_from(_origin(server=None, author="a1")) is False
    assert p.is_addressable_from(ANONYMOUS) is False
    for transport in ("portal", "gmail", "zammad", "internal", "test"):
        assert p.is_addressable_from(
            _origin(transport=transport, channel="portal")) is False
    # A forged server_id on a non-Discord transport must not pass either: the
    # portal takes server_id from a caller-supplied request body.
    assert p.is_addressable_from(
        _origin(transport="portal", server="12345")) is False


def test_origin_allowlist_narrows_to_channel_and_author(base_persona_args):
    p = Persona(**base_persona_args, origin_allowlist=["12345/678", "999/*/42"])
    assert p.is_addressable_from(_origin(server="12345", channel="678")) is True
    assert p.is_addressable_from(_origin(server="12345", channel="000")) is False
    assert p.is_addressable_from(
        _origin(server="999", channel="anything", author="42")) is True
    assert p.is_addressable_from(
        _origin(server="999", channel="anything", author="43")) is False


def test_origin_allowlist_drops_malformed_entries(base_persona_args):
    """Fail closed: a wildcard server would grant every guild the bot is in, so
    the shared parser drops it rather than honouring it."""
    p = Persona(**base_persona_args, origin_allowlist=["*", "12345/1/2/3", "77"])
    assert p.get_origin_allowlist() == ["*", "12345/1/2/3", "77"]  # as authored
    assert p.is_addressable_from(_origin(server="12345", channel="1")) is False
    assert p.is_addressable_from(_origin(server="anything")) is False
    assert p.is_addressable_from(_origin(server="77")) is True


def test_set_origin_allowlist_normalizes_and_clears(persona):
    assert persona.set_origin_allowlist(["12345", "9/8/7"]) == ["12345", "9/8/7"]
    assert persona.is_addressable_from(_origin(server="12345")) is True
    # A list whose entries are ALL malformed must not collapse to empty, which
    # means unrestricted — the persona fails closed instead, and says so.
    assert persona.set_origin_allowlist(["*"]) == ["*"]
    assert persona.origin_allowlist_is_malformed() is True
    assert persona.is_addressable_from(_origin(server="12345")) is False
    assert persona.set_origin_allowlist([]) == []
    assert persona.origin_allowlist_is_malformed() is False
    assert persona.get_origin_allowlist() == []


def test_set_and_load_report_the_same_allowlist(base_persona_args, persona):
    """Both write paths run one normalizer, so the field reads back identically
    however it was last written — a persona's on-disk shape must not depend on
    whether it was last edited or last loaded."""
    authored = ["12345/*", "9//7"]
    persona.set_origin_allowlist(authored)
    loaded = Persona(**base_persona_args, origin_allowlist=authored)
    assert persona.get_origin_allowlist() == loaded.get_origin_allowlist()


def test_origin_allowlist_accepts_unquoted_guild_ids(base_persona_args):
    """A Discord guild id is a number, so the natural JSON authoring is
    unquoted. It must not raise — the exception was swallowed by
    load_personas_from_file's blanket except, which dropped EVERY user persona
    and let the next mutating dev command save an empty list over the file."""
    p = Persona(**base_persona_args, origin_allowlist=[347812763093172225])
    assert p.get_origin_allowlist() == ["347812763093172225"]
    assert p.is_addressable_from(_origin(server="347812763093172225")) is True
    assert p.is_addressable_from(_origin(server="99999")) is False


def test_persona_file_with_unquoted_guild_id_still_loads(tmp_path):
    """The end-to-end version of the above: one unquoted number in the file
    must not take the whole persona set down with it."""
    import json

    from src.personas import store

    save_file = tmp_path / "personas.json"
    save_file.write_text(json.dumps({"personas": [
        {"name": "gated", "model_name": "m", "prompt": "p",
         "origin_allowlist": [347812763093172225]},
        {"name": "other", "model_name": "m", "prompt": "p"},
    ], "models": {}}))
    loaded = store.load_personas_from_file(str(save_file))
    assert loaded is not None, "a bad allowlist must not drop every persona"
    assert set(loaded) == {"gated", "other"}
    assert loaded["gated"].is_addressable_from(
        _origin(server="347812763093172225")) is True


def test_origin_allowlist_bare_string_is_one_entry(base_persona_args):
    """`"origin_allowlist": "12345"` (a one-character JSON typo) must not
    explode into one entry per character, which bricked the persona from every
    origin including the intended one."""
    p = Persona(**base_persona_args, origin_allowlist="12345")
    assert p.get_origin_allowlist() == ["12345"]
    assert p.is_addressable_from(_origin(server="12345")) is True
    assert p.is_addressable_from(_origin(server="1")) is False


def test_origin_allowlist_comma_inside_an_entry_never_widens(base_persona_args):
    """Entries are parsed one at a time. Joining them with ',' first let a
    comma pasted INSIDE one entry grant a guild the operator never listed —
    a typo that widens reachability, which this field must never do."""
    p = Persona(**base_persona_args, origin_allowlist=["12345,99999"])
    assert p.is_addressable_from(_origin(server="12345")) is False
    assert p.is_addressable_from(_origin(server="99999")) is False
    assert p.origin_allowlist_is_malformed() is True


def test_origin_allowlist_rejects_non_list_values(base_persona_args):
    """Anything that is not a list/tuple/str fails closed rather than being
    coerced into a policy nobody wrote."""
    p = Persona(**base_persona_args, origin_allowlist={"guild": "12345"})
    assert p.origin_allowlist_is_malformed() is True
    assert p.is_addressable_from(_origin(server="12345")) is False


def test_declared_empty_origin_allowlist_survives_a_save(base_persona_args):
    """A persona shipped with `"origin_allowlist": []` keeps the key on save —
    it is the operator's only in-file hint that the knob exists, and erasing it
    leaves the persona unrestricted with nothing suggesting it could be
    otherwise."""
    from src.personas import store

    declared = Persona(**base_persona_args, origin_allowlist=[])
    never_set = Persona(persona_name="plain", model_name="m", prompt="p")
    assert declared.origin_allowlist_is_declared() is True
    assert never_set.origin_allowlist_is_declared() is False

    by_name = {e["name"]: e for e in
               store.to_dict({"tester": declared, "plain": never_set})}
    assert by_name["tester"]["origin_allowlist"] == []
    assert "origin_allowlist" not in by_name["plain"]


def test_origin_allowlist_roundtrips_through_store(tmp_path, base_persona_args):
    import json

    from src.personas import store

    gated = Persona(**base_persona_args, origin_allowlist=["12345"])
    plain = Persona(persona_name="plain", model_name="m", prompt="p")
    serialized = store.to_dict({"tester": gated, "plain": plain})
    by_name = {e["name"]: e for e in serialized}
    assert by_name["tester"]["origin_allowlist"] == ["12345"]
    # An unrestricted persona keeps its on-disk shape — no new key appears.
    assert "origin_allowlist" not in by_name["plain"]

    save_file = tmp_path / "personas.json"
    save_file.write_text(json.dumps({"personas": serialized, "models": {}}))
    loaded = store.load_personas_from_file(str(save_file))
    assert loaded["tester"].get_origin_allowlist() == ["12345"]
    assert loaded["tester"].is_addressable_from(_origin(server="12345")) is True
    assert loaded["tester"].is_addressable_from(_origin(server="1")) is False
    assert loaded["plain"].get_origin_allowlist() == []
    assert loaded["plain"].is_addressable_from(_origin(server="1")) is True


def test_load_persona_json_without_origin_allowlist(tmp_path):
    """A persona file written before DP-330 loads unrestricted, not blocked."""
    import json

    from src.personas import store

    save_file = tmp_path / "old.json"
    save_file.write_text(json.dumps(
        {"personas": [{"name": "legacy", "model_name": "m", "prompt": "hi"}],
         "models": {}}))
    loaded = store.load_personas_from_file(str(save_file))
    assert loaded["legacy"].get_origin_allowlist() == []
    assert loaded["legacy"].is_addressable_from(_origin(server="1")) is True


def test_system_persona_load_accepts_origin_allowlist(tmp_path, monkeypatch):
    """The system-persona loader is a second, independently maintained call
    into Persona() — a field wired into only one of them is the classic miss."""
    import json

    from src.personas import store

    sys_file = tmp_path / "system.json"
    sys_file.write_text(json.dumps({"personas": [
        {"name": "sysgated", "model_name": "m", "prompt": "p",
         "origin_allowlist": ["12345"]},
        {"name": "sysplain", "model_name": "m", "prompt": "p"},
    ]}))
    monkeypatch.setattr(global_config, "SYSTEM_PERSONA_FILE", str(sys_file))
    loaded = store.load_system_personas_from_file()
    assert loaded["sysgated"].get_origin_allowlist() == ["12345"]
    assert loaded["sysgated"].is_addressable_from(_origin(server="1")) is False
    assert loaded["sysplain"].is_addressable_from(_origin(server="1")) is True


def test_hypr_ships_with_an_empty_origin_allowlist():
    """DP-330: the guild id is the operator's, and this template lives in a
    public repo — it ships with the field present but empty, and the operator
    fills it in their own data/personas.json. `to_dict` keys off the field
    being DECLARED so that empty list survives the first mutating dev
    command."""
    assert _hypr_entry()["origin_allowlist"] == []
