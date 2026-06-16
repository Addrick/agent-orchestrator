# tests/test_persona_fields.py
"""Registry-level invariants for the persona-field table (DP-200 slice D).

Behavioral coverage of individual fields lives in the message-handler tests
(dev-command surface) and the kobold adapter tests (PATCH surface) — both
dispatch through this registry now. These tests pin the registry's own
contracts: the surfaces it feeds can't drift, and the factory semantics
match the legacy hand-written handlers they replaced.
"""

import pytest

from src.interfaces._persona_patch import _KNOWN_PATCH_KEYS_ENGINE
from src.persona import Persona
from src.persona_fields import (
    PERSONA_FIELDS,
    apply_patch_fields,
    cli_set_handlers,
    cli_what_handlers,
    patchable_fields,
    registry_patch_keys,
)


@pytest.fixture
def persona():
    return Persona(persona_name="reg_test", model_name="gemini-2.5-flash", prompt="hi")


def test_field_names_are_unique():
    names = [f.name for f in PERSONA_FIELDS]
    assert len(names) == len(set(names))


def test_every_field_has_at_least_one_surface():
    for f in PERSONA_FIELDS:
        assert f.describe or f.set_cli or f.patch_key, (
            f"Registry field '{f.name}' is dead weight — no what/set/patch surface."
        )


def test_patchable_fields_have_apply():
    for f in patchable_fields():
        assert f.patch_apply is not None, f"'{f.name}' has patch_key but no patch_apply"


def test_patch_keys_are_known_to_the_patch_route():
    """The PATCH route's known-key set must include every registry patch key,
    or the route would warn 'unknown field' on keys it actually applies."""
    assert registry_patch_keys() <= _KNOWN_PATCH_KEYS_ENGINE


def test_what_set_round_trip(persona):
    """Every field with both surfaces: set a value, then `what` reflects it."""
    set_table = cli_set_handlers()
    what_table = cli_what_handlers()

    msg, mutated = set_table['temp'](['temp', '0.7'], persona)
    assert mutated and "0.7" in msg
    assert "0.7" in what_table['temp']([], persona)[0]

    msg, mutated = set_table['long_term_memory'](['long_term_memory', 'on'], persona)
    assert mutated
    assert "enabled" in what_table['long_term_memory']([], persona)[0]

    msg, mutated = set_table['memory_mode'](['memory_mode', 'global'], persona)
    assert mutated
    assert "global" in what_table['memory_mode']([], persona)[0]


def test_set_chat_template_validates_slug(persona):
    """`set chat_template` accepts a known CHAT_TEMPLATES slug, rejects an
    unknown one (with the available list), and clears on 'none'."""
    from src.stream_engine import CHAT_TEMPLATES
    set_ct = cli_set_handlers()['chat_template']

    known = sorted(CHAT_TEMPLATES)[0]
    msg, mutated = set_ct(['chat_template', known], persona)
    assert mutated and known in msg
    assert persona.get_chat_template() == known

    msg, mutated = set_ct(['chat_template', 'not_a_real_template'], persona)
    assert not mutated
    assert "unknown chat template" in msg.lower()
    assert known in msg  # available list surfaced
    assert persona.get_chat_template() == known  # unchanged

    msg, mutated = set_ct(['chat_template', 'none'], persona)
    assert mutated and persona.get_chat_template() is None


def test_optional_numeric_legacy_semantics(persona):
    """Factory must keep the quirky legacy contract: missing arg falls
    through, non-numeric input MUTATES (clears to default), range error
    does not mutate."""
    set_temp = cli_set_handlers()['temp']

    assert set_temp(['temp'], persona) == (None, False)

    msg, mutated = set_temp(['temp', 'banana'], persona)
    assert mutated is True
    assert "Non-numeric temperature 'banana'" in msg
    assert persona.get_temperature() is None

    persona.set_temperature(0.5)
    msg, mutated = set_temp(['temp', '9'], persona)
    assert mutated is False
    assert "between 0 and 2" in msg
    assert persona.get_temperature() == 0.5


def test_apply_patch_fields_rejection_semantics(persona):
    """PATCH applies must reject coerced-away values like the old if-chain."""
    rejected: list = []
    apply_patch_fields(persona, {"temperature": "abc", "top_p": 0.9, "prompt": "new"}, rejected)
    assert rejected == ["temperature"]
    assert persona.get_top_p() == 0.9
    assert persona.get_prompt() == "new"

    rejected = []
    before = persona.get_memory_mode()
    apply_patch_fields(persona, {"memory_mode": "not_a_mode"}, rejected)
    assert rejected == ["memory_mode"]
    assert persona.get_memory_mode() == before


def test_memory_mode_patch_accepts_current_mode_any_case(persona):
    """Re-sending the persona's current memory mode must not be reported as a
    rejection — including the lowercase form the API itself displays
    (get_memory_mode().name.lower()), which Persona.set_memory_mode accepts."""
    current = persona.get_memory_mode()

    for same_value in (current.name, current.name.lower(), current):
        rejected: list = []
        apply_patch_fields(persona, {"memory_mode": same_value}, rejected)
        assert rejected == [], f"no-op memory_mode={same_value!r} falsely rejected"
        assert persona.get_memory_mode() == current


def test_inject_timestamp_field(persona):
    set_table = cli_set_handlers()
    what_table = cli_what_handlers()

    # Default is True
    assert "enabled" in what_table['inject_timestamp']([], persona)[0]

    # Turn off via CLI setter
    msg, mutated = set_table['inject_timestamp'](['inject_timestamp', 'off'], persona)
    assert mutated
    assert "disabled" in msg
    assert persona.get_inject_timestamp() is False
    assert "disabled" in what_table['inject_timestamp']([], persona)[0]

    # Turn on via CLI setter
    msg, mutated = set_table['inject_timestamp'](['inject_timestamp', 'on'], persona)
    assert mutated
    assert "enabled" in msg
    assert persona.get_inject_timestamp() is True

    # Check PATCH apply
    rejected = []
    apply_patch_fields(persona, {"inject_timestamp": False}, rejected)
    assert not rejected
    assert persona.get_inject_timestamp() is False


def test_self_edit_field(persona):
    """DP-227: self_edit is queryable, settable, and patchable through the
    registry, default disabled."""
    set_table = cli_set_handlers()
    what_table = cli_what_handlers()

    # Default disabled
    assert "disabled" in what_table['self_edit']([], persona)[0]

    # Turn on via CLI setter
    msg, mutated = set_table['self_edit'](['self_edit', 'on'], persona)
    assert mutated
    assert "enabled" in msg
    assert persona.get_self_edit() is True
    assert "enabled" in what_table['self_edit']([], persona)[0]

    # Turn off via CLI setter
    msg, mutated = set_table['self_edit'](['self_edit', 'off'], persona)
    assert mutated
    assert "disabled" in msg
    assert persona.get_self_edit() is False

    # PATCH apply
    rejected = []
    apply_patch_fields(persona, {"self_edit": True}, rejected)
    assert not rejected
    assert persona.get_self_edit() is True

