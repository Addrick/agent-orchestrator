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
