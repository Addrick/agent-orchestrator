# tests/tools/test_tool_definitions.py

"""
Phase 1 of tool_security_framework: every tool must declare static
`capabilities` flags (`produces_untrusted`, `irreversible`, optional
`irreversible_if`). The validator enforces this at registration time.
"""

import pytest

from src.tools.definitions import (
    ALL_TOOL_DEFINITIONS,
    validate_tool_capabilities,
)


@pytest.mark.parametrize(
    "tool",
    ALL_TOOL_DEFINITIONS,
    ids=[t.get("function", {}).get("name", "<unknown>") for t in ALL_TOOL_DEFINITIONS],
)
def test_every_tool_has_valid_capabilities(tool):
    validate_tool_capabilities(tool)
    caps = tool["capabilities"]
    assert isinstance(caps["produces_untrusted"], bool)
    assert isinstance(caps["irreversible"], bool)


def test_validator_rejects_missing_capabilities():
    bad = {"type": "function", "function": {"name": "x"}}
    with pytest.raises(ValueError, match="missing 'capabilities'"):
        validate_tool_capabilities(bad)


def test_validator_rejects_missing_required_flag():
    bad = {
        "type": "function",
        "function": {"name": "x"},
        "capabilities": {"produces_untrusted": False},
    }
    with pytest.raises(ValueError, match="missing required flag 'irreversible'"):
        validate_tool_capabilities(bad)


def test_validator_rejects_non_bool_flag():
    bad = {
        "type": "function",
        "function": {"name": "x"},
        "capabilities": {"produces_untrusted": "yes", "irreversible": False},
    }
    with pytest.raises(ValueError, match="must be bool"):
        validate_tool_capabilities(bad)


def test_validator_rejects_bad_irreversible_if_format():
    bad = {
        "type": "function",
        "function": {"name": "x"},
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "irreversible_if": "no_colon_path",
        },
    }
    with pytest.raises(ValueError, match="dotted path"):
        validate_tool_capabilities(bad)


def test_validator_rejects_unresolvable_irreversible_if_module():
    bad = {
        "type": "function",
        "function": {"name": "x"},
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "irreversible_if": "src.tools.does_not_exist:fn",
        },
    }
    with pytest.raises(ValueError, match="could not be imported"):
        validate_tool_capabilities(bad)


def test_validator_rejects_unresolvable_irreversible_if_function():
    bad = {
        "type": "function",
        "function": {"name": "x"},
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "irreversible_if": "src.tools.classifiers:no_such_function",
        },
    }
    with pytest.raises(ValueError, match="did not resolve to a callable"):
        validate_tool_capabilities(bad)


def test_add_note_classifier_internal_is_reversible():
    from src.tools.classifiers import add_note_irreversible_check
    assert add_note_irreversible_check({"internal": True}) is False


def test_add_note_classifier_customer_visible_is_irreversible():
    from src.tools.classifiers import add_note_irreversible_check
    assert add_note_irreversible_check({"internal": False}) is True
    assert add_note_irreversible_check({}) is True  # default visible
