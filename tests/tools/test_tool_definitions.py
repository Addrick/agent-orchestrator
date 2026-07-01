# tests/tools/test_tool_definitions.py

"""
Phase 1 of tool_security_framework: every tool must declare static
`capabilities` flags (`produces_untrusted`, `irreversible`, optional
`irreversible_if`). The validator enforces this at registration time.
"""

import pytest

import src.tools.definitions as definitions
from src.tools.definitions import (
    ALL_TOOL_DEFINITIONS,
    ToolDefinitionRegistry,
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
        "capabilities": {
            "locality": "local",
            "sensitivity": "public",
            "produces_untrusted": False
        },
    }
    with pytest.raises(ValueError, match="missing required flag 'irreversible'"):
        validate_tool_capabilities(bad)


def test_validator_rejects_non_bool_flag():
    bad = {
        "type": "function",
        "function": {"name": "x"},
        "capabilities": {
            "locality": "local",
            "sensitivity": "public",
            "produces_untrusted": "yes",
            "irreversible": False
        },
    }
    with pytest.raises(ValueError, match="must be bool"):
        validate_tool_capabilities(bad)


def test_validator_rejects_bad_irreversible_if_format():
    bad = {
        "type": "function",
        "function": {"name": "x"},
        "capabilities": {
            "locality": "local",
            "sensitivity": "public",
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
            "locality": "local",
            "sensitivity": "public",
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
            "locality": "local",
            "sensitivity": "public",
            "produces_untrusted": False,
            "irreversible": False,
            "irreversible_if": "src.tools.classifiers:no_such_function",
        },
    }
    with pytest.raises(ValueError, match="did not resolve to a callable"):
        validate_tool_capabilities(bad)


# --- ToolDefinitionRegistry (DP-268: dynamic registration for MCP tools) ---

def _make_tool(name, *, is_write=False, caps=None):
    tool = {
        "type": "function",
        "function": {"name": name, "description": "t", "parameters": {}},
        "capabilities": caps or {
            "produces_untrusted": True,
            "irreversible": True,
            "locality": "network",
            "sensitivity": "pii",
        },
    }
    if is_write:
        tool["is_write"] = True
    return tool


def test_registry_seeds_static_definitions():
    reg = ToolDefinitionRegistry(ALL_TOOL_DEFINITIONS)
    assert reg.all_definitions() == ALL_TOOL_DEFINITIONS
    assert reg.get("web_search") is not None
    assert reg.is_write("update_ticket") is True
    assert reg.is_write("web_search") is False


def test_registry_register_reflects_in_all_accessors():
    reg = ToolDefinitionRegistry([])
    tool = _make_tool("mcp__srv__do_thing", is_write=True)
    reg.register(tool)
    assert tool in reg.all_definitions()
    assert reg.get("mcp__srv__do_thing") is tool
    assert reg.is_write("mcp__srv__do_thing") is True
    assert reg.capabilities("mcp__srv__do_thing")["produces_untrusted"] is True


def test_registry_rejects_duplicate_name():
    reg = ToolDefinitionRegistry([])
    reg.register(_make_tool("mcp__srv__x"))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_make_tool("mcp__srv__x"))
    assert len(reg.all_definitions()) == 1


def test_registry_rejects_invalid_capabilities_without_registering():
    reg = ToolDefinitionRegistry([])
    bad = {"type": "function", "function": {"name": "x"}}
    with pytest.raises(ValueError, match="missing 'capabilities'"):
        reg.register(bad)
    assert reg.all_definitions() == []
    assert reg.get("x") is None


def test_registry_lists_but_does_not_index_non_function_entries():
    reg = ToolDefinitionRegistry([])
    grounding = {
        "type": "google_grounding",
        "capabilities": {
            "produces_untrusted": True,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "public",
        },
    }
    reg.register(grounding)
    assert grounding in reg.all_definitions()
    assert reg.get("") is None


def test_module_accessors_route_through_registry(monkeypatch):
    fresh = ToolDefinitionRegistry(ALL_TOOL_DEFINITIONS)
    monkeypatch.setattr(definitions, "_REGISTRY", fresh)

    tool = _make_tool("mcp__srv__dyn", is_write=True)
    definitions.register_tool_definition(tool)

    assert tool in definitions.get_all_tool_definitions()
    assert definitions.get_tool_definition("mcp__srv__dyn") is tool
    assert definitions.is_write_tool("mcp__srv__dyn") is True
    assert definitions.get_tool_capabilities("mcp__srv__dyn")["sensitivity"] == "pii"
    # Static seed is NOT mutated by dynamic registration
    assert tool not in ALL_TOOL_DEFINITIONS


def test_add_note_classifier_internal_is_reversible():
    from src.tools.classifiers import add_note_irreversible_check
    assert add_note_irreversible_check({"internal": True}) is False


def test_add_note_classifier_customer_visible_is_irreversible():
    from src.tools.classifiers import add_note_irreversible_check
    assert add_note_irreversible_check({"internal": False}) is True
    assert add_note_irreversible_check({}) is True  # default visible
