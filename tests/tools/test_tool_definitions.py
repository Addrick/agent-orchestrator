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
from src.tools.mcp_bridge import McpBridge

# Shared parametrize ids. One definition so changing the id scheme (or the
# source list) is one edit, not one per test.
_TOOL_IDS = [
    t.get("function", {}).get("name", "<unknown>") for t in ALL_TOOL_DEFINITIONS
]


@pytest.mark.parametrize("tool", ALL_TOOL_DEFINITIONS, ids=_TOOL_IDS)
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
    # `is_write` is always set, never conditionally: DP-306 made an omitted flag
    # a registration error, because omitting it silently exempts a tool from the
    # write audit and from MCP-bridge gating.
    tool = {
        "type": "function",
        "is_write": is_write,
        "function": {"name": name, "description": "t", "parameters": {}},
        "capabilities": caps or {
            "produces_untrusted": True,
            "irreversible": True,
            "locality": "network",
            "sensitivity": "pii",
        },
    }
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


def test_registry_unregister_removes_dynamic_tool():
    reg = ToolDefinitionRegistry([])
    tool = _make_tool("mcp__srv__x", is_write=True)
    tool["dynamic"] = True
    reg.register(tool)
    assert reg.unregister("mcp__srv__x") is True
    assert reg.get("mcp__srv__x") is None
    assert reg.is_write("mcp__srv__x") is False
    assert tool not in reg.all_definitions()


def test_registry_unregister_refuses_static_tool():
    reg = ToolDefinitionRegistry(ALL_TOOL_DEFINITIONS)
    with pytest.raises(ValueError, match="static"):
        reg.unregister("web_search")
    assert reg.get("web_search") is not None


def test_registry_unregister_unknown_returns_false():
    reg = ToolDefinitionRegistry([])
    assert reg.unregister("nope") is False


def test_module_unregister_accessor_routes_through_registry(monkeypatch):
    fresh = ToolDefinitionRegistry([])
    monkeypatch.setattr(definitions, "_REGISTRY", fresh)
    tool = _make_tool("mcp__srv__dyn")
    tool["dynamic"] = True
    definitions.register_tool_definition(tool)
    assert definitions.unregister_tool_definition("mcp__srv__dyn") is True
    assert definitions.get_tool_definition("mcp__srv__dyn") is None


def test_add_note_def_is_internal_only_and_not_exfil_capable():
    # The internal-only clamp and the exfil_capable=False classification are
    # one decision: the def must not re-expose a customer-visible knob while
    # claiming the tool can't exfiltrate.
    tool = next(t for t in ALL_TOOL_DEFINITIONS
                if t["function"]["name"] == "add_note_to_ticket")
    assert tool["capabilities"]["exfil_capable"] is False
    assert "irreversible_if" not in tool["capabilities"]
    assert "internal" not in tool["function"]["parameters"]["properties"]


# --- DP-306: approval classification must be unskippable --------------------
#
# Three predicates answer "does this call need human approval?":
#   1. definitions.is_write_tool()        -> tool_loop's universal write audit
#   2. mcp_bridge._is_gated()             -> is_write OR capabilities.irreversible
#   3. definitions.is_irreversible()      -> capabilities + argument-aware classifier
# Two of the three read top-level `is_write`. Nothing used to require it, so a
# definition could omit it, register cleanly, and be silently ungated. These
# tests pin the flag as mandatory and pin the predicates to each other.


def test_validator_rejects_missing_is_write():
    tool = _make_tool("no_write_flag")
    del tool["is_write"]
    with pytest.raises(ValueError, match="missing top-level 'is_write'"):
        validate_tool_capabilities(tool)


def test_validator_rejects_non_bool_is_write():
    tool = _make_tool("stringy_write_flag")
    tool["is_write"] = "true"  # truthy string would have gated by accident
    with pytest.raises(ValueError, match="'is_write' must be bool"):
        validate_tool_capabilities(tool)


def test_registry_refuses_to_register_tool_without_is_write():
    """The gate is at registration, not merely at import of the static seed —
    runtime-registered (MCP) tools go through the same door."""
    reg = ToolDefinitionRegistry(ALL_TOOL_DEFINITIONS)
    tool = _make_tool("mcp__rogue__unflagged")
    del tool["is_write"]
    with pytest.raises(ValueError, match="missing top-level 'is_write'"):
        reg.register(tool)
    assert reg.get("mcp__rogue__unflagged") is None
    assert "mcp__rogue__unflagged" not in reg._write_tools


def test_validator_does_not_exempt_a_named_tool_with_a_missing_type():
    """The exemption must key on `function.name`, not on `type`.

    The registry indexes and write-flags a definition `if name:` -- it never
    looks at `type`. So a definition with a name but a missing or misspelled
    `type` is fully live (answers `is_write_tool`, `_is_gated`, listed to
    subagents) and must not skip the check. Keying the exemption on
    `type != "function"` made exactly that shape a bypass.
    """
    tool = _make_tool("wipe_everything")
    del tool["type"]
    del tool["is_write"]
    with pytest.raises(ValueError, match="missing top-level 'is_write'"):
        validate_tool_capabilities(tool)


@pytest.mark.parametrize("tool", ALL_TOOL_DEFINITIONS, ids=_TOOL_IDS)
def test_every_tool_declares_is_write(tool):
    """Every entry with a callable name -- including non-`function` types like
    `google_grounding`, which the registry still indexes -- declares the flag."""
    if not tool.get("function", {}).get("name"):
        pytest.skip("unnamed entry: the registry cannot index or gate it")
    assert "is_write" in tool, (
        f"{tool.get('function', {}).get('name')} does not declare is_write"
    )
    assert isinstance(tool["is_write"], bool)


# Verb heuristic: an INDEPENDENT read of whether a tool mutates something,
# derived from the name rather than from the flag under test. Comparing
# `is_write` against itself (or against anything computed from it) can only
# catch an OMITTED flag -- and every definition already carried one. The
# likelier and far more dangerous failure is a WRONG flag: a mutating tool
# declared `is_write: False` drops out of the tool-loop write audit AND out of
# `_is_gated`, so an autonomous agent executes it with no human approval.
_MUTATING_VERBS = {
    "add", "answer", "approve", "cancel", "create", "delete", "deny",
    "dispatch", "ingest", "kill", "manage", "merge", "prune", "reboot",
    "remove", "retire", "send", "set", "start", "stop", "update",
}

# Tools whose name reads as mutating but that are deliberately NOT writes.
# Every entry is a reviewed decision, not a default -- adding a name here is
# how you overrule the heuristic, and it is meant to be conspicuous in review.
_REVIEWED_NON_WRITES = {
    # fixr subagent control-plane: acts on derpr's own in-process agents, not
    # on any external system or persisted record.
    "kill_agent",
    "prune_agents",
    "answer_agent",
    # voice timers: process-local, self-expiring, trivially undone.
    "set_timer",
    "cancel_timer",
    # notification egress, no state mutated at the destination.
    "send_discord",
}


@pytest.mark.parametrize("tool", ALL_TOOL_DEFINITIONS, ids=_TOOL_IDS)
def test_mutating_tool_names_are_declared_writes(tool):
    """Cross-check `is_write` against a source that is not `is_write`.

    A tool named with a mutating verb must be `is_write: True` unless it is in
    `_REVIEWED_NON_WRITES`. Flipping e.g. `update_user` to False fails here.
    """
    name = tool.get("function", {}).get("name")
    if not name or name in _REVIEWED_NON_WRITES:
        return
    if name.split("_", 1)[0] not in _MUTATING_VERBS:
        return
    assert tool.get("is_write") is True, (
        f"'{name}' reads as a mutating tool but declares is_write="
        f"{tool.get('is_write')!r}. If that is correct, add it to "
        f"_REVIEWED_NON_WRITES with the reason; otherwise fix the flag."
    )


def test_reviewed_non_writes_has_not_rotted():
    """The exception list is the heuristic's only escape hatch, so it must not
    silently accumulate names that no longer exist or that are now writes."""
    by_name = {t.get("function", {}).get("name"): t for t in ALL_TOOL_DEFINITIONS}
    for name in _REVIEWED_NON_WRITES:
        assert name in by_name, f"_REVIEWED_NON_WRITES names '{name}', not a real tool"
        assert by_name[name].get("is_write") is False, (
            f"'{name}' is listed as a reviewed non-write but declares is_write=True"
        )


@pytest.mark.parametrize("tool", ALL_TOOL_DEFINITIONS, ids=_TOOL_IDS)
def test_write_audit_and_bridge_gate_agree(tool):
    """Predicate 1 vs predicate 2, calling BOTH real implementations.

    Re-deriving the bridge's expression here instead of importing it would make
    this unfailable: gutting `_is_gated` to `return False` -- removing the
    autonomous-agent approval gate outright -- left the old version green.
    """
    name = tool.get("function", {}).get("name")
    if not name:
        pytest.skip("unnamed entry: not dispatchable")
    if definitions.is_write_tool(name):
        assert McpBridge._is_gated(tool), (
            f"{name} is write-audited but ungated at the MCP bridge"
        )


@pytest.mark.parametrize("tool", ALL_TOOL_DEFINITIONS, ids=_TOOL_IDS)
def test_statically_irreversible_tools_are_gated(tool):
    """Predicate 3 vs predicate 2: a tool whose capabilities call it
    irreversible must be gated, with no argument needed to get there."""
    name = tool.get("function", {}).get("name")
    if not name:
        pytest.skip("unnamed entry: not dispatchable")
    if definitions.is_irreversible(name, {}):
        assert McpBridge._is_gated(tool), (
            f"{name} is irreversible but ungated at the MCP bridge"
        )


@pytest.mark.parametrize("tool", ALL_TOOL_DEFINITIONS, ids=_TOOL_IDS)
def test_gated_tools_are_announced_to_subagents(tool):
    """Predicate 5: `_to_mcp_tool`'s description annotation. A gated tool with
    no warning in its description is the mid-task surprise the note exists to
    prevent, so the annotation must not be narrower than the gate itself."""
    name = tool.get("function", {}).get("name")
    if not name or not McpBridge._is_gated(tool):
        pytest.skip("not gated")
    assert "requires human approval" in McpBridge._to_mcp_tool(tool).description


def test_always_confirm_tools_exist_and_are_writes():
    """`ALWAYS_CONFIRM_TOOLS` is a name-keyed list consumed by tool_loop.

    NOTE (DP-315): despite its name it does NOT gate anything today. Its only
    runtime effect is `tool_loop.py:378`, which appends a "HIGH-IMPACT" badge to
    the confirmation text -- every write already parks unconditionally. So this
    test keeps a *label* honest, not a predicate. A typo still matters: it would
    silently drop the badge on the tool that most needs it.
    """
    assert definitions.ALWAYS_CONFIRM_TOOLS, (
        "ALWAYS_CONFIRM_TOOLS is empty -- every assertion below passes vacuously"
    )
    names = {t["function"]["name"] for t in ALL_TOOL_DEFINITIONS
             if t.get("function", {}).get("name")}
    for tool_name in definitions.ALWAYS_CONFIRM_TOOLS:
        assert tool_name in names, (
            f"ALWAYS_CONFIRM_TOOLS names '{tool_name}', which is not a real tool"
        )
        assert definitions.is_write_tool(tool_name), (
            f"ALWAYS_CONFIRM_TOOLS names '{tool_name}' but it is not a write tool"
        )


# NOTE: the inverse direction -- "every statically irreversible tool is in
# ALWAYS_CONFIRM_TOOLS" -- is NOT asserted. It does not hold (`create_ticket`
# and `reboot_node` are irreversible and unlisted), but the omission is harmless
# today because the set only drives a badge: every write parks regardless. It
# becomes a real gap only if an autonomy tier is ever built. See DP-315.
