# tests/test_tool_policy.py

import pytest
from src.tool_policy import ToolPolicy
from src.tools.definitions import ALL_TOOL_DEFINITIONS
from src.persona import Persona

def test_policy_filtering():
    # Only allow web_search
    policy = ToolPolicy(allow=["web_search"])
    filtered = policy.filter_tools(ALL_TOOL_DEFINITIONS)
    
    assert len(filtered) == 1
    assert filtered[0]["function"]["name"] == "web_search"

def test_policy_wildcard():
    # Allow all
    policy = ToolPolicy(default="allow", allow=["*"])
    filtered = policy.filter_tools(ALL_TOOL_DEFINITIONS)
    
    assert len(filtered) == len(ALL_TOOL_DEFINITIONS)

def test_composition_rule_network_read_local_write():
    policy = ToolPolicy()
    # web_search is network:read, update_core_memory is local:write
    tools = [
        next(t for t in ALL_TOOL_DEFINITIONS if t["function"]["name"] == "web_search"),
        next(t for t in ALL_TOOL_DEFINITIONS if t["function"]["name"] == "update_core_memory"),
    ]
    
    errors = policy.validate_composition(tools)
    assert any("network:read + local:write" in e for e in errors)

def test_composition_rule_untrusted_read_network_write():
    policy = ToolPolicy()
    # web_search is untrusted:read, add_mcp_server is network:write
    tools = [
        next(t for t in ALL_TOOL_DEFINITIONS if t["function"]["name"] == "web_search"),
        next(t for t in ALL_TOOL_DEFINITIONS if t["function"]["name"] == "add_mcp_server"),
    ]
    
    errors = policy.validate_composition(tools)
    assert any("untrusted:read + network:write" in e for e in errors)

def test_composition_rule_pii_read_network_any():
    policy = ToolPolicy()
    # search_user is pii:read, web_search is network:any
    tools = [
        next(t for t in ALL_TOOL_DEFINITIONS if t["function"]["name"] == "search_user"),
        next(t for t in ALL_TOOL_DEFINITIONS if t["function"]["name"] == "web_search"),
    ]
    
    errors = policy.validate_composition(tools)
    assert any("pii:read + network:*" in e for e in errors)

def _tools_by_name(*names):
    return [next(t for t in ALL_TOOL_DEFINITIONS if t["function"]["name"] == n) for n in names]


def test_egress_domain_mapping():
    by_name = {t["function"]["name"]: t for t in ALL_TOOL_DEFINITIONS}
    # network + service_binding -> the binding
    assert ToolPolicy._egress_domain(by_name["update_ticket"]) == "zammad"
    # network + no binding -> open
    assert ToolPolicy._egress_domain(by_name["web_search"]) == "open"
    # local -> local
    assert ToolPolicy._egress_domain(by_name["update_core_memory"]) == "local"


def test_same_origin_zammad_read_write_is_clean():
    # joy's scope: untrusted read + network write + pii read, all within zammad.
    # Same-origin -> no violation.
    policy = ToolPolicy()
    tools = _tools_by_name(
        "get_ticket_details", "search_tickets", "update_ticket",
        "add_note_to_ticket", "create_ticket", "search_user",
    )
    assert policy.validate_composition(tools) == []


def test_same_origin_breaks_when_foreign_egress_added():
    # The instant a foreign-domain (open) tool joins the zammad scope, the
    # trifecta rules re-arm: untrusted zammad read -> open write would exfiltrate,
    # and pii read + open egress would leak PII.
    policy = ToolPolicy()
    tools = _tools_by_name("get_ticket_details", "search_user", "web_search")
    errors = policy.validate_composition(tools)
    # pii (zammad) + web_search (open) egress
    assert any("pii:read + network:*" in e and "open" in e for e in errors)


def test_local_untrusted_read_plus_network_write_is_foreign():
    # recall_memory is a local untrusted read (domain "local"); a poisoned
    # memory hit driving an MCP write is cross-domain (local -> mcp) and
    # must trip Rule 2 — provenance is domain-tagged even for local reads.
    policy = ToolPolicy()
    tools = _tools_by_name("recall_memory", "add_mcp_server")
    errors = policy.validate_composition(tools)
    assert any("untrusted:read + network:write" in e and "mcp" in e for e in errors)


def test_zammad_writes_opted_out_of_exfil_accounting():
    # All zammad tools are exfil_capable=False (internal-only instance, no
    # customer-facing output): a foreign untrusted read + a zammad write must
    # NOT trip Rule 2, because the zammad egress carries nothing outside
    # trusted infra.
    policy = ToolPolicy()
    tools = _tools_by_name("web_search", "update_ticket", "add_note_to_ticket")
    assert policy.validate_composition(tools) == []


def test_triage_websearch_only_is_clean():
    # triage: web_search alone, no pii/write tools -> no trifecta.
    policy = ToolPolicy()
    assert policy.validate_composition(_tools_by_name("web_search")) == []


def test_wildcard_full_toolset_still_denied():
    # Same-origin does NOT rescue ["*"] — a wildcard spans open + local + zammad
    # and genuinely has the trifecta. It must still fail to load.
    policy = ToolPolicy(default="allow", allow=["*"])
    errors = policy.validate_composition(ALL_TOOL_DEFINITIONS)
    assert errors, "Wildcard over the full toolset must trip composition rules"


def test_explicit_override():
    policy = ToolPolicy(explicit_overrides=["network_read_local_write"])
    tools = [
        next(t for t in ALL_TOOL_DEFINITIONS if t["function"]["name"] == "web_search"),
        next(t for t in ALL_TOOL_DEFINITIONS if t["function"]["name"] == "update_core_memory"),
    ]
    
    errors = policy.validate_composition(tools)
    assert not any("network:read + local:write" in e for e in errors)

def test_persona_legacy_migration():
    persona = Persona(
        persona_name="test",
        model_name="test",
        prompt="test",
        enabled_tools=["web_search", "get_ticket_details"]
    )
    
    assert persona.get_tool_policy().allow == ["web_search", "get_ticket_details"]
    assert persona.get_enabled_tools() == ["get_ticket_details", "web_search"]

def test_persona_structured_policy():
    policy_dict = {
        "allow": ["web_search"],
        "ask": ["update_ticket"]
    }
    persona = Persona(
        persona_name="test",
        model_name="test",
        prompt="test",
        tool_policy=policy_dict
    )
    
    assert persona.get_tool_policy().allow == ["web_search"]
    assert persona.get_tool_policy().ask == ["update_ticket"]
    # get_enabled_tools should return both for the engine to consider them
    assert set(persona.get_enabled_tools()) == {"web_search", "update_ticket"}


# --- DP-268: wildcard never auto-includes dynamic (MCP) tools ------------------

def _dynamic_tool(name, *, is_write=True):
    return {
        "type": "function",
        "dynamic": True,
        "is_write": is_write,
        "service_binding": "mcp:srv",
        "capabilities": {
            "produces_untrusted": True,
            "irreversible": True,
            "locality": "network",
            "sensitivity": "pii",
        },
        "function": {"name": name, "description": "d", "parameters": {}},
    }


def test_wildcard_excludes_dynamic_tools():
    policy = ToolPolicy(default="allow", allow=["*"])
    catalog = ALL_TOOL_DEFINITIONS + [_dynamic_tool("mcp__srv__hack")]
    filtered = policy.filter_tools(catalog)
    names = {t.get("function", {}).get("name") for t in filtered}
    assert "mcp__srv__hack" not in names
    assert len(filtered) == len(ALL_TOOL_DEFINITIONS)


def test_wildcard_includes_explicitly_allowed_dynamic_tool():
    policy = ToolPolicy(default="allow", allow=["*", "mcp__srv__ok"])
    catalog = ALL_TOOL_DEFINITIONS + [
        _dynamic_tool("mcp__srv__ok"), _dynamic_tool("mcp__srv__other"),
    ]
    names = {t.get("function", {}).get("name") for t in policy.filter_tools(catalog)}
    assert "mcp__srv__ok" in names
    assert "mcp__srv__other" not in names


def test_wildcard_includes_ask_listed_dynamic_tool():
    policy = ToolPolicy(default="allow", allow=["*"], ask=["mcp__srv__ok"])
    catalog = ALL_TOOL_DEFINITIONS + [_dynamic_tool("mcp__srv__ok")]
    names = {t.get("function", {}).get("name") for t in policy.filter_tools(catalog)}
    assert "mcp__srv__ok" in names


def test_explicit_allow_of_dynamic_tool_without_wildcard():
    policy = ToolPolicy(allow=["mcp__srv__ok"])
    catalog = ALL_TOOL_DEFINITIONS + [_dynamic_tool("mcp__srv__ok")]
    filtered = policy.filter_tools(catalog)
    assert len(filtered) == 1
    assert filtered[0]["function"]["name"] == "mcp__srv__ok"


def test_resolve_policy_tools_wildcard_excludes_dynamic(monkeypatch):
    """composition.resolve_policy_tools must expand '*' identically to
    filter_tools: a runtime-registered MCP tool never enters wildcard
    composition validation → no quarantine cascade on server install."""
    from src.tools import composition, definitions
    from src.tools.definitions import ToolDefinitionRegistry

    fresh = ToolDefinitionRegistry(ALL_TOOL_DEFINITIONS)
    monkeypatch.setattr(definitions, "_REGISTRY", fresh)
    definitions.register_tool_definition(_dynamic_tool("mcp__srv__evil"))

    resolved = composition.resolve_policy_tools(ToolPolicy(default="allow", allow=["*"]))
    names = {t.get("function", {}).get("name") for t in resolved}
    assert "mcp__srv__evil" not in names

    # Explicit listing brings it into composition validation.
    resolved = composition.resolve_policy_tools(
        ToolPolicy(default="allow", allow=["*", "mcp__srv__evil"])
    )
    names = {t.get("function", {}).get("name") for t in resolved}
    assert "mcp__srv__evil" in names


# --- DP-263: exfil_capable opt-out --------------------------------------------

def test_set_active_model_exempt_from_untrusted_read_exfil_rule():
    """set_active_model (exfil_capable=False) must NOT trip Rule 2 even beside an
    untrusted:read tool — its SSH carries no model-controlled payload."""
    policy = ToolPolicy()
    tools = _tools_by_name("web_search", "set_active_model")
    assert policy.validate_composition(tools) == []


def test_set_active_model_exempt_from_pii_read_exfil_rule():
    """Rule 3 (pii:read + network egress) also must not fire on set_active_model."""
    policy = ToolPolicy()
    tools = _tools_by_name("search_user", "set_active_model")
    assert policy.validate_composition(tools) == []


def test_other_proxmox_writes_still_trip_exfil_rule():
    """The exemption is scoped: a normal proxmox network:write (reboot_guest,
    exfil_capable defaults True) beside an untrusted read STILL trips Rule 2."""
    policy = ToolPolicy()
    tools = _tools_by_name("web_search", "reboot_guest")
    errors = policy.validate_composition(tools)
    assert any("untrusted:read + network:write" in e for e in errors)


def test_exfil_capable_capability_validation():
    from src.tools.definitions import validate_tool_capabilities
    good = {
        "function": {"name": "x"},
        "capabilities": {
            "produces_untrusted": False, "irreversible": False,
            "locality": "network", "sensitivity": "internal",
            "exfil_capable": False,
        },
    }
    validate_tool_capabilities(good)  # no raise
    bad = {
        "function": {"name": "x"},
        "capabilities": {
            "produces_untrusted": False, "irreversible": False,
            "locality": "network", "sensitivity": "internal",
            "exfil_capable": "no",
        },
    }
    with pytest.raises(ValueError, match="exfil_capable"):
        validate_tool_capabilities(bad)
