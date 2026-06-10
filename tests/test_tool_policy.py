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
    # web_search is untrusted:read, update_ticket is network:write
    tools = [
        next(t for t in ALL_TOOL_DEFINITIONS if t["function"]["name"] == "web_search"),
        next(t for t in ALL_TOOL_DEFINITIONS if t["function"]["name"] == "update_ticket"),
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
    # memory hit driving a zammad write is cross-domain (local -> zammad) and
    # must trip Rule 2 — provenance is domain-tagged even for local reads.
    policy = ToolPolicy()
    tools = _tools_by_name("recall_memory", "update_ticket")
    errors = policy.validate_composition(tools)
    assert any("untrusted:read + network:write" in e and "zammad" in e for e in errors)


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
