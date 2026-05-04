# tests/tools/test_policy.py

import pytest
from src.tools.policy import ToolPolicy
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
    assert persona.get_enabled_tools() == ["web_search", "get_ticket_details"]

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
