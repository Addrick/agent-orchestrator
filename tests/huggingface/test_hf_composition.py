"""The composition invariant DP-265 has to keep true.

Putting untrusted HuggingFace reads on the persona that already holds node power
is the whole risk of this ticket. ``ToolPolicy`` Rule 2 is the thing that would
notice; these tests pin that it stays **armed** and still validates clean, and
that it would fire if the honest tagging were undone.

The failure mode being guarded is quiet: a later tool addition re-trips Rule 2,
`hypr` is quarantined at load, and the only symptom is a persona that stopped
answering.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.tool_policy import ToolPolicy
from src.tools.definitions import ALL_TOOL_DEFINITIONS

_HYPR = Path(__file__).resolve().parents[2] / "config" / "optional_personas" / "hypr.json"


def _hypr_persona():
    return json.loads(_HYPR.read_text(encoding="utf-8"))["personas"][0]


def _hypr_tools(definitions=None):
    """The exact toolset hypr is exposed, as `RequestBuilder` computes it."""
    persona = _hypr_persona()
    policy = ToolPolicy(**persona["tool_policy"])
    bindings = set(persona["service_bindings"])
    available = [
        t for t in (definitions or ALL_TOOL_DEFINITIONS)
        if not t.get("service_binding") or t.get("service_binding") in bindings
    ]
    return policy, policy.filter_tools(available)


def test_hypr_exposes_the_four_new_tools():
    persona = _hypr_persona()
    for name in ("hf_search", "hf_files", "install_model", "install_status"):
        assert name in persona["enabled_tools"]
        assert name in persona["tool_policy"]["allow"]
    assert "huggingface" in persona["service_bindings"]
    _, tools = _hypr_tools()
    assert {t["function"]["name"] for t in tools} >= {
        "hf_search", "hf_files", "install_model", "install_status",
    }


def test_hypr_composition_validates_clean_with_no_overrides():
    """The success criterion this ticket was written around: same-origin closed
    loop, achieved by honest tagging rather than by disarming the rule."""
    policy, tools = _hypr_tools()
    assert policy.explicit_overrides == []
    assert policy.validate_composition(tools) == []


def test_the_shipped_persona_carries_no_explicit_overrides():
    """`explicit_overrides` is the kill switch for the composition invariants.
    It is not settable from a generic policy dict (DP-277); this asserts nobody
    added it to the shipped file either."""
    persona = _hypr_persona()
    assert "explicit_overrides" not in persona.get("tool_policy", {})
    assert "explicit_overrides" not in persona


def test_rule_2_is_still_armed_for_anything_added_later():
    """A foreign-domain write added to hypr must still be caught. If this test
    ever passes with an empty error list, the protection is off and the clean
    result above means nothing."""
    policy, tools = _hypr_tools()
    intruder = {
        "type": "function",
        "is_write": True,
        "capabilities": {
            "produces_untrusted": False,
            "irreversible": False,
            "locality": "network",
            "sensitivity": "internal",
        },
        "function": {"name": "post_anywhere", "parameters": {}},
    }
    errors = policy.validate_composition(list(tools) + [intruder])
    assert errors
    assert "untrusted:read + network:write" in errors[0]


def test_undoing_the_exfil_flag_on_a_proxmox_write_re_trips_rule_2():
    """The counterpart edit's justification, made checkable. The guest/node
    write tools are `exfil_capable: False` because their arguments are values
    discovered from the node — not because it was convenient. If a future tool
    genuinely does carry a model-authored string out, it must leave the default
    True, and this shows what that costs: the composition has to be re-reasoned,
    not overridden."""
    definitions = copy.deepcopy(ALL_TOOL_DEFINITIONS)
    for tool in definitions:
        if tool.get("function", {}).get("name") == "stop_guest":
            tool["capabilities"].pop("exfil_capable", None)
    policy, tools = _hypr_tools(definitions)
    errors = policy.validate_composition(tools)
    assert errors
    assert "proxmox" in errors[0]


@pytest.mark.parametrize("name,expected", [
    ("hf_search", "huggingface"),
    ("hf_files", "huggingface"),
    ("install_model", "huggingface"),
    # Its egress really is the node, so it is tagged for the node — the
    # `capabilities["domain"]` seam used for accuracy, not to dodge a rule.
    ("install_status", "proxmox"),
])
def test_egress_domains_are_tagged_where_the_data_actually_goes(name, expected):
    tool = next(t for t in ALL_TOOL_DEFINITIONS if t.get("function", {}).get("name") == name)
    assert ToolPolicy._egress_domain(tool) == expected


def test_untrusted_reads_are_declared_on_the_hub_tools_only():
    by_name = {
        t["function"]["name"]: t for t in ALL_TOOL_DEFINITIONS
        if t.get("service_binding") == "huggingface"
    }
    assert by_name["hf_search"]["capabilities"]["produces_untrusted"] is True
    assert by_name["hf_files"]["capabilities"]["produces_untrusted"] is True
    # The node emits a fixed vocabulary and `_clean_status` whitelists it.
    assert by_name["install_status"]["capabilities"]["produces_untrusted"] is False
    assert by_name["install_model"]["capabilities"]["produces_untrusted"] is False


def test_install_model_is_a_gated_write():
    tool = next(
        t for t in ALL_TOOL_DEFINITIONS
        if t.get("function", {}).get("name") == "install_model"
    )
    assert tool["is_write"] is True
