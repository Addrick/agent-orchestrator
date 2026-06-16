# tests/test_fixr_persona.py
"""DP-227: the fixr supervisor persona loads from default_personas.json with the
right model, bindings and tools, and is not quarantined. Also covers the
config-absent case (old configs without fixr load fine)."""

import json
import os

from config import global_config
from src.personas.store import load_personas_from_file


def _default_config_path():
    return os.path.join(global_config.CONFIG_DIR, "default_personas.json")


def test_fixr_persona_loads_clean():
    personas = load_personas_from_file(file_path_override=_default_config_path())
    assert personas is not None
    assert "fixr" in personas
    fixr = personas["fixr"]
    # gemini-2.5-flash: an API tool-calling model that's actually keyed AND
    # quota'd on prod. (gemini-3.1-flash-lite resolved to the zero-quota
    # *-preview tier → 429; claude-opus-4-8 was wrong too — this deployment has
    # no Anthropic API key, and cc-* can't host the supervisor since it bypasses
    # derpr's tool loop. Live-smoked PONG on prod :5003 2026-06-16.)
    assert fixr.get_model_name() == "gemini-2.5-flash"
    assert "fixr" in fixr.get_service_bindings()
    # Not quarantined — its tool policy composes cleanly.
    assert not fixr.get_security_block_reasons()
    tools = set(fixr.get_enabled_tools())
    assert {"dispatch_fix", "inspect_agents", "answer_agent",
            "kill_agent", "send_discord"} <= tools


def test_config_without_fixr_still_loads(tmp_path):
    """An old config that predates the fixr persona must load without error."""
    cfg = {"personas": [{
        "name": "legacy", "prompt": "hi", "model_name": "gemini-2.5-flash",
        "enabled_tools": [], "memory_mode": "CHANNEL_ISOLATED",
        "history_messages": 5,
    }]}
    p = tmp_path / "personas.json"
    p.write_text(json.dumps(cfg))
    personas = load_personas_from_file(file_path_override=str(p))
    assert personas is not None
    assert "legacy" in personas
    assert "fixr" not in personas


def test_dispatch_fix_is_the_only_parked_fixr_write():
    """Autonomy contract: only dispatch_fix parks (is_write). The coordination +
    reporting tools run ungated so the woken supervisor isn't approval-gated per
    event."""
    from src.tools.definitions import ALL_TOOL_DEFINITIONS
    fixr_tools = {
        t["function"]["name"]: t
        for t in ALL_TOOL_DEFINITIONS
        if t.get("service_binding") == "fixr"
    }
    assert set(fixr_tools) == {
        "dispatch_fix", "inspect_agents", "answer_agent", "kill_agent", "send_discord"
    }
    writes = {n for n, t in fixr_tools.items() if t.get("is_write")}
    assert writes == {"dispatch_fix"}
