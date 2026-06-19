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
    # fixr inherits the global default via the "default" sentinel so it moves
    # with DEFAULT_MODEL_NAME instead of pinning a soon-deprecated id. The raw
    # value stays "default" (round-trips through save/UI); get_model_name()
    # resolves it for the engine. (A concrete id like claude-opus-4-8 was wrong
    # — this deployment has no Anthropic API key — and cc-* can't host the
    # supervisor since it bypasses derpr's tool loop. Loop live-smoked on prod
    # 2026-06-16 → PR; the *-preview gemini tier 429s, non-preview is fine.)
    assert fixr.get_raw_model_name() == "default"
    assert fixr.get_model_name() == global_config.DEFAULT_MODEL_NAME
    assert "fixr" in fixr.get_service_bindings()
    # Not quarantined — its tool policy composes cleanly.
    assert not fixr.get_security_block_reasons()
    tools = set(fixr.get_enabled_tools())
    assert {"dispatch_fix", "inspect_agents", "answer_agent",
            "kill_agent", "prune_agents", "send_discord"} <= tools


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
        "dispatch_fix", "inspect_agents", "answer_agent", "kill_agent",
        "prune_agents", "send_discord"
    }
    writes = {n for n, t in fixr_tools.items() if t.get("is_write")}
    assert writes == {"dispatch_fix"}
