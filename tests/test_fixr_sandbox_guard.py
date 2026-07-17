# tests/test_fixr_sandbox_guard.py
"""DP-234: fixr must not (mis)direct sandboxed coding agents at host/infra
outages — the DP-ZAM-001 failure mode. Guard lives in two prompts:
the dispatched-agent system prompt self-aborts, and the fixr supervisor persona
is told not to dispatch such tasks."""

import json

from config import global_config
from src.self_edit.events import SENTINEL_ERROR
from src.self_edit.prompts import DISPATCH_AGENT_PROMPT


def test_dispatch_prompt_has_sandbox_scope_self_abort():
    p = DISPATCH_AGENT_PROMPT
    assert "SANDBOX SCOPE" in p
    # Names the out-of-reach surfaces and instructs a FIXR_ERROR self-abort.
    assert "cloudflared" in p
    assert "outside your sandbox" in p
    assert SENTINEL_ERROR in p
    # Explicitly forbids the proxy-code-change failure mode.
    assert "proxy" in p


def test_dispatch_prompt_mandates_pull_request():
    """DP-291: a live run committed but never opened a PR — the deliverable
    silently vanished. The prompt must make the PR non-optional and forbid a
    DONE sentinel without a PR URL."""
    p = DISPATCH_AGENT_PROMPT
    assert "gh pr create" in p
    assert "git push" in p
    assert "THE PULL REQUEST IS YOUR DELIVERABLE" in p
    assert "never report success without a PR URL" in p
    assert "without a PR URL is a" in p  # DONE-sentinel guard


def test_fixr_persona_prompt_warns_against_dispatching_infra():
    with open(global_config.DEFAULT_PERSONA_SAVE_FILE, encoding="utf-8") as f:
        data = json.load(f)  # also guards: my edit kept the JSON valid
    fixr = next(p for p in data["personas"] if p["name"] == "fixr")
    prompt = fixr["prompt"]
    assert "SCOPE" in prompt
    assert "Do NOT dispatch" in prompt
    assert "escalate to a human" in prompt
