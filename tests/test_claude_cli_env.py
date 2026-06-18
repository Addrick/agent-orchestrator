# tests/test_claude_cli_env.py
"""DP-232: the cc-*/fixr `claude` CLI must run on the Claude subscription, not
the metered API. `build_claude_cli_env` strips the API-key vars from the child
env so `-p` mode falls through to CLAUDE_CODE_OAUTH_TOKEN."""

import pytest

from config import global_config
from src.utils.claude_cli_env import build_claude_cli_env


def test_strips_api_keys_when_subscription_on(monkeypatch):
    monkeypatch.setattr(global_config, "CC_USE_SUBSCRIPTION", True)
    base = {
        "ANTHROPIC_API_KEY": "sk-ant-secret",
        "ANTHROPIC_AUTH_TOKEN": "bearer-secret",
        "CLAUDE_CODE_OAUTH_TOKEN": "oauth-keep",
        "PATH": "/usr/bin",
    }
    env = build_claude_cli_env(base)
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    # Subscription token + unrelated vars survive so the CLI can still auth + run.
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-keep"
    assert env["PATH"] == "/usr/bin"


def test_keeps_api_keys_when_subscription_off(monkeypatch):
    monkeypatch.setattr(global_config, "CC_USE_SUBSCRIPTION", False)
    base = {"ANTHROPIC_API_KEY": "sk-ant-secret", "PATH": "/usr/bin"}
    env = build_claude_cli_env(base)
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-secret"


def test_does_not_mutate_input(monkeypatch):
    monkeypatch.setattr(global_config, "CC_USE_SUBSCRIPTION", True)
    base = {"ANTHROPIC_API_KEY": "sk-ant-secret"}
    build_claude_cli_env(base)
    assert base["ANTHROPIC_API_KEY"] == "sk-ant-secret"  # caller's dict untouched


def test_defaults_to_os_environ(monkeypatch):
    monkeypatch.setattr(global_config, "CC_USE_SUBSCRIPTION", True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("DERPR_MARKER", "present")
    env = build_claude_cli_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert env["DERPR_MARKER"] == "present"
