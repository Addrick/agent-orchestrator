# tests/test_claude_cli_env.py
"""DP-232: the cc-*/fixr `claude` CLI must run on the Claude subscription, not
the metered API. `build_claude_cli_env` strips the API-key vars from the child
env so `-p` mode falls through to CLAUDE_CODE_OAUTH_TOKEN.

DP-277: the same builder (and `build_agy_cli_env`) must also strip derpr's
machine secrets so a sandboxed-but-untrusted agent can't read or exfiltrate them.
"""

import pytest

from config import global_config
from src.utils.claude_cli_env import (
    _DERPR_SECRET_VARS,
    build_agy_cli_env,
    build_claude_cli_env,
)
from src.security.vault import CredentialVault


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


# --- DP-277 secret isolation -------------------------------------------------

def _secret_base():
    return {
        "DERPR_CONTROL_TOKEN": "portal-gate-secret",
        "OPENAI_API_KEY": "sk-openai",
        "GOOGLE_GENERATIVEAI_API_KEY": "goog-key",
        "ZAMMAD_API_KEY": "zammad-key",
        "DISCORD_API_KEY": "discord-token",
        "GH_TOKEN": "gh-push-token",
        "CLAUDE_CODE_OAUTH_TOKEN": "oauth-keep",
        "PATH": "/usr/bin",
    }


def test_claude_env_strips_derpr_secrets(monkeypatch):
    monkeypatch.setattr(global_config, "CC_USE_SUBSCRIPTION", True)
    env = build_claude_cli_env(_secret_base())
    # Every derpr-managed secret gone by default — including the control token
    # that gates the whole portal (DP-277's own boundary).
    for var in _DERPR_SECRET_VARS:
        assert var not in env, f"{var} leaked into cc-* child env"
    # The child's own auth + benign vars survive so it can still run.
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-keep"
    assert env["PATH"] == "/usr/bin"


def test_fixr_keeps_gh_token_only(monkeypatch):
    monkeypatch.setattr(global_config, "CC_USE_SUBSCRIPTION", True)
    env = build_claude_cli_env(_secret_base(), keep_gh_token=True)
    # fixr must push; GH_TOKEN is the ONE secret it retains.
    assert env["GH_TOKEN"] == "gh-push-token"
    assert "DERPR_CONTROL_TOKEN" not in env
    assert "ZAMMAD_API_KEY" not in env


def test_agy_env_strips_derpr_secrets_and_gh_token(monkeypatch):
    # agy never pushes — GH_TOKEN goes too; and the Anthropic billing strip is
    # cc-specific, so ANTHROPIC_API_KEY is NOT the concern here.
    env = build_agy_cli_env(_secret_base())
    for var in _DERPR_SECRET_VARS:
        assert var not in env, f"{var} leaked into agy child env"
    assert env["PATH"] == "/usr/bin"


def test_vault_known_refs_are_all_scrubbed():
    """Guard against a new host secret being added to the vault but forgotten
    in the cli-env denylist (the 'list rots' failure DP-277 warns about)."""
    for ref in CredentialVault.KNOWN_REFS:
        if ref == "ANTHROPIC_API_KEY":
            continue  # handled by the subscription strip, tested above
        assert ref in _DERPR_SECRET_VARS, (
            f"vault secret {ref} is not scrubbed from spawned-agent env"
        )
