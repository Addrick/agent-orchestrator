"""Unit tests for the credential vault."""

import pytest

from src.security.scrubber import SecretScrubber
from src.security.vault import CredentialVault, get_vault, reset_vault


def test_package_reexports_vault_accessors():
    """The src.security package must re-export the vault accessors, symmetric
    with get_scrubber/reset_scrubber, so `from src.security import get_vault`
    works."""
    import src.security as security

    assert security.get_vault is get_vault
    assert security.reset_vault is reset_vault
    assert security.CredentialVault is CredentialVault
    for name in ("get_vault", "reset_vault", "CredentialVault"):
        assert name in security.__all__


def _fake_source(values: dict):
    """Return a source callable backed by a dict (never touches real env)."""
    return lambda ref: values.get(ref)


def test_get_returns_injected_source_value():
    vault = CredentialVault(source=_fake_source({"OPENAI_API_KEY": "abc123val"}))
    assert vault.get("OPENAI_API_KEY") == "abc123val"


def test_get_returns_none_when_absent():
    vault = CredentialVault(source=_fake_source({}))
    assert vault.get("OPENAI_API_KEY") is None


def test_require_returns_value_when_present():
    vault = CredentialVault(source=_fake_source({"ZAMMAD_API_KEY": "zammadval1"}))
    assert vault.require("ZAMMAD_API_KEY") == "zammadval1"


def test_require_raises_keyerror_when_absent():
    vault = CredentialVault(source=_fake_source({}))
    with pytest.raises(KeyError) as exc:
        vault.require("ANTHROPIC_API_KEY")
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_require_raises_keyerror_on_empty_value():
    vault = CredentialVault(source=_fake_source({"OPENAI_API_KEY": ""}))
    with pytest.raises(KeyError):
        vault.require("OPENAI_API_KEY")


def test_known_refs_returns_managed_keys():
    vault = CredentialVault(source=_fake_source({}))
    assert vault.known_refs() == (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_GENERATIVEAI_API_KEY",
        "ZAMMAD_API_KEY",
    )


def test_register_into_registers_resolved_known_refs():
    vault = CredentialVault(
        source=_fake_source(
            {
                "OPENAI_API_KEY": "openaikey123",
                "ANTHROPIC_API_KEY": "anthropickey123",
                # GOOGLE + ZAMMAD intentionally absent.
            }
        )
    )
    scrubber = SecretScrubber()
    count = vault.register_into(scrubber)
    assert count == 2
    assert scrubber.active_secret_count() == 2
    assert scrubber.scrub("openaikey123") == "[REDACTED:OPENAI_API_KEY]"
    assert scrubber.scrub("anthropickey123") == "[REDACTED:ANTHROPIC_API_KEY]"


def test_register_into_skips_empty_values():
    vault = CredentialVault(
        source=_fake_source({"OPENAI_API_KEY": "", "ZAMMAD_API_KEY": "zammadval1"})
    )
    scrubber = SecretScrubber()
    count = vault.register_into(scrubber)
    assert count == 1
    assert scrubber.active_secret_count() == 1


def test_register_into_zero_when_nothing_resolves():
    vault = CredentialVault(source=_fake_source({}))
    scrubber = SecretScrubber()
    assert vault.register_into(scrubber) == 0
    assert scrubber.active_secret_count() == 0


def test_fake_source_does_not_touch_real_env():
    # A vault with an explicit fake source must resolve only from that dict.
    vault = CredentialVault(source=_fake_source({"OPENAI_API_KEY": "fakeonly1"}))
    assert vault.get("OPENAI_API_KEY") == "fakeonly1"
    # An unrelated ref the fake source doesn't define stays None.
    assert vault.get("PATH") is None


# --- sanitized_env() ----------------------------------------------------------


def test_sanitized_env_drops_all_known_refs(monkeypatch):
    vault = CredentialVault(source=_fake_source({}))
    for ref in vault.known_refs():
        monkeypatch.setenv(ref, f"{ref.lower()}_secret_value")
    monkeypatch.setenv("UNRELATED_VAR", "keep-me")

    env = vault.sanitized_env()

    for ref in vault.known_refs():
        assert ref not in env
    assert env["UNRELATED_VAR"] == "keep-me"


def test_sanitized_env_drops_extra_refs(monkeypatch):
    vault = CredentialVault(source=_fake_source({}))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-secret")

    env = vault.sanitized_env(extra_refs=("CLAUDE_CODE_OAUTH_TOKEN",))

    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env


def test_sanitized_env_does_not_mutate_parent_env(monkeypatch):
    vault = CredentialVault(source=_fake_source({}))
    monkeypatch.setenv("OPENAI_API_KEY", "still-here-after")

    vault.sanitized_env()

    import os

    assert os.environ["OPENAI_API_KEY"] == "still-here-after"


def test_sanitized_env_respects_explicit_base():
    vault = CredentialVault(source=_fake_source({}))
    base = {"OPENAI_API_KEY": "secretval1", "PATH": "/usr/bin"}

    env = vault.sanitized_env(base=base)

    assert "OPENAI_API_KEY" not in env
    assert env["PATH"] == "/usr/bin"
    # The base dict itself is untouched.
    assert base["OPENAI_API_KEY"] == "secretval1"


def test_sanitized_env_missing_refs_are_fine():
    vault = CredentialVault(source=_fake_source({}))
    # No known refs present in base at all — must not raise.
    env = vault.sanitized_env(base={"HOME": "/home/x"})
    assert env == {"HOME": "/home/x"}


# --- get_vault() / reset_vault() singleton behavior ---------------------------


def test_get_vault_returns_singleton():
    reset_vault()
    try:
        first = get_vault()
        second = get_vault()
        assert first is second
        assert isinstance(first, CredentialVault)
    finally:
        reset_vault()


def test_reset_vault_replaces_singleton():
    reset_vault()
    try:
        first = get_vault()
        reset_vault()
        second = get_vault()
        assert first is not second
    finally:
        reset_vault()


def test_reset_vault_injects_fake_source():
    injected = CredentialVault(
        source=_fake_source({"OPENAI_API_KEY": "injectedkey1"})
    )
    reset_vault(injected)
    try:
        assert get_vault() is injected
        assert get_vault().get("OPENAI_API_KEY") == "injectedkey1"
    finally:
        reset_vault()


def test_default_vault_reads_env_live(monkeypatch):
    reset_vault()
    try:
        monkeypatch.setenv("OPENAI_API_KEY", "liveenvkey123")
        # Vault was built before the env was set, proving it reads live.
        assert get_vault().get("OPENAI_API_KEY") == "liveenvkey123"
        monkeypatch.setenv("OPENAI_API_KEY", "changedkey456")
        assert get_vault().get("OPENAI_API_KEY") == "changedkey456"
    finally:
        reset_vault()
