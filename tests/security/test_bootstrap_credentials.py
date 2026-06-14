"""Tests for credential registration into the egress scrubber at startup.

Covers the bootstrap helper ``register_credentials()`` and that the live
entrypoint factory ``create_chat_system`` triggers it so production assembly
always populates the scrubber.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.bootstrap import create_chat_system, register_credentials
from src.engine import TextEngine
from src.memory.memory_manager import MemoryManager
from src.security.scrubber import get_scrubber, reset_scrubber
from src.security.vault import CredentialVault, reset_vault


@pytest.fixture(autouse=True)
def _clean_security_singletons():
    """Reset scrubber + vault around each test for isolation."""
    reset_scrubber()
    reset_vault()
    yield
    reset_scrubber()
    reset_vault()


def _fake_vault(values: dict) -> CredentialVault:
    return CredentialVault(source=lambda ref: values.get(ref))


def test_register_credentials_registers_resolved_secrets():
    reset_vault(_fake_vault({
        "ANTHROPIC_API_KEY": "sometestsecret123",
        "OPENAI_API_KEY": "openaisecret456",
    }))
    count = register_credentials()
    assert count == 2
    scrubber = get_scrubber()
    assert scrubber.scrub("leak sometestsecret123 here") == (
        "leak [REDACTED:ANTHROPIC_API_KEY] here"
    )
    assert scrubber.scrub("openaisecret456") == "[REDACTED:OPENAI_API_KEY]"


def test_register_credentials_counts_only_set_secrets():
    reset_vault(_fake_vault({
        "ZAMMAD_API_KEY": "zammadsecret789",
        # Others intentionally absent.
    }))
    assert register_credentials() == 1
    assert get_scrubber().active_secret_count() == 1


def test_register_credentials_is_idempotent():
    reset_vault(_fake_vault({"ANTHROPIC_API_KEY": "sometestsecret123"}))
    register_credentials()
    register_credentials()
    # Dedup by value: still one registered secret.
    assert get_scrubber().active_secret_count() == 1


def test_register_credentials_via_env_monkeypatch(monkeypatch):
    # Default env-backed vault: monkeypatched env should still flow through.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "envsecretvalue123")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENERATIVEAI_API_KEY", raising=False)
    monkeypatch.delenv("ZAMMAD_API_KEY", raising=False)
    count = register_credentials()
    assert count == 1
    assert get_scrubber().scrub("envsecretvalue123") == (
        "[REDACTED:ANTHROPIC_API_KEY]"
    )


def test_create_chat_system_registers_credentials_into_scrubber():
    """The live entrypoint factory must populate the scrubber on assembly."""
    reset_vault(_fake_vault({"ANTHROPIC_API_KEY": "bootsecret12345"}))
    mm = MagicMock(spec=MemoryManager)
    mm.backend = MagicMock()
    with patch("src.bootstrap.load_personas_from_file", return_value={}), \
            patch("src.bootstrap.load_system_personas_from_file", return_value={}), \
            patch("src.bootstrap.get_model_list", return_value={"Local": ["local"]}):
        create_chat_system(
            memory_manager=mm, text_engine=MagicMock(spec=TextEngine),
        )
    assert get_scrubber().scrub("bootsecret12345") == (
        "[REDACTED:ANTHROPIC_API_KEY]"
    )
