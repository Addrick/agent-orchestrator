"""Unit tests for src.agents.base recipient resolution (DP-422)."""

from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from src.agents.base import Agent, resolve_recipient
from src.chat_system import ChatSystem


class _StubAgent(Agent):
    """Minimal Agent subclass to exercise Agent base methods."""
    agent_name = "stub_agent"

    def __init__(self, chat_system: ChatSystem, agent_config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(chat_system, inject_personas=False)
        self.agent_config = agent_config or {}

    async def deploy(self) -> None:
        pass


@pytest.fixture
def sample_recipients() -> Dict[str, Any]:
    return {
        "debug": {
            "discord_channel_id": 1498,
        },
        "adrich": {
            "discord_user_id": "321",
            "email": "adrich@example.com",
        },
        "alerts": {
            "discord_channel_id": "9999",
            "email": "alerts@example.com",
        },
        "empty_entry": {},
        "none_entry": {
            "discord_channel_id": None,
            "discord_user_id": None,
            "email": None,
        },
    }


def test_resolve_recipient_empty_or_falsy_key(sample_recipients):
    assert resolve_recipient(sample_recipients, "discord_channel", None) is None
    assert resolve_recipient(sample_recipients, "discord_channel", "") is None
    assert resolve_recipient(sample_recipients, "discord_channel", "   ") is None
    assert resolve_recipient(sample_recipients, None, "debug") is None
    assert resolve_recipient(sample_recipients, "discord_channel", 0) == "0"


def test_resolve_recipient_numeric_id_passthrough(sample_recipients):
    assert resolve_recipient(sample_recipients, "discord_channel", "12345") == "12345"
    assert resolve_recipient(sample_recipients, "discord_dm", 98765) == "98765"
    assert resolve_recipient(None, "discord_channel", "12345") == "12345"


def test_resolve_recipient_discord_channel(sample_recipients):
    assert resolve_recipient(sample_recipients, "discord_channel", "debug") == "1498"
    assert resolve_recipient(sample_recipients, "discord_channel", "alerts") == "9999"
    assert resolve_recipient(sample_recipients, "discord_channel", "adrich") is None


def test_resolve_recipient_discord_dm(sample_recipients):
    assert resolve_recipient(sample_recipients, "discord_dm", "adrich") == "321"
    assert resolve_recipient(sample_recipients, "discord_dm", "debug") is None


def test_resolve_recipient_email_channels(sample_recipients):
    assert resolve_recipient(sample_recipients, "email", "adrich") == "adrich@example.com"
    assert resolve_recipient(sample_recipients, "zammad_email", "alerts") == "alerts@example.com"
    assert resolve_recipient(sample_recipients, "email", "debug") is None


def test_resolve_recipient_missing_or_invalid_recipients_mapping():
    assert resolve_recipient(None, "discord_channel", "debug") is None
    assert resolve_recipient({}, "discord_channel", "debug") is None
    assert resolve_recipient("not_a_mapping", "discord_channel", "debug") is None  # type: ignore[arg-type]


def test_resolve_recipient_missing_or_invalid_entry(sample_recipients):
    assert resolve_recipient(sample_recipients, "discord_channel", "unknown_key") is None
    assert resolve_recipient(sample_recipients, "discord_channel", "empty_entry") is None
    assert resolve_recipient(sample_recipients, "discord_channel", "none_entry") is None
    recipients_with_bad_entry = {"bad": "scalar_value"}
    assert resolve_recipient(recipients_with_bad_entry, "discord_channel", "bad") is None  # type: ignore[arg-type]


def test_resolve_recipient_unsupported_channel(sample_recipients):
    assert resolve_recipient(sample_recipients, "unknown_channel", "debug") is None


def _make_stub_agent(agent_config: Optional[Dict[str, Any]] = None) -> _StubAgent:
    mock_chat = MagicMock()
    mock_chat.text_engine = MagicMock()
    mock_chat.memory_manager = MagicMock()
    return _StubAgent(mock_chat, agent_config=agent_config)


def test_agent_resolve_recipient_passthrough(sample_recipients):
    agent = _make_stub_agent(agent_config={"_recipients": sample_recipients})

    # Resolved via _recipients
    assert agent._resolve_recipient("discord_channel", "debug") == "1498"
    assert agent._resolve_recipient("discord_dm", "adrich") == "321"
    assert agent._resolve_recipient("email", "adrich") == "adrich@example.com"

    # Digits pass through
    assert agent._resolve_recipient("discord_dm", "12345") == "12345"

    # Unmapped keys pass through as-is
    assert agent._resolve_recipient("discord_channel", "unknown") == "unknown"


def test_agent_resolve_recipient_without_config():
    agent = _make_stub_agent(agent_config=None)

    assert agent._resolve_recipient("discord_dm", "12345") == "12345"
    assert agent._resolve_recipient("discord_dm", "adrich") == "adrich"
    assert agent._resolve_recipient("discord_dm", None) == ""
