"""Unit tests for DispatchAgent (DP-422 recipient resolution)."""

from unittest.mock import MagicMock

from src.agents.dispatch_agent import DispatchAgent


def _make_dispatch_agent(agent_config=None) -> DispatchAgent:
    chat_system = MagicMock()
    chat_system.text_engine = MagicMock()
    chat_system.memory_manager = MagicMock()
    zammad = MagicMock()
    router = MagicMock()
    return DispatchAgent(chat_system, zammad, router, agent_config=agent_config)


def test_dispatch_agent_resolves_mapped_recipient():
    agent = _make_dispatch_agent(agent_config={
        "notification_defaults": {"recipient": "alice"},
        "_recipients": {
            "alice": {
                "discord_channel_id": "12345",
                "discord_user_id": "67890",
                "email": "alice@example.com",
            },
        },
    })
    assert agent._resolve_recipient("discord_channel", 42) == "12345"
    assert agent._resolve_recipient("discord_dm", 42) == "67890"
    assert agent._resolve_recipient("email", 42) == "alice@example.com"


def test_dispatch_agent_falls_back_to_ticket_id_when_unmapped():
    agent = _make_dispatch_agent(agent_config={
        "notification_defaults": {"recipient": "alice"},
        "_recipients": {
            "alice": {
                "discord_channel_id": "12345",
            },
        },
    })
    # email not in alice's mapping -> falls back to ticket ID
    assert agent._resolve_recipient("email", 99) == "99"
    # unknown channel -> falls back to ticket ID
    assert agent._resolve_recipient("unknown_chan", 99) == "99"


def test_dispatch_agent_falls_back_when_no_defaults_or_no_recipients():
    agent = _make_dispatch_agent(agent_config={})
    assert agent._resolve_recipient("discord_dm", 77) == "77"

    agent_none = _make_dispatch_agent(agent_config=None)
    assert agent_none._resolve_recipient("discord_dm", 77) == "77"
