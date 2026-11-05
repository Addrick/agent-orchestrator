"""DP-418: the drift check is only useful if startup actually schedules it.

Drives the real composition step (`main._register_interfaces`) with stub
interfaces and records which tasks it registers.

DP-422: Fast and mock-only startup wiring tests. Stays unmarked by
pytest.mark.integration so CI's -m "not integration" gate guards startup wiring.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import src.main as main


class _App:
    def __init__(self) -> None:
        self.tasks: dict = {}

    def register_task(self, name, coro, **_kw) -> None:
        self.tasks[name] = coro


def _register(monkeypatch, *, pve: bool, channel: int, on_create_bot=None) -> dict:
    monkeypatch.setattr(main, "DISCORD_BOT", True)
    monkeypatch.setattr(main, "GMAIL_BOT", False)
    monkeypatch.setattr(main, "WEB_INTERFACE", False)
    monkeypatch.setattr(main, "PVE_TOOLS_ENABLED", pve)
    monkeypatch.setenv("DISCORD_API_KEY", "x")

    async def _start(_token):
        return None

    def _create_bot(_bot, **kw):
        if on_create_bot:
            on_create_bot(_bot, **kw)
        return SimpleNamespace(start=_start)

    monkeypatch.setattr(main, "create_discord_bot", _create_bot)
    app = _App()
    bot = SimpleNamespace(get_service=lambda _name: None)
    main._register_interfaces(app, bot, main.NotificationRouter(), debug_channel=channel)
    for coro in app.tasks.values():  # never awaited here; close to avoid warnings
        if inspect.iscoroutine(coro):
            coro.close()
    return app.tasks


def test_startup_schedules_the_check(monkeypatch):
    assert "node_artifact_check" in _register(monkeypatch, pve=True, channel=123)


def test_register_interfaces_passes_debug_channel_to_discord_bot(monkeypatch):
    """DP-422: discord_bot must receive the resolved debug_channel_id so on_message
    can exclude messages in the debug channel."""
    passed: dict = {}
    _register(monkeypatch, pve=False, channel=1498, on_create_bot=lambda _bot, **kw: passed.update(kw))
    assert passed.get("debug_channel_id") == 1498


@pytest.mark.parametrize("pve,channel", [(False, 123), (True, 0)])
def test_no_check_without_transport_or_channel(monkeypatch, pve, channel):
    assert "node_artifact_check" not in _register(monkeypatch, pve=pve, channel=channel)


@pytest.mark.parametrize("env,recipients,want", [
    (0, {"debug": {"discord_channel_id": "1498"}}, 1498),   # prod: agents.json only
    (77, {"debug": {"discord_channel_id": "1498"}}, 77),    # env overrides
    (0, {"debug": {"discord_channel_id": None}}, 0),
    (0, {}, 0),
])
def test_debug_channel_comes_from_the_recipients_prod_configures(monkeypatch, env, recipients, want):
    """DP-420: DP-418 keyed on DISCORD_DEBUG_CHANNEL alone, which prod does not
    set, and shipped inert. Prod's debug channel is agents.json recipients.debug,
    the same entry managr's boot report resolves."""
    monkeypatch.setattr(main, "DISCORD_DEBUG_CHANNEL", env)
    assert main._debug_channel_id({"recipients": recipients}) == want


def test_the_shipped_agents_json_resolves_a_debug_channel(monkeypatch):
    """The real config file, not a fixture: if `debug` disappears from it the
    check silently stops again."""
    import json
    from pathlib import Path
    monkeypatch.setattr(main, "DISCORD_DEBUG_CHANNEL", 0)
    cfg = json.loads((Path(main.__file__).resolve().parents[1] / "config" / "agents.json")
                     .read_text(encoding="utf-8"))
    assert main._debug_channel_id(cfg) > 0
