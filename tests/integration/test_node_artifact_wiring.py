"""DP-418: the drift check is only useful if startup actually schedules it.

Drives the real composition step (`main._register_interfaces`) with stub
interfaces and records which tasks it registers.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import src.main as main

pytestmark = pytest.mark.integration


class _App:
    def __init__(self) -> None:
        self.tasks: dict = {}

    def register_task(self, name, coro, **_kw) -> None:
        self.tasks[name] = coro


def _register(monkeypatch, *, pve: bool, channel: int) -> dict:
    monkeypatch.setattr(main, "DISCORD_BOT", True)
    monkeypatch.setattr(main, "GMAIL_BOT", False)
    monkeypatch.setattr(main, "WEB_INTERFACE", False)
    monkeypatch.setattr(main, "PVE_TOOLS_ENABLED", pve)
    monkeypatch.setattr(main, "DISCORD_DEBUG_CHANNEL", channel)
    monkeypatch.setenv("DISCORD_API_KEY", "x")

    async def _start(_token):
        return None
    monkeypatch.setattr(main, "create_discord_bot",
                        lambda _bot: SimpleNamespace(start=_start))
    app = _App()
    bot = SimpleNamespace(get_service=lambda _name: None)
    main._register_interfaces(app, bot, main.NotificationRouter())
    for coro in app.tasks.values():  # never awaited here; close to avoid warnings
        if inspect.iscoroutine(coro):
            coro.close()
    return app.tasks


def test_startup_schedules_the_check(monkeypatch):
    assert "node_artifact_check" in _register(monkeypatch, pve=True, channel=123)


@pytest.mark.parametrize("pve,channel", [(False, 123), (True, 0)])
def test_no_check_without_transport_or_channel(monkeypatch, pve, channel):
    assert "node_artifact_check" not in _register(monkeypatch, pve=pve, channel=channel)
