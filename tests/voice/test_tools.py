# tests/voice/test_tools.py
import asyncio

from src.tools.turn_context import TurnContext, turn_scope
from src.voice.timer import TimerService
from src.voice.tools import VoiceTimerToolHandler


def _svc():
    return TimerService(lambda t: asyncio.sleep(0), sleep=lambda s: asyncio.sleep(10))


def _ctx(channel):
    return TurnContext(persona_name="p", user_identifier="u", channel=channel, server_id=None)


async def test_set_timer_uses_numeric_channel_context():
    handler = VoiceTimerToolHandler(_svc())
    with turn_scope(_ctx("555")):
        result = await handler._set_timer("10 minutes", label="pasta")
    assert result["status"] == "scheduled"
    assert result["seconds"] == 600
    assert result["label"] == "pasta"


async def test_set_timer_falls_back_to_config(monkeypatch):
    from config import global_config
    monkeypatch.setattr(global_config, "VOICE_NOTIFY_CHANNEL_ID", "999")
    handler = VoiceTimerToolHandler(_svc())
    with turn_scope(_ctx("not-numeric")):
        result = await handler._set_timer("30 seconds")
    assert result["status"] == "scheduled"
    assert result["seconds"] == 30


async def test_set_timer_bad_duration():
    handler = VoiceTimerToolHandler(_svc())
    with turn_scope(_ctx("555")):
        result = await handler._set_timer("whenever")
    assert result["status"] == "error"


async def test_set_timer_no_target(monkeypatch):
    from config import global_config
    monkeypatch.setattr(global_config, "VOICE_NOTIFY_CHANNEL_ID", "")
    handler = VoiceTimerToolHandler(_svc())
    with turn_scope(_ctx("not-numeric")):
        result = await handler._set_timer("10 minutes")
    assert result["status"] == "error"


async def test_list_and_cancel():
    svc = _svc()
    handler = VoiceTimerToolHandler(svc)
    with turn_scope(_ctx("555")):
        scheduled = await handler._set_timer("10 minutes")
    listing = await handler._list_timers()
    assert listing["count"] == 1
    assert listing["timers"][0]["id"] == scheduled["id"]

    cancelled = await handler._cancel_timer(scheduled["id"])
    assert cancelled["status"] == "cancelled"
    assert (await handler._list_timers())["count"] == 0


async def test_cancel_unknown():
    handler = VoiceTimerToolHandler(_svc())
    result = await handler._cancel_timer("zzz")
    assert result["status"] == "not_found"
