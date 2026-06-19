# tests/voice/test_timer.py
import asyncio

import pytest

from src.voice.timer import TimerService
from src.voice.types import TimerTarget

TARGET = TimerTarget(channel="discord_channel", recipient="123")


async def test_schedule_fires_on_fire():
    fired = []
    ev = asyncio.Event()

    async def on_fire(timer):
        fired.append(timer)
        ev.set()

    async def instant_sleep(_seconds):
        return None

    svc = TimerService(on_fire, sleep=instant_sleep)
    timer = await svc.schedule(600, TARGET, label="pasta")
    await asyncio.wait_for(ev.wait(), timeout=1)
    assert len(fired) == 1
    assert fired[0].id == timer.id
    assert fired[0].label == "pasta"
    # Fired timers leave the live set.
    assert svc.get(timer.id) is None


async def test_cancel_prevents_fire():
    fired = []

    async def on_fire(timer):
        fired.append(timer)

    svc = TimerService(on_fire, sleep=lambda s: asyncio.sleep(10))
    timer = await svc.schedule(10, TARGET)
    assert await svc.cancel(timer.id) is True
    await asyncio.sleep(0.05)
    assert fired == []
    assert svc.list() == []


async def test_cancel_unknown_returns_false():
    svc = TimerService(lambda t: asyncio.sleep(0), sleep=lambda s: asyncio.sleep(10))
    assert await svc.cancel("nope") is False


async def test_list_sorted_and_remaining():
    now = [1000.0]

    svc = TimerService(
        lambda t: asyncio.sleep(0),
        clock=lambda: now[0],
        sleep=lambda s: asyncio.sleep(10),
    )
    t1 = await svc.schedule(300, TARGET)
    t2 = await svc.schedule(60, TARGET)
    listed = svc.list()
    assert [t.id for t in listed] == [t2.id, t1.id]  # soonest first
    assert listed[0].remaining(now[0]) == 60


async def test_schedule_rejects_nonpositive():
    svc = TimerService(lambda t: asyncio.sleep(0))
    with pytest.raises(ValueError):
        await svc.schedule(0, TARGET)


async def test_shutdown_cancels_all():
    fired = []
    svc = TimerService(
        lambda t: fired.append(t), sleep=lambda s: asyncio.sleep(10),
    )
    await svc.schedule(10, TARGET)
    await svc.schedule(20, TARGET)
    await svc.shutdown()
    assert svc.list() == []
    await asyncio.sleep(0.05)
    assert fired == []
