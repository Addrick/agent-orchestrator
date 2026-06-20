# tests/voice/test_alarm_bus.py
"""Fired-timer alarm fan-out for the portal (DP-238)."""
from src.voice.alarm_bus import AlarmBus, WebAlarmNotifier


async def test_publish_fans_out_to_all_subscribers():
    bus = AlarmBus()
    q1 = bus.subscribe()
    q2 = bus.subscribe()
    assert bus.subscriber_count == 2
    await bus.publish({"text": "a"})
    assert (await q1.get())["text"] == "a"
    assert (await q2.get())["text"] == "a"


async def test_unsubscribe_stops_delivery():
    bus = AlarmBus()
    q = bus.subscribe()
    bus.unsubscribe(q)
    await bus.publish({"text": "a"})
    assert q.empty()
    assert bus.subscriber_count == 0


async def test_full_queue_drops_oldest_not_newest():
    # A wedged browser must not grow memory or lose the most recent alarm: a full
    # queue drops its oldest entry to make room (last-32-win, maxsize=32).
    bus = AlarmBus()
    q = bus.subscribe()
    for i in range(40):
        await bus.publish({"i": i})
    drained = []
    while not q.empty():
        drained.append(q.get_nowait()["i"])
    assert len(drained) == 32
    assert drained[0] == 8 and drained[-1] == 39


async def test_web_alarm_notifier_publishes_with_channel_tag():
    bus = AlarmBus()
    q = bus.subscribe()
    notifier = WebAlarmNotifier(bus)
    ok = await notifier.send(recipient="web_ui:kitchen", subject="Timer", body="⏰ up")
    assert ok is True
    assert await q.get() == {
        "channel": "web_ui:kitchen",
        "subject": "Timer",
        "text": "⏰ up",
    }
