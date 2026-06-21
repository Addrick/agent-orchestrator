# tests/voice/test_integration.py
import asyncio

from src.voice.integration import VoiceIntegration, _format_fire
from src.voice.intent import TimerIntent
from src.voice.timer import Timer, TimerService
from src.voice.types import TimerTarget, VoiceCommand


class _FakeRouter:
    def __init__(self):
        self.sent = []

    async def send(self, channel, recipient, subject, body):
        self.sent.append({"channel": channel, "recipient": recipient,
                          "subject": subject, "body": body})
        return True


class _FakeToolManager:
    def __init__(self):
        self.registered = {}

    def register(self, name, fn):
        self.registered[name] = fn


def test_name_is_voice():
    assert VoiceIntegration(_FakeRouter()).name == "voice"


def test_registers_timer_tools():
    integ = VoiceIntegration(_FakeRouter())
    tm = _FakeToolManager()
    integ.register_tools(tm)
    assert set(tm.registered) == {"set_timer", "list_timers", "cancel_timer"}


async def test_on_fire_sends_notification_with_mention():
    router = _FakeRouter()
    integ = VoiceIntegration(router)
    timer = Timer(
        id="abc", seconds=90, fire_at=0.0,
        target=TimerTarget(channel="discord_channel", recipient="555", mention_user_id=77),
        label="pasta",
    )
    await integ._on_fire(timer)
    assert len(router.sent) == 1
    msg = router.sent[0]
    assert msg["recipient"] == "555"
    assert "<@77>" in msg["body"]
    assert "pasta" in msg["body"]


async def test_on_intent_schedules_to_config_channel(monkeypatch):
    from config import global_config
    monkeypatch.setattr(global_config, "VOICE_NOTIFY_CHANNEL_ID", "999")
    scheduled = []

    async def fake_schedule(seconds, target, *, label=None):
        scheduled.append((seconds, target, label))

    integ = VoiceIntegration(_FakeRouter())
    monkeypatch.setattr(integ.timer_service, "schedule", fake_schedule)

    await integ._on_intent(
        TimerIntent(seconds=600, label="x"),
        VoiceCommand(raw_text="set a timer for 10 minutes", user_id=77, source_channel_id=42),
    )
    assert len(scheduled) == 1
    seconds, target, label = scheduled[0]
    assert seconds == 600
    assert target.recipient == "999"  # config wins over the source VC
    assert target.mention_user_id == 77


async def test_on_intent_falls_back_to_source_channel(monkeypatch):
    from config import global_config
    monkeypatch.setattr(global_config, "VOICE_NOTIFY_CHANNEL_ID", "")
    scheduled = []

    async def fake_schedule(seconds, target, *, label=None):
        scheduled.append(target)

    integ = VoiceIntegration(_FakeRouter())
    monkeypatch.setattr(integ.timer_service, "schedule", fake_schedule)
    await integ._on_intent(
        TimerIntent(seconds=60),
        VoiceCommand(raw_text="timer 1 minute", user_id=1, source_channel_id=42),
    )
    assert scheduled[0].recipient == "42"


def test_attach_discord_disabled_builds_no_pipeline(monkeypatch):
    from config import global_config
    monkeypatch.setattr(global_config, "VOICE_ENABLED", False)
    integ = VoiceIntegration(_FakeRouter())
    integ.attach_discord(object())
    assert integ._pipeline is None


def test_attach_discord_no_autojoin_without_experiment_flag(monkeypatch):
    """VOICE_ENABLED alone must NOT join the (dead) Discord voice path — the join
    is gated behind the explicit VOICE_DISCORD_EXPERIMENT escape hatch."""
    from config import global_config
    monkeypatch.setattr(global_config, "VOICE_ENABLED", True)
    monkeypatch.setattr(global_config, "VOICE_DISCORD_CHANNEL_ID", "12345")
    monkeypatch.setattr(global_config, "VOICE_DISCORD_EXPERIMENT", False)
    integ = VoiceIntegration(_FakeRouter())
    integ.attach_discord(object())
    assert integ._pipeline is None


def test_format_fire_variants():
    base = TimerTarget(channel="discord_channel", recipient="1")
    assert "1 minute" in _format_fire(Timer("a", 60, 0.0, base))
    assert "30 second" in _format_fire(Timer("a", 30, 0.0, base))
    assert "1m 30s" in _format_fire(Timer("a", 90, 0.0, base))
    assert "pasta" in _format_fire(Timer("a", 60, 0.0, base, label="pasta"))


async def test_shutdown_clears_timers():
    integ = VoiceIntegration(_FakeRouter())
    integ.timer_service = TimerService(lambda t: asyncio.sleep(0), sleep=lambda s: asyncio.sleep(10))
    await integ.timer_service.schedule(10, TimerTarget("discord_channel", "1"))
    await integ.shutdown()
    assert integ.timer_service.list() == []
