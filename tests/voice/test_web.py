# tests/voice/test_web.py
"""Browser/phone push-to-talk web capture (DP-238 web)."""
import asyncio

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.voice.alarm_bus import AlarmBus
from src.voice.integration import VoiceIntegration
from src.voice.intent import KeywordTimerRouter
from src.voice.pipeline import VoicePipeline
from src.voice.transcriber import NullTranscriber
from src.voice.web import _alarm_events, register_voice_web


class _FakeRouter:
    def __init__(self):
        self.sent = []
        self.registered = {}

    def register(self, channel, notifier):
        self.registered[channel] = notifier

    async def send(self, channel, recipient, subject, body):
        self.sent.append({"recipient": recipient, "body": body})
        return True


class _FakeStream:
    """Closes an utterance every 2nd pushed frame; flush emits a tail once."""

    def __init__(self):
        self.frames = 0
        self.flushed = False

    async def push(self, pcm, sample_rate, channels):
        self.frames += 1
        return "hello world" if self.frames % 2 == 0 else None

    async def flush(self):
        if self.flushed:
            return None
        self.flushed = True
        return "tail words"


def _noop_stream():
    return _FakeStream()


def _ptt_pipeline(integ, transcript):
    """A capture-less pipeline wired to the integration's intent handler."""
    return VoicePipeline(
        capture=None,
        vad_factory=lambda: None,  # type: ignore[arg-type,return-value]  # unused in submit path
        transcriber=NullTranscriber(transcript),
        intent_router=KeywordTimerRouter(),
        on_intent=integ._on_intent,
    )


# -- HTTP contract (register_voice_web) --------------------------------------

def test_register_voice_web_serves_page_and_accepts_utterance():
    app = FastAPI()

    async def handler(pcm, sample_rate):
        return {"text": "set a timer for 1 minute", "matched": True, "message": "ok"}

    async def transcribe(pcm, sample_rate):
        return {"text": "hello there"}

    register_voice_web(app, handler, transcribe, _noop_stream, AlarmBus())
    client = TestClient(app)

    page = client.get("/voice")
    assert page.status_code == 200
    assert "hold to talk" in page.text

    pcm = np.zeros(2000, dtype=np.int16).tobytes()
    ok = client.post("/voice/utterance", content=pcm, headers={"X-Sample-Rate": "16000"})
    assert ok.status_code == 200
    assert ok.json()["matched"] is True

    # dictation route: STT-only, returns just the transcript
    tr = client.post("/voice/transcribe", content=pcm, headers={"X-Sample-Rate": "16000"})
    assert tr.status_code == 200
    assert tr.json() == {"text": "hello there"}


def test_utterance_route_rejects_missing_rate_and_empty_body():
    app = FastAPI()

    async def handler(pcm, sample_rate):  # pragma: no cover - not reached on 400
        return {}

    register_voice_web(app, handler, handler, _noop_stream, AlarmBus())
    client = TestClient(app)
    pcm = np.zeros(2000, dtype=np.int16).tobytes()

    assert client.post("/voice/utterance", content=pcm).status_code == 400  # no header
    assert client.post(
        "/voice/utterance", content=b"", headers={"X-Sample-Rate": "16000"}
    ).status_code == 400  # empty body
    # the transcribe route shares the same validation
    assert client.post("/voice/transcribe", content=pcm).status_code == 400  # no header
    assert client.post(
        "/voice/transcribe", content=b"", headers={"X-Sample-Rate": "16000"}
    ).status_code == 400  # empty body


def test_utterance_route_handler_error_is_500_not_crash():
    app = FastAPI()

    async def handler(pcm, sample_rate):
        raise RuntimeError("boom")

    register_voice_web(app, handler, handler, _noop_stream, AlarmBus())
    client = TestClient(app, raise_server_exceptions=False)
    pcm = np.zeros(2000, dtype=np.int16).tobytes()
    r = client.post("/voice/utterance", content=pcm, headers={"X-Sample-Rate": "16000"})
    assert r.status_code == 500
    r = client.post("/voice/transcribe", content=pcm, headers={"X-Sample-Rate": "16000"})
    assert r.status_code == 500


# -- streaming WebSocket (/voice/stream) -------------------------------------

def test_stream_ws_returns_transcript_per_closed_utterance():
    app = FastAPI()

    async def handler(pcm, sample_rate):  # pragma: no cover - not used here
        return {}

    register_voice_web(app, handler, handler, _noop_stream, AlarmBus())
    client = TestClient(app)
    pcm = np.zeros(2000, dtype=np.int16).tobytes()

    with client.websocket_connect("/voice/stream") as ws:
        ws.send_json({"sample_rate": 48000})
        ws.send_bytes(pcm)  # frame 1 → None (no message)
        ws.send_bytes(pcm)  # frame 2 → closes an utterance
        assert ws.receive_json() == {"text": "hello world"}


def test_stream_ws_ignores_frames_before_sample_rate():
    app = FastAPI()

    async def handler(pcm, sample_rate):  # pragma: no cover - not used here
        return {}

    register_voice_web(app, handler, handler, _noop_stream, AlarmBus())
    client = TestClient(app)
    pcm = np.zeros(2000, dtype=np.int16).tobytes()

    with client.websocket_connect("/voice/stream") as ws:
        # No sample-rate control frame yet → these binary frames are dropped, so
        # the first transcript only arrives after the 2nd POST-rate frame.
        ws.send_bytes(pcm)
        ws.send_json({"sample_rate": 16000})
        ws.send_bytes(pcm)
        ws.send_bytes(pcm)
        assert ws.receive_json() == {"text": "hello world"}


def test_stream_ws_non_dict_control_frame_does_not_crash():
    # A valid-JSON but non-dict control frame ("5", '"x"') must be ignored, not
    # raise AttributeError on .get() and kill the stream handler.
    app = FastAPI()

    async def handler(pcm, sample_rate):  # pragma: no cover - not used here
        return {}

    register_voice_web(app, handler, handler, _noop_stream, AlarmBus())
    client = TestClient(app)
    pcm = np.zeros(2000, dtype=np.int16).tobytes()

    with client.websocket_connect("/voice/stream") as ws:
        ws.send_text("5")  # non-dict JSON — ignored, sample_rate stays 0
        ws.send_text('"x"')  # also non-dict — ignored
        ws.send_json({"sample_rate": 16000})
        ws.send_bytes(pcm)
        ws.send_bytes(pcm)  # closes an utterance
        assert ws.receive_json() == {"text": "hello world"}


# -- integration handler wiring ----------------------------------------------

async def test_handle_web_utterance_matched_schedules_timer(monkeypatch):
    from config import global_config
    monkeypatch.setattr(global_config, "VOICE_NOTIFY_CHANNEL_ID", "999")
    integ = VoiceIntegration(_FakeRouter())
    integ._pipeline = _ptt_pipeline(integ, "set a timer for 5 minutes")

    scheduled = []

    async def fake_schedule(seconds, target, *, label=None):
        scheduled.append((seconds, label, target.recipient))

    monkeypatch.setattr(integ.timer_service, "schedule", fake_schedule)

    pcm = np.full(16000, 1000, dtype=np.int16).tobytes()
    res = await integ._handle_web_utterance(pcm, 16000)

    assert res["matched"] is True
    assert res["text"] == "set a timer for 5 minutes"
    assert "5 minute" in res["message"]
    assert scheduled == [(300, None, "999")]


async def test_handle_web_utterance_non_command(monkeypatch):
    integ = VoiceIntegration(_FakeRouter())
    integ._pipeline = _ptt_pipeline(integ, "what is the weather today")
    pcm = np.full(16000, 1000, dtype=np.int16).tobytes()
    res = await integ._handle_web_utterance(pcm, 16000)
    assert res["matched"] is False
    assert res["text"] == "what is the weather today"


async def test_handle_web_utterance_no_notify_channel_not_matched(monkeypatch):
    # With VOICE_NOTIFY_CHANNEL_ID unset, _on_intent drops the schedule for a web
    # utterance (no source channel), so the reply must NOT claim the timer was set.
    from config import global_config
    monkeypatch.setattr(global_config, "VOICE_NOTIFY_CHANNEL_ID", "")
    integ = VoiceIntegration(_FakeRouter())
    integ._pipeline = _ptt_pipeline(integ, "set a timer for 5 minutes")

    scheduled = []

    async def fake_schedule(seconds, target, *, label=None):  # pragma: no cover
        scheduled.append(seconds)

    monkeypatch.setattr(integ.timer_service, "schedule", fake_schedule)
    pcm = np.full(16000, 1000, dtype=np.int16).tobytes()
    res = await integ._handle_web_utterance(pcm, 16000)

    assert res["matched"] is False
    assert res["text"] == "set a timer for 5 minutes"
    assert "VOICE_NOTIFY_CHANNEL_ID" in res["message"]
    assert scheduled == []  # nothing scheduled


async def test_handle_web_utterance_empty_transcript():
    integ = VoiceIntegration(_FakeRouter())
    integ._pipeline = _ptt_pipeline(integ, "")
    pcm = np.full(16000, 1000, dtype=np.int16).tobytes()
    res = await integ._handle_web_utterance(pcm, 16000)
    assert res["matched"] is False
    assert res["text"] == ""
    assert "didn't catch" in res["message"]


async def test_handle_web_transcribe_returns_text_without_routing(monkeypatch):
    # Dictation must NOT schedule a timer even for a timer-shaped utterance —
    # the LLM owns intents on the SPA path.
    integ = VoiceIntegration(_FakeRouter())
    integ._pipeline = _ptt_pipeline(integ, "set a timer for 5 minutes")

    scheduled = []

    async def fake_schedule(seconds, target, *, label=None):  # pragma: no cover
        scheduled.append(seconds)

    monkeypatch.setattr(integ.timer_service, "schedule", fake_schedule)
    pcm = np.full(16000, 1000, dtype=np.int16).tobytes()
    res = await integ._handle_web_transcribe(pcm, 16000)
    assert res == {"text": "set a timer for 5 minutes"}
    assert scheduled == []


async def test_handle_web_transcribe_empty():
    integ = VoiceIntegration(_FakeRouter())
    integ._pipeline = _ptt_pipeline(integ, "")
    pcm = np.full(16000, 1000, dtype=np.int16).tobytes()
    res = await integ._handle_web_transcribe(pcm, 16000)
    assert res == {"text": ""}


# -- attach_web enable/disable ------------------------------------------------

def test_attach_web_disabled_mounts_nothing(monkeypatch):
    from config import global_config
    monkeypatch.setattr(global_config, "VOICE_WEB_ENABLED", False)
    integ = VoiceIntegration(_FakeRouter())
    app = FastAPI()
    integ.attach_web(app)
    assert integ._pipeline is None
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/voice" not in paths


async def test_attach_web_enabled_builds_pipeline_and_routes(monkeypatch):
    from config import global_config
    monkeypatch.setattr(global_config, "VOICE_WEB_ENABLED", True)
    integ = VoiceIntegration(_FakeRouter())
    # Avoid constructing the real Moonshine transcriber in a unit test.
    monkeypatch.setattr(integ, "_build_pipeline", lambda *, capture: _ptt_pipeline(integ, "x"))
    app = FastAPI()
    integ.attach_web(app)  # schedules a background STT warmup (needs a running loop)
    assert integ._pipeline is not None
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/voice" in paths and "/voice/utterance" in paths
    assert "/voice/transcribe" in paths and "/voice/stream" in paths
    assert "/voice/alarms" in paths
    # the "web" NotificationRouter channel must be registered so a web-targeted
    # fired timer reaches the alarm bus instead of falling back to the logger
    assert "web" in integ._notifier.registered
    assert integ._warmup_task is not None
    await integ._warmup_task  # NullTranscriber warmup is a no-op; just don't leak it


# -- alarm SSE back-channel (GET /voice/alarms) ------------------------------

async def test_alarm_events_streams_published_alarm_then_unsubscribes():
    bus = AlarmBus()

    class _Req:
        async def is_disconnected(self):
            return False

    gen = _alarm_events(_Req(), bus)
    # First __anext__ subscribes, then blocks on the queue; publish while blocked.
    pending = asyncio.ensure_future(gen.__anext__())
    await asyncio.sleep(0)
    assert bus.subscriber_count == 1
    await bus.publish({"text": "⏰ Timer up!", "channel": "web_ui"})
    frame = await asyncio.wait_for(pending, 1.0)
    assert frame.startswith("data: ")
    assert "⏰ Timer up!" in frame
    await gen.aclose()  # finally: unsubscribe
    assert bus.subscriber_count == 0


async def test_alarm_events_stops_on_disconnect():
    bus = AlarmBus()

    class _Req:
        async def is_disconnected(self):
            return True

    gen = _alarm_events(_Req(), bus)
    with __import__("pytest").raises(StopAsyncIteration):
        await gen.__anext__()
    assert bus.subscriber_count == 0
