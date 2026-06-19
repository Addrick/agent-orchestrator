# tests/voice/test_web.py
"""Browser/phone push-to-talk web capture (DP-238 web)."""
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.voice.integration import VoiceIntegration
from src.voice.intent import KeywordTimerRouter
from src.voice.pipeline import VoicePipeline
from src.voice.transcriber import NullTranscriber
from src.voice.web import register_voice_web


class _FakeRouter:
    def __init__(self):
        self.sent = []

    async def send(self, channel, recipient, subject, body):
        self.sent.append({"recipient": recipient, "body": body})
        return True


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

    register_voice_web(app, handler)
    client = TestClient(app)

    page = client.get("/voice")
    assert page.status_code == 200
    assert "hold to talk" in page.text

    pcm = np.zeros(2000, dtype=np.int16).tobytes()
    ok = client.post("/voice/utterance", content=pcm, headers={"X-Sample-Rate": "16000"})
    assert ok.status_code == 200
    assert ok.json()["matched"] is True


def test_utterance_route_rejects_missing_rate_and_empty_body():
    app = FastAPI()

    async def handler(pcm, sample_rate):  # pragma: no cover - not reached on 400
        return {}

    register_voice_web(app, handler)
    client = TestClient(app)
    pcm = np.zeros(2000, dtype=np.int16).tobytes()

    assert client.post("/voice/utterance", content=pcm).status_code == 400  # no header
    assert client.post(
        "/voice/utterance", content=b"", headers={"X-Sample-Rate": "16000"}
    ).status_code == 400  # empty body


def test_utterance_route_handler_error_is_500_not_crash():
    app = FastAPI()

    async def handler(pcm, sample_rate):
        raise RuntimeError("boom")

    register_voice_web(app, handler)
    client = TestClient(app, raise_server_exceptions=False)
    pcm = np.zeros(2000, dtype=np.int16).tobytes()
    r = client.post("/voice/utterance", content=pcm, headers={"X-Sample-Rate": "16000"})
    assert r.status_code == 500


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


async def test_handle_web_utterance_empty_transcript():
    integ = VoiceIntegration(_FakeRouter())
    integ._pipeline = _ptt_pipeline(integ, "")
    pcm = np.full(16000, 1000, dtype=np.int16).tobytes()
    res = await integ._handle_web_utterance(pcm, 16000)
    assert res["matched"] is False
    assert res["text"] == ""
    assert "didn't catch" in res["message"]


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
    assert integ._warmup_task is not None
    await integ._warmup_task  # NullTranscriber warmup is a no-op; just don't leak it
