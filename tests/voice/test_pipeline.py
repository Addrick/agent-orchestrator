# tests/voice/test_pipeline.py
import numpy as np

from src.voice.intent import KeywordTimerRouter
from src.voice.pipeline import VoicePipeline
from src.voice.transcriber import NullTranscriber
from src.voice.types import AudioFrame
from src.voice.vad import VAD


class _ImmediateVAD(VAD):
    """Emits whatever it's fed as a completed utterance at once."""

    def add_chunk(self, pcm16k_mono: bytes):
        return pcm16k_mono

    def flush(self):
        return None


class _NoCapture:
    def __init__(self):
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


def _frame(user_id):
    pcm = np.full(960 * 2, 1000, dtype=np.int16).tobytes()  # 48k stereo chunk
    return AudioFrame(pcm=pcm, sample_rate=48000, channels=2, user_id=user_id,
                      source_channel_id=42)


def _pipeline(transcriber, on_intent, vad_factory=None):
    factory = vad_factory or (lambda: _ImmediateVAD())
    return VoicePipeline(
        capture=_NoCapture(),
        vad_factory=factory,
        transcriber=transcriber,
        intent_router=KeywordTimerRouter(),
        on_intent=on_intent,
    )


async def test_matched_command_calls_on_intent():
    seen = []

    async def on_intent(intent, command):
        seen.append((intent, command))

    pipe = _pipeline(NullTranscriber("set a timer for 10 minutes"), on_intent)
    await pipe.on_frame(_frame(user_id=7))

    assert len(seen) == 1
    intent, command = seen[0]
    assert intent.seconds == 600
    assert command.user_id == 7
    assert command.source_channel_id == 42


async def test_non_command_does_not_fire():
    seen = []

    async def on_intent(intent, command):
        seen.append(intent)

    pipe = _pipeline(NullTranscriber("what's the weather"), on_intent)
    await pipe.on_frame(_frame(user_id=7))
    assert seen == []


async def test_empty_transcript_ignored():
    seen = []

    async def on_intent(intent, command):
        seen.append(intent)

    pipe = _pipeline(NullTranscriber(""), on_intent)
    await pipe.on_frame(_frame(user_id=7))
    assert seen == []


async def test_per_user_vad_isolation():
    created = []

    def factory():
        v = _ImmediateVAD()
        created.append(v)
        return v

    async def on_intent(intent, command):
        pass

    pipe = _pipeline(NullTranscriber("timer for 1 minute"), on_intent, vad_factory=factory)
    await pipe.on_frame(_frame(user_id=1))
    await pipe.on_frame(_frame(user_id=2))
    await pipe.on_frame(_frame(user_id=1))  # reuses user 1's VAD
    assert len(created) == 2


async def test_warmup_delegates_to_transcriber():
    class _WarmTranscriber(NullTranscriber):
        def __init__(self):
            super().__init__("")
            self.warmed = False

        async def warmup(self):
            self.warmed = True

    t = _WarmTranscriber()
    pipe = VoicePipeline(
        capture=None,
        vad_factory=lambda: _ImmediateVAD(),
        transcriber=t,
        intent_router=KeywordTimerRouter(),
        on_intent=lambda i, c: None,  # type: ignore[arg-type,return-value]
    )
    await pipe.warmup()
    assert t.warmed is True


async def test_submit_utterance_returns_text_and_intent():
    seen = []

    async def on_intent(intent, command):
        seen.append(intent)

    # No capture (web push-to-talk path): the browser delimits the utterance.
    pipe = VoicePipeline(
        capture=None,
        vad_factory=lambda: _ImmediateVAD(),
        transcriber=NullTranscriber("set a timer for 10 minutes"),
        intent_router=KeywordTimerRouter(),
        on_intent=on_intent,
    )
    pcm = np.full(16000, 1000, dtype=np.int16).tobytes()  # 1s 16k mono
    text, intent = await pipe.submit_utterance(pcm, 16000, 1)
    assert text == "set a timer for 10 minutes"
    assert intent is not None and intent.seconds == 600
    assert len(seen) == 1


async def test_submit_utterance_non_command_returns_text_no_intent():
    pipe = VoicePipeline(
        capture=None,
        vad_factory=lambda: _ImmediateVAD(),
        transcriber=NullTranscriber("what's the weather"),
        intent_router=KeywordTimerRouter(),
        on_intent=lambda i, c: None,  # type: ignore[arg-type,return-value]
    )
    pcm = np.full(16000, 1000, dtype=np.int16).tobytes()
    text, intent = await pipe.submit_utterance(pcm, 16000, 1)
    assert text == "what's the weather"
    assert intent is None


async def test_submit_utterance_empty_pcm():
    pipe = VoicePipeline(
        capture=None,
        vad_factory=lambda: _ImmediateVAD(),
        transcriber=NullTranscriber("ignored"),
        intent_router=KeywordTimerRouter(),
        on_intent=lambda i, c: None,  # type: ignore[arg-type,return-value]
    )
    text, intent = await pipe.submit_utterance(b"", 16000, 1)
    assert text is None and intent is None


async def test_start_stop_noop_without_capture():
    pipe = VoicePipeline(
        capture=None,
        vad_factory=lambda: _ImmediateVAD(),
        transcriber=NullTranscriber(""),
        intent_router=KeywordTimerRouter(),
        on_intent=lambda i, c: None,  # type: ignore[arg-type,return-value]
    )
    await pipe.start()  # must not raise with no capture
    await pipe.stop()


async def test_start_stop_delegate_to_capture():
    cap = _NoCapture()
    pipe = VoicePipeline(
        capture=cap,
        vad_factory=lambda: _ImmediateVAD(),
        transcriber=NullTranscriber(""),
        intent_router=KeywordTimerRouter(),
        on_intent=lambda i, c: None,  # type: ignore[arg-type,return-value]
    )
    await pipe.start()
    await pipe.stop()
    assert cap.started and cap.stopped
