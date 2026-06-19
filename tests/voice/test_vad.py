# tests/voice/test_vad.py
import numpy as np

from src.voice.vad import EnergyVAD

_FRAME = 320  # samples @16k = 20ms


def _loud(frames):
    return np.full(_FRAME * frames, 8000, dtype=np.int16).tobytes()


def _quiet(frames):
    return np.zeros(_FRAME * frames, dtype=np.int16).tobytes()


def test_speech_then_silence_emits_utterance():
    vad = EnergyVAD(silence_ms=100, min_speech_ms=40)  # 5 silence frames, 2 speech
    assert vad.add_chunk(_loud(10)) is None  # speaking, no close yet
    out = vad.add_chunk(_quiet(6))  # 6 silent frames > 5 needed -> close
    assert out is not None
    assert len(out) > 0


def test_short_blip_discarded():
    vad = EnergyVAD(silence_ms=100, min_speech_ms=200)  # need 10 speech frames
    vad.add_chunk(_loud(2))  # only 2 speech frames
    out = vad.add_chunk(_quiet(6))
    assert out is None  # below min_speech -> dropped


def test_pure_silence_never_emits():
    vad = EnergyVAD(silence_ms=100)
    assert vad.add_chunk(_quiet(20)) is None


def test_flush_emits_buffered_speech():
    vad = EnergyVAD(silence_ms=2000, min_speech_ms=40)
    vad.add_chunk(_loud(10))
    out = vad.flush()
    assert out is not None and len(out) > 0


def test_max_utterance_caps():
    vad = EnergyVAD(silence_ms=5000, min_speech_ms=40, max_utterance_ms=100)  # ~5 frames cap
    out = vad.add_chunk(_loud(20))
    assert out is not None  # hit the hard cap mid-speech


def test_sub_frame_chunks_accumulate():
    vad = EnergyVAD(silence_ms=100, min_speech_ms=40)
    # feed 10ms (half-frame) pieces of loud audio
    half = np.full(_FRAME // 2, 8000, dtype=np.int16).tobytes()
    for _ in range(20):
        vad.add_chunk(half)
    out = vad.add_chunk(_quiet(6))
    assert out is not None
