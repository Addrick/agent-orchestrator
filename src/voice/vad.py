# src/voice/vad.py
"""Voice activity detection — segment a PCM stream into utterances (DP-238).

A ``VAD`` is fed mono 16 kHz int16 PCM in arbitrary-sized chunks and emits a
complete utterance (the buffered speech) once it sees enough trailing silence.
One VAD instance is stateful and per-speaker; the pipeline keeps one per user.

``EnergyVAD`` is a dependency-free RMS-threshold detector — good enough to bound
a short spoken command. ``SileroVAD`` (neural, far better in noise) can drop in
behind the same ABC later; it's intentionally not built yet to avoid the torch
dependency on the always-on path.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

# A VAD "frame" granularity. 16 kHz int16 → 320 samples = 20 ms.
_FRAME_SAMPLES = 320
_BYTES_PER_SAMPLE = 2


class VAD(ABC):
    """Stateful per-speaker speech segmenter."""

    @abstractmethod
    def add_chunk(self, pcm16k_mono: bytes) -> Optional[bytes]:
        """Feed mono 16 kHz PCM. Returns a completed utterance's PCM when a
        speech segment ends, else None."""

    @abstractmethod
    def flush(self) -> Optional[bytes]:
        """Force-emit any buffered speech (e.g. on capture stop)."""


class EnergyVAD(VAD):
    """RMS-energy VAD with a trailing-silence hangover.

    Parameters
    ----------
    rms_threshold: int
        Per-frame RMS above which a 20 ms frame counts as speech.
    silence_ms: int
        Trailing silence that closes an utterance.
    min_speech_ms: int
        Discard segments shorter than this (coughs, clicks, key taps).
    max_utterance_ms: int
        Hard cap so a noisy room can't buffer forever.
    """

    def __init__(
        self,
        *,
        rms_threshold: int = 500,
        silence_ms: int = 700,
        min_speech_ms: int = 250,
        max_utterance_ms: int = 15000,
    ) -> None:
        self._rms_threshold = rms_threshold
        self._silence_frames_needed = max(1, silence_ms // 20)
        self._min_speech_frames = max(1, min_speech_ms // 20)
        self._max_frames = max(1, max_utterance_ms // 20)
        self._buf = bytearray()
        self._tail = bytearray()  # carries a sub-frame remainder between chunks
        self._in_speech = False
        self._silence_run = 0
        self._speech_frames = 0

    @staticmethod
    def _rms(frame: bytes) -> float:
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float64)
        if samples.size == 0:
            return 0.0
        return math.sqrt(float(np.mean(samples * samples)))

    def add_chunk(self, pcm16k_mono: bytes) -> Optional[bytes]:
        if not pcm16k_mono:
            return None
        self._tail.extend(pcm16k_mono)
        frame_bytes = _FRAME_SAMPLES * _BYTES_PER_SAMPLE
        completed: Optional[bytes] = None
        while len(self._tail) >= frame_bytes:
            frame = bytes(self._tail[:frame_bytes])
            del self._tail[:frame_bytes]
            emitted = self._consume_frame(frame)
            if emitted is not None:
                completed = emitted  # keep the most recent; chunks rarely close >1
        return completed

    def _consume_frame(self, frame: bytes) -> Optional[bytes]:
        is_speech = self._rms(frame) >= self._rms_threshold
        if is_speech:
            self._in_speech = True
            self._silence_run = 0
            self._speech_frames += 1
            self._buf.extend(frame)
        elif self._in_speech:
            # Hangover: keep buffering silence until the gap is long enough.
            self._silence_run += 1
            self._buf.extend(frame)
            if self._silence_run >= self._silence_frames_needed:
                return self._close()
        # Hard cap regardless of trailing silence.
        if self._in_speech and len(self._buf) >= self._max_frames * len(frame):
            return self._close()
        return None

    def _close(self) -> Optional[bytes]:
        speech_frames = self._speech_frames
        out = bytes(self._buf)
        self._buf.clear()
        self._in_speech = False
        self._silence_run = 0
        self._speech_frames = 0
        if speech_frames < self._min_speech_frames:
            return None
        return out

    def flush(self) -> Optional[bytes]:
        if self._in_speech:
            return self._close()
        return None
