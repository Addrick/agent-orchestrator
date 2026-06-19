# src/voice/resample.py
"""Downmix + resample PCM to what the STT model wants (DP-238).

Discord voice-recv hands us 48 kHz **stereo** 16-bit PCM; Moonshine (and every
Whisper-family model) wants 16 kHz **mono**. numpy is already a hard dep of the
project, so we lean on it rather than the removed stdlib ``audioop``.

The resample is a straight linear interpolation — adequate for a 2-second voice
command, and STT accuracy here is not gated by resampler quality. A fancier
polyphase filter can replace this behind the same signature if needed.
"""
from __future__ import annotations

import numpy as np

TARGET_RATE = 16000


def to_16k_mono(pcm: bytes, sample_rate: int, channels: int) -> bytes:
    """Convert interleaved int16 PCM to 16 kHz mono int16 PCM.

    Returns ``b""`` for empty input. Safe for mono input (skips the downmix) and
    for input already at 16 kHz (skips the resample)."""
    if not pcm:
        return b""
    samples = np.frombuffer(pcm, dtype=np.int16)
    if channels > 1:
        # Trim a trailing partial frame, then average channels → mono.
        usable = (samples.size // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    mono = samples.astype(np.float32)
    if mono.size == 0:
        return b""
    if sample_rate != TARGET_RATE:
        n_out = int(round(mono.size * TARGET_RATE / sample_rate))
        if n_out <= 0:
            return b""
        # Linear interpolation onto the target grid.
        src_idx = np.linspace(0.0, mono.size - 1, num=n_out, dtype=np.float64)
        mono = np.interp(src_idx, np.arange(mono.size), mono).astype(np.float32)
    return np.clip(mono, -32768, 32767).astype(np.int16).tobytes()
