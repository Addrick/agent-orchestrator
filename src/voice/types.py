# src/voice/types.py
"""Value types passed between the voice-pipeline stages (DP-238).

Frozen dataclasses so a frame/utterance/command can't be mutated as it flows
capture → VAD → transcribe → intent. Audio is raw little-endian 16-bit signed
PCM throughout; the sample rate / channel count travel with the frame so a
stage never has to assume them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AudioFrame:
    """A chunk of PCM straight off a capture source (pre-resample)."""

    pcm: bytes
    sample_rate: int
    channels: int
    user_id: Optional[int] = None
    source_channel_id: Optional[int] = None


@dataclass(frozen=True)
class Utterance:
    """A complete speech segment (one VAD-bounded utterance), 16 kHz mono."""

    pcm16k_mono: bytes
    user_id: Optional[int] = None
    source_channel_id: Optional[int] = None


@dataclass(frozen=True)
class VoiceCommand:
    """A transcribed utterance ready for intent routing."""

    raw_text: str
    user_id: Optional[int] = None
    source_channel_id: Optional[int] = None


@dataclass(frozen=True)
class TimerTarget:
    """Where a fired timer announces itself.

    ``channel`` is a NotificationRouter channel key (e.g. ``discord_channel``);
    ``recipient`` is that channel's id. ``mention_user_id`` optionally pings the
    person who set it.
    """

    channel: str
    recipient: str
    mention_user_id: Optional[int] = None
