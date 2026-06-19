# src/voice/__init__.py
"""Voice command subsystem (DP-238).

Discord always-listening → VAD-segmented utterances → Moonshine STT (CPU) →
keyword intent → timer, with a text-callable timer tool sharing the same
service. Stage ABCs (CaptureSource / VAD / Transcriber / IntentRouter) make the
later sources (wake-word, push-to-talk, mic endpoints), STT backends, and intent
strategies drop-in replacements.

Importing this package never requires the optional voice deps
(``discord-ext-voice-recv``, Moonshine) — those are lazy-loaded only when voice
is enabled and capture/transcription actually run.
"""
from src.voice.integration import VoiceIntegration

__all__ = ["VoiceIntegration"]
