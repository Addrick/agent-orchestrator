# src/voice/pipeline.py
"""The voice pipeline orchestrator (DP-238).

Wires the stages together:

    capture → (per-user) resample → VAD → transcribe → intent → on_intent

Each speaker gets their own VAD instance (so two people talking don't merge into
one utterance). When the VAD closes an utterance the audio is transcribed, the
text routed through the intent router, and any matched intent handed to the
``on_intent`` callback (the integration schedules the timer). Every stage is the
injected ABC, so swapping capture source / STT model / intent strategy is a
constructor change, not a rewrite.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable, Dict, Optional, Tuple

from src.voice.capture import CaptureSource
from src.voice.intent import IntentRouter, TimerIntent
from src.voice.resample import to_16k_mono
from src.voice.transcriber import Transcriber
from src.voice.types import AudioFrame, Utterance, VoiceCommand
from src.voice.vad import VAD

logger = logging.getLogger(__name__)

OnIntent = Callable[[TimerIntent, VoiceCommand], Awaitable[None]]
VADFactory = Callable[[], VAD]


class VoicePipeline:
    def __init__(
        self,
        *,
        capture: Optional[CaptureSource] = None,
        vad_factory: VADFactory,
        transcriber: Transcriber,
        intent_router: IntentRouter,
        on_intent: OnIntent,
    ) -> None:
        # ``capture`` is the streaming pull source (Discord VC). It is optional:
        # the browser push-to-talk path (DP-238 web) hands complete utterances to
        # ``submit_utterance`` and has no always-on capture to start.
        self._capture = capture
        self._vad_factory = vad_factory
        self._transcriber = transcriber
        self._intent_router = intent_router
        self._on_intent = on_intent
        self._vads: Dict[Optional[int], VAD] = {}
        # DP-238d diagnostics: prove where the spoken path stops (receive → VAD →
        # STT). Cheap counters; the first-frame/first-utterance logs fire once.
        self._frames_seen = 0
        self._utterances_seen = 0

    async def warmup(self) -> None:
        """Pre-load the STT model so the first utterance isn't lost to cold-start."""
        await self._transcriber.warmup()

    async def start(self) -> None:
        if self._capture is not None:
            await self._capture.start()

    async def stop(self) -> None:
        if self._capture is not None:
            await self._capture.stop()
        # Flush any speech still buffered per speaker.
        for user_id, vad in list(self._vads.items()):
            tail = vad.flush()
            if tail:
                await self._handle_utterance(Utterance(pcm16k_mono=tail, user_id=user_id))
        self._vads.clear()

    async def on_frame(self, frame: AudioFrame) -> None:
        """Capture callback. One frame of raw PCM for one speaker."""
        self._frames_seen += 1
        if self._frames_seen == 1:
            logger.info("Voice: first audio frame received (user=%s)", frame.user_id)
        mono16k = to_16k_mono(frame.pcm, frame.sample_rate, frame.channels)
        if not mono16k:
            return
        vad = self._vads.get(frame.user_id)
        if vad is None:
            vad = self._vad_factory()
            self._vads[frame.user_id] = vad
        completed = vad.add_chunk(mono16k)
        if completed:
            await self._handle_utterance(
                Utterance(
                    pcm16k_mono=completed,
                    user_id=frame.user_id,
                    source_channel_id=frame.source_channel_id,
                )
            )

    async def submit_utterance(
        self,
        pcm: bytes,
        sample_rate: int,
        channels: int,
        *,
        user_id: Optional[int] = None,
        source_channel_id: Optional[int] = None,
    ) -> Tuple[Optional[str], Optional[TimerIntent]]:
        """Push a *complete* pre-segmented utterance (DP-238 web push-to-talk).

        The browser delimits the utterance with the talk button, so there is no
        VAD in this path — the raw PCM is resampled to 16 kHz mono and routed
        straight through transcribe → intent. Returns ``(transcript, intent)``:
        ``transcript`` is ``None`` when nothing was transcribed, ``intent`` is the
        matched ``TimerIntent`` (already scheduled) or ``None``. The caller echoes
        both back to the browser.
        """
        mono16k = to_16k_mono(pcm, sample_rate, channels)
        if not mono16k:
            return None, None
        return await self._handle_utterance(
            Utterance(pcm16k_mono=mono16k, user_id=user_id, source_channel_id=source_channel_id)
        )

    def dictation_stream(self) -> "DictationStream":
        """A per-connection streaming dictation session (DP-238 item B): its own
        VAD endpoints continuous mic frames into utterances, each transcribed
        WITHOUT intent routing (the LLM owns intents, as with ``transcribe``).
        One per WebSocket so two clients never share VAD state."""
        return DictationStream(self._vad_factory(), self._transcriber)

    async def transcribe(
        self, pcm: bytes, sample_rate: int, channels: int
    ) -> Optional[str]:
        """Transcribe a complete utterance WITHOUT intent routing (DP-238 web
        dictation). Used by the SPA mic button: the transcript is dropped into
        the composer so the LLM — which already owns the timer tools — handles
        any command, rather than the keyword router second-guessing it. Returns
        the stripped transcript or ``None`` when nothing was transcribed."""
        mono16k = to_16k_mono(pcm, sample_rate, channels)
        if not mono16k:
            return None
        text = (await self._transcriber.transcribe(mono16k)).strip()
        return text or None

    async def _handle_utterance(
        self, utterance: Utterance
    ) -> Tuple[Optional[str], Optional[TimerIntent]]:
        self._utterances_seen += 1
        logger.info(
            "Voice: VAD closed utterance #%d (user=%s, %d bytes); transcribing…",
            self._utterances_seen, utterance.user_id, len(utterance.pcm16k_mono),
        )
        text = (await self._transcriber.transcribe(utterance.pcm16k_mono)).strip()
        if not text:
            logger.info("Voice: transcription empty (user=%s)", utterance.user_id)
            return None, None
        logger.info("Voice utterance (user=%s): %r", utterance.user_id, text)
        command = VoiceCommand(
            raw_text=text,
            user_id=utterance.user_id,
            source_channel_id=utterance.source_channel_id,
        )
        intent = await self._intent_router.route(text)
        if intent is None:
            logger.info("Voice: no timer intent in %r (needs 'timer'/'alarm' + duration)", text)
            return text, None
        logger.info("Voice: matched timer intent %ds (user=%s)", intent.seconds, utterance.user_id)
        try:
            await self._on_intent(intent, command)
        except Exception:  # noqa: BLE001 - an action failure must not kill the pipeline
            logger.exception("voice intent handler failed")
        return text, intent


class DictationStream:
    """One streaming dictation session (DP-238 item B). Continuous mic frames are
    resampled to 16 kHz mono and fed to a private ``VAD`` that endpoints them into
    utterances on trailing silence; each closed utterance is transcribed (no intent
    routing — the LLM owns intents). Stateful and single-speaker, so the caller
    builds one per WebSocket connection.

    Endpointing quality is bounded by the VAD: too short a silence cuts mid-sentence,
    too long adds latency (tuned via ``VOICE_VAD_SILENCE_MS``). A smarter
    "done talking?" endpointer (Silero VAD / a tiny LLM turn-end classifier /
    streaming partial transcripts) is a known follow-up — see tasks/DP-238.md.
    """

    def __init__(self, vad: VAD, transcriber: Transcriber) -> None:
        self._vad = vad
        self._transcriber = transcriber

    async def push(self, pcm: bytes, sample_rate: int, channels: int) -> Optional[str]:
        """Feed one frame. Returns the transcript when this frame closes an
        utterance (trailing silence reached), else ``None``."""
        mono16k = to_16k_mono(pcm, sample_rate, channels)
        if not mono16k:
            return None
        completed = self._vad.add_chunk(mono16k)
        if not completed:
            return None
        return (await self._transcriber.transcribe(completed)).strip() or None

    async def flush(self) -> Optional[str]:
        """Transcribe any speech still buffered (e.g. on disconnect)."""
        tail = self._vad.flush()
        if not tail:
            return None
        return (await self._transcriber.transcribe(tail)).strip() or None
