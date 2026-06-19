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
from typing import Awaitable, Callable, Dict, Optional

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
        capture: CaptureSource,
        vad_factory: VADFactory,
        transcriber: Transcriber,
        intent_router: IntentRouter,
        on_intent: OnIntent,
    ) -> None:
        self._capture = capture
        self._vad_factory = vad_factory
        self._transcriber = transcriber
        self._intent_router = intent_router
        self._on_intent = on_intent
        self._vads: Dict[Optional[int], VAD] = {}

    async def start(self) -> None:
        await self._capture.start()

    async def stop(self) -> None:
        await self._capture.stop()
        # Flush any speech still buffered per speaker.
        for user_id, vad in list(self._vads.items()):
            tail = vad.flush()
            if tail:
                await self._handle_utterance(Utterance(pcm16k_mono=tail, user_id=user_id))
        self._vads.clear()

    async def on_frame(self, frame: AudioFrame) -> None:
        """Capture callback. One frame of raw PCM for one speaker."""
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

    async def _handle_utterance(self, utterance: Utterance) -> None:
        text = (await self._transcriber.transcribe(utterance.pcm16k_mono)).strip()
        if not text:
            return
        logger.info("Voice utterance (user=%s): %r", utterance.user_id, text)
        command = VoiceCommand(
            raw_text=text,
            user_id=utterance.user_id,
            source_channel_id=utterance.source_channel_id,
        )
        intent = await self._intent_router.route(text)
        if intent is None:
            return
        try:
            await self._on_intent(intent, command)
        except Exception:  # noqa: BLE001 - an action failure must not kill the pipeline
            logger.exception("voice intent handler failed")
