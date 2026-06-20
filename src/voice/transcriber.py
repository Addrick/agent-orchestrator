# src/voice/transcriber.py
"""Speech-to-text backends (DP-238).

``Transcriber`` is the swap point for the STT model. The MVP uses Moonshine on
CPU (the idle 5800X on the .70 host) — purpose-built for short voice commands,
~100 ms latency, CPU-first, so it never touches the R9700 that serves the prod
gemma model. A whisper.cpp-Vulkan backend for long-form dictation on the R9700
can be added later behind this same ABC.

The heavy ``moonshine_onnx`` import is lazy and the model loads once on first use, so
importing this module (and the whole ``src.voice`` package) costs nothing and
never fails when the optional dependency is absent — ``main.py`` and the unit
tests import it unconditionally.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_TranscribeFn = Callable[[bytes], str]

# Map the friendly tier names Adam configures to Moonshine model ids.
_MOONSHINE_MODELS = {
    "tiny": "moonshine/tiny",
    "base": "moonshine/base",
}


class Transcriber(ABC):
    @abstractmethod
    async def transcribe(self, pcm16k_mono: bytes) -> str:
        """Transcribe one 16 kHz mono int16 PCM utterance to text."""

    async def warmup(self) -> None:
        """Load the model and run one throwaway inference so the first *real*
        utterance isn't dropped or stalled by cold-start. No-op by default."""
        return None


class NullTranscriber(Transcriber):
    """Test/dev double. Returns scripted text (or a fixed string) — no model.

    Pass a list to pop one transcript per call (raises nothing when exhausted —
    returns ``""``), or a single string to return every call.
    """

    def __init__(self, scripted: "str | List[str]" = "") -> None:
        self._fixed: Optional[str]
        self._queue: Optional[List[str]]
        if isinstance(scripted, list):
            self._fixed = None
            self._queue = list(scripted)
        else:
            self._fixed = scripted
            self._queue = None
        self.calls: List[bytes] = []

    async def transcribe(self, pcm16k_mono: bytes) -> str:
        self.calls.append(pcm16k_mono)
        if self._queue is not None:
            return self._queue.pop(0) if self._queue else ""
        return self._fixed or ""


class MoonshineTranscriber(Transcriber):
    """Moonshine STT on CPU. Lazy-loads the model; runs inference in a worker
    thread so the event loop (Discord, the engine) is never blocked."""

    def __init__(self, model: str = "base") -> None:
        self._model_id = _MOONSHINE_MODELS.get(model, _MOONSHINE_MODELS["base"])
        self._transcribe_fn: Optional[_TranscribeFn] = None  # loaded on first use
        self._load_lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._transcribe_fn is not None:
            return
        async with self._load_lock:
            if self._transcribe_fn is None:
                self._transcribe_fn = await asyncio.to_thread(self._load)

    def _load(self) -> _TranscribeFn:
        try:
            import moonshine_onnx as moonshine
        except ImportError as e:  # pragma: no cover - exercised only without the dep
            raise RuntimeError(
                "Moonshine is not installed. `pip install useful-moonshine-onnx` "
                "(CPU onnx backend) to enable voice transcription."
            ) from e
        model_id = self._model_id

        def _run(pcm16k_mono: bytes) -> str:
            audio = np.frombuffer(pcm16k_mono, dtype=np.int16).astype(np.float32) / 32768.0
            result = moonshine.transcribe(audio, model_id)
            if isinstance(result, (list, tuple)):
                return " ".join(str(r) for r in result).strip()
            return str(result).strip()

        return _run

    async def transcribe(self, pcm16k_mono: bytes) -> str:
        if not pcm16k_mono:
            return ""
        await self._ensure_loaded()
        assert self._transcribe_fn is not None
        try:
            return await asyncio.to_thread(self._transcribe_fn, pcm16k_mono)
        except Exception:  # noqa: BLE001 - a bad clip must not kill the pipeline
            logger.exception("Moonshine transcription failed")
            return ""

    async def warmup(self) -> None:
        """Load the model + run one inference on 0.5 s of silence. The first
        onnxruntime inference compiles/optimizes the graph (seconds) and the
        model download happens here too — doing it at startup means the first
        spoken command is fast and isn't lost to cold-start (DP-238 web)."""
        try:
            loop = asyncio.get_event_loop()
            start = loop.time()
            await self._ensure_loaded()
            assert self._transcribe_fn is not None
            silence = np.zeros(8000, dtype=np.int16).tobytes()  # 0.5 s @ 16 kHz
            await asyncio.to_thread(self._transcribe_fn, silence)
            logger.info(
                "Moonshine (%s) warmed up in %.1fs", self._model_id, loop.time() - start
            )
        except Exception:  # noqa: BLE001 - warmup is best-effort, never fatal
            logger.exception("Moonshine warmup failed (will lazy-load on first use)")
