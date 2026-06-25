# src/voice/capture.py
"""Audio capture sources (DP-238).

``CaptureSource`` is the swap point for *where speech comes from*. The MVP
``DiscordVoiceCapture`` joins a Discord voice channel and always-listens; later
sources (a wake-word mic, push-to-talk, a Pi/ESP32/phone push endpoint) are new
implementations behind the same ABC, feeding the same pipeline.

Stock discord.py can only *send* voice, so receiving needs the community
extension ``discord-ext-voice-recv``. Its import is lazy: importing this module
(and ``src.voice``) never requires the optional dep, and ``main.py`` can build
the integration unconditionally — the dep is only touched when voice is actually
enabled and ``start()`` is called.

The voice-recv sink callback runs on a non-async thread, so frames are bridged
back onto the bot's event loop with ``run_coroutine_threadsafe``.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Optional

from src.voice.types import AudioFrame

logger = logging.getLogger(__name__)

OnFrame = Callable[[AudioFrame], Awaitable[None]]

# Discord voice-recv decodes to this format.
_DISCORD_RATE = 48000
_DISCORD_CHANNELS = 2


_opus_resilience_patched = False


def _patch_voice_recv_opus_resilience() -> None:
    """Stop a single corrupted Opus packet from killing voice receive.

    discord-ext-voice-recv (experimental alpha) runs Opus decode in a
    ``PacketRouter`` thread whose ``run()`` has no per-packet error handling: any
    ``OpusError`` ("corrupted stream") propagates out of the loop, and the
    ``finally`` calls ``stop_listening()`` — permanently tearing down receive for
    the whole session (upstream issue #27 "stops listening even with the example
    script"). The error is upstream of our sink, so we can't catch it there.

    Wrap ``PacketDecoder.pop_data`` to drop a bad packet and return ``None`` (which
    the router already treats as "no data this tick"), so the loop survives.
    Idempotent; a no-op if the library internals move.
    """
    global _opus_resilience_patched
    if _opus_resilience_patched:
        return
    try:
        from discord.ext.voice_recv.opus import PacketDecoder
        from discord.opus import OpusError
    except Exception:  # pragma: no cover - lib missing / internals changed
        return

    original = PacketDecoder.pop_data

    def _safe_pop_data(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return original(self, *args, **kwargs)
        except OpusError:
            logger.warning("Dropped corrupted Opus packet (voice-recv decode error)")
            return None

    PacketDecoder.pop_data = _safe_pop_data  # type: ignore[method-assign]
    _opus_resilience_patched = True
    logger.info("Applied voice-recv Opus-decode resilience patch")


class CaptureSource(ABC):
    @abstractmethod
    async def start(self) -> None:
        ...

    @abstractmethod
    async def stop(self) -> None:
        ...


class DiscordVoiceCapture(CaptureSource):
    def __init__(
        self,
        discord_client: Any,
        channel_id: int,
        on_frame: OnFrame,
    ) -> None:
        self._client = discord_client
        self._channel_id = channel_id
        self._on_frame = on_frame
        self._voice: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        try:
            from discord.ext import voice_recv
        except ImportError as e:  # pragma: no cover - only without the dep
            raise RuntimeError(
                "discord-ext-voice-recv is not installed. "
                "`pip install discord-ext-voice-recv` (and libopus/PyNaCl) to "
                "enable voice capture."
            ) from e

        _patch_voice_recv_opus_resilience()

        # ``attach_discord`` launches this before the Discord client's ``start()``
        # task has run, so the client may not be logged in yet. ``wait_until_ready``
        # *raises* ("Client has not been properly initialised") if called pre-login,
        # whereas ``is_ready`` only returns False — poll it until the gateway is up.
        while not self._client.is_ready():
            await asyncio.sleep(0.2)
        channel = self._client.get_channel(self._channel_id) or await self._client.fetch_channel(
            self._channel_id
        )
        if channel is None:
            raise RuntimeError(f"voice channel {self._channel_id} not found")

        self._loop = asyncio.get_running_loop()
        self._voice = await channel.connect(cls=voice_recv.VoiceRecvClient)

        def _sink_callback(user: Any, data: Any) -> None:
            # Runs off-loop (voice-recv reader thread). Bounce to the loop.
            pcm = getattr(data, "pcm", None)
            if not pcm or self._loop is None:
                return
            frame = AudioFrame(
                pcm=pcm,
                sample_rate=_DISCORD_RATE,
                channels=_DISCORD_CHANNELS,
                user_id=getattr(user, "id", None),
                source_channel_id=self._channel_id,
            )
            asyncio.run_coroutine_threadsafe(self._dispatch(frame), self._loop)

        self._voice.listen(voice_recv.BasicSink(_sink_callback))
        logger.info("Voice capture listening in channel %s", self._channel_id)

    async def _dispatch(self, frame: AudioFrame) -> None:
        try:
            await self._on_frame(frame)
        except Exception:  # noqa: BLE001 - a bad frame must not kill capture
            logger.exception("voice frame handler failed")

    async def stop(self) -> None:
        if self._voice is not None:
            try:
                self._voice.stop_listening()
            except Exception:  # noqa: BLE001
                pass
            try:
                await self._voice.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self._voice = None
        logger.info("Voice capture stopped for channel %s", self._channel_id)
