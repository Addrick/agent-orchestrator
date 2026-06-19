# src/voice/integration.py
"""ServiceIntegration for the voice command subsystem (DP-238).

Owns the shared ``TimerService`` (used by both the spoken pipeline and the
text-callable tools) and, when voice is enabled, the ``VoicePipeline``.

Mirrors the fixr pattern (DP-227/230): registered in ``main.py`` after the
NotificationRouter, with the Discord client late-bound via ``attach_discord``
(the client is constructed after services). Personas opt into the timer tools
via ``service_bindings: ["voice"]``.

Everything is default-off: with ``VOICE_ENABLED`` false (or no Discord client /
no configured channel) only the text tools are live — no voice channel is
joined, and the optional voice deps are never imported.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, TYPE_CHECKING

from config import global_config
from src.clients.service_integration import ServiceIntegration
from src.voice.intent import KeywordTimerRouter, TimerIntent
from src.voice.timer import Timer, TimerService
from src.voice.transcriber import MoonshineTranscriber
from src.voice.types import TimerTarget, VoiceCommand
from src.voice.vad import EnergyVAD

if TYPE_CHECKING:
    from src.clients.notification import NotificationRouter
    from src.tools.tool_manager import ToolManager
    from src.voice.pipeline import VoicePipeline

logger = logging.getLogger(__name__)


def _format_fire(timer: Timer) -> str:
    minutes, seconds = divmod(timer.seconds, 60)
    if minutes and seconds:
        dur = f"{minutes}m {seconds}s"
    elif minutes:
        dur = f"{minutes} minute{'s' if minutes != 1 else ''}"
    else:
        dur = f"{seconds} second{'s' if seconds != 1 else ''}"
    label = f" — {timer.label}" if timer.label else ""
    return f"⏰ Timer up ({dur}){label}!"


class VoiceIntegration(ServiceIntegration):
    def __init__(
        self,
        notification_router: "NotificationRouter",
        *,
        timer_service: Optional[TimerService] = None,
    ) -> None:
        self._notifier = notification_router
        self.timer_service = timer_service or TimerService(on_fire=self._on_fire)
        self._discord: Any = None
        self._pipeline: Optional["VoicePipeline"] = None
        self._start_task: Optional["asyncio.Task[None]"] = None

    @property
    def name(self) -> str:
        return "voice"

    def register_tools(self, tool_manager: "ToolManager") -> None:
        from src.voice.tools import VoiceTimerToolHandler
        VoiceTimerToolHandler(self.timer_service).register(tool_manager)

    # -- timer fire ----------------------------------------------------------

    async def _on_fire(self, timer: Timer) -> None:
        body = _format_fire(timer)
        if timer.target.mention_user_id:
            body = f"<@{timer.target.mention_user_id}> {body}"
        try:
            await self._notifier.send(
                channel=timer.target.channel,
                recipient=timer.target.recipient,
                subject="Timer",
                body=body,
            )
        except Exception:  # noqa: BLE001 - a failed ping must not crash the loop
            logger.exception("timer %s notification failed", timer.id)

    # -- voice pipeline (DP-238) --------------------------------------------

    def attach_discord(self, discord_client: Any) -> None:
        """Late-bind the Discord client and, if voice is enabled, build + start
        the capture pipeline. Called from ``_register_interfaces`` in main."""
        self._discord = discord_client
        if not self._voice_enabled():
            logger.info("Voice pipeline disabled (VOICE_ENABLED / channel unset).")
            return
        self._pipeline = self._build_pipeline()
        # capture.start() waits for the bot to be ready, so launch it detached.
        self._start_task = asyncio.create_task(self._start_pipeline())

    def _voice_enabled(self) -> bool:
        return bool(
            global_config.VOICE_ENABLED
            and self._discord is not None
            and global_config.VOICE_DISCORD_CHANNEL_ID
        )

    def _build_pipeline(self) -> "VoicePipeline":
        from src.voice.capture import DiscordVoiceCapture
        from src.voice.pipeline import VoicePipeline

        channel_id = int(global_config.VOICE_DISCORD_CHANNEL_ID)
        capture = DiscordVoiceCapture(
            self._discord, channel_id, on_frame=lambda f: self._pipeline.on_frame(f),  # type: ignore[union-attr]
        )
        return VoicePipeline(
            capture=capture,
            vad_factory=lambda: EnergyVAD(silence_ms=global_config.VOICE_VAD_SILENCE_MS),
            transcriber=MoonshineTranscriber(global_config.VOICE_STT_MODEL),
            intent_router=KeywordTimerRouter(
                wake_word=global_config.VOICE_WAKEWORD or None,
            ),
            on_intent=self._on_intent,
        )

    async def _start_pipeline(self) -> None:
        try:
            assert self._pipeline is not None
            await self._pipeline.start()
        except Exception:  # noqa: BLE001 - voice failing must not take down the bot
            logger.exception("voice pipeline failed to start")

    async def _on_intent(self, intent: TimerIntent, command: VoiceCommand) -> None:
        # Prefer the configured text channel; fall back to the source VC's id.
        recipient = global_config.VOICE_NOTIFY_CHANNEL_ID or (
            str(command.source_channel_id) if command.source_channel_id else ""
        )
        if not recipient:
            logger.warning("voice timer has no notify channel; dropping")
            return
        target = TimerTarget(
            channel="discord_channel",
            recipient=recipient,
            mention_user_id=command.user_id,
        )
        await self.timer_service.schedule(intent.seconds, target, label=intent.label)

    async def shutdown(self) -> None:
        if self._pipeline is not None:
            await self._pipeline.stop()
        await self.timer_service.shutdown()
