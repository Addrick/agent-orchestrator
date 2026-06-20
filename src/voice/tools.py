# src/voice/tools.py
"""Text-callable timer tools (DP-238), behind ``service_binding: "voice"``.

The same ``TimerService`` the voice pipeline uses is exposed to the LLM so a
persona can set/list/cancel timers from a typed conversation too — "remind me in
5 minutes" works in text exactly like the spoken command does in a voice
channel. The fire target for a text-set timer is the current turn's channel
(from ``TurnContext``) when it's a numeric id, else the configured default
notify channel.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from config import global_config
from src.tools.turn_context import get_turn_context
from src.voice.intent import parse_duration
from src.voice.timer import Timer, TimerService
from src.voice.types import TimerTarget

if TYPE_CHECKING:
    from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)


def _timer_view(t: Timer, now: float) -> Dict[str, Any]:
    return {
        "id": t.id,
        "seconds": t.seconds,
        "remaining_seconds": int(t.remaining(now)),
        "label": t.label,
    }


class VoiceTimerToolHandler:
    def __init__(self, timer_service: TimerService) -> None:
        self._timers = timer_service

    def register(self, manager: "ToolManager") -> None:
        manager.register("set_timer", self._set_timer)
        manager.register("list_timers", self._list_timers)
        manager.register("cancel_timer", self._cancel_timer)

    def _default_target(self) -> Optional[TimerTarget]:
        """Fire a text-set timer back through the channel the turn came from.

        - A portal turn carries a ``web_ui``-prefixed channel tag (DP-136); it has
          no Discord id, so the alarm routes to the ``web`` NotificationRouter
          channel (an SSE push back to the portal) with the tag as the recipient,
          so it surfaces in the same conversation it was set from.
        - A Discord turn carries a numeric channel id → announce there.
        - Otherwise fall back to the configured ``VOICE_NOTIFY_CHANNEL_ID``.
        """
        ctx = get_turn_context()
        if ctx is not None and ctx.channel:
            if ctx.channel.startswith("web_ui"):
                return TimerTarget(channel="web", recipient=ctx.channel)
            if ctx.channel.isdigit():
                return TimerTarget(channel="discord_channel", recipient=ctx.channel)
        if global_config.VOICE_NOTIFY_CHANNEL_ID:
            return TimerTarget(
                channel="discord_channel",
                recipient=str(global_config.VOICE_NOTIFY_CHANNEL_ID),
            )
        return None

    async def _set_timer(
        self, duration: str, label: Optional[str] = None,
    ) -> Dict[str, Any]:
        seconds = parse_duration(duration)
        if seconds is None:
            return {
                "status": "error",
                "message": f"Could not parse a duration from {duration!r} "
                           f"(try '10 minutes', '30 seconds').",
            }
        target = self._default_target()
        if target is None:
            return {
                "status": "error",
                "message": "No channel to announce the timer in. Set "
                           "VOICE_NOTIFY_CHANNEL_ID or run from a channel.",
            }
        timer = await self._timers.schedule(seconds, target, label=label)
        return {"status": "scheduled", "id": timer.id, "seconds": seconds, "label": label}

    async def _list_timers(self) -> Dict[str, Any]:
        import time
        now = time.time()
        timers = self._timers.list()
        return {"count": len(timers), "timers": [_timer_view(t, now) for t in timers]}

    async def _cancel_timer(self, timer_id: str) -> Dict[str, Any]:
        ok = await self._timers.cancel(timer_id)
        return {"status": "cancelled" if ok else "not_found", "id": timer_id}
