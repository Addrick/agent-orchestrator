# src/voice/alarm_bus.py
"""In-process fan-out for fired-timer alarms destined for the web portal (DP-238).

A timer set from the portal has no Discord channel to ping — its ``TimerTarget``
carries ``channel="web"`` and the originating ``web_ui`` channel tag as the
recipient. When such a timer fires, ``VoiceIntegration._on_fire`` routes it
through the ``NotificationRouter`` exactly like a Discord ping, but the notifier
registered for the ``web`` channel is ``WebAlarmNotifier``: instead of a network
send it publishes the alarm onto this bus. The portal subscribes via the
``GET /voice/alarms`` SSE endpoint and surfaces each alarm as a chat line + beep.

The bus is a tiny pub/sub over per-subscriber ``asyncio.Queue``s — no external
broker, no persistence. An alarm that fires while no browser is connected is
dropped (bare-minimum: a fired timer only matters to a live listener). Bounded
queues mean a stuck subscriber drops the oldest pending alarm rather than growing
without limit.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Set

from src.clients.notification import Notifier

logger = logging.getLogger(__name__)

# Per-subscriber backlog cap. A connected-but-stalled browser drops its oldest
# pending alarm past this — alarms are ephemeral, so a backlog is never worth
# unbounded memory.
_QUEUE_MAXSIZE = 32


class AlarmBus:
    """Fan-out of fired-timer alarms to every connected portal SSE stream."""

    def __init__(self) -> None:
        self._subscribers: Set["asyncio.Queue[Dict[str, Any]]"] = set()

    def subscribe(self) -> "asyncio.Queue[Dict[str, Any]]":
        q: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue[Dict[str, Any]]") -> None:
        self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    async def publish(self, event: Dict[str, Any]) -> None:
        """Deliver ``event`` to every live subscriber (non-blocking).

        A full queue means that browser is wedged — drop the oldest alarm to make
        room rather than block the timer-fire path on one stuck client.
        """
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):  # pragma: no cover - race
                    pass


class WebAlarmNotifier(Notifier):
    """``NotificationRouter`` channel ``web`` → publish onto an ``AlarmBus``.

    Mirrors the ``Notifier`` contract so a web-targeted timer fires through the
    same ``_on_fire`` path as a Discord one. ``recipient`` is the originating
    ``web_ui`` channel tag, forwarded so the portal can show the alarm only in the
    channel it was set in.
    """

    def __init__(self, bus: AlarmBus) -> None:
        self._bus = bus

    async def send(self, recipient: str, subject: str, body: str) -> bool:
        await self._bus.publish({
            "channel": recipient,
            "subject": subject,
            "text": body,
        })
        return True
