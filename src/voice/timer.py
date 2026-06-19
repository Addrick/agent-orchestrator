# src/voice/timer.py
"""Async timer scheduling (DP-238).

``TimerService`` owns the live set of pending timers. Each timer is one asyncio
task that sleeps until its deadline, then invokes an injected ``on_fire``
callback (the integration wires this to a Discord ping). ``clock`` and ``sleep``
are injectable so tests drive fired/cancelled behaviour without real time.

Used by two callers that share one service instance:
- the voice pipeline (``KeywordTimerRouter`` → ``schedule``), and
- the text-callable ``set_timer``/``list_timers``/``cancel_timer`` tools.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from src.voice.types import TimerTarget

logger = logging.getLogger(__name__)


@dataclass
class Timer:
    id: str
    seconds: int
    fire_at: float
    target: TimerTarget
    label: Optional[str] = None
    _task: Optional["asyncio.Task[None]"] = field(default=None, repr=False)

    def remaining(self, now: float) -> float:
        return max(0.0, self.fire_at - now)


OnFire = Callable[[Timer], Awaitable[None]]


class TimerService:
    def __init__(
        self,
        on_fire: OnFire,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._on_fire = on_fire
        self._clock = clock
        self._sleep = sleep
        self._timers: Dict[str, Timer] = {}

    async def schedule(
        self,
        seconds: int,
        target: TimerTarget,
        *,
        label: Optional[str] = None,
    ) -> Timer:
        if seconds <= 0:
            raise ValueError("timer duration must be positive")
        timer = Timer(
            id=uuid.uuid4().hex[:8],
            seconds=seconds,
            fire_at=self._clock() + seconds,
            target=target,
            label=label,
        )
        self._timers[timer.id] = timer
        timer._task = asyncio.create_task(self._run(timer))
        logger.info("Scheduled timer %s for %ds (label=%r)", timer.id, seconds, label)
        return timer

    async def _run(self, timer: Timer) -> None:
        try:
            await self._sleep(timer.seconds)
        except asyncio.CancelledError:
            return
        # Drop from the live set before firing so a re-entrant list() is accurate.
        self._timers.pop(timer.id, None)
        try:
            await self._on_fire(timer)
        except Exception:  # noqa: BLE001 - a failed fire must not crash the loop
            logger.exception("timer %s on_fire callback failed", timer.id)

    async def cancel(self, timer_id: str) -> bool:
        timer = self._timers.pop(timer_id, None)
        if timer is None:
            return False
        if timer._task is not None and not timer._task.done():
            timer._task.cancel()
        logger.info("Cancelled timer %s", timer_id)
        return True

    def list(self) -> List[Timer]:
        return sorted(self._timers.values(), key=lambda t: t.fire_at)

    def get(self, timer_id: str) -> Optional[Timer]:
        return self._timers.get(timer_id)

    async def shutdown(self) -> None:
        for timer in list(self._timers.values()):
            if timer._task is not None and not timer._task.done():
                timer._task.cancel()
        self._timers.clear()
