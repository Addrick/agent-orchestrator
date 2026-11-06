# src/interfaces/memory_rebuild.py
"""Portal "purge & rebuild all persona memory" job (DP-506).

The Imports panel's button. The work itself is
``ChatSystem.rebuild_persona_memory``, handed in as a callable; this module
only runs it for every memory persona as one background job, holds the
progress the panel polls, and guards the destructive entry point:

- the request must carry ``confirm: "PURGE"`` (the phrase the dialog makes
  the operator type), so a stray POST or a client bug cannot wipe banks;
- one job at a time: a second start while one runs is refused;
- the control-plane middleware already gates the POST on the operator token.

Personas run one after another, not concurrently: each one's resend is a burst
of POSTs into a single Hindsight, and a failure in one (Hindsight went away)
should not leave several half-rebuilt at once.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

#: `ChatSystem.rebuild_persona_memory`: (name, progress=) -> counts.
Rebuild = Callable[..., Awaitable[Dict[str, int]]]

logger = logging.getLogger(__name__)

CONFIRM_PHRASE = "PURGE"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryRebuildJob:
    """The single rebuild job of one process, and its last result."""

    def __init__(self) -> None:
        self._task: Optional["asyncio.Task[None]"] = None
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.personas: List[Dict[str, Any]] = []

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def snapshot(self) -> Dict[str, Any]:
        status = "running" if self.running else ("finished" if self.started_at else "idle")
        return {
            "status": status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "personas": [dict(p) for p in self.personas],
        }

    def start(self, rebuild: Rebuild, names: List[str]) -> Dict[str, Any]:
        """Begin rebuilding `names` with `rebuild`. The caller checked
        confirmation and that no job runs."""
        self.started_at, self.finished_at = _now(), None
        self.personas = [{"name": n, "state": "waiting", "counts": {}, "error": None}
                         for n in names]
        self._task = asyncio.create_task(self._run(rebuild))
        return self.snapshot()

    async def _run(self, rebuild: Rebuild) -> None:
        try:
            for entry in self.personas:
                entry["state"] = "running"

                def progress(counts: Dict[str, int], entry: Dict[str, Any] = entry) -> None:
                    entry["counts"] = counts

                try:
                    entry["counts"] = await rebuild(entry["name"], progress=progress)
                    entry["state"] = "done"
                except Exception as e:  # noqa: BLE001 — one persona never stops the job
                    logger.warning(f"Memory rebuild of {entry['name']} failed: {e}")
                    entry["state"], entry["error"] = "error", str(e)
        finally:
            self.finished_at = _now()
            logger.info(f"Memory rebuild finished: {self.personas}")
