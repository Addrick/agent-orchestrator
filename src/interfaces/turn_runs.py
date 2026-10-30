"""Portal turns that outlive the connection that started them (DP-487).

Before this, the SSE request WAS the turn: uvicorn cancels the response
generator on disconnect, the `CancelledError` reached the engine, and the turn
was saved as stopped. A phone screen going dark therefore stopped every reply.

Now a turn runs as its own task. It drains the engine generator and appends
the transcoded SSE frames to a buffer; any number of followers read that
buffer by index — the request that started the turn, and any later re-attach
(`from=N`). A follower going away ends only the follower. The one way to end a
turn early is `cancel`, which cancels the task — exactly the cancellation the
disconnect used to deliver, so the engine's stop path (DP-444 seal-and-save)
is unchanged.

Bounds: one running turn per conversation key; a per-run byte cap past which
the oldest frames are dropped (a re-attach from before the window is refused,
live followers are unaffected); finished runs are kept `FINISHED_GRACE_S` so a
client waking just after the end still gets the tail and `[DONE]`; shutdown
cancels every run, and each saves what it had.

⚠️ In-process only. An engine restart ends every running turn — the same
"we all die together" premise the registry relies on, deliberately kept.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    Any, AsyncIterator, Callable, Dict, Iterable, List, Optional, Tuple,
)

logger = logging.getLogger(__name__)

# (persona, user_identifier, channel, server_id) — the conversation a turn
# writes into. Same key the engine's history and taint cache use.
TurnKey = Tuple[str, str, str, Optional[str]]

DONE_FRAME = b"data: [DONE]\n\n"
MAX_RUN_BYTES = 32 * 1024 * 1024
FINISHED_GRACE_S = 120.0
CANCEL_WAIT_S = 30.0
# A stop that arrives before its turn started (the client names the run, so it
# can stop one whose first frame it never saw) is remembered this long.
EARLY_CANCEL_TTL_S = 60.0
_MAX_EARLY_CANCELS = 256
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


class ReplayGone(Exception):
    """The requested start index fell out of the run's byte window."""


@dataclass
class TurnRun:
    run_id: str
    key: TurnKey
    kind: str  # "chat" | "confirm"
    continue_id: Optional[int] = None
    frames: List[bytes] = field(default_factory=list)
    # Index of frames[0] in the run's full frame sequence; > 0 once trimmed.
    base: int = 0
    nbytes: int = 0
    done: bool = False
    finished_at: Optional[float] = None
    task: Optional["asyncio.Task[None]"] = None
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)

    @property
    def end(self) -> int:
        """Index one past the last frame produced."""
        return self.base + len(self.frames)

    def info(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "continue_id": self.continue_id,
            "done": self.done,
            "replay_from": self.base,
        }

    async def append(self, frames: Iterable[bytes], max_bytes: int) -> None:
        async with self.cond:
            for f in frames:
                self.frames.append(f)
                self.nbytes += len(f)
            # Keep at least the newest frame, however large: a live follower
            # waiting on it must still receive it.
            while self.nbytes > max_bytes and len(self.frames) > 1:
                self.nbytes -= len(self.frames.pop(0))
                self.base += 1
            self.cond.notify_all()

    async def finish(self) -> None:
        async with self.cond:
            self.done = True
            self.finished_at = time.monotonic()
            self.cond.notify_all()

    async def follow(self, start: int = 0) -> AsyncIterator[bytes]:
        """Replay from `start`, then tail until the run ends.

        Raises ReplayGone if `start` is older than the retained window — a
        follower that fell behind the trim, or a re-attach from too far back.
        Cancelling this (client disconnect) leaves the run untouched.
        """
        i = start
        while True:
            async with self.cond:
                while i >= self.end and not self.done:
                    await self.cond.wait()
                if i < self.base:
                    raise ReplayGone(f"run {self.run_id}: frame {i} < window {self.base}")
                batch = self.frames[i - self.base:]
                finished = self.done
            for f in batch:
                yield f
            i += len(batch)
            if finished and not batch:
                return


def _run_frame(run: TurnRun) -> bytes:
    payload = {"run_id": run.run_id, "kind": run.kind, "continue_id": run.continue_id}
    return f"event: derpr-run\ndata: {json.dumps(payload)}\n\n".encode("utf-8")


class TurnRuns:
    """Registry of portal turns running independently of any connection."""

    def __init__(self, *, max_run_bytes: int = MAX_RUN_BYTES,
                 finished_grace_s: float = FINISHED_GRACE_S) -> None:
        self._runs: Dict[str, TurnRun] = {}
        self._max_run_bytes = max_run_bytes
        self._grace = finished_grace_s
        self._early_cancels: Dict[str, float] = {}

    @staticmethod
    def valid_run_id(run_id: Any) -> bool:
        return isinstance(run_id, str) and bool(_RUN_ID_RE.match(run_id))

    def cancelled_before_start(self, run_id: Optional[str]) -> bool:
        """True if a stop for `run_id` arrived before the turn started — the
        route then refuses the turn, so a stop pressed in the first instant
        (before the client saw any frame) still means nothing is written."""
        if run_id is None:
            return False
        at = self._early_cancels.pop(run_id, None)
        return at is not None and time.monotonic() - at <= EARLY_CANCEL_TTL_S

    def _purge(self) -> None:
        now = time.monotonic()
        for rid, run in list(self._runs.items()):
            if run.done and run.finished_at is not None and now - run.finished_at > self._grace:
                del self._runs[rid]

    def get(self, run_id: str) -> Optional[TurnRun]:
        self._purge()
        return self._runs.get(run_id)

    def active(self, key: TurnKey) -> Optional[TurnRun]:
        """The running (not finished) turn for this conversation, if any."""
        self._purge()
        for run in self._runs.values():
            if run.key == key and not run.done:
                return run
        return None

    def start(
        self,
        key: TurnKey,
        kind: str,
        events: Callable[[], Any],
        to_frames: Callable[[Any], Iterable[bytes]],
        *,
        error_frames: Callable[[str], Iterable[bytes]],
        continue_id: Optional[int] = None,
        on_finish: Optional[Callable[[str], None]] = None,
        run_id: Optional[str] = None,
    ) -> TurnRun:
        """Start a turn. `events()` builds the engine's async generator; it is
        called inside the task so the turn's context is the task's own.

        The caller must have checked `active(key)` — the registry does not
        refuse here, because the route owns the refusal's wire shape.
        `run_id` is the client's name for the run; a malformed or already-used
        one is replaced.
        """
        self._purge()
        if not self.valid_run_id(run_id) or run_id in self._runs:
            run_id = uuid.uuid4().hex
        assert run_id is not None
        run = TurnRun(run_id=run_id, key=key, kind=kind, continue_id=continue_id)
        run.frames.append(_run_frame(run))
        run.nbytes = len(run.frames[0])
        self._runs[run.run_id] = run
        run.task = asyncio.create_task(
            self._drive(run, events, to_frames, error_frames, on_finish),
            name=f"turn-run-{kind}-{run.run_id[:8]}",
        )
        return run

    async def _drive(
        self,
        run: TurnRun,
        events: Callable[[], Any],
        to_frames: Callable[[Any], Iterable[bytes]],
        error_frames: Callable[[str], Iterable[bytes]],
        on_finish: Optional[Callable[[str], None]],
    ) -> None:
        outcome = "done"
        try:
            # aclosing: cancelling this task must close the engine generator
            # (and everything below it down to the HTTP stream, DP-417) rather
            # than leave it suspended holding the conversation lock.
            async with contextlib.aclosing(events()) as agen:
                async for ev in agen:
                    await run.append(to_frames(ev), self._max_run_bytes)
        except asyncio.CancelledError:
            # Stopped: the engine saved what it had on its way out. Tell any
            # follower the turn is over so it re-syncs to the saved row.
            outcome = "cancelled"
            await run.append([DONE_FRAME], self._max_run_bytes)
            raise
        except Exception as e:  # the engine yields ErrorEvent for known failures
            outcome = "error"
            logger.error("turn run %s failed: %s", run.run_id, e, exc_info=True)
            await run.append(error_frames("engine error: turn failed"), self._max_run_bytes)
        finally:
            await run.finish()
            if on_finish is not None:
                try:
                    on_finish(outcome)
                except Exception:
                    logger.debug("turn run %s on_finish failed", run.run_id, exc_info=True)

    async def cancel(self, run_id: str) -> bool:
        """Stop a running turn and wait for the engine to save it. False if
        there is no such run or it already finished."""
        run = self._runs.get(run_id)
        if run is None:
            if self.valid_run_id(run_id):
                now = time.monotonic()
                for rid, at in list(self._early_cancels.items()):
                    if now - at > EARLY_CANCEL_TTL_S:
                        del self._early_cancels[rid]
                if len(self._early_cancels) >= _MAX_EARLY_CANCELS:
                    self._early_cancels.pop(next(iter(self._early_cancels)))
                self._early_cancels[run_id] = now
            return False
        if run.done or run.task is None:
            return False
        run.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
            # shield: a cancel request that itself disconnects must not
            # abandon the wait half-way — the task is already cancelled.
            await asyncio.wait_for(asyncio.shield(run.task), CANCEL_WAIT_S)
        await self._seal_unstarted(run)
        return True

    async def _seal_unstarted(self, run: TurnRun) -> None:
        """A task cancelled before its first step never enters `_drive`, so
        nothing marks its run finished: followers would wait forever and
        `active(key)` would refuse every later send in that conversation."""
        if run.task is not None and run.task.done() and not run.done:
            await run.append([DONE_FRAME], self._max_run_bytes)
            await run.finish()

    async def shutdown(self) -> None:
        """Cancel every running turn; each saves what it had."""
        tasks = [r.task for r in self._runs.values() if r.task is not None and not r.done]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for run in list(self._runs.values()):
            await self._seal_unstarted(run)
