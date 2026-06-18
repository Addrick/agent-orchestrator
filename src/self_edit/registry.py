# src/self_edit/registry.py
"""Registry of dispatched coding agents (DP-227), SQLite-backed (DP-233).

The "management view" the supervisor (fixr) inspects and acts on. One record per
dispatched bug-fix agent: where its worktree is, its OS pid, the resumable
Claude session id, current status, log paths, and the eventual PR url.

An in-memory dict is the hot path; when an ``AgentStore`` is injected, every
mutation write-throughs to SQLite so in-flight agents survive a derpr restart
(DP-233). On construction with a store, records are loaded and any still marked
RUNNING/WAITING are flipped to ORPHANED — their detached process + in-process
bridge task did not survive the restart, so they can't be resumed. Access is
serialized by an asyncio.Lock since the dispatch tool, the per-agent bridge
tasks, and inspect/kill tools all touch it concurrently (and the store's single
connection relies on that serialization).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.self_edit.store import AgentStore

# Status lifecycle.
RUNNING = "running"
WAITING = "waiting"     # asked a question, awaiting answer_agent
DONE = "done"
ERROR = "error"
KILLED = "killed"
ORPHANED = "orphaned"   # was RUNNING/WAITING at a restart; process + bridge gone


@dataclass
class AgentRecord:
    agent_id: str
    bug_id: str
    description: str
    worktree: str
    branch: str
    raw_log: str
    events_log: str
    pid: Optional[int] = None
    session_id: Optional[str] = None
    status: str = RUNNING
    pr_url: Optional[str] = None
    last_event: Optional[str] = None
    #: Discord thread carrying this agent's transcript + Q&A (DP-230). Set when
    #: the per-agent thread is created; resolves inbound thread→agent_id.
    discord_thread_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentRegistry:
    """Async-safe store of ``AgentRecord``s keyed by ``agent_id``."""

    def __init__(self, store: Optional["AgentStore"] = None) -> None:
        self._records: Dict[str, AgentRecord] = {}
        self._lock = asyncio.Lock()
        self._store = store
        if store is not None:
            # Restart recovery: orphan stale rows in the DB, then load the
            # (now-corrected) records into the in-memory hot path.
            store.orphan_stale()
            for rec in store.load_all():
                self._records[rec.agent_id] = rec

    async def add(self, record: AgentRecord) -> None:
        async with self._lock:
            if record.agent_id in self._records:
                raise KeyError(f"agent_id already registered: {record.agent_id}")
            self._records[record.agent_id] = record
            if self._store is not None:
                await asyncio.to_thread(self._store.upsert, record)

    async def get(self, agent_id: str) -> Optional[AgentRecord]:
        async with self._lock:
            return self._records.get(agent_id)

    async def update(self, agent_id: str, **fields: Any) -> Optional[AgentRecord]:
        async with self._lock:
            rec = self._records.get(agent_id)
            if rec is None:
                return None
            for k, v in fields.items():
                if hasattr(rec, k):
                    setattr(rec, k, v)
            rec.updated_at = time.time()
            if self._store is not None:
                await asyncio.to_thread(self._store.upsert, rec)
            return rec

    async def list(self, *, active_only: bool = False) -> List[AgentRecord]:
        async with self._lock:
            recs = list(self._records.values())
        if active_only:
            recs = [r for r in recs if r.status in (RUNNING, WAITING)]
        return sorted(recs, key=lambda r: r.created_at)

    async def remove(self, agent_id: str) -> Optional[AgentRecord]:
        async with self._lock:
            rec = self._records.pop(agent_id, None)
            if rec is not None and self._store is not None:
                await asyncio.to_thread(self._store.delete, agent_id)
            return rec

    async def get_by_thread(self, thread_id: str) -> Optional[AgentRecord]:
        """Resolve the agent whose Discord thread is ``thread_id`` (DP-230).

        Inbound thread messages route to ``answer_agent`` via this lookup."""
        async with self._lock:
            for rec in self._records.values():
                if rec.discord_thread_id == thread_id:
                    return rec
            return None

    async def has_active_for_bug(self, bug_id: str) -> bool:
        """True if a still-running/waiting agent already owns this bug — the
        dispatch tool uses this to refuse duplicate dispatches."""
        async with self._lock:
            return any(
                r.bug_id == bug_id and r.status in (RUNNING, WAITING)
                for r in self._records.values()
            )
