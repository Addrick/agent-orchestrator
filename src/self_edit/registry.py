# src/self_edit/registry.py
"""In-memory registry of dispatched coding agents (DP-227).

The "management view" the supervisor (fixr) inspects and acts on. One record per
dispatched bug-fix agent: where its worktree is, its OS pid, the resumable
Claude session id, current status, log paths, and the eventual PR url.

In-memory for v1 (fits the woken-supervisor model — derpr is up whenever fixr
runs). Persisting to SQLite (so in-flight agents survive a derpr restart) is the
documented next step, hung off the same `dynamic_tasks_and_watches` storage.
Access is serialized by an asyncio.Lock since the dispatch tool, the per-agent
bridge tasks, and inspect/kill tools all touch it concurrently.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Status lifecycle.
RUNNING = "running"
WAITING = "waiting"     # asked a question, awaiting answer_agent
DONE = "done"
ERROR = "error"
KILLED = "killed"


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
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentRegistry:
    """Async-safe store of ``AgentRecord``s keyed by ``agent_id``."""

    def __init__(self) -> None:
        self._records: Dict[str, AgentRecord] = {}
        self._lock = asyncio.Lock()

    async def add(self, record: AgentRecord) -> None:
        async with self._lock:
            if record.agent_id in self._records:
                raise KeyError(f"agent_id already registered: {record.agent_id}")
            self._records[record.agent_id] = record

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
            return rec

    async def list(self, *, active_only: bool = False) -> List[AgentRecord]:
        async with self._lock:
            recs = list(self._records.values())
        if active_only:
            recs = [r for r in recs if r.status in (RUNNING, WAITING)]
        return sorted(recs, key=lambda r: r.created_at)

    async def remove(self, agent_id: str) -> Optional[AgentRecord]:
        async with self._lock:
            return self._records.pop(agent_id, None)

    async def has_active_for_bug(self, bug_id: str) -> bool:
        """True if a still-running/waiting agent already owns this bug — the
        dispatch tool uses this to refuse duplicate dispatches."""
        async with self._lock:
            return any(
                r.bug_id == bug_id and r.status in (RUNNING, WAITING)
                for r in self._records.values()
            )
