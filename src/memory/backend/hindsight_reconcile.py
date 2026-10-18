# src/memory/backend/hindsight_reconcile.py
"""Re-run Hindsight retains that were accepted and then failed (DP-443).

A retain is POSTed with `async: true`: Hindsight accepts it, and a worker
extracts its facts later. When that worker's LLM is down, Hindsight retries
three times within minutes, marks the operation `failed` and never looks at it
again — while every caller (DP-423's confirmed flush included) already counted
the accept as delivery. This pass finds those operations and re-runs them with
Hindsight's own `/retry`, which replays the stored payload.

Blind retry is unsafe. A `replace` rewrites its document whole (a session's
first turn, or a DP-409 retry rebuild), so re-running an old failed replace
after newer operations on the same document landed would erase them. Hindsight
claims one retain per document, oldest first, so the plan works per document
on the operations in submission order:

- a failed operation older than a later *completed* replace is moot: that
  replace already holds everything it carried (`superseded`);
- a failed replace with a later completed operation is replayed as ONE new
  replace of its content plus everything after it, in order, so a partial
  replay can never leave the document worse than it was (`replayed`);
- anything else is retried in place, with backoff, and given up after
  RETAIN_RECONCILE_MAX_ATTEMPTS (`given_up`);
- a document with an operation still pending or processing is left alone.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

from .base import (
    RETAIN_RECONCILE_BACKOFF_CAP_SECONDS,
    RETAIN_RECONCILE_INTERVAL_SECONDS,
    RETAIN_RECONCILE_MAX_ATTEMPTS,
)

logger = logging.getLogger(__name__)

PAGE_SIZE = 100  # Hindsight rejects a larger `limit`
IN_FLIGHT = ("pending", "processing")
TERMINAL_STATES = ("superseded", "replayed", "given_up")


class _OpsClient(Protocol):
    async def alist_banks(self) -> Dict[str, Any]: ...
    async def alist_operations(
        self, bank_id: str, *, status: Optional[str] = None,
        op_type: Optional[str] = None, limit: Optional[int] = None,
        offset: Optional[int] = None, exclude_parents: Optional[bool] = None,
    ) -> Dict[str, Any]: ...
    async def aget_operation(
        self, bank_id: str, operation_id: str, *, include_payload: bool = False,
    ) -> Dict[str, Any]: ...
    async def aretry_operation(self, bank_id: str, operation_id: str) -> Dict[str, Any]: ...
    async def aretain(
        self, bank_id: str, items: List[Dict[str, Any]], *, async_: bool = True,
    ) -> Dict[str, Any]: ...


@dataclass
class Op:
    """One retain operation on a document, as the planner sees it."""
    id: str
    status: str
    created_at: datetime
    mode: Optional[str] = None  # 'replace' / 'append'; None until fetched


@dataclass
class DocPlan:
    superseded: List[str] = field(default_factory=list)
    retry: List[str] = field(default_factory=list)
    # Ops whose payloads are merged into one replace, oldest first; the failed
    # ones among them are what the replay resolves.
    replay: List[str] = field(default_factory=list)
    replay_failed: List[str] = field(default_factory=list)


def plan_document(ops: List[Op]) -> DocPlan:
    """Decide what to do with one document's failed operations.

    `ops` are that document's operations from its oldest unresolved failure
    on, in submission order. A completed op needs `mode`; so does a failed
    one, unless it is older than a completed replace.
    """
    plan = DocPlan()
    if any(o.status in IN_FLIGHT for o in ops):
        return plan
    last_full = max(
        (i for i, o in enumerate(ops) if o.status == "completed" and o.mode == "replace"),
        default=-1,
    )
    plan.superseded = [o.id for o in ops[:last_full + 1] if o.status == "failed"]
    tail = [o for o in ops[last_full + 1:] if o.status in ("failed", "completed")]
    for j, o in enumerate(tail):
        if (o.status == "failed" and o.mode == "replace"
                and any(p.status == "completed" for p in tail[j + 1:])):
            # A replace holds the whole document up to itself, so failed
            # appends before it are covered by replaying it too.
            plan.replay = [p.id for p in tail[j:]]
            plan.replay_failed = [p.id for p in tail if p.status == "failed"]
            return plan
    plan.retry = [o.id for o in tail if o.status == "failed"]
    return plan


def merge_replay(payloads: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """One replace item carrying the replayed ops' content, in order.

    Tags mirror DP-409's `replace_session`: the document's scope and persona
    tags, no per-turn role tag, and untrusted if any part was.
    """
    items = [it for p in payloads for it in p]
    tags: List[str] = []
    for it in items:
        for t in it.get("tags") or []:
            if not t.startswith("role:") and t not in tags:
                tags.append(t)
    if "untrusted:true" in tags and "untrusted:false" in tags:
        tags.remove("untrusted:false")
    last = items[-1]
    merged: Dict[str, Any] = {
        "content": "\n\n".join(str(it.get("content") or "") for it in items),
        "document_id": last.get("document_id"),
        "update_mode": "replace",
        "tags": tags,
    }
    for key in ("event_date", "timestamp", "mentioned_at", "metadata"):
        if last.get(key) is not None:
            merged[key] = last[key]
    return merged


def backoff_seconds(attempts: int) -> float:
    if attempts <= 0:
        return 0.0
    return float(min(RETAIN_RECONCILE_INTERVAL_SECONDS * 2 ** (attempts - 1),
                     RETAIN_RECONCILE_BACKOFF_CAP_SECONDS))


class ReconcileStore:
    """Per-operation reconcile state, kept beside the doc-scope store.

    Hindsight resets an operation's own retry_count on every `/retry`, so the
    attempt count and the terminal verdicts have to live here.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS Retain_Reconcile (
        operation_id TEXT PRIMARY KEY,
        bank_id      TEXT NOT NULL,
        state        TEXT NOT NULL,
        attempts     INTEGER NOT NULL DEFAULT 0,
        last_at      TEXT NOT NULL
    );
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None

    def _get(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.db_path != ":memory:":
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(self.SCHEMA)
            self._conn.commit()
        return self._conn

    def get(self, operation_id: str) -> Optional[Tuple[str, int, datetime]]:
        with self._lock:
            row = self._get().execute(
                "SELECT state, attempts, last_at FROM Retain_Reconcile WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
        if row is None:
            return None
        return str(row["state"]), int(row["attempts"]), _parse_ts(row["last_at"])

    def record(self, operation_id: str, bank_id: str, state: str,
               attempts: int, now: datetime) -> None:
        with self._lock:
            conn = self._get()
            conn.execute(
                "INSERT INTO Retain_Reconcile (operation_id, bank_id, state, attempts, last_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(operation_id) DO UPDATE SET "
                "state=excluded.state, attempts=excluded.attempts, last_at=excluded.last_at",
                (operation_id, bank_id, state, attempts, now.isoformat()),
            )
            conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


def _parse_ts(value: Any) -> datetime:
    ts = datetime.fromisoformat(str(value))
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _mode(payload_items: List[Dict[str, Any]]) -> str:
    return "replace" if any(it.get("update_mode") == "replace" for it in payload_items) else "append"


class Reconciler:
    """Drives one pass over every bank; see the module docstring."""

    def __init__(self, store: ReconcileStore):
        self.store = store
        self._payloads: Dict[str, List[Dict[str, Any]]] = {}

    def _due(self, op_id: str, now: datetime) -> bool:
        rec = self.store.get(op_id)
        if rec is None:
            return True
        state, attempts, last_at = rec
        if state in TERMINAL_STATES:
            return False
        return now >= last_at + timedelta(seconds=backoff_seconds(attempts))

    async def _items(self, client: _OpsClient, bank_id: str, op_id: str) -> List[Dict[str, Any]]:
        if op_id not in self._payloads:
            op = await client.aget_operation(bank_id, op_id, include_payload=True)
            payload = op.get("task_payload") or {}
            self._payloads[op_id] = list(payload.get("contents") or [])
        return self._payloads[op_id]

    async def _list(self, client: _OpsClient, bank_id: str, *, status: Optional[str],
                    stop_before: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Retain ops newest first; with `stop_before`, stop paging once
        a page reaches back past it."""
        out: List[Dict[str, Any]] = []
        offset = 0
        while True:
            page = await client.alist_operations(
                bank_id, status=status, op_type="retain", limit=PAGE_SIZE,
                offset=offset, exclude_parents=True,
            )
            ops = page.get("operations") or []
            out.extend(ops)
            offset += len(ops)
            if not ops or offset >= int(page.get("total") or 0):
                return out
            if stop_before is not None and _parse_ts(ops[-1]["created_at"]) < stop_before:
                return out

    async def run(self, client: _OpsClient, now: Optional[datetime] = None) -> int:
        """Reconcile every bank; returns how many operations were acted on."""
        now = now or datetime.now(timezone.utc)
        self._payloads = {}
        banks = (await client.alist_banks()).get("banks") or []
        acted = 0
        for bank in banks:
            bank_id = bank.get("bank_id")
            if not bank_id:
                continue
            try:
                acted += await self._bank(client, str(bank_id), now)
            except Exception as e:  # noqa: BLE001 — one bank must not stop the rest
                logger.warning("Retain reconcile failed for bank %s: %s", bank_id, e)
        return acted

    async def _bank(self, client: _OpsClient, bank_id: str, now: datetime) -> int:
        failed = [o for o in await self._list(client, bank_id, status="failed")
                  if self._due(str(o["id"]), now)]
        if not failed:
            return 0
        oldest = min(_parse_ts(o["created_at"]) for o in failed)
        peers = await self._list(client, bank_id, status=None, stop_before=oldest)
        by_doc: Dict[str, List[Dict[str, Any]]] = {}
        loose: List[Dict[str, Any]] = []
        for o in peers:
            if _parse_ts(o["created_at"]) < oldest:
                continue
            doc = o.get("document_id")
            if doc is None:
                loose.append(o)  # multi-document batch: no peers to order against
            else:
                by_doc.setdefault(str(doc), []).append(o)
        due_ids = {str(o["id"]) for o in failed}
        acted = 0
        for raw in loose:
            if str(raw["id"]) in due_ids and raw["status"] == "failed":
                acted += await self._retry(client, bank_id, [str(raw["id"])], now)
        for raw_ops in by_doc.values():
            if not any(str(o["id"]) in due_ids for o in raw_ops):
                continue
            acted += await self._document(client, bank_id, raw_ops, due_ids, now)
        return acted

    async def _document(self, client: _OpsClient, bank_id: str,
                        raw_ops: List[Dict[str, Any]], due_ids: Set[str], now: datetime) -> int:
        raw_ops = sorted(raw_ops, key=lambda o: (_parse_ts(o["created_at"]), str(o["id"])))
        # Start at this document's oldest failure that is still unresolved.
        start = next(i for i, o in enumerate(raw_ops) if str(o["id"]) in due_ids)
        ops = [Op(str(o["id"]), str(o["status"]), _parse_ts(o["created_at"]))
               for o in raw_ops[start:]
               if o["status"] != "failed" or str(o["id"]) in due_ids]
        if any(o.status in IN_FLIGHT for o in ops):
            return 0
        for o in ops:
            if o.status in ("failed", "completed"):
                o.mode = _mode(await self._items(client, bank_id, o.id))
        plan = plan_document(ops)
        for op_id in plan.superseded:
            self.store.record(op_id, bank_id, "superseded", 0, now)
        acted = len(plan.superseded)
        if plan.replay:
            payloads = [await self._items(client, bank_id, op_id) for op_id in plan.replay]
            await client.aretain(bank_id, [merge_replay(payloads)], async_=True)
            for op_id in plan.replay_failed:
                self.store.record(op_id, bank_id, "replayed", 0, now)
            logger.info("Retain reconcile: replayed %d op(s) on %s/%s as one replace",
                        len(plan.replay), bank_id, raw_ops[start].get("document_id"))
            acted += len(plan.replay_failed)
        acted += await self._retry(client, bank_id, plan.retry, now)
        return acted

    async def _retry(self, client: _OpsClient, bank_id: str,
                     op_ids: List[str], now: datetime) -> int:
        acted = 0
        for op_id in op_ids:
            rec = self.store.get(op_id)
            attempts = rec[1] if rec else 0
            if attempts >= RETAIN_RECONCILE_MAX_ATTEMPTS:
                self.store.record(op_id, bank_id, "given_up", attempts, now)
                logger.warning(
                    "Retain reconcile: giving up on %s/%s after %d retries; it stays "
                    "failed in Hindsight's operations list", bank_id, op_id, attempts,
                )
                continue
            await client.aretry_operation(bank_id, op_id)
            self.store.record(op_id, bank_id, "retrying", attempts + 1, now)
            acted += 1
        if acted:
            logger.info("Retain reconcile: retried %d failed op(s) in bank %s", acted, bank_id)
        return acted
