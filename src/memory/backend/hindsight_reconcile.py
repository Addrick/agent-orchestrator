# src/memory/backend/hindsight_reconcile.py
"""Re-run Hindsight retains that were accepted and then failed (DP-443).

A retain is POSTed with `async: true`: Hindsight accepts it, and a worker
extracts its facts later. When that worker's LLM is down, Hindsight retries
three times within minutes, marks the operation `failed` and never looks at it
again — while every caller (DP-423's confirmed flush included) already counted
the accept as delivery. This pass finds those operations and re-runs them with
Hindsight's own `/retry`, which replays the stored payload.

Blind retry is unsafe. A `replace` rewrites its document whole (a session's
first turn, a DP-409 retry rebuild, an experience or upload re-ingest), so an
old failed replace re-run after newer operations on the same document landed
would erase them. Hindsight claims one retain per document, oldest first, so
the plan works per document on the operations in submission order:

- a failed operation older than a later *completed* replace is moot: that
  replace holds everything it carried (`superseded`);
- a failed append is always retried: appending can only add;
- a failed replace is retried only when nothing completed on its document
  after it; otherwise it is `blocked` and reported — re-running it would
  erase that later content, and rebuilding the document is a human call;
- a document with an operation still pending or processing waits.

Which documents an operation touched comes from its stored payload, never
from the list's `document_id`: Hindsight leaves that null on most operations,
multi-document batches included. An operation is retried only if every
document it touches allows it. Retries back off to once a day and are never
given up (Adam, 2026-10-18).
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

from src.utils.timeutil import to_utc

from .base import RETAIN_RECONCILE_BACKOFF_CAP_SECONDS, RETAIN_RECONCILE_INTERVAL_SECONDS
from .sqlite_store import LazySqliteStore

logger = logging.getLogger(__name__)

PAGE_SIZE = 100  # Hindsight rejects a larger `limit`
IN_FLIGHT = ("pending", "processing")
TERMINAL_STATES = ("superseded",)
PAYLOAD_FETCH_CONCURRENCY = 8
# Past this many retries an op is reported on every retry, so a deterministic
# failure is visible by name rather than only as a daily count.
LONG_FAILING_ATTEMPTS = 8
STUCK_IN_FLIGHT_SECONDS = 3600

# {document_id: 'replace' | 'append'}; None when the payload is gone.
DocModes = Optional[Dict[str, str]]


class _OpsClient(Protocol):
    async def alist_operations(
        self, bank_id: str, *, status: Optional[str] = None,
        op_type: Optional[str] = None, limit: Optional[int] = None,
        offset: Optional[int] = None, exclude_parents: Optional[bool] = None,
    ) -> Dict[str, Any]: ...
    async def aget_operation(
        self, bank_id: str, operation_id: str, *, include_payload: bool = False,
    ) -> Dict[str, Any]: ...
    async def aretry_operation(self, bank_id: str, operation_id: str) -> Dict[str, Any]: ...


@dataclass
class Op:
    """One retain operation, as the planner sees it."""
    id: str
    status: str
    created_at: datetime
    docs: DocModes = None


@dataclass
class Plan:
    superseded: List[str] = field(default_factory=list)
    retry: List[str] = field(default_factory=list)
    blocked: List[str] = field(default_factory=list)


def doc_modes(op_id: str, items: List[Dict[str, Any]]) -> Dict[str, str]:
    """Documents a payload touches, and whether it replaces each one. An item
    with no document_id is a document of its own."""
    modes: Dict[str, str] = {}
    for it in items:
        doc = str(it.get("document_id") or f"op:{op_id}")
        if it.get("update_mode") == "replace" or modes.get(doc) == "replace":
            modes[doc] = "replace"
        else:
            modes[doc] = "append"
    return modes


def plan_bank(ops: List[Op], unresolved: Set[str]) -> Plan:
    """Decide what to do with a bank's unresolved failed operations.

    `ops` are every retain op of the bank from its oldest unresolved failure
    on, in submission order; `unresolved` are the failed ones still to settle.
    An op whose payload is gone (`docs` None) cannot be placed on a document:
    completed, it may follow any failed replace; failed, it cannot be retried.
    """
    by_doc: Dict[str, List[Tuple[int, Op, str]]] = {}
    unknown_completed = [i for i, o in enumerate(ops) if o.status == "completed" and o.docs is None]
    for i, o in enumerate(ops):
        for doc, mode in (o.docs or {}).items():
            by_doc.setdefault(doc, []).append((i, o, mode))

    verdicts: Dict[str, Set[str]] = {}
    for entries in by_doc.values():
        waiting = any(o.status in IN_FLIGHT for _, o, _ in entries)
        last_full = max((i for i, o, m in entries if o.status == "completed" and m == "replace"),
                        default=-1)
        for i, o, mode in entries:
            if o.status != "failed" or o.id not in unresolved:
                continue
            if waiting:
                verdict = "wait"
            elif i < last_full:
                verdict = "superseded"
            elif mode == "append":
                verdict = "retry"
            elif (any(j > i and p.status == "completed" for j, p, _ in entries)
                  or any(j > i for j in unknown_completed)):
                verdict = "blocked"
            else:
                verdict = "retry"
            verdicts.setdefault(o.id, set()).add(verdict)

    plan = Plan()
    for o in ops:
        if o.status != "failed" or o.id not in unresolved:
            continue
        v = verdicts.get(o.id)
        if not v:  # payload gone: nothing to retry it with
            plan.blocked.append(o.id)
        elif "wait" in v:
            continue
        elif v == {"superseded"}:
            plan.superseded.append(o.id)
        elif v == {"retry"}:
            plan.retry.append(o.id)
        else:
            # A batch is re-run whole: one document's veto, or one it
            # would replay behind a newer replace, stops all of it.
            plan.blocked.append(o.id)
    return plan


def backoff_seconds(attempts: int) -> float:
    if attempts <= 0:
        return 0.0
    return float(min(RETAIN_RECONCILE_INTERVAL_SECONDS * 2 ** (attempts - 1),
                     RETAIN_RECONCILE_BACKOFF_CAP_SECONDS))


def _ts(value: Any) -> datetime:
    return to_utc(datetime.fromisoformat(str(value)))


class ReconcileStore(LazySqliteStore):
    """Per-operation reconcile state, in the doc-scope DB file.

    Hindsight resets an operation's own retry_count on every `/retry`, so the
    attempt count and verdicts live here. `Retain_Op_Docs` caches what a
    completed operation touched, which never changes, so a long window is
    not re-fetched payload by payload on every pass.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS Retain_Reconcile (
        operation_id TEXT PRIMARY KEY,
        bank_id      TEXT NOT NULL,
        state        TEXT NOT NULL,
        attempts     INTEGER NOT NULL DEFAULT 0,
        last_at      TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS Retain_Op_Docs (
        operation_id TEXT PRIMARY KEY,
        docs         TEXT NOT NULL
    );
    """

    def get(self, operation_id: str) -> Optional[Tuple[str, int, datetime]]:
        with self._lock:
            row = self._get().execute(
                "SELECT state, attempts, last_at FROM Retain_Reconcile WHERE operation_id=?",
                (operation_id,),
            ).fetchone()
        if row is None:
            return None
        return str(row["state"]), int(row["attempts"]), _ts(row["last_at"])

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

    def cached_docs(self, operation_id: str) -> Optional[Dict[str, str]]:
        with self._lock:
            row = self._get().execute(
                "SELECT docs FROM Retain_Op_Docs WHERE operation_id=?", (operation_id,),
            ).fetchone()
        return None if row is None else dict(json.loads(row["docs"]))

    def cache_docs(self, operation_id: str, docs: Dict[str, str]) -> None:
        with self._lock:
            conn = self._get()
            conn.execute(
                "INSERT OR REPLACE INTO Retain_Op_Docs (operation_id, docs) VALUES (?, ?)",
                (operation_id, json.dumps(docs)),
            )
            conn.commit()


class Reconciler:
    """Drives one pass over the given banks; see the module docstring."""

    def __init__(self, store: ReconcileStore):
        self.store = store

    def _due(self, op_id: str, now: datetime) -> bool:
        rec = self.store.get(op_id)
        if rec is None:
            return True
        _state, attempts, last_at = rec
        return now >= last_at + timedelta(seconds=backoff_seconds(attempts))

    async def _list(self, client: _OpsClient, bank_id: str, *, status: Optional[str],
                    stop_before: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Retain ops newest first, without duplicates (offset paging over a
        list that grows at its head repeats rows); with `stop_before`, stop
        once a page reaches back past it."""
        out: Dict[str, Dict[str, Any]] = {}
        offset = 0
        while True:
            page = await client.alist_operations(
                bank_id, status=status, op_type="retain", limit=PAGE_SIZE,
                offset=offset, exclude_parents=True,
            )
            ops = page.get("operations") or []
            for o in ops:
                out.setdefault(str(o["id"]), o)
            offset += len(ops)
            if not ops or offset >= int(page.get("total") or 0):
                return list(out.values())
            if stop_before is not None and _ts(ops[-1]["created_at"]) < stop_before:
                return list(out.values())

    async def _docs(self, client: _OpsClient, bank_id: str, raw: Dict[str, Any],
                    sem: asyncio.Semaphore) -> DocModes:
        op_id = str(raw["id"])
        if raw["status"] == "completed":
            cached = self.store.cached_docs(op_id)
            if cached is not None:
                return cached
        try:
            async with sem:
                op = await client.aget_operation(bank_id, op_id, include_payload=True)
        except Exception as e:  # noqa: BLE001 — an unreadable op is planned as unknown
            logger.info("Retain reconcile: no payload for %s/%s: %s", bank_id, op_id, e)
            return None
        items = list((op.get("task_payload") or {}).get("contents") or [])
        if not items:
            return None
        docs = doc_modes(op_id, items)
        if raw["status"] == "completed":
            self.store.cache_docs(op_id, docs)
        return docs

    async def run(self, client: _OpsClient, banks: List[str], now: datetime) -> int:
        """Reconcile `banks`; returns how many operations were acted on."""
        acted = 0
        for bank_id in banks:
            try:
                acted += await self._bank(client, bank_id, now)
            except Exception as e:  # noqa: BLE001 — one bank must not stop the rest
                logger.warning("Retain reconcile failed for bank %s: %s", bank_id, e)
        return acted

    async def _bank(self, client: _OpsClient, bank_id: str, now: datetime) -> int:
        failed = await self._list(client, bank_id, status="failed")
        unresolved = {str(o["id"]) for o in failed
                      if (self.store.get(str(o["id"])) or ("",))[0] not in TERMINAL_STATES}
        due = {op_id for op_id in unresolved if self._due(op_id, now)}
        if not due:
            return 0
        oldest = min(_ts(o["created_at"]) for o in failed if str(o["id"]) in unresolved)
        window = [o for o in await self._list(client, bank_id, status=None, stop_before=oldest)
                  if _ts(o["created_at"]) >= oldest]
        window.sort(key=lambda o: (_ts(o["created_at"]), str(o["id"])))
        sem = asyncio.Semaphore(PAYLOAD_FETCH_CONCURRENCY)
        docs = await asyncio.gather(*(self._docs(client, bank_id, o, sem) for o in window))
        ops = [Op(str(o["id"]), str(o["status"]), _ts(o["created_at"]), d)
               for o, d in zip(window, docs)]
        self._report_stuck(bank_id, window, now)

        plan = plan_bank(ops, unresolved)
        for op_id in plan.superseded:
            self.store.record(op_id, bank_id, "superseded", 0, now)
        acted = len(plan.superseded)
        created = {o.id: o.created_at for o in ops}
        for op_id in plan.blocked:
            if op_id not in due:
                continue
            attempts = self._attempts(op_id) + 1
            self.store.record(op_id, bank_id, "blocked", attempts, now)
            logger.warning(
                "Retain reconcile: %s/%s (failed %s) is not retried — it would rewrite its "
                "document over content that landed after it, or its payload is gone. "
                "Rebuild that document by hand if it matters.",
                bank_id, op_id, created[op_id].date(),
            )
        acted += await self._retry(client, bank_id, [i for i in plan.retry if i in due],
                                   created, now)
        return acted

    def _attempts(self, op_id: str) -> int:
        rec = self.store.get(op_id)
        return rec[1] if rec else 0

    def _report_stuck(self, bank_id: str, window: List[Dict[str, Any]], now: datetime) -> None:
        for o in window:
            if o["status"] not in IN_FLIGHT:
                continue
            seen = _ts(o.get("updated_at") or o["created_at"])
            if (now - seen).total_seconds() > STUCK_IN_FLIGHT_SECONDS:
                logger.warning(
                    "Retain reconcile: %s/%s has been %s since %s; failures on its "
                    "document wait until it settles", bank_id, o["id"], o["status"], seen,
                )

    async def _retry(self, client: _OpsClient, bank_id: str, op_ids: List[str],
                     created: Dict[str, datetime], now: datetime) -> int:
        acted = 0
        for op_id in op_ids:
            attempts = self._attempts(op_id) + 1
            try:
                await client.aretry_operation(bank_id, op_id)
            except Exception as e:  # noqa: BLE001 — one op must not stop the rest
                logger.warning("Retain reconcile: retry of %s/%s refused: %s", bank_id, op_id, e)
                self.store.record(op_id, bank_id, "retrying", attempts, now)
                continue
            self.store.record(op_id, bank_id, "retrying", attempts, now)
            acted += 1
            if attempts >= LONG_FAILING_ATTEMPTS:
                logger.warning(
                    "Retain reconcile: %s/%s still failing after %d retries (since %s)",
                    bank_id, op_id, attempts - 1, created[op_id].date(),
                )
        if acted:
            logger.info("Retain reconcile: retried %d failed op(s) in bank %s", acted, bank_id)
        return acted
