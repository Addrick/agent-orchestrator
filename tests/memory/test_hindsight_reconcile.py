# tests/memory/test_hindsight_reconcile.py
#
# DP-443: re-running retains Hindsight accepted and then failed to extract.
# The planner is pure; the Reconciler runs against an in-memory fake of the
# operations API that keeps Hindsight's semantics as measured on prod
# 2026-10-18: newest-first listing, `limit` <= 100, list rows whose
# `document_id` is null even for one-document ops (only the payload says which
# documents an op touched), `/retry` flips failed -> pending.

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.memory.backend.base import RETAIN_RECONCILE_INTERVAL_SECONDS
from src.memory.backend.hindsight import HindsightBackend, _DocScopeStore
from src.memory.backend.hindsight_reconcile import (
    LONG_FAILING_ATTEMPTS,
    Op,
    Reconciler,
    ReconcileStore,
    backoff_seconds,
    doc_modes,
    plan_bank,
)

T0 = datetime(2026, 10, 15, 5, 0, tzinfo=timezone.utc)
DOC = "explr:web_ui:2026-10-09T20:51:00+00:00"
OTHER = "explr:web_ui:2026-10-01T00:00:00+00:00"


def _op(i: int, status: str, docs: Optional[Dict[str, str]] = None) -> Op:
    return Op(f"op{i}", status, T0 + timedelta(minutes=i), docs)


def R(doc: str = DOC) -> Dict[str, str]:
    return {doc: "replace"}


def A(doc: str = DOC) -> Dict[str, str]:
    return {doc: "append"}


def _unresolved(ops: List[Op]) -> set:
    return {o.id for o in ops if o.status == "failed"}


# ---------- plan_bank ----------


def test_failures_older_than_a_completed_replace_are_superseded() -> None:
    """The live explr case: 17 failures, then DP-409 rebuilds that completed.
    Retrying the oldest (a short replace) would wipe the rebuilt document."""
    ops = [_op(0, "failed", R()), _op(1, "failed", A()), _op(2, "failed", R()),
           _op(3, "completed", R()), _op(4, "completed", R())]
    plan = plan_bank(ops, _unresolved(ops))
    assert plan.superseded == ["op0", "op1", "op2"]
    assert plan.retry == [] and plan.blocked == []


def test_failed_tail_is_retried_in_place_in_order() -> None:
    ops = [_op(0, "failed", R()), _op(1, "failed", A()), _op(2, "failed", A())]
    plan = plan_bank(ops, _unresolved(ops))
    assert plan.retry == ["op0", "op1", "op2"]


def test_failed_append_before_a_completed_append_is_retried() -> None:
    ops = [_op(0, "failed", A()), _op(1, "completed", A())]
    assert plan_bank(ops, _unresolved(ops)).retry == ["op0"]


def test_failed_replace_with_later_content_is_blocked_but_appends_still_retry() -> None:
    """The hypr case: the session-opening replace failed, later appends
    landed, then more appends failed. Re-running the replace would erase the
    landed appends; the failed appends are safe."""
    ops = [_op(0, "failed", R()), _op(1, "completed", A()),
           _op(2, "failed", A()), _op(3, "failed", A())]
    plan = plan_bank(ops, _unresolved(ops))
    assert plan.blocked == ["op0"]
    assert plan.retry == ["op2", "op3"]


def test_a_later_failed_rebuild_is_retried_and_the_earlier_replace_waits_on_it() -> None:
    """A later failed replace already holds everything before it: retry it,
    and do not re-run (or concatenate) the earlier one."""
    ops = [_op(0, "failed", R()), _op(1, "completed", A()),
           _op(2, "failed", R()), _op(3, "failed", A())]
    plan = plan_bank(ops, _unresolved(ops))
    assert plan.blocked == ["op0"]
    assert plan.retry == ["op2", "op3"]
    # Once the rebuild lands, the earlier replace is covered by it.
    ops[2] = _op(2, "completed", R())
    plan = plan_bank(ops, {"op0", "op3"})
    assert plan.superseded == ["op0"]


def test_document_with_an_op_in_flight_waits() -> None:
    for status in ("pending", "processing"):
        ops = [_op(0, "failed", R()), _op(1, status, A())]
        plan = plan_bank(ops, _unresolved(ops))
        assert (plan.superseded, plan.retry, plan.blocked) == ([], [], [])


def test_documents_are_independent() -> None:
    ops = [_op(0, "failed", R(OTHER)), _op(1, "completed", A()), _op(2, "failed", R())]
    plan = plan_bank(ops, _unresolved(ops))
    assert plan.retry == ["op0", "op2"]


def test_a_batch_is_retried_only_if_every_document_allows_it() -> None:
    # op0 appends to DOC and replaces OTHER; OTHER took content after it.
    ops = [_op(0, "failed", {DOC: "append", OTHER: "replace"}), _op(1, "completed", A(OTHER))]
    assert plan_bank(ops, _unresolved(ops)).blocked == ["op0"]
    # Superseded on one document and retryable on the other: re-running the
    # batch would replay the stale part too.
    ops = [_op(0, "failed", {DOC: "append", OTHER: "append"}), _op(1, "completed", R(OTHER))]
    assert plan_bank(ops, _unresolved(ops)).blocked == ["op0"]
    ops = [_op(0, "failed", {DOC: "append", OTHER: "append"})]
    assert plan_bank(ops, _unresolved(ops)).retry == ["op0"]


def test_an_unreadable_completed_op_blocks_older_replaces_only() -> None:
    """A completed op whose payload is gone could be on any document."""
    ops = [_op(0, "failed", R()), _op(1, "failed", A()), _op(2, "completed", None)]
    plan = plan_bank(ops, _unresolved(ops))
    assert plan.blocked == ["op0"]
    assert plan.retry == ["op1"]


def test_a_failed_op_without_payload_is_blocked() -> None:
    ops = [_op(0, "failed", None)]
    assert plan_bank(ops, _unresolved(ops)).blocked == ["op0"]


def test_doc_modes_reads_the_payload() -> None:
    assert doc_modes("x", [
        {"document_id": DOC, "update_mode": "append"},
        {"document_id": DOC, "update_mode": "replace"},
        {"document_id": OTHER},
        {"content": "no id"},
    ]) == {DOC: "replace", OTHER: "append", "op:x": "append"}


def test_backoff_doubles_to_once_a_day() -> None:
    assert backoff_seconds(0) == 0
    assert backoff_seconds(1) == RETAIN_RECONCILE_INTERVAL_SECONDS
    assert backoff_seconds(2) == 2 * RETAIN_RECONCILE_INTERVAL_SECONDS
    assert backoff_seconds(8) == 24 * 3600
    assert backoff_seconds(10_000) == 24 * 3600


# ---------- Reconciler against a fake operations API ----------


class FakeHindsight:
    def __init__(self) -> None:
        self.ops: Dict[str, Dict[str, Any]] = {}
        self.retried: List[str] = []
        self.payload_fetches: List[str] = []
        self.on_page: Optional[Any] = None
        self.refuse: set = set()

    def add(self, op_id: str, status: str, minute: float, *, bank: str = "explr",
            items: Optional[List[Tuple[str, str]]] = None, payload: bool = True,
            updated: Optional[datetime] = None) -> None:
        created = T0 + timedelta(minutes=minute)
        self.ops[op_id] = {
            "id": op_id, "bank": bank, "status": status, "task_type": "retain",
            "document_id": None,  # as on prod: the list does not say
            "created_at": created.isoformat(),
            "updated_at": (updated or created).isoformat(),
            "items": None if not payload else [
                {"document_id": d, "update_mode": m, "content": op_id}
                for d, m in (items or [(DOC, "append")])
            ],
        }

    async def alist_operations(self, bank_id: str, *, status: Optional[str] = None,
                               op_type: Optional[str] = None, limit: Optional[int] = None,
                               offset: Optional[int] = None,
                               exclude_parents: Optional[bool] = None) -> Dict[str, Any]:
        assert limit is not None and limit <= 100
        if self.on_page is not None:
            self.on_page(offset)
        rows = [o for o in self.ops.values() if o["bank"] == bank_id
                and (status is None or o["status"] == status)]
        rows.sort(key=lambda o: o["created_at"], reverse=True)
        start = offset or 0
        page = [{k: v for k, v in o.items() if k != "items"} for o in rows[start:start + limit]]
        return {"total": len(rows), "operations": page}

    async def aget_operation(self, bank_id: str, operation_id: str, *,
                             include_payload: bool = False) -> Dict[str, Any]:
        self.payload_fetches.append(operation_id)
        items = self.ops[operation_id]["items"]
        return {"operation_id": operation_id,
                "task_payload": None if items is None else {"contents": items}}

    async def aretry_operation(self, bank_id: str, operation_id: str) -> Dict[str, Any]:
        if operation_id in self.refuse or self.ops[operation_id]["status"] != "failed":
            raise httpx.HTTPError("409 not retryable")
        self.ops[operation_id]["status"] = "pending"
        self.retried.append(operation_id)
        return {"success": True}

    def settle(self, status: str) -> None:
        for o in self.ops.values():
            if o["status"] == "pending":
                o["status"] = status


@pytest.fixture
def reconciler(tmp_path) -> Reconciler:
    return Reconciler(ReconcileStore(str(tmp_path / "doc_scope.db")))


async def _run(r: Reconciler, fake: FakeHindsight, now: datetime,
               banks: Optional[List[str]] = None) -> int:
    return await r.run(fake, banks or sorted({o["bank"] for o in fake.ops.values()}), now)


@pytest.mark.asyncio
async def test_superseded_document_is_left_untouched(reconciler: Reconciler) -> None:
    fake = FakeHindsight()
    fake.add("f0", "failed", 0, items=[(DOC, "replace")])
    fake.add("f1", "failed", 1)
    fake.add("c2", "completed", 60, items=[(DOC, "replace")])
    assert await _run(reconciler, fake, T0 + timedelta(days=2)) == 2
    assert fake.retried == []
    # Resolved for good: the next pass lists failures and stops there.
    fake.payload_fetches.clear()
    assert await _run(reconciler, fake, T0 + timedelta(days=3)) == 0
    assert fake.payload_fetches == []


@pytest.mark.asyncio
async def test_failed_ops_are_retried_with_backoff_and_never_given_up(reconciler: Reconciler) -> None:
    fake = FakeHindsight()
    fake.add("f0", "failed", 0, items=[(DOC, "replace")])
    fake.add("f1", "failed", 1)
    now = T0 + timedelta(hours=1)
    assert await _run(reconciler, fake, now) == 2
    assert fake.retried == ["f0", "f1"]

    fake.settle("failed")  # the model server is still down
    fake.retried.clear()
    assert await _run(reconciler, fake, now + timedelta(minutes=5)) == 0
    now += timedelta(seconds=backoff_seconds(1))
    assert await _run(reconciler, fake, now) == 2

    for _ in range(40):  # a month and more of daily failures
        fake.settle("failed")
        now += timedelta(days=1)
        await _run(reconciler, fake, now)
    fake.settle("failed")
    fake.retried.clear()
    assert await _run(reconciler, fake, now + timedelta(hours=23)) == 0
    assert await _run(reconciler, fake, now + timedelta(days=1)) == 2


@pytest.mark.asyncio
async def test_long_failing_op_is_named_in_a_warning(reconciler: Reconciler, caplog) -> None:
    fake = FakeHindsight()
    fake.add("f0", "failed", 0)
    now = T0
    with caplog.at_level(logging.WARNING):
        for _ in range(LONG_FAILING_ATTEMPTS):
            now += timedelta(days=1)
            await _run(reconciler, fake, now)
            fake.settle("failed")
    assert any("f0 still failing" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_recovered_retry_drops_out(reconciler: Reconciler) -> None:
    fake = FakeHindsight()
    fake.add("f0", "failed", 0)
    await _run(reconciler, fake, T0 + timedelta(hours=1))
    fake.settle("completed")
    fake.retried.clear()
    assert await _run(reconciler, fake, T0 + timedelta(days=1)) == 0


@pytest.mark.asyncio
async def test_hypr_case_blocks_the_replace_and_retries_the_appends(
        reconciler: Reconciler, caplog) -> None:
    fake = FakeHindsight()
    fake.add("r0", "failed", 0, items=[(DOC, "replace")])
    fake.add("c1", "completed", 5)
    fake.add("a2", "failed", 10)
    with caplog.at_level(logging.WARNING):
        assert await _run(reconciler, fake, T0 + timedelta(hours=1)) == 1
    assert fake.retried == ["a2"]
    assert reconciler.store.get("r0")[0] == "blocked"  # type: ignore[index]
    assert any("r0" in r.getMessage() and "Rebuild" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_failures_in_backoff_still_count_as_peers(reconciler: Reconciler) -> None:
    """A failure that is not due this pass is still on its document: a later
    decision must not treat it as absent."""
    fake = FakeHindsight()
    fake.add("a1", "failed", 0)
    fake.add("r2", "failed", 1, items=[(DOC, "replace")])
    now = T0 + timedelta(hours=1)
    reconciler.store.record("a1", "explr", "retrying", 3, now)  # in backoff
    await _run(reconciler, fake, now)
    assert fake.retried == ["r2"]  # a1 waits its turn, and is not superseded
    assert reconciler.store.get("a1")[0] == "retrying"  # type: ignore[index]


@pytest.mark.asyncio
async def test_completed_payloads_are_fetched_once(reconciler: Reconciler) -> None:
    fake = FakeHindsight()
    fake.add("f0", "failed", 0)
    fake.add("c1", "completed", 1)
    await _run(reconciler, fake, T0 + timedelta(hours=1))
    fake.settle("failed")
    fake.payload_fetches.clear()
    await _run(reconciler, fake, T0 + timedelta(days=1))
    assert fake.payload_fetches == ["f0"]


@pytest.mark.asyncio
async def test_in_flight_document_waits_and_a_stuck_op_is_reported(
        reconciler: Reconciler, caplog) -> None:
    fake = FakeHindsight()
    fake.add("f0", "failed", 0, items=[(DOC, "replace")])
    fake.add("p1", "processing", 3)
    with caplog.at_level(logging.WARNING):
        assert await _run(reconciler, fake, T0 + timedelta(hours=3)) == 0
    assert fake.retried == []
    assert any("p1 has been processing" in r.getMessage() for r in caplog.records)
    fake.ops["p1"]["status"] = "completed"
    await _run(reconciler, fake, T0 + timedelta(hours=4))
    assert reconciler.store.get("f0")[0] == "blocked"  # type: ignore[index]


@pytest.mark.asyncio
async def test_paging_dedupes_ops_shifted_by_new_arrivals(reconciler: Reconciler) -> None:
    fake = FakeHindsight()
    fake.add("f0", "failed", 0)
    for i in range(1, 151):
        fake.add(f"c{i}", "completed", i, items=[(f"other:{i}", "append")])
    arrivals = iter(range(1000, 1100))

    def new_head(offset: Optional[int]) -> None:
        if offset:  # a turn lands between pages
            n = next(arrivals)
            fake.add(f"n{n}", "completed", n, items=[(f"new:{n}", "append")])

    fake.on_page = new_head
    assert await _run(reconciler, fake, T0 + timedelta(days=1)) == 1
    assert fake.retried == ["f0"]
    assert len(fake.payload_fetches) == len(set(fake.payload_fetches))


@pytest.mark.asyncio
async def test_a_refused_retry_does_not_stop_the_rest(reconciler: Reconciler) -> None:
    fake = FakeHindsight()
    fake.add("bad", "failed", 0, items=[("doc:a", "append")])
    fake.add("good", "failed", 1, items=[("doc:b", "append")])
    fake.refuse.add("bad")
    assert await _run(reconciler, fake, T0 + timedelta(hours=1)) == 1
    assert fake.retried == ["good"]


@pytest.mark.asyncio
async def test_one_bank_failing_does_not_stop_the_rest(reconciler: Reconciler) -> None:
    fake = FakeHindsight()
    fake.add("a", "failed", 0, bank="broken")
    fake.add("b", "failed", 0, bank="fine")
    real = fake.alist_operations

    async def flaky(bank_id: str, **kw: Any) -> Dict[str, Any]:
        if bank_id == "broken":
            raise httpx.ConnectError("boom")
        return await real(bank_id, **kw)

    fake.alist_operations = flaky  # type: ignore[method-assign]
    assert await _run(reconciler, fake, T0 + timedelta(hours=1)) == 1
    assert fake.retried == ["b"]


# ---------- Store: lives beside an existing doc-scope DB ----------


def test_store_adds_its_tables_to_an_existing_doc_scope_db(tmp_path) -> None:
    path = str(tmp_path / "doc_scope.db")
    scope = _DocScopeStore(path)
    doc_id, _ = scope.resolve("explr:web_ui", T0)
    scope.close()

    store = ReconcileStore(path)
    store.record("op1", "explr", "blocked", 2, T0)
    store.cache_docs("op2", {DOC: "replace"})
    store.close()
    store = ReconcileStore(path)  # idempotent reopen
    assert store.get("op1") == ("blocked", 2, T0)
    assert store.cached_docs("op2") == {DOC: "replace"}
    store.close()

    conn = sqlite3.connect(path)
    assert conn.execute("SELECT document_id FROM Doc_Scope").fetchone()[0] == doc_id
    conn.close()


# ---------- Backend entry point ----------


@pytest.fixture
def backend(tmp_path) -> HindsightBackend:
    return HindsightBackend(
        url="http://stub:8888",
        override_db_path=str(tmp_path / "overrides.db"),
        doc_scope_db_path=str(tmp_path / "doc_scope.db"),
    )


@pytest.mark.asyncio
async def test_backend_reconciles_only_its_own_banks(backend: HindsightBackend) -> None:
    run = AsyncMock(return_value=0)
    client = backend._get_client()
    with patch.object(backend._reconciler, "run", run):
        # Nothing provisioned yet: no pass, and the throttle stays open.
        assert await backend.reconcile_failed_retains() == 0
        assert run.await_count == 0
        with patch.object(client, "_request", AsyncMock(return_value={})):
            await backend.ensure_bank("explr")
        await backend.reconcile_failed_retains()
    assert run.await_args.args[1] == ["explr"]  # never lme_* or other foreign banks
    await backend.aclose()


@pytest.mark.asyncio
async def test_backend_pass_is_throttled(backend: HindsightBackend) -> None:
    backend._own_banks.add("explr")
    run = AsyncMock(return_value=3)
    with patch.object(backend._reconciler, "run", run):
        assert await backend.reconcile_failed_retains() == 3
        assert await backend.reconcile_failed_retains() == 0
        assert run.await_count == 1
        backend._reconciled_at = datetime.now(timezone.utc) - timedelta(
            seconds=RETAIN_RECONCILE_INTERVAL_SECONDS + 1)
        assert await backend.reconcile_failed_retains() == 3
    await backend.aclose()


@pytest.mark.asyncio
async def test_backend_pass_never_raises(backend: HindsightBackend) -> None:
    backend._own_banks.add("explr")
    with patch.object(backend._reconciler, "run",
                      AsyncMock(side_effect=httpx.ConnectError("down"))):
        assert await backend.reconcile_failed_retains() == 0
    backend._reconciled_at = None
    with patch.object(backend._reconciler, "run", AsyncMock(side_effect=RuntimeError("bug"))):
        assert await backend.reconcile_failed_retains() == 0
    await backend.aclose()


@pytest.mark.asyncio
async def test_retry_route(backend: HindsightBackend) -> None:
    client = backend._get_client()
    seen: Dict[str, Any] = {}

    async def fake_request(method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        seen.update(method=method, path=path)
        return {"success": True}

    with patch.object(client, "_request", side_effect=fake_request):
        await client.aretry_operation("explr", "abc")
    assert seen == {"method": "POST",
                    "path": "/v1/default/banks/explr/operations/abc/retry"}
    await backend.aclose()
