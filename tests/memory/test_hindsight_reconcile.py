# tests/memory/test_hindsight_reconcile.py
#
# DP-443: re-running retains Hindsight accepted and then failed to extract.
# The planner is pure; the Reconciler runs against an in-memory fake of the
# operations API that keeps Hindsight's semantics: newest-first listing,
# `limit` <= 100, `/retry` flips failed -> pending, payloads stored forever.

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.memory.backend.base import (
    RETAIN_RECONCILE_INTERVAL_SECONDS,
    RETAIN_RECONCILE_MAX_ATTEMPTS,
)
from src.memory.backend.hindsight import HindsightBackend, _DocScopeStore
from src.memory.backend.hindsight_reconcile import (
    Op,
    Reconciler,
    ReconcileStore,
    backoff_seconds,
    merge_replay,
    plan_document,
)

T0 = datetime(2026, 10, 15, 5, 0, tzinfo=timezone.utc)
DOC = "explr:web_ui:2026-10-09T20:51:00+00:00"


def _op(i: int, status: str, mode: Optional[str] = None) -> Op:
    return Op(f"op{i}", status, T0 + timedelta(minutes=i), mode)


# ---------- plan_document ----------


def test_failures_older_than_a_completed_replace_are_superseded() -> None:
    """The live explr case: 17 failures, then DP-409 rebuilds that completed.
    Retrying the oldest (a short replace) would wipe the rebuilt document."""
    ops = [_op(0, "failed", "replace"), _op(1, "failed", "append"),
           _op(2, "failed", "replace"), _op(3, "completed", "replace"),
           _op(4, "completed", "replace")]
    plan = plan_document(ops)
    assert plan.superseded == ["op0", "op1", "op2"]
    assert plan.retry == [] and plan.replay == []


def test_failed_tail_is_retried_in_place_in_order() -> None:
    ops = [_op(0, "failed", "replace"), _op(1, "failed", "append"), _op(2, "failed", "append")]
    plan = plan_document(ops)
    assert plan.retry == ["op0", "op1", "op2"]
    assert plan.superseded == [] and plan.replay == []


def test_failed_append_before_a_completed_append_is_retried() -> None:
    plan = plan_document([_op(0, "failed", "append"), _op(1, "completed", "append")])
    assert plan.retry == ["op0"] and plan.replay == []


def test_failed_replace_with_a_later_completed_op_is_replayed() -> None:
    """Retrying the replace in place would erase the append that landed after it."""
    ops = [_op(0, "failed", "append"), _op(1, "failed", "replace"),
           _op(2, "completed", "append"), _op(3, "failed", "append")]
    plan = plan_document(ops)
    assert plan.replay == ["op1", "op2", "op3"]
    assert plan.replay_failed == ["op0", "op1", "op3"]
    assert plan.retry == []


def test_document_with_an_op_in_flight_is_left_alone() -> None:
    for status in ("pending", "processing"):
        plan = plan_document([_op(0, "failed", "replace"), _op(1, status)])
        assert (plan.superseded, plan.retry, plan.replay) == ([], [], [])


def test_cancelled_ops_are_ignored() -> None:
    plan = plan_document([_op(0, "failed", "append"), _op(1, "cancelled")])
    assert plan.retry == ["op0"]


# ---------- merge_replay / backoff ----------


def test_merge_replay_concatenates_in_order_as_one_replace() -> None:
    merged = merge_replay([
        [{"content": "A", "document_id": DOC, "update_mode": "replace",
          "tags": ["channel:c", "persona:p", "untrusted:false"], "event_date": "t1"}],
        [{"content": "B", "document_id": DOC, "update_mode": "append",
          "tags": ["channel:c", "persona:p", "role:user", "untrusted:true"],
          "event_date": "t2", "metadata": {"interaction_id": "7"}}],
    ])
    assert merged["content"] == "A\n\nB"
    assert merged["update_mode"] == "replace"
    assert merged["document_id"] == DOC
    # Trust is sticky across the document (DP-409); role tags are per-turn.
    assert merged["tags"] == ["channel:c", "persona:p", "untrusted:true"]
    assert merged["event_date"] == "t2"
    assert merged["metadata"] == {"interaction_id": "7"}


def test_backoff_doubles_to_the_cap_and_spans_days() -> None:
    assert backoff_seconds(0) == 0
    assert backoff_seconds(1) == RETAIN_RECONCILE_INTERVAL_SECONDS
    assert backoff_seconds(2) == 2 * RETAIN_RECONCILE_INTERVAL_SECONDS
    assert backoff_seconds(50) == 6 * 3600
    total = sum(backoff_seconds(n) for n in range(1, RETAIN_RECONCILE_MAX_ATTEMPTS))
    assert total > 3 * 24 * 3600  # a multi-day outage still recovers


# ---------- Reconciler against a fake operations API ----------


class FakeHindsight:
    def __init__(self) -> None:
        self.ops: Dict[str, Dict[str, Any]] = {}
        self.retried: List[str] = []
        self.posted: List[List[Dict[str, Any]]] = []
        self.payload_fetches = 0

    def add(self, op_id: str, status: str, minute: int, *, doc: Optional[str] = DOC,
            mode: str = "append", content: str = "", bank: str = "explr",
            tags: Optional[List[str]] = None) -> None:
        self.ops[op_id] = {
            "id": op_id, "bank": bank, "status": status, "task_type": "retain",
            "document_id": doc,
            "created_at": (T0 + timedelta(minutes=minute)).isoformat(),
            "items": [{"content": content or op_id, "document_id": doc, "update_mode": mode,
                       "tags": tags or ["channel:web_ui", "persona:explr"]}],
        }

    async def alist_banks(self) -> Dict[str, Any]:
        return {"banks": [{"bank_id": b} for b in sorted({o["bank"] for o in self.ops.values()})]}

    async def alist_operations(self, bank_id: str, *, status: Optional[str] = None,
                               op_type: Optional[str] = None, limit: Optional[int] = None,
                               offset: Optional[int] = None,
                               exclude_parents: Optional[bool] = None) -> Dict[str, Any]:
        assert limit is not None and limit <= 100
        rows = [o for o in self.ops.values() if o["bank"] == bank_id
                and (status is None or o["status"] == status)]
        rows.sort(key=lambda o: o["created_at"], reverse=True)
        start = offset or 0
        page = [{k: v for k, v in o.items() if k != "items"} for o in rows[start:start + limit]]
        return {"total": len(rows), "operations": page}

    async def aget_operation(self, bank_id: str, operation_id: str, *,
                             include_payload: bool = False) -> Dict[str, Any]:
        self.payload_fetches += 1
        return {"operation_id": operation_id,
                "task_payload": {"contents": self.ops[operation_id]["items"]}}

    async def aretry_operation(self, bank_id: str, operation_id: str) -> Dict[str, Any]:
        assert self.ops[operation_id]["status"] == "failed"
        self.ops[operation_id]["status"] = "pending"
        self.retried.append(operation_id)
        return {"success": True}

    async def aretain(self, bank_id: str, items: List[Dict[str, Any]], *,
                      async_: bool = True) -> Dict[str, Any]:
        self.posted.append(items)
        return {"success": True}

    def settle(self, status: str) -> None:
        for o in self.ops.values():
            if o["status"] == "pending":
                o["status"] = status


@pytest.fixture
def reconciler(tmp_path) -> Reconciler:
    return Reconciler(ReconcileStore(str(tmp_path / "doc_scope.db")))


@pytest.mark.asyncio
async def test_superseded_document_is_left_untouched(reconciler: Reconciler) -> None:
    fake = FakeHindsight()
    fake.add("f0", "failed", 0, mode="replace", content="short")
    fake.add("f1", "failed", 1)
    fake.add("c2", "completed", 60, mode="replace", content="whole session")
    assert await reconciler.run(fake, T0 + timedelta(days=2)) == 2
    assert fake.retried == [] and fake.posted == []
    # Resolved for good: the next pass does not even fetch payloads.
    fetches = fake.payload_fetches
    assert await reconciler.run(fake, T0 + timedelta(days=3)) == 0
    assert fake.payload_fetches == fetches


@pytest.mark.asyncio
async def test_failed_ops_are_retried_with_backoff_then_given_up(reconciler: Reconciler) -> None:
    fake = FakeHindsight()
    fake.add("f0", "failed", 0, mode="replace")
    fake.add("f1", "failed", 1)
    now = T0 + timedelta(hours=1)
    assert await reconciler.run(fake, now) == 2
    assert fake.retried == ["f0", "f1"]

    fake.settle("failed")  # the model server is still down
    fake.retried.clear()
    assert await reconciler.run(fake, now + timedelta(minutes=5)) == 0
    assert fake.retried == []  # inside the backoff window
    now += timedelta(seconds=backoff_seconds(1))
    assert await reconciler.run(fake, now) == 2

    for _ in range(RETAIN_RECONCILE_MAX_ATTEMPTS):
        fake.settle("failed")
        now += timedelta(seconds=backoff_seconds(RETAIN_RECONCILE_MAX_ATTEMPTS))
        await reconciler.run(fake, now)
    assert reconciler.store.get("f0")[0] == "given_up"  # type: ignore[index]
    fake.retried.clear()
    await reconciler.run(fake, now + timedelta(days=30))
    assert fake.retried == []


@pytest.mark.asyncio
async def test_recovered_retry_drops_out(reconciler: Reconciler) -> None:
    fake = FakeHindsight()
    fake.add("f0", "failed", 0)
    await reconciler.run(fake, T0 + timedelta(hours=1))
    fake.settle("completed")
    fake.retried.clear()
    assert await reconciler.run(fake, T0 + timedelta(days=1)) == 0
    assert fake.retried == []


@pytest.mark.asyncio
async def test_failed_replace_superseded_by_append_is_replayed_once(reconciler: Reconciler) -> None:
    fake = FakeHindsight()
    fake.add("f0", "failed", 0, mode="replace", content="rebuilt session")
    fake.add("c1", "completed", 5, content="later turn",
             tags=["channel:web_ui", "persona:explr", "role:user"])
    assert await reconciler.run(fake, T0 + timedelta(hours=1)) == 1
    assert fake.retried == []
    assert len(fake.posted) == 1 and len(fake.posted[0]) == 1
    item = fake.posted[0][0]
    assert item["update_mode"] == "replace"
    assert item["content"] == "rebuilt session\n\nlater turn"
    assert item["document_id"] == DOC
    assert "role:user" not in item["tags"]
    assert reconciler.store.get("f0")[0] == "replayed"  # type: ignore[index]
    assert await reconciler.run(fake, T0 + timedelta(days=1)) == 0
    assert len(fake.posted) == 1


@pytest.mark.asyncio
async def test_in_flight_document_waits(reconciler: Reconciler) -> None:
    fake = FakeHindsight()
    fake.add("f0", "failed", 0, mode="replace")
    fake.add("p1", "pending", 3)
    assert await reconciler.run(fake, T0 + timedelta(hours=1)) == 0
    assert fake.retried == []
    fake.ops["p1"]["status"] = "completed"
    assert await reconciler.run(fake, T0 + timedelta(hours=2)) == 1
    assert fake.posted  # f0 is a replace with a completed op after it


@pytest.mark.asyncio
async def test_documentless_batch_and_other_banks_are_retried(reconciler: Reconciler) -> None:
    fake = FakeHindsight()
    fake.add("loose", "failed", 0, doc=None)
    fake.add("exp", "failed", 1, doc="agent_action:246", mode="replace", bank="managr")
    assert await reconciler.run(fake, T0 + timedelta(hours=1)) == 2
    assert sorted(fake.retried) == ["exp", "loose"]


@pytest.mark.asyncio
async def test_paging_reaches_back_to_the_oldest_failure(reconciler: Reconciler) -> None:
    """More than one page of newer ops sits between the failure and now."""
    fake = FakeHindsight()
    fake.add("f0", "failed", 0, mode="replace", content="old")
    for i in range(1, 251):
        fake.add(f"c{i}", "completed", i, doc=f"other:{i}")
    fake.add("c_last", "completed", 300, content="new turn")
    await reconciler.run(fake, T0 + timedelta(days=1))
    assert fake.posted and fake.posted[0][0]["content"] == "old\n\nnew turn"


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
    assert await reconciler.run(fake, T0 + timedelta(hours=1)) == 1
    assert fake.retried == ["b"]


# ---------- Store: lives beside an existing doc-scope DB ----------


def test_store_adds_its_table_to_an_existing_doc_scope_db(tmp_path) -> None:
    path = str(tmp_path / "doc_scope.db")
    scope = _DocScopeStore(path)
    doc_id, _ = scope.resolve("explr:web_ui", T0)
    scope.close()

    store = ReconcileStore(path)
    store.record("op1", "explr", "superseded", 0, T0)
    assert store.get("op1") == ("superseded", 0, T0)
    store.close()
    store = ReconcileStore(path)  # idempotent reopen
    assert store.get("op1") == ("superseded", 0, T0)
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
async def test_backend_pass_is_throttled_unless_forced(backend: HindsightBackend) -> None:
    run = AsyncMock(return_value=3)
    with patch.object(backend._reconciler, "run", run):
        assert await backend.reconcile_failed_retains() == 3
        assert await backend.reconcile_failed_retains() == 0
        assert run.await_count == 1
        assert await backend.reconcile_failed_retains(force=True) == 3
        assert run.await_count == 2
    await backend.aclose()


@pytest.mark.asyncio
async def test_backend_pass_never_raises(backend: HindsightBackend) -> None:
    client = backend._get_client()
    with patch.object(client, "alist_banks", AsyncMock(side_effect=httpx.ConnectError("down"))):
        assert await backend.reconcile_failed_retains() == 0
    with patch.object(backend._reconciler, "run", AsyncMock(side_effect=RuntimeError("bug"))):
        assert await backend.reconcile_failed_retains(force=True) == 0
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
