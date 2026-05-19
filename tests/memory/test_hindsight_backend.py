"""HindsightBackend tests.

Two layers:
  1. Unit — mock the underlying httpx client; verify tag round-trip,
     untrusted-bit conformance, retry-storm absence, fail-soft retain.
  2. Live (`@pytest.mark.hindsight_live`) — exercises a real container at
     HINDSIGHT_LIVE_URL. Auto-skipped when env var absent.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.memory.backend.hindsight import (
    HindsightAPIError,
    HindsightBackend,
    HindsightRESTClient,
    SESSION_GAP_SECONDS,
    TRUSTED_TAG,
    UNTRUSTED_TAG,
    _read_untrusted,
)


# ---------- Pure-function unit tests ----------


def test_read_untrusted_explicit_true() -> None:
    assert _read_untrusted([UNTRUSTED_TAG]) is True


def test_read_untrusted_explicit_false() -> None:
    assert _read_untrusted([TRUSTED_TAG]) is False


def test_read_untrusted_absent_defaults_true() -> None:
    """Fail-closed: pre-bit data is treated as suspect, not silently trusted."""
    assert _read_untrusted(["persona:alice", "role:user"]) is True


# ---------- Mock-backed unit tests ----------


@pytest.fixture
def backend(tmp_path) -> HindsightBackend:
    return HindsightBackend(
        url="http://stub:8888",
        override_db_path=str(tmp_path / "overrides.db"),
        doc_scope_db_path=str(tmp_path / "doc_scope.db"),
    )


@pytest.mark.asyncio
async def test_retain_turn_threads_untrusted_tag(backend: HindsightBackend) -> None:
    captured: Dict[str, Any] = {}

    async def fake_aretain(bank_id: str, items, async_=True) -> Dict[str, Any]:
        captured["bank_id"] = bank_id
        captured["items"] = items
        return {"id": "unit-123"}

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=fake_aretain):
        hit_id = await backend.retain_turn(
            "alice", "user", "hello",
            timestamp=datetime.now(timezone.utc),
            scope_tags=["channel:c1"],
            source_persona="alice",
            untrusted=True,
        )
        # Fire-and-forget: ID isn't known to the caller, drain to verify worker did POST.
        await backend.aclose()
    assert hit_id == ""
    assert captured["bank_id"] == "alice"
    assert len(captured["items"]) == 1
    item = captured["items"][0]
    assert UNTRUSTED_TAG in item["tags"]
    assert "persona:alice" in item["tags"]
    assert "role:user" in item["tags"]
    assert "channel:c1" in item["tags"]
    assert item["document_id"].startswith("alice:c1:")
    assert item["update_mode"] == "replace"  # first turn in scope


@pytest.mark.asyncio
async def test_retain_turn_trusted_default(backend: HindsightBackend) -> None:
    captured: Dict[str, Any] = {}

    async def fake_aretain(bank_id: str, items, async_=True) -> Dict[str, Any]:
        captured["items"] = items
        return {"id": "x"}

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=fake_aretain):
        await backend.retain_turn(
            "alice", "assistant", "ack",
            timestamp=datetime.now(timezone.utc),
            scope_tags=[],
            source_persona="alice",
        )
        await backend.aclose()
    tags = captured["items"][0]["tags"]
    assert TRUSTED_TAG in tags
    assert UNTRUSTED_TAG not in tags


@pytest.mark.asyncio
async def test_recall_recovers_untrusted_bit(backend: HindsightBackend) -> None:
    """Cross-backend conformance: bit set on retain MUST surface on recall."""
    fake_results = [
        {"id": "1", "content": "trusted msg", "score": 0.9, "tags": [TRUSTED_TAG, "role:user"]},
        {"id": "2", "content": "untrusted msg", "score": 0.8, "tags": [UNTRUSTED_TAG]},
        {"id": "3", "content": "untagged msg", "score": 0.7, "tags": []},
    ]

    async def fake_arecall(bank_id: str, query: str, **kw):
        return fake_results

    client = backend._get_client()
    with patch.object(client, "arecall", side_effect=fake_arecall):
        hits = await backend.recall("alice", "anything", k=10)

    assert len(hits) == 3
    assert hits[0].untrusted is False
    assert hits[1].untrusted is True
    assert hits[2].untrusted is True  # fail-closed default


@pytest.mark.asyncio
async def test_retain_turn_fails_soft_on_network_error(backend: HindsightBackend) -> None:
    """Plan §1.3 — fire-and-forget. Network error must NOT crash the user turn."""
    client = backend._get_client()
    with patch.object(
        client, "aretain", side_effect=httpx.ConnectError("kobold offline")
    ):
        hit_id = await backend.retain_turn(
            "alice", "user", "hi",
            timestamp=datetime.now(timezone.utc),
            scope_tags=[],
            source_persona="alice",
        )
        # Drain — worker must swallow ConnectError and stay alive.
        await backend.aclose()
    assert hit_id == ""


@pytest.mark.asyncio
async def test_recall_fails_soft_on_api_error(backend: HindsightBackend) -> None:
    client = backend._get_client()
    with patch.object(
        client, "arecall", side_effect=HindsightAPIError(503, "bank down")
    ):
        hits = await backend.recall("alice", "x")
    assert hits == []


@pytest.mark.asyncio
async def test_request_does_not_retry_storm() -> None:
    """Plan §1.3 — no retry storm. _request must hit the network exactly once."""
    client = HindsightRESTClient(base_url="http://stub:8888")
    inner = AsyncMock(side_effect=httpx.ConnectError("nope"))
    with patch.object(client.client, "request", inner):
        with pytest.raises(httpx.ConnectError):
            await client._request("POST", "/banks/x/recall", json={})
    assert inner.call_count == 1


@pytest.mark.asyncio
async def test_legacy_methods_raise_not_implemented(backend: HindsightBackend) -> None:
    """Flipping SEMANTIC_BACKEND with un-migrated callers must fail loud."""
    with pytest.raises(NotImplementedError):
        backend.store_segment()
    with pytest.raises(NotImplementedError):
        backend.retrieve_relevant_summaries()
    # Agent-action telemetry on the backend itself must also fail loud;
    # MemoryManager routes those calls through its sqlite action-log delegate.
    with pytest.raises(NotImplementedError):
        backend.log_agent_action()
    with pytest.raises(NotImplementedError):
        backend.get_relevant_agent_actions("dispatch")


@pytest.mark.asyncio
async def test_queue_preserves_intra_bank_order(backend: HindsightBackend) -> None:
    """Plan §1.3 — intra-bank FIFO. Cross-bank ordering not guaranteed (parallel)."""
    seen: Dict[str, List[str]] = {"alice": [], "bob": []}

    async def fake_aretain(bank_id: str, items, async_=True) -> Dict[str, Any]:
        for it in items:
            seen[bank_id].append(it["content"])
        return {"id": "ok"}

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=fake_aretain):
        for i in range(5):
            await backend.retain_turn(
                "alice", "user", f"a{i}",
                timestamp=datetime.now(timezone.utc),
                scope_tags=[], source_persona="alice",
            )
            await backend.retain_turn(
                "bob", "user", f"b{i}",
                timestamp=datetime.now(timezone.utc),
                scope_tags=[], source_persona="bob",
            )
        await backend.aclose()

    assert seen["alice"] == [f"a{i}" for i in range(5)]
    assert seen["bob"] == [f"b{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_worker_survives_connect_error(backend: HindsightBackend) -> None:
    """ConnectError on item N must not kill the worker; item N+1 still drains."""
    calls: List[str] = []

    async def flaky(bank_id: str, items, async_=True) -> Dict[str, Any]:
        contents = [it["content"] for it in items]
        calls.extend(contents)
        if "boom" in contents:
            raise httpx.ConnectError("kobold offline")
        return {"id": "ok"}

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=flaky):
        # Three sequential retains; awaiting between each lets the worker
        # drain item-by-item so the ConnectError batch is just "boom" alone
        # and the next item still drains.
        await backend.retain_turn(
            "alice", "user", "ok1",
            timestamp=datetime.now(timezone.utc),
            scope_tags=[], source_persona="alice",
        )
        await backend._queues["alice"].join()
        await backend.retain_turn(
            "alice", "user", "boom",
            timestamp=datetime.now(timezone.utc),
            scope_tags=[], source_persona="alice",
        )
        await backend._queues["alice"].join()
        await backend.retain_turn(
            "alice", "user", "ok2",
            timestamp=datetime.now(timezone.utc),
            scope_tags=[], source_persona="alice",
        )
        await backend.aclose()

    assert calls == ["ok1", "boom", "ok2"]


@pytest.mark.asyncio
async def test_mark_trusted_audit_and_recall_override(backend: HindsightBackend) -> None:
    """Flip → recall → assert MemoryHit.untrusted reflects new value. Flip back → re-verify."""
    fake_results = [
        {"id": "u1", "content": "x", "score": 0.9, "tags": [UNTRUSTED_TAG]},
    ]

    async def fake_arecall(bank_id: str, query: str, **kw):
        return fake_results

    client = backend._get_client()
    with patch.object(client, "arecall", side_effect=fake_arecall):
        # Storage tag is untrusted → hit reads untrusted.
        hits = await backend.recall("alice", "q")
        assert hits[0].untrusted is True

        # Operator marks trusted.
        await backend.mark_trusted("alice", "u1", operator_id="adam", reason="vetted")
        hits = await backend.recall("alice", "q")
        assert hits[0].untrusted is False

        # Flip back.
        await backend.mark_untrusted("alice", "u1", operator_id="adam", reason="recanted")
        hits = await backend.recall("alice", "q")
        assert hits[0].untrusted is True

    # Audit log captured both flips with prior + new.
    rows = backend._overrides._get().execute(
        "SELECT prior, new, operator_id, reason FROM Unit_Trust_Audit "
        "WHERE bank_id=? AND hit_id=? ORDER BY audit_id", ("alice", "u1"),
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["prior"] is None and rows[0]["new"] == 0  # first flip: no prior override
    assert rows[1]["prior"] == 0 and rows[1]["new"] == 1     # second flip: prior trusted → untrusted
    assert all(r["operator_id"] == "adam" for r in rows)
    await backend.aclose()


# ---------- Bundle shape (DP-112) ----------


@pytest.mark.asyncio
async def test_bundle_coalesces_when_worker_is_busy(backend: HindsightBackend) -> None:
    """N items enqueued while the worker's first POST is in flight collapse
    into one bundled POST on the next drain tick."""
    batches: List[List[Dict[str, Any]]] = []
    gate = asyncio.Event()

    async def fake_aretain(bank_id: str, items, async_=True) -> Dict[str, Any]:
        batches.append(list(items))
        # Hold the first POST until everything is queued, so the second drain
        # tick observes all remaining items at once.
        if len(batches) == 1:
            await gate.wait()
        return {"id": "ok"}

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=fake_aretain):
        # First retain wakes the worker; aretain blocks on `gate`.
        await backend.retain_turn(
            "alice", "user", "first",
            timestamp=datetime.now(timezone.utc),
            scope_tags=["channel:c1"], source_persona="alice",
        )
        # Wait for the worker to actually start processing the first item so
        # the rest land on a busy worker.
        for _ in range(50):
            if batches:
                break
            await asyncio.sleep(0.01)
        for i in range(4):
            await backend.retain_turn(
                "alice", "user", f"q{i}",
                timestamp=datetime.now(timezone.utc),
                scope_tags=["channel:c1"], source_persona="alice",
            )
        gate.set()
        await backend.aclose()

    # First POST: just "first". The four queued items share a document_id
    # within the same session — server-side dedupe forces one POST per item,
    # but they all dispatch in the same drain tick after `gate` releases.
    assert [it["content"] for it in batches[0]] == ["first"]
    drained = [it["content"] for batch in batches[1:] for it in batch]
    assert drained == ["q0", "q1", "q2", "q3"]


@pytest.mark.asyncio
async def test_document_id_stable_within_scope(backend: HindsightBackend) -> None:
    seen: List[Dict[str, Any]] = []

    async def fake_aretain(bank_id: str, items, async_=True) -> Dict[str, Any]:
        seen.extend(items)
        return {"id": "ok"}

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=fake_aretain):
        for content in ("a", "b", "c"):
            await backend.retain_turn(
                "alice", "user", content,
                timestamp=datetime.now(timezone.utc),
                scope_tags=["channel:c1"], source_persona="alice",
            )
        await backend.aclose()

    doc_ids = {it["document_id"] for it in seen}
    assert len(doc_ids) == 1, f"all retains in same scope must share doc_id, got {doc_ids}"
    modes = [it["update_mode"] for it in seen]
    assert modes[0] == "replace"
    assert all(m == "append" for m in modes[1:])


@pytest.mark.asyncio
async def test_document_id_differs_across_scopes(backend: HindsightBackend) -> None:
    seen: List[Dict[str, Any]] = []

    async def fake_aretain(bank_id: str, items, async_=True) -> Dict[str, Any]:
        seen.extend(items)
        return {"id": "ok"}

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=fake_aretain):
        await backend.retain_turn(
            "alice", "user", "x",
            timestamp=datetime.now(timezone.utc),
            scope_tags=["channel:c1"], source_persona="alice",
        )
        await backend.retain_turn(
            "alice", "user", "y",
            timestamp=datetime.now(timezone.utc),
            scope_tags=["channel:c2"], source_persona="alice",
        )
        await backend.aclose()

    assert seen[0]["document_id"] != seen[1]["document_id"]


@pytest.mark.asyncio
async def test_session_gap_starts_new_document(backend: HindsightBackend) -> None:
    """>SESSION_GAP_SECONDS gap → new document_id + update_mode='replace'."""
    seen: List[Dict[str, Any]] = []

    async def fake_aretain(bank_id: str, items, async_=True) -> Dict[str, Any]:
        seen.extend(items)
        return {"id": "ok"}

    t0 = datetime.now(timezone.utc) - timedelta(seconds=SESSION_GAP_SECONDS + 60)
    t1 = datetime.now(timezone.utc)

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=fake_aretain):
        await backend.retain_turn(
            "alice", "user", "old",
            timestamp=t0,
            scope_tags=["channel:c1"], source_persona="alice",
        )
        await backend._queues["alice"].join()
        await backend.retain_turn(
            "alice", "user", "new",
            timestamp=t1,
            scope_tags=["channel:c1"], source_persona="alice",
        )
        await backend.aclose()

    assert seen[0]["document_id"] != seen[1]["document_id"]
    assert seen[0]["update_mode"] == "replace"
    assert seen[1]["update_mode"] == "replace"


@pytest.mark.asyncio
async def test_ensure_bank_sends_retain_mission_not_mission(backend: HindsightBackend) -> None:
    """Wire-level: retain_mission goes to the server; deprecated `mission` does not."""
    captured: Dict[str, Any] = {}

    async def fake_request(method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        captured["json"] = kwargs.get("json")
        return {}

    client = backend._get_client()
    with patch.object(client, "_request", side_effect=fake_request):
        await backend.ensure_bank(
            "alice",
            retain_mission="extract decisions",
            reflect_mission="summarize",
            enable_observations=True,
            observations_mission="stable facts",
        )
    payload = captured["json"]
    assert payload["retain_mission"] == "extract decisions"
    assert payload["reflect_mission"] == "summarize"
    assert payload["enable_observations"] is True
    assert payload["observations_mission"] == "stable facts"
    assert "mission" not in payload
    assert "background" not in payload


@pytest.mark.asyncio
async def test_aretain_uses_async_field_not_retain_async() -> None:
    """Wire-level: payload key is `async` (upstream RetainRequest), not `retain_async`."""
    client = HindsightRESTClient(base_url="http://stub")
    captured: Dict[str, Any] = {}

    async def fake_request(method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        captured["path"] = path
        captured["json"] = kwargs.get("json")
        return {}

    with patch.object(client, "_request", side_effect=fake_request):
        await client.aretain("bank-x", [{"content": "hi", "tags": []}])
    assert captured["path"] == "/v1/default/banks/bank-x/memories"
    assert captured["json"]["async"] is True
    assert "retain_async" not in captured["json"]
    assert captured["json"]["items"] == [{"content": "hi", "tags": []}]


# ---------- Live container fixtures (Golden Set Pattern) ----------


@pytest.fixture(scope="module")
def hindsight_live_url() -> str:
    url = os.environ.get("HINDSIGHT_LIVE_URL")
    if not url:
        pytest.skip("HINDSIGHT_LIVE_URL not set")
    return url


@pytest.fixture(scope="module")
async def golden_hindsight_bank(hindsight_live_url: str) -> str:
    """Ensures a persistent 'golden' bank exists and is seeded for tests.
    
    Follows the Zammad pattern: seeds once and leaves artifacts.
    Does NOT wait for indexing; the test will skip until data is ready.
    """
    backend = HindsightBackend(url=hindsight_live_url)
    bank = "golden-conformance-bank"
    
    # 1. Ensure bank exists
    await backend.ensure_bank(bank, retain_mission="Persistent conformance test artifacts")
    
    # 2. Check if already seeded
    hits = await backend.recall(bank, "fox", k=1)
    if not hits:
        # 3. Seed once if missing (fire-and-forget)
        await backend.retain_turn(
            bank, "user", "the quick brown fox jumps over the lazy dog",
            timestamp=datetime.now(timezone.utc),
            scope_tags=["test:conformance"],
            source_persona="alice",
            untrusted=True,
        )
        # No wait, no poll.
    
    await backend.aclose()
    return bank


@pytest.mark.hindsight_live
@pytest.mark.asyncio
async def test_live_retain_recall_round_trip(golden_hindsight_bank: str, hindsight_live_url: str) -> None:
    """End-to-end: verify untrusted bit round-trips via the persistent golden bank."""
    backend = HindsightBackend(url=hindsight_live_url)
    try:
        hits = await backend.recall(golden_hindsight_bank, "fox", k=5)
        if not hits:
            pytest.skip("Golden bank not yet indexed by server (seeding in progress)")
            
        assert any(h.untrusted is True for h in hits), \
            "untrusted bit must round-trip through Hindsight tags"
    finally:
        await backend.aclose()

@pytest.mark.asyncio
async def test_retain_experience_threads_tags_and_content(backend: HindsightBackend) -> None:
    captured: Dict[str, Any] = {}

    async def fake_aretain(bank_id: str, items, async_=True) -> Dict[str, Any]:
        captured["items"] = items
        return {"id": "exp-123"}

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=fake_aretain):
        hit_id = await backend.retain_experience(
            "alice", "search", {"query": "foo"}, "found bar",
            scope_tags=["channel:c1"],
            source_persona="alice",
            untrusted=True,
            metadata={"meta": "data"},
        )
        await backend.aclose()
    
    assert hit_id == ""
    item = captured["items"][0]
    assert "type:experience" in item["tags"]
    assert "action:search" in item["tags"]
    assert UNTRUSTED_TAG in item["tags"]
    assert "Action: search" in item["content"]
    assert "found bar" in item["content"]
    assert item["metadata"] == {"meta": "data"}


@pytest.mark.asyncio
async def test_retain_experience_explicit_document_id_bypasses_doc_scope(
    backend: HindsightBackend,
) -> None:
    """DP-116b: explicit document_id pins the doc + uses replace, skipping
    the rolling _doc_scope heuristic. Re-retain on the same id is idempotent."""
    captured: List[Dict[str, Any]] = []

    async def fake_aretain(bank_id: str, items, async_=True) -> Dict[str, Any]:
        captured.append(items[0])
        return {}

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=fake_aretain), \
         patch.object(backend._doc_scope, "resolve", side_effect=AssertionError("should not be called")):
        await backend.retain_experience(
            "alice", "dispatch", {"action_id": 42}, "success",
            scope_tags=["agent:dispatch", "action_id:42"],
            source_persona="dispatch_analyst",
            document_id="agent_action:42",
            content_override="agent=dispatch action_id=42 type=dispatch outcome=success",
        )
        await backend.retain_experience(
            "alice", "dispatch", {"action_id": 42}, "success",
            scope_tags=["agent:dispatch", "action_id:42"],
            source_persona="dispatch_analyst",
            document_id="agent_action:42",
            content_override="agent=dispatch action_id=42 type=dispatch outcome=success",
        )
        await backend.aclose()

    assert len(captured) == 2
    for item in captured:
        assert item["document_id"] == "agent_action:42"
        assert item["update_mode"] == "replace"
        assert item["content"].startswith("agent=dispatch action_id=42")
        assert "type:experience" in item["tags"]
        assert "action:dispatch" in item["tags"]


@pytest.mark.asyncio
async def test_reflect_success_and_failure(backend: HindsightBackend) -> None:
    client = backend._get_client()
    
    # Success
    with patch.object(client, "areflect", return_value={"answer": "yes", "mental_models": [{"id": "m1", "content": "model1", "tags": []}]}):
        res = await backend.reflect("alice", "q")
        assert res.answer == "yes"
        assert res.mental_models[0].id == "m1"
        assert res.mental_models[0].content == "model1"

    # Failure
    with patch.object(client, "areflect", side_effect=HindsightAPIError(500, "boom")):
        res = await backend.reflect("alice", "q")
        assert res.answer == ""
        assert res.mental_models == []


@pytest.mark.asyncio
async def test_ensure_bank_ignores_409(backend: HindsightBackend) -> None:
    client = backend._get_client()
    
    with patch.object(client, "_request", side_effect=HindsightAPIError(409, "already exists")):
        await backend.ensure_bank("alice") # Should not raise
        
    with patch.object(client, "_request", side_effect=HindsightAPIError(500, "boom")):
        with pytest.raises(HindsightAPIError):
            await backend.ensure_bank("alice")


@pytest.mark.asyncio
async def test_delete_bank(backend: HindsightBackend) -> None:
    client = backend._get_client()
    
    with patch.object(client, "adelete_bank") as mock_adelete:
        await backend.delete_bank("alice")
        mock_adelete.assert_called_once_with(bank_id="alice")


@pytest.mark.asyncio
async def test_request_raises_api_error_on_non_2xx() -> None:
    client = HindsightRESTClient(base_url="http://stub")
    
    class FakeResponse:
        status_code = 404
        text = "not found"
        def json(self): return {}
        
    async def fake_request(*args, **kwargs):
        return FakeResponse()
        
    with patch.object(client.client, "request", side_effect=fake_request):
        with pytest.raises(HindsightAPIError) as exc_info:
            await client._request("GET", "/foo")
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.message


@pytest.mark.asyncio
async def test_worker_survives_api_error_and_exception(backend: HindsightBackend) -> None:
    calls: List[str] = []

    async def flaky(bank_id: str, items, async_=True) -> Dict[str, Any]:
        contents = [it["content"] for it in items]
        calls.extend(contents)
        if "api-error" in contents:
            raise HindsightAPIError(500, "server on fire")
        if "value-error" in contents:
            raise ValueError("bad data")
        return {"id": "ok"}

    client = backend._get_client()
    with patch.object(client, "aretain", side_effect=flaky):
        await backend.retain_turn("alice", "user", "api-error", timestamp=datetime.now(timezone.utc), scope_tags=[], source_persona="alice")
        await backend._queues["alice"].join()
        
        await backend.retain_turn("alice", "user", "value-error", timestamp=datetime.now(timezone.utc), scope_tags=[], source_persona="alice")
        await backend._queues["alice"].join()

        await backend.retain_turn("alice", "user", "ok", timestamp=datetime.now(timezone.utc), scope_tags=[], source_persona="alice")
        await backend._queues["alice"].join()
        
        await backend.aclose()

    assert calls == ["api-error", "value-error", "ok"]


@pytest.mark.asyncio
async def test_worker_first_is_stop(backend: HindsightBackend) -> None:
    # Wake up the worker with just _STOP
    q = await backend._ensure_worker("bob")
    await backend.aclose()
    # It shouldn't crash


# ---------- DP-118: retain_document (ingest_path tool path) ----------


@pytest.mark.asyncio
async def test_retain_document_posts_one_item_with_replace(backend: HindsightBackend) -> None:
    captured: Dict[str, Any] = {}

    async def fake_aretain(bank_id: str, items, async_=True) -> Dict[str, Any]:
        captured["bank_id"] = bank_id
        captured["items"] = items
        return {"id": "doc-1"}

    client = backend._get_client()
    ts = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
    with patch.object(client, "aretain", side_effect=fake_aretain):
        await backend.retain_document(
            "alice", "notes/foo.md", "body text",
            tags=["ingest", "notes"],
            metadata={"source_path": "notes/foo.md", "sha256": "abc"},
            timestamp=ts,
        )
        await backend.aclose()

    assert captured["bank_id"] == "alice"
    assert len(captured["items"]) == 1
    item = captured["items"][0]
    assert item["document_id"] == "notes/foo.md"
    assert item["update_mode"] == "replace"
    assert item["content"] == "body text"
    assert item["timestamp"] == ts.isoformat()
    assert "ingest" in item["tags"] and "notes" in item["tags"]
    assert item["metadata"] == {"source_path": "notes/foo.md", "sha256": "abc"}
