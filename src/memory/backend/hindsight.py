# src/memory/backend/hindsight.py
from __future__ import annotations
import asyncio
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Tuple, TYPE_CHECKING, cast
import httpx
from .base import MemoryBackend, MemoryHit, Experience, ReflectResult, MentalModel

if TYPE_CHECKING:
    from src.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

UNTRUSTED_TAG = "untrusted:true"
TRUSTED_TAG = "untrusted:false"


class HindsightAPIError(Exception):
    """Raised for non-2xx responses from the Hindsight API."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Hindsight API Error {status_code}: {message}")


class HindsightRESTClient:
    """Audited, native implementation of the Hindsight SDK logic.

    Apes the official `hindsight-client` using the project's trusted httpx,
    bypassing the install-time attack surface during the supply-chain quarantine.
    See memory/project/decisions/2026-05-05-hindsight-paranoid-mode.md.

    Failure policy: log + raise once. NO retry storm — alpha system, retain is
    fire-and-forget at the backend layer; recall is user-facing and should fail
    fast so the caller can degrade gracefully.
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def _request(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = await self.client.request(method, url, **kwargs)
        except httpx.RequestError as e:
            logger.warning("Hindsight network error %s %s: %s", method, path, e)
            raise
        if not (200 <= response.status_code < 300):
            raise HindsightAPIError(response.status_code, response.text)
        return cast(Dict[str, Any], response.json())

    async def aretain(self, bank_id: str, content: str, tags: List[str]) -> Dict[str, Any]:
        payload = {
            "content": content,
            "tags": tags,
            "retain_async": True,  # async consolidation per plan §1.3
        }
        return await self._request("POST", f"/banks/{bank_id}/retain", json=payload)

    async def arecall(
        self, bank_id: str, query: str, k: int = 10, tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        payload = {"query": query, "k": k, "tags": tags or []}
        result = await self._request("POST", f"/banks/{bank_id}/recall", json=payload)
        return cast(List[Dict[str, Any]], result.get("results", []))

    async def areflect(
        self, bank_id: str, query: str, tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        payload = {"query": query, "tags": tags or []}
        return await self._request("POST", f"/banks/{bank_id}/reflect", json=payload)

    async def acreate_bank(
        self,
        bank_id: str,
        mission: Optional[str] = None,
        reflect_mission: Optional[str] = None,
    ) -> None:
        payload = {
            "bank_id": bank_id,
            "mission": mission,
            "reflect_mission": reflect_mission,
        }
        try:
            await self._request("POST", "/banks", json=payload)
        except HindsightAPIError as e:
            if e.status_code == 409:  # already exists
                return
            raise

    async def adelete_bank(self, bank_id: str) -> None:
        await self._request("DELETE", f"/banks/{bank_id}")


def _untrusted_tag(untrusted: bool) -> str:
    return UNTRUSTED_TAG if untrusted else TRUSTED_TAG


def _read_untrusted(tags: List[str]) -> bool:
    """Recover the bit from the recall result's tag list.

    Defaults to True (untrusted) when the tag is absent — fail-closed:
    pre-bit data should be treated as suspect rather than silently trusted.
    """
    if UNTRUSTED_TAG in tags:
        return True
    if TRUSTED_TAG in tags:
        return False
    return True


# Sentinel pushed onto a per-bank queue to stop the worker.
_STOP = object()


class _TrustOverrideStore:
    """Parallel SQLite store for per-unit trust flips (DP-110 option c).

    Rationale: upstream Hindsight v0.5.0 has no unit-tag PATCH endpoint. Rather
    than supersede via delete+re-retain (loses chunk identity, may break the
    cross-encoder reranker) or block on an upstream PR, we maintain a small
    override table here. Recall post-filters and rewrites the untrusted bit.

    Two tables:
      - `Unit_Trust_State` — current effective override per (bank, hit).
      - `Unit_Trust_Audit` — append-only log of every flip with operator + reason.

    Sync sqlite3 in async land is OK at this volume: flips are rare operator
    actions; the read path is one indexed SELECT per recall.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS Unit_Trust_State (
        bank_id   TEXT NOT NULL,
        hit_id    TEXT NOT NULL,
        untrusted INTEGER NOT NULL CHECK(untrusted IN (0, 1)),
        updated_at TEXT NOT NULL,
        PRIMARY KEY (bank_id, hit_id)
    );
    CREATE TABLE IF NOT EXISTS Unit_Trust_Audit (
        audit_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        bank_id     TEXT NOT NULL,
        hit_id      TEXT NOT NULL,
        prior       INTEGER,
        new         INTEGER NOT NULL CHECK(new IN (0, 1)),
        operator_id TEXT NOT NULL,
        reason      TEXT NOT NULL,
        ts          TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_trust_audit_bank_hit
        ON Unit_Trust_Audit(bank_id, hit_id);
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

    def set(self, bank_id: str, hit_id: str, untrusted: bool,
            operator_id: str, reason: str) -> Optional[bool]:
        """Upsert state + append audit row. Return the prior override value (or None)."""
        with self._lock:
            conn = self._get()
            now = datetime.now(timezone.utc).isoformat()
            row = conn.execute(
                "SELECT untrusted FROM Unit_Trust_State WHERE bank_id=? AND hit_id=?",
                (bank_id, hit_id),
            ).fetchone()
            prior = bool(row["untrusted"]) if row is not None else None
            conn.execute(
                "INSERT INTO Unit_Trust_Audit (bank_id, hit_id, prior, new, operator_id, reason, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (bank_id, hit_id,
                 (1 if prior else 0) if prior is not None else None,
                 1 if untrusted else 0,
                 operator_id, reason, now),
            )
            conn.execute(
                "INSERT INTO Unit_Trust_State (bank_id, hit_id, untrusted, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(bank_id, hit_id) DO UPDATE SET "
                "untrusted=excluded.untrusted, updated_at=excluded.updated_at",
                (bank_id, hit_id, 1 if untrusted else 0, now),
            )
            conn.commit()
            return prior

    def get_overrides(self, bank_id: str, hit_ids: List[str]) -> Dict[str, bool]:
        """Bulk-read overrides for a recall result set."""
        if not hit_ids:
            return {}
        with self._lock:
            conn = self._get()
            placeholders = ",".join("?" * len(hit_ids))
            rows = conn.execute(
                f"SELECT hit_id, untrusted FROM Unit_Trust_State "
                f"WHERE bank_id=? AND hit_id IN ({placeholders})",
                (bank_id, *hit_ids),
            ).fetchall()
            return {r["hit_id"]: bool(r["untrusted"]) for r in rows}

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


class HindsightBackend(MemoryBackend):
    """MemoryBackend implementation using the native HindsightRESTClient.

    Retain path is fire-and-forget through a per-bank async queue (plan §1.3):
    one worker task per bank drains in FIFO, preserving intra-bank ordering
    without serializing across banks. User turns enqueue + return; they never
    block on the retain LLM round-trip. Recall stays synchronous (caller awaits).
    """

    def __init__(self, url: str, override_db_path: Optional[str] = None):
        self.url = url
        self._client: Optional[HindsightRESTClient] = None
        self._queues: Dict[str, "asyncio.Queue[Any]"] = {}
        self._workers: Dict[str, "asyncio.Task[None]"] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        if override_db_path is None:
            override_db_path = str(Path(__file__).resolve().parent.parent / "hindsight_overrides.db")
        self._overrides = _TrustOverrideStore(override_db_path)

    def _get_client(self) -> HindsightRESTClient:
        if self._client is None:
            self._client = HindsightRESTClient(base_url=self.url)
        return self._client

    async def __aenter__(self) -> "HindsightBackend":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    async def _ensure_worker(self, bank_id: str) -> "asyncio.Queue[Any]":
        """Lazy-init one queue + worker per bank under a single lock."""
        async with self._lock:
            q = self._queues.get(bank_id)
            if q is None:
                q = asyncio.Queue()
                self._queues[bank_id] = q
                self._workers[bank_id] = asyncio.create_task(
                    self._worker_loop(bank_id, q),
                    name=f"hindsight-retain-{bank_id}",
                )
        return q

    async def _worker_loop(self, bank_id: str, q: "asyncio.Queue[Any]") -> None:
        """Drain retain payloads in FIFO. Drop on transport errors, never crash."""
        client = self._get_client()
        while True:
            item = await q.get()
            if item is _STOP:
                q.task_done()
                return
            content, tags = item
            try:
                await client.aretain(bank_id=bank_id, content=content, tags=tags)
            except httpx.ConnectError as e:
                # Kobold offline / proxy down — expected operational state, log+drop.
                logger.info("Hindsight retain dropped (kobold offline): %s", e)
            except (httpx.RequestError, HindsightAPIError) as e:
                logger.warning("Hindsight retain dropped: %s", e)
            except Exception:  # noqa: BLE001 — worker must never die
                logger.exception("Hindsight retain worker: unexpected error, dropping payload")
            finally:
                q.task_done()

    async def aclose(self) -> None:
        """Drain queues, stop workers, close httpx client. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # Signal each worker to exit after draining.
        for q in self._queues.values():
            await q.put(_STOP)
        if self._workers:
            await asyncio.gather(*self._workers.values(), return_exceptions=True)
        self._queues.clear()
        self._workers.clear()
        if self._client is not None:
            await self._client.client.aclose()
            self._client = None
        self._overrides.close()

    # ---------- Legacy SQLite-shape: fail loud when flag is flipped early ----------
    # Plan §5 (Cleanup): legacy callers must migrate to new-shape methods before
    # SEMANTIC_BACKEND can flip to "hindsight". Silent no-ops would mask data loss;
    # NotImplementedError surfaces missing migrations on first call.
    def _no_legacy(self, name: str) -> NoReturn:
        raise NotImplementedError(
            f"{name} is a legacy SQLite-shape method; HindsightBackend has no equivalent. "
            "Migrate the caller to retain_turn/recall before flipping SEMANTIC_BACKEND."
        )

    def log_agent_action(self, *args: Any, **kwargs: Any) -> int: self._no_legacy("log_agent_action")
    def update_agent_action_outcome(self, *args: Any, **kwargs: Any) -> None: self._no_legacy("update_agent_action_outcome")
    def add_action_contexts(self, *args: Any, **kwargs: Any) -> None: self._no_legacy("add_action_contexts")
    def get_relevant_agent_actions(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]: self._no_legacy("get_relevant_agent_actions")
    def get_action_steps(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]: self._no_legacy("get_action_steps")
    def store_message_embedding(self, *args: Any, **kwargs: Any) -> None: self._no_legacy("store_message_embedding")
    def get_unembedded_messages(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]: self._no_legacy("get_unembedded_messages")
    def store_segment(self, *args: Any, **kwargs: Any) -> int: self._no_legacy("store_segment")
    def store_summary(self, *args: Any, **kwargs: Any) -> int: self._no_legacy("store_summary")
    def get_summaries_for_channel(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]: self._no_legacy("get_summaries_for_channel")
    def get_unsegmented_embedded_messages(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]: self._no_legacy("get_unsegmented_embedded_messages")
    def retrieve_relevant_summaries(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]: self._no_legacy("retrieve_relevant_summaries")
    def record_segment_failure(self, *args: Any, **kwargs: Any) -> None: self._no_legacy("record_segment_failure")
    def get_failed_segment_ranges(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]: self._no_legacy("get_failed_segment_ranges")
    def clear_segment_failure(self, *args: Any, **kwargs: Any) -> None: self._no_legacy("clear_segment_failure")
    def get_active_channels(self, *args: Any, **kwargs: Any) -> List[Tuple[str, str, Optional[str]]]: self._no_legacy("get_active_channels")
    def get_last_segment_tail_embeddings(self, *args: Any, **kwargs: Any) -> Optional[List[bytes]]: self._no_legacy("get_last_segment_tail_embeddings")

    # ---------- New Hindsight-shape Methods ----------

    async def retain_turn(
        self,
        bank_id: str,
        role: str,
        content: str,
        *,
        timestamp: datetime,
        scope_tags: List[str],
        source_persona: str,
        untrusted: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Fire-and-forget: enqueue + return. ID isn't known until the worker
        # POSTs; callers that need a synchronous handle use the legacy path.
        tags = list(scope_tags) + [
            f"persona:{source_persona}",
            f"role:{role}",
            _untrusted_tag(untrusted),
        ]
        q = await self._ensure_worker(bank_id)
        await q.put((content, tags))
        return ""

    async def retain_experience(
        self,
        bank_id: str,
        action_type: str,
        context: Dict[str, Any],
        outcome: Optional[str],
        *,
        scope_tags: List[str],
        source_persona: str,
        untrusted: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        content = f"Action: {action_type}\nContext: {json.dumps(context)}\nOutcome: {outcome}"
        tags = list(scope_tags) + [
            f"persona:{source_persona}",
            "type:experience",
            f"action:{action_type}",
            _untrusted_tag(untrusted),
        ]
        q = await self._ensure_worker(bank_id)
        await q.put((content, tags))
        return ""

    async def recall(
        self,
        bank_id: str,
        query: str,
        *,
        k: int = 10,
        types: Optional[List[str]] = None,
        tag_filter: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        budget: Optional[float] = None,
    ) -> List[MemoryHit]:
        client = self._get_client()
        try:
            results = await client.arecall(bank_id=bank_id, query=query, k=k, tags=tag_filter)
        except (httpx.RequestError, HindsightAPIError) as e:
            logger.warning("Hindsight recall failed: %s", e)
            return []

        hits: List[MemoryHit] = []
        for r in results:
            tags = r.get("tags", []) or []
            hits.append(
                MemoryHit(
                    id=str(r.get("id", "")),
                    content=r.get("content", ""),
                    score=float(r.get("score", 0.0)),
                    untrusted=_read_untrusted(tags),
                    metadata=r.get("metadata", {}) or {},
                    tags=tags,
                    timestamp=(
                        datetime.fromisoformat(r["timestamp"]) if "timestamp" in r else None
                    ),
                )
            )
        # Apply operator overrides on top of the storage-side bit (DP-110 option c).
        overrides = self._overrides.get_overrides(bank_id, [h.id for h in hits if h.id])
        if overrides:
            for h in hits:
                if h.id in overrides:
                    h.untrusted = overrides[h.id]
        return hits

    async def reflect(
        self,
        bank_id: str,
        query: str,
        *,
        tag_filter: Optional[List[str]] = None,
    ) -> ReflectResult:
        client = self._get_client()
        try:
            result = await client.areflect(bank_id=bank_id, query=query, tags=tag_filter)
            return ReflectResult(
                answer=result.get("answer", ""),
                mental_models=[
                    MentalModel(id=str(m["id"]), content=m["content"], tags=m.get("tags", []))
                    for m in result.get("mental_models", [])
                ],
            )
        except (httpx.RequestError, HindsightAPIError) as e:
            logger.warning("Hindsight reflect failed: %s", e)
            return ReflectResult(answer="", mental_models=[])

    async def ensure_bank(
        self,
        bank_id: str,
        *,
        mission: Optional[str] = None,
        reflect_mission: Optional[str] = None,
    ) -> None:
        client = self._get_client()
        await client.acreate_bank(
            bank_id=bank_id, mission=mission, reflect_mission=reflect_mission
        )

    async def delete_bank(self, bank_id: str) -> None:
        client = self._get_client()
        await client.adelete_bank(bank_id=bank_id)

    def _flip(self, bank_id: str, hit_id: str, untrusted: bool,
              operator_id: str, reason: str) -> None:
        prior = self._overrides.set(bank_id, hit_id, untrusted, operator_id, reason)
        logger.info(
            "Trust flip bank=%s hit=%s prior_override=%s new=%s operator=%s reason=%s",
            bank_id, hit_id, prior, untrusted, operator_id, reason,
        )

    async def mark_trusted(
        self,
        bank_id: str,
        hit_id: str,
        *,
        operator_id: str,
        reason: str,
    ) -> None:
        self._flip(bank_id, hit_id, untrusted=False, operator_id=operator_id, reason=reason)

    async def mark_untrusted(
        self,
        bank_id: str,
        hit_id: str,
        *,
        operator_id: str,
        reason: str,
    ) -> None:
        self._flip(bank_id, hit_id, untrusted=True, operator_id=operator_id, reason=reason)
