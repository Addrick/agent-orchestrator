# src/memory/backend/hindsight.py
from __future__ import annotations
import json
import logging
from datetime import datetime
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


class HindsightBackend(MemoryBackend):
    """MemoryBackend implementation using the native HindsightRESTClient."""

    def __init__(self, url: str):
        self.url = url
        self._client: Optional[HindsightRESTClient] = None

    def _get_client(self) -> HindsightRESTClient:
        if self._client is None:
            self._client = HindsightRESTClient(base_url=self.url)
        return self._client

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
        client = self._get_client()
        tags = list(scope_tags) + [
            f"persona:{source_persona}",
            f"role:{role}",
            _untrusted_tag(untrusted),
        ]
        try:
            result = await client.aretain(bank_id=bank_id, content=content, tags=tags)
            return str(result.get("id", ""))
        except (httpx.RequestError, HindsightAPIError) as e:
            # Fire-and-forget per plan §1.3 — log and drop, don't crash the user turn.
            logger.warning("Hindsight retain_turn dropped: %s", e)
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
        client = self._get_client()
        content = f"Action: {action_type}\nContext: {json.dumps(context)}\nOutcome: {outcome}"
        tags = list(scope_tags) + [
            f"persona:{source_persona}",
            "type:experience",
            f"action:{action_type}",
            _untrusted_tag(untrusted),
        ]
        try:
            result = await client.aretain(bank_id=bank_id, content=content, tags=tags)
            return str(result.get("id", ""))
        except (httpx.RequestError, HindsightAPIError) as e:
            logger.warning("Hindsight retain_experience dropped: %s", e)
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

    async def mark_trusted(
        self,
        bank_id: str,
        hit_id: str,
        *,
        operator_id: str,
        reason: str,
    ) -> None:
        # TODO(DP-110): upstream Hindsight has no documented unit-tag PATCH
        # endpoint as of v0.5.0. Options: (1) extend upstream, (2) supersede
        # via delete+re-retain, (3) maintain a parallel override table.
        # Tracked separately so DP-109 can close on the storage-side bit.
        raise NotImplementedError(
            "mark_trusted: pending upstream tag-patch endpoint verification (DP-110)"
        )

    async def mark_untrusted(
        self,
        bank_id: str,
        hit_id: str,
        *,
        operator_id: str,
        reason: str,
    ) -> None:
        raise NotImplementedError(
            "mark_untrusted: pending upstream tag-patch endpoint verification (DP-110)"
        )
