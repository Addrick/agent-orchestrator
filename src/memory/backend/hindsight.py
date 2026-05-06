# src/memory/backend/hindsight.py
from __future__ import annotations
import json
import logging
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import httpx
from .base import MemoryBackend, MemoryHit, Experience, ReflectResult, MentalModel

if TYPE_CHECKING:
    from src.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class HindsightAPIError(Exception):
    """Raised for non-2xx responses from the Hindsight API."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Hindsight API Error {status_code}: {message}")

class HindsightRESTClient:
    """An audited, native implementation of the Hindsight SDK logic.
    
    This class 'apes' the behavior of the official hindsight-client but uses 
    our project's trusted httpx library, bypassing the need to install the 
    untrusted external package during quarantine.
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """Core request handler with retry logic and error parsing."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        
        # Simple retry loop for transient network errors
        for attempt in range(3):
            try:
                response = await self.client.request(method, url, **kwargs)
                if response.status_code >= 500 and attempt < 2:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
                
                if not (200 <= response.status_code < 300):
                    raise HindsightAPIError(response.status_code, response.text)
                
                return response.json()
            except (httpx.RequestError, HindsightAPIError) as e:
                if attempt == 2:
                    logger.error(f"Hindsight request failed after 3 attempts: {e}")
                    raise
                await asyncio.sleep(1 * (attempt + 1))
        
        return {} # Should not reach here

    async def aretain(self, bank_id: str, content: str, tags: List[str]) -> Dict[str, Any]:
        """Ape of the aretain method."""
        payload = {
            "content": content,
            "tags": tags,
            "retain_async": True # Default to async consolidation per plan
        }
        return await self._request("POST", f"/banks/{bank_id}/retain", json=payload)

    async def arecall(self, bank_id: str, query: str, k: int = 10, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Ape of the arecall method."""
        payload = {
            "query": query,
            "k": k,
            "tags": tags or []
        }
        result = await self._request("POST", f"/banks/{bank_id}/recall", json=payload)
        return result.get("results", [])

    async def areflect(self, bank_id: str, query: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Ape of the areflect method."""
        payload = {
            "query": query,
            "tags": tags or []
        }
        return await self._request("POST", f"/banks/{bank_id}/reflect", json=payload)

    async def acreate_bank(self, bank_id: str, mission: Optional[str] = None, reflect_mission: Optional[str] = None) -> None:
        """Ape of bank creation logic."""
        payload = {
            "bank_id": bank_id,
            "mission": mission,
            "reflect_mission": reflect_mission
        }
        try:
            await self._request("POST", "/banks", json=payload)
        except HindsightAPIError as e:
            if e.status_code == 409: # Conflict - already exists
                return
            raise

    async def adelete_bank(self, bank_id: str) -> None:
        """Ape of bank deletion logic."""
        await self._request("DELETE", f"/banks/{bank_id}")

class HindsightBackend(MemoryBackend):
    """MemoryBackend implementation using our native HindsightRESTClient."""

    def __init__(self, url: str):
        self.url = url
        self._client: Optional[HindsightRESTClient] = None

    def _get_client(self) -> HindsightRESTClient:
        if self._client is None:
            self._client = HindsightRESTClient(base_url=self.url)
        return self._client

    # ---------- Legacy SQLite-shape stubs (Satisfy ABC) ----------
    def log_agent_action(self, *args, **kwargs) -> int: return 0
    def update_agent_action_outcome(self, *args, **kwargs) -> None: pass
    def add_action_contexts(self, *args, **kwargs) -> None: pass
    def get_relevant_agent_actions(self, *args, **kwargs) -> List[Dict[str, Any]]: return []
    def get_action_steps(self, *args, **kwargs) -> List[Dict[str, Any]]: return []
    def store_message_embedding(self, *args, **kwargs) -> None: pass
    def get_unembedded_messages(self, *args, **kwargs) -> List[Dict[str, Any]]: return []
    def store_segment(self, *args, **kwargs) -> int: return 0
    def store_summary(self, *args, **kwargs) -> int: return 0
    def get_summaries_for_channel(self, *args, **kwargs) -> List[Dict[str, Any]]: return []
    def get_unsegmented_embedded_messages(self, *args, **kwargs) -> List[Dict[str, Any]]: return []
    def retrieve_relevant_summaries(self, *args, **kwargs) -> List[Dict[str, Any]]: return []
    def record_segment_failure(self, *args, **kwargs) -> None: pass
    def get_failed_segment_ranges(self, *args, **kwargs) -> List[Dict[str, Any]]: return []
    def clear_segment_failure(self, *args, **kwargs) -> None: pass
    def get_active_channels(self, *args, **kwargs) -> List[Tuple[str, str, Optional[str]]]: return []
    def get_last_segment_tail_embeddings(self, *args, **kwargs) -> Optional[List[bytes]]: return None

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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        client = self._get_client()
        tags = list(scope_tags) + [f"persona:{source_persona}", f"role:{role}"]
        try:
            result = await client.aretain(bank_id=bank_id, content=content, tags=tags)
            return str(result.get("id", ""))
        except Exception:
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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        client = self._get_client()
        content = f"Action: {action_type}\nContext: {json.dumps(context)}\nOutcome: {outcome}"
        tags = list(scope_tags) + [f"persona:{source_persona}", "type:experience", f"action:{action_type}"]
        try:
            result = await client.aretain(bank_id=bank_id, content=content, tags=tags)
            return str(result.get("id", ""))
        except Exception:
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
            hits = []
            for r in results:
                hits.append(MemoryHit(
                    id=str(r.get("id", "")),
                    content=r.get("content", ""),
                    score=float(r.get("score", 0.0)),
                    metadata=r.get("metadata", {}),
                    tags=r.get("tags", []),
                    timestamp=datetime.fromisoformat(r["timestamp"]) if "timestamp" in r else None
                ))
            return hits
        except Exception:
            return []

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
                ]
            )
        except Exception:
            return ReflectResult(answer="", mental_models=[])

    async def ensure_bank(
        self,
        bank_id: str,
        *,
        mission: Optional[str] = None,
        reflect_mission: Optional[str] = None,
    ) -> None:
        client = self._get_client()
        await client.acreate_bank(bank_id=bank_id, mission=mission, reflect_mission=reflect_mission)

    async def delete_bank(self, bank_id: str) -> None:
        client = self._get_client()
        await client.adelete_bank(bank_id=bank_id)
