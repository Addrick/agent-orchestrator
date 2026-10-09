"""DP-424: `get_memory` / `get_document` — scoped drill-down from a recall hit.

Asserts:
- Both tools are defined with recall_memory's trust shape (read, untrusted).
- They register only on a backend that implements the reads (Hindsight), not
  on one inheriting the ABC stubs (SQLite has `drill_down_memory` instead).
- Bank and tag scope come from the active turn, exactly as `recall_memory`
  scopes: an id outside the persona's memory-mode scope answers the same
  "not found" a nonexistent id does, and out-of-scope source facts are
  dropped from an observation's `source_memories`.
- A transient backend failure surfaces as a tool error, not as "not found".
- `get_document` pages the original text, and `chunk_id` returns one passage.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.memory.backend.base import MemoryBackend, MemoryBackendError, MemoryHit
from src.persona import MemoryMode
from src.tools.definitions import ALL_TOOL_DEFINITIONS, get_tool_capabilities
from src.tools.tool_manager import MemoryRecallHandler, MemorySourceHandler, ToolManager
from src.tools.turn_context import TurnContext, reset_turn_context, set_turn_context

OFFERED = {"get_memory", "get_document", "recall_memory"}


class _Backend:
    """In-memory stand-in implementing just the drill-down reads."""

    def __init__(self, units: Dict[str, Dict[str, Any]], docs: Dict[str, Dict[str, Any]],
                 chunks: Optional[Dict[str, list]] = None) -> None:
        self.units, self.docs, self.chunks = units, docs, chunks or {}
        self.banks_seen: list = []

    async def get_memory(self, bank_id: str, memory_id: str) -> Dict[str, Any]:
        self.banks_seen.append(bank_id)
        if memory_id not in self.units:
            raise MemoryBackendError("Hindsight API Error 404: not found")
        return dict(self.units[memory_id])

    async def get_document(self, bank_id: str, document_id: str) -> Dict[str, Any]:
        self.banks_seen.append(bank_id)
        if document_id not in self.docs:
            raise MemoryBackendError("Hindsight API Error 404: not found")
        return dict(self.docs[document_id])

    async def list_document_chunks(self, bank_id: str, document_id: str, *,
                                   limit: Optional[int] = None,
                                   offset: Optional[int] = None) -> Dict[str, Any]:
        items = self.chunks.get(document_id, [])
        start = offset or 0
        page = items[start:start + (limit or 100)]
        return {"items": page, "total": len(items), "limit": limit, "offset": start}


def _units() -> Dict[str, Dict[str, Any]]:
    return {
        "obs": {"id": "obs", "text": "observation", "type": "observation",
                "tags": ["channel:c1"], "document_id": None, "chunk_id": None,
                "source_memory_ids": ["f-mine", "f-other"], "untrusted": False},
        "f-mine": {"id": "f-mine", "text": "my fact", "type": "experience",
                   "tags": ["channel:c1"], "document_id": "doc-mine", "chunk_id": "ck-1",
                   "untrusted": False},
        "f-other": {"id": "f-other", "text": "other channel's fact", "type": "experience",
                    "tags": ["channel:c2"], "document_id": "doc-other", "chunk_id": "ck-9",
                    "untrusted": False},
    }


def _docs() -> Dict[str, Dict[str, Any]]:
    return {
        "doc-mine": {"id": "doc-mine", "original_text": "abcdefghij",
                     "tags": ["channel:c1"], "untrusted": False},
        "doc-other": {"id": "doc-other", "original_text": "secret",
                      "tags": ["channel:c2"], "untrusted": False},
    }


async def _call(backend: Any, tool: str, mode: MemoryMode = MemoryMode.CHANNEL_ISOLATED,
                **kwargs: Any) -> Dict[str, Any]:
    manager = ToolManager()
    MemorySourceHandler(backend).register(manager)
    token = set_turn_context(TurnContext(
        persona_name="alice", user_identifier="u1",
        channel="c1", server_id=None, memory_mode=mode,
    ))
    try:
        return await manager.execute_tool(tool, OFFERED, **kwargs)
    finally:
        reset_turn_context(token)


def test_definitions_share_recall_trust_shape() -> None:
    names = {t.get("function", {}).get("name") for t in ALL_TOOL_DEFINITIONS}
    for tool in ("get_memory", "get_document"):
        assert tool in names
        caps = get_tool_capabilities(tool)
        assert caps["produces_untrusted"] is True
        assert caps["irreversible"] is False
        assert caps["locality"] == "local"


def test_registers_only_on_a_backend_with_the_reads() -> None:
    class _Stub:
        # What a backend that doesn't override the read inherits from the ABC.
        get_memory = MemoryBackend.get_memory

    manager = ToolManager()
    MemorySourceHandler(_Stub()).register(manager)
    names = {t.get("function", {}).get("name") for t in manager.get_tool_definitions()}
    assert not names & {"get_memory", "get_document"}

    MemorySourceHandler(_Backend({}, {})).register(manager)
    names = {t.get("function", {}).get("name") for t in manager.get_tool_definitions()}
    assert {"get_memory", "get_document"} <= names


@pytest.mark.asyncio
async def test_get_memory_drops_out_of_scope_sources() -> None:
    backend = _Backend(_units(), _docs())
    out = await _call(backend, "get_memory", memory_id="obs")

    view = out["result"]
    assert view["id"] == "obs" and view["type"] == "observation"
    assert [s["id"] for s in view["source_memories"]] == ["f-mine"]
    assert view["source_memories"][0]["document_id"] == "doc-mine"
    assert view["source_memories"][0]["chunk_id"] == "ck-1"
    assert set(backend.banks_seen) == {"alice"}


@pytest.mark.asyncio
async def test_out_of_scope_reads_exactly_like_nonexistent() -> None:
    """The not-found answer must not reveal that another channel holds the id."""
    backend = _Backend(_units(), _docs())
    other = await _call(backend, "get_memory", memory_id="f-other")
    missing = await _call(backend, "get_memory", memory_id="nope")
    assert other["result"]["error"] == "Memory 'f-other' not found."
    assert missing["result"]["error"] == "Memory 'nope' not found."

    doc_other = await _call(backend, "get_document", document_id="doc-other")
    assert doc_other["result"] == {"error": "Document 'doc-other' not found."}
    assert "secret" not in str(doc_other)


@pytest.mark.asyncio
async def test_global_mode_reads_across_channels() -> None:
    """Scope follows memory mode, as recall_memory does (DP-407)."""
    backend = _Backend(_units(), _docs())
    out = await _call(backend, "get_document", MemoryMode.GLOBAL, document_id="doc-other")
    assert out["result"]["text"] == "secret"
    obs = await _call(backend, "get_memory", MemoryMode.GLOBAL, memory_id="obs")
    assert [s["id"] for s in obs["result"]["source_memories"]] == ["f-mine", "f-other"]


@pytest.mark.asyncio
async def test_untagged_item_is_in_scope_like_recall() -> None:
    """Hindsight's default tags_match='any' includes untagged items."""
    backend = _Backend({"bare": {"id": "bare", "text": "t", "tags": []}}, {})
    out = await _call(backend, "get_memory", memory_id="bare")
    assert out["result"]["id"] == "bare"


@pytest.mark.asyncio
async def test_transient_failure_is_an_error_not_not_found() -> None:
    backend = _Backend({}, {})
    backend.get_memory = AsyncMock(  # type: ignore[method-assign]
        side_effect=MemoryBackendError("Hindsight API Error 503", transient=True))
    out = await _call(backend, "get_memory", memory_id="x")
    assert "error" in out and "not found" not in out["error"]


@pytest.mark.asyncio
async def test_auth_failure_is_an_error_not_not_found() -> None:
    """Only a missing/malformed id is "not found"; a 403 is an outage the model
    must not read as an empty bank."""
    err = MemoryBackendError("Hindsight API Error 403: forbidden")
    err.status_code = 403  # type: ignore[attr-defined]
    backend = _Backend({}, {})
    backend.get_memory = AsyncMock(side_effect=err)  # type: ignore[method-assign]
    out = await _call(backend, "get_memory", memory_id="x")
    assert "error" in out and "not found" not in out["error"]


@pytest.mark.asyncio
async def test_get_document_null_paging_args_mean_defaults() -> None:
    backend = _Backend(_units(), _docs())
    out = (await _call(backend, "get_document", document_id="doc-mine",
                       offset=None, max_chars=None))["result"]
    assert (out["text"], out["offset"], out["next_offset"]) == ("abcdefghij", 0, None)


@pytest.mark.asyncio
async def test_get_document_pages_the_original_text() -> None:
    backend = _Backend(_units(), _docs())
    first = (await _call(backend, "get_document", document_id="doc-mine", max_chars=4))["result"]
    assert (first["text"], first["offset"], first["next_offset"]) == ("abcd", 0, 4)
    assert first["total_chars"] == 10

    last = (await _call(backend, "get_document", document_id="doc-mine",
                        offset=8, max_chars=4))["result"]
    assert (last["text"], last["next_offset"]) == ("ij", None)


@pytest.mark.asyncio
async def test_get_document_caps_max_chars() -> None:
    backend = _Backend({}, {"big": {"id": "big", "original_text": "x" * 50000, "tags": []}})
    out = (await _call(backend, "get_document", document_id="big", max_chars=10**9))["result"]
    assert len(out["text"]) == MemorySourceHandler._MAX_DOC_CHARS
    assert out["next_offset"] == MemorySourceHandler._MAX_DOC_CHARS


@pytest.mark.asyncio
async def test_get_document_chunk_pages_until_found() -> None:
    """The chunk lookup must follow pagination, not stop at the first page."""
    chunks = [{"chunk_id": f"ck-{i}", "chunk_index": i, "chunk_text": f"passage {i}"}
              for i in range(1500)]
    backend = _Backend(_units(), _docs(), chunks={"doc-mine": chunks})

    out = (await _call(backend, "get_document", document_id="doc-mine",
                       chunk_id="ck-1200"))["result"]
    assert out["text"] == "passage 1200" and out["chunk_index"] == 1200

    miss = (await _call(backend, "get_document", document_id="doc-mine",
                        chunk_id="ck-nope"))["result"]
    assert "not found" in miss["error"]


@pytest.mark.asyncio
async def test_chunk_of_out_of_scope_document_is_not_reachable() -> None:
    backend = _Backend(_units(), _docs(),
                       chunks={"doc-other": [{"chunk_id": "ck-9", "chunk_text": "secret"}]})
    out = (await _call(backend, "get_document", document_id="doc-other",
                       chunk_id="ck-9"))["result"]
    assert out == {"error": "Document 'doc-other' not found."}


@pytest.mark.asyncio
async def test_no_turn_context_is_an_error_without_backend_calls() -> None:
    backend = _Backend(_units(), _docs())
    manager = ToolManager()
    MemorySourceHandler(backend).register(manager)
    out = await manager.execute_tool("get_memory", OFFERED, memory_id="obs")
    assert "error" in out["result"]
    assert backend.banks_seen == []


@pytest.mark.asyncio
async def test_recall_memory_hits_expose_the_drill_handles() -> None:
    backend = MagicMock()
    backend.recall = AsyncMock(return_value=[
        MemoryHit(id="f-mine", content="my fact", score=1.0, tags=["channel:c1"],
                  document_id="doc-mine", chunk_id="ck-1"),
    ])
    manager = ToolManager()
    MemoryRecallHandler(backend).register(manager)
    token = set_turn_context(TurnContext(
        persona_name="alice", user_identifier="u1", channel="c1", server_id=None,
    ))
    try:
        out = await manager.execute_tool("recall_memory", OFFERED, query="q")
    finally:
        reset_turn_context(token)
    hit = out["result"][0]
    assert (hit["document_id"], hit["chunk_id"]) == ("doc-mine", "ck-1")
