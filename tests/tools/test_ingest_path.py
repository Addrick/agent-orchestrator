"""DP-118: tests for the `ingest_path` tool wiring + handler."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.tools.definitions import ALL_TOOL_DEFINITIONS, get_tool_capabilities
from src.tools.ingest_path import IngestPathHandler
from src.tools.tool_manager import ToolManager
from src.tools.turn_context import TurnContext, set_turn_context, reset_turn_context


# ----- tool registration / def -----

def test_ingest_path_tool_definition_present() -> None:
    names = {t.get("function", {}).get("name") for t in ALL_TOOL_DEFINITIONS}
    assert "ingest_path" in names


def test_ingest_path_capabilities() -> None:
    caps = get_tool_capabilities("ingest_path")
    assert caps["produces_untrusted"] is True
    assert caps["irreversible"] is False


# ----- helpers -----

def _make_backend(class_name: str = "HindsightBackend") -> MagicMock:
    backend = MagicMock()
    # Stamp class name so handler's noop detection works.
    backend.__class__.__name__ = class_name
    backend.retain_document = AsyncMock(return_value=None)
    return backend


def _make_handler(tmp_path: Path, backend: MagicMock,
                  persona_lookup=None) -> IngestPathHandler:
    return IngestPathHandler(
        memory_backend=backend,
        cache_dir=tmp_path / "cache",
        persona_lookup=persona_lookup,
    )


def _ctx(persona: str = "alice") -> TurnContext:
    return TurnContext(
        persona_name=persona, user_identifier="u1",
        channel="c1", server_id="s9",
    )


# ----- bank resolution -----

@pytest.mark.asyncio
async def test_bank_resolution_arg_beats_persona(tmp_path: Path) -> None:
    persona = MagicMock()
    persona.get_ingest_bank.return_value = "from_persona"
    backend = _make_backend()
    handler = _make_handler(tmp_path, backend, persona_lookup=lambda _: persona)

    f = tmp_path / "n.md"
    f.write_text("hi")

    token = set_turn_context(_ctx())
    try:
        out = await handler._ingest_path(str(f), bank="from_arg")
    finally:
        reset_turn_context(token)

    assert out["bank"] == "from_arg"
    backend.retain_document.assert_awaited_once()
    assert backend.retain_document.call_args.args[0] == "from_arg"


@pytest.mark.asyncio
async def test_bank_resolution_persona_beats_name(tmp_path: Path) -> None:
    persona = MagicMock()
    persona.get_ingest_bank.return_value = "from_persona"
    backend = _make_backend()
    handler = _make_handler(tmp_path, backend, persona_lookup=lambda _: persona)

    f = tmp_path / "n.md"
    f.write_text("hi")

    token = set_turn_context(_ctx())
    try:
        out = await handler._ingest_path(str(f))
    finally:
        reset_turn_context(token)

    assert out["bank"] == "from_persona"


@pytest.mark.asyncio
async def test_bank_resolution_falls_back_to_persona_name(tmp_path: Path) -> None:
    persona = MagicMock()
    persona.get_ingest_bank.return_value = None
    backend = _make_backend()
    handler = _make_handler(tmp_path, backend, persona_lookup=lambda _: persona)

    f = tmp_path / "n.md"
    f.write_text("hi")

    token = set_turn_context(_ctx("zoe"))
    try:
        out = await handler._ingest_path(str(f))
    finally:
        reset_turn_context(token)

    assert out["bank"] == "zoe"


# ----- file vs dir, glob -----

@pytest.mark.asyncio
async def test_single_file_ingest(tmp_path: Path) -> None:
    backend = _make_backend()
    handler = _make_handler(tmp_path, backend)
    f = tmp_path / "note.md"
    f.write_text("hello world", encoding="utf-8")

    token = set_turn_context(_ctx())
    try:
        out = await handler._ingest_path(str(f))
    finally:
        reset_turn_context(token)

    assert out == {"status": "ok", "bank": "alice",
                   "ingested": 1, "skipped": 0, "failed": 0}
    call = backend.retain_document.call_args
    # positional: (bank_id, document_id, content)
    assert call.args[1] == "note.md"
    assert call.args[2] == "hello world"


@pytest.mark.asyncio
async def test_directory_ingest_with_glob(tmp_path: Path) -> None:
    backend = _make_backend()
    handler = _make_handler(tmp_path, backend)
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "b.md").write_text("b")
    (tmp_path / "ignore.txt").write_text("nope")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.md").write_text("c")

    token = set_turn_context(_ctx())
    try:
        out = await handler._ingest_path(str(tmp_path))
    finally:
        reset_turn_context(token)

    assert out["ingested"] == 3
    assert out["skipped"] == 0
    assert backend.retain_document.await_count == 3


# ----- cache behavior -----

@pytest.mark.asyncio
async def test_hash_cache_skips_unchanged(tmp_path: Path) -> None:
    backend = _make_backend()
    handler = _make_handler(tmp_path, backend)
    f = tmp_path / "n.md"
    f.write_text("stable", encoding="utf-8")

    token = set_turn_context(_ctx())
    try:
        first = await handler._ingest_path(str(f))
        second = await handler._ingest_path(str(f))
    finally:
        reset_turn_context(token)

    assert first["ingested"] == 1
    assert second["ingested"] == 0
    assert second["skipped"] == 1
    assert backend.retain_document.await_count == 1


@pytest.mark.asyncio
async def test_modified_file_re_ingests(tmp_path: Path) -> None:
    backend = _make_backend()
    handler = _make_handler(tmp_path, backend)
    f = tmp_path / "n.md"
    f.write_text("v1")

    token = set_turn_context(_ctx())
    try:
        await handler._ingest_path(str(f))
        f.write_text("v2")
        out = await handler._ingest_path(str(f))
    finally:
        reset_turn_context(token)

    assert out["ingested"] == 1
    assert backend.retain_document.await_count == 2


@pytest.mark.asyncio
async def test_force_bypasses_cache(tmp_path: Path) -> None:
    backend = _make_backend()
    handler = _make_handler(tmp_path, backend)
    f = tmp_path / "n.md"
    f.write_text("same")

    token = set_turn_context(_ctx())
    try:
        await handler._ingest_path(str(f))
        out = await handler._ingest_path(str(f), force=True)
    finally:
        reset_turn_context(token)

    assert out["ingested"] == 1
    assert backend.retain_document.await_count == 2


# ----- metadata + types -----

@pytest.mark.asyncio
async def test_metadata_values_are_all_strings(tmp_path: Path) -> None:
    backend = _make_backend()
    handler = _make_handler(tmp_path, backend)
    f = tmp_path / "n.md"
    f.write_text("hi")

    token = set_turn_context(_ctx())
    try:
        await handler._ingest_path(str(f))
    finally:
        reset_turn_context(token)

    md = backend.retain_document.call_args.kwargs["metadata"]
    assert {"source_path", "sha256", "file_mtime"} <= set(md.keys())
    for v in md.values():
        assert isinstance(v, str)


# ----- error paths -----

@pytest.mark.asyncio
async def test_missing_path_returns_error(tmp_path: Path) -> None:
    backend = _make_backend()
    handler = _make_handler(tmp_path, backend)
    token = set_turn_context(_ctx())
    try:
        out = await handler._ingest_path(str(tmp_path / "does_not_exist"))
    finally:
        reset_turn_context(token)
    assert out["status"] == "error"
    backend.retain_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_turn_context_returns_error(tmp_path: Path) -> None:
    backend = _make_backend()
    handler = _make_handler(tmp_path, backend)
    f = tmp_path / "n.md"
    f.write_text("hi")
    out = await handler._ingest_path(str(f))
    assert out["status"] == "error"


@pytest.mark.asyncio
async def test_disabled_flag_short_circuits(tmp_path: Path, monkeypatch) -> None:
    from config import global_config as gc
    monkeypatch.setattr(gc, "INGEST_PATH_ENABLED", False)

    backend = _make_backend()
    handler = _make_handler(tmp_path, backend)
    f = tmp_path / "n.md"
    f.write_text("hi")

    token = set_turn_context(_ctx())
    try:
        out = await handler._ingest_path(str(f))
    finally:
        reset_turn_context(token)

    assert out["status"] == "disabled"
    backend.retain_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_sqlite_backend_returns_noop_note(tmp_path: Path) -> None:
    backend = _make_backend(class_name="SqliteSemanticBackend")
    handler = _make_handler(tmp_path, backend)
    f = tmp_path / "n.md"
    f.write_text("hi")

    token = set_turn_context(_ctx())
    try:
        out = await handler._ingest_path(str(f))
    finally:
        reset_turn_context(token)

    assert "note" in out
    assert "sqlite" in out["note"]


# ----- cache file shape -----

@pytest.mark.asyncio
async def test_cache_file_written_per_bank(tmp_path: Path) -> None:
    backend = _make_backend()
    handler = _make_handler(tmp_path, backend)
    f = tmp_path / "n.md"
    f.write_text("hi")

    token = set_turn_context(_ctx())
    try:
        await handler._ingest_path(str(f))
    finally:
        reset_turn_context(token)

    cache_file = tmp_path / "cache" / "alice.json"
    assert cache_file.exists()
    data = json.loads(cache_file.read_text())
    assert "n.md" in data
    assert "sha256" in data["n.md"]
