# tests/scripts/test_backfill_hindsight.py
"""DP-115: smoke tests for scripts/backfill_hindsight.py.

Backfill is one-shot, fire-and-forget — it never gets exercised under
normal CI, so the tests below pin the behaviors that would silently
corrupt history if they regressed:

  - One ensure_bank per distinct persona
  - retain_turn called per row, ordered by (timestamp, interaction_id)
  - reasoning_content gets wrapped in <thought> blocks (not dropped)
  - metadata carries the legacy interaction_id for audit traceability
  - aclose() drains the queue at the end
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from memory.memory_manager import MemoryManager


def _load_backfill_module():
    """The script lives at scripts/backfill_hindsight.py — not on the
    package path, so import it by file."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "backfill_hindsight.py"
    spec = importlib.util.spec_from_file_location("backfill_hindsight_script", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["backfill_hindsight_script"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def populated_mm(tmp_path: Path) -> MemoryManager:
    db_path = str(tmp_path / "backfill.db")
    mm = MemoryManager(db_path=db_path)
    mm.create_schema()
    rows = [
        # (user, persona, channel, role, content, ts, reasoning, tool_ctx)
        ("u1", "alice", "c1", "user", "hello", "2026-05-01T10:00:00", None, None),
        ("u1", "alice", "c1", "assistant", "hi back", "2026-05-01T10:00:05",
         "user greeted", None),
        ("u2", "bob", "c2", "user", "what's up", "2026-05-01T11:00:00", None, "{\"k\":1}"),
    ]
    with mm.transaction() as conn:
        for user, persona, channel, role, content, ts, reasoning, tool_ctx in rows:
            conn.execute(
                "INSERT INTO User_Interactions "
                "(user_identifier, persona_name, channel, author_role, "
                " author_name, content, timestamp, reasoning_content, tool_context) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user, persona, channel, role, role, content, ts, reasoning, tool_ctx),
            )
    yield mm
    mm.close()


@pytest.mark.asyncio
async def test_backfill_provisions_each_persona_once(populated_mm: MemoryManager) -> None:
    fake = AsyncMock()
    fake.ensure_bank = AsyncMock()
    fake.retain_turn = AsyncMock()
    fake.aclose = AsyncMock()
    mod = _load_backfill_module()

    with patch.object(mod, "MemoryManager", return_value=populated_mm), \
         patch.object(mod, "HindsightBackend", return_value=fake):
        await mod.backfill()

    banks = sorted(c.kwargs["bank_id"] for c in fake.ensure_bank.await_args_list)
    assert banks == ["alice", "bob"]


@pytest.mark.asyncio
async def test_backfill_retains_rows_in_temporal_order(populated_mm: MemoryManager) -> None:
    fake = AsyncMock()
    fake.ensure_bank = AsyncMock()
    fake.retain_turn = AsyncMock()
    fake.aclose = AsyncMock()
    mod = _load_backfill_module()

    with patch.object(mod, "MemoryManager", return_value=populated_mm), \
         patch.object(mod, "HindsightBackend", return_value=fake):
        await mod.backfill()

    calls = fake.retain_turn.await_args_list
    assert len(calls) == 3
    ts_seq = [c.kwargs["timestamp"] for c in calls]
    assert ts_seq == sorted(ts_seq)
    assert all(isinstance(t, datetime) for t in ts_seq)


@pytest.mark.asyncio
async def test_backfill_wraps_reasoning_in_thought_block(populated_mm: MemoryManager) -> None:
    fake = AsyncMock()
    fake.ensure_bank = AsyncMock()
    fake.retain_turn = AsyncMock()
    fake.aclose = AsyncMock()
    mod = _load_backfill_module()

    with patch.object(mod, "MemoryManager", return_value=populated_mm), \
         patch.object(mod, "HindsightBackend", return_value=fake):
        await mod.backfill()

    contents = [c.kwargs["content"] for c in fake.retain_turn.await_args_list]
    reasoning_payloads = [c for c in contents if "<thought>" in c]
    assert len(reasoning_payloads) == 1
    assert "user greeted" in reasoning_payloads[0]
    assert "hi back" in reasoning_payloads[0]


@pytest.mark.asyncio
async def test_backfill_metadata_carries_legacy_id(populated_mm: MemoryManager) -> None:
    fake = AsyncMock()
    fake.ensure_bank = AsyncMock()
    fake.retain_turn = AsyncMock()
    fake.aclose = AsyncMock()
    mod = _load_backfill_module()

    with patch.object(mod, "MemoryManager", return_value=populated_mm), \
         patch.object(mod, "HindsightBackend", return_value=fake):
        await mod.backfill()

    metas = [c.kwargs["metadata"] for c in fake.retain_turn.await_args_list]
    legacy_ids = [m["legacy_id"] for m in metas]
    assert legacy_ids == sorted(legacy_ids)
    assert all(isinstance(i, int) for i in legacy_ids)


@pytest.mark.asyncio
async def test_backfill_closes_backend(populated_mm: MemoryManager) -> None:
    fake = AsyncMock()
    fake.ensure_bank = AsyncMock()
    fake.retain_turn = AsyncMock()
    fake.aclose = AsyncMock()
    mod = _load_backfill_module()

    with patch.object(mod, "MemoryManager", return_value=populated_mm), \
         patch.object(mod, "HindsightBackend", return_value=fake):
        await mod.backfill()

    fake.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_backfill_normalizes_iso_timestamps(populated_mm: MemoryManager) -> None:
    """SQLite returns timestamps as strings; the script must hand datetimes
    to the backend (the new-shape ABC types `timestamp` as datetime)."""
    fake = AsyncMock()
    fake.ensure_bank = AsyncMock()
    fake.retain_turn = AsyncMock()
    fake.aclose = AsyncMock()
    mod = _load_backfill_module()

    with patch.object(mod, "MemoryManager", return_value=populated_mm), \
         patch.object(mod, "HindsightBackend", return_value=fake):
        await mod.backfill()

    first_ts = fake.retain_turn.await_args_list[0].kwargs["timestamp"]
    assert isinstance(first_ts, datetime)
    assert first_ts == datetime(2026, 5, 1, 10, 0, 0)
