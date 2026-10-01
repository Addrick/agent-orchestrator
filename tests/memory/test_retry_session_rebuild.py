"""DP-409: a retry replaces the session's Hindsight document with the
canonical conversation instead of appending the new attempt beside the old."""

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest

from config import global_config
from src.memory.backend.base import MemoryBackend
from src.memory.backend.hindsight import (
    HindsightBackend, TRUSTED_TAG, UNTRUSTED_TAG, _DocScopeStore,
)
from src.memory.memory_manager import MemoryManager
from src.turn_persistence import TurnPersistence, format_retained_turn


@pytest.fixture(autouse=True)
def eastern(monkeypatch):
    monkeypatch.setattr(global_config, "LOCAL_TZ", "America/New_York")


@pytest.fixture
def mm():
    manager = MemoryManager(db_path=":memory:")
    manager.create_schema()
    yield manager
    manager.close()


@pytest.fixture
def backend(tmp_path) -> HindsightBackend:
    return HindsightBackend(
        url="http://stub:8888",
        override_db_path=str(tmp_path / "overrides.db"),
        doc_scope_db_path=str(tmp_path / "doc_scope.db"),
    )


def _log(mm: MemoryManager, role: str, content: str, ts: datetime,
         channel: str = "web_ui", persona: str = "explr") -> int:
    return mm.log_message(
        user_identifier="portal", persona_name=persona, channel=channel,
        author_role=role, author_name="Adam" if role == "user" else persona,
        content=content, timestamp=ts,
    )


# ---------- format_retained_turn ----------

def test_inline_think_is_stripped_from_assistant_turn():
    ts = datetime(2026, 9, 26, 18, 5, tzinfo=timezone.utc)
    out = format_retained_turn(
        "assistant", "explr", "<think>\nthe user is a researcher\n</think>\nHello.", ts,
    )
    assert out == "explr: Hello."


def test_user_text_mentioning_think_is_untouched():
    ts = datetime(2026, 9, 26, 18, 5, tzinfo=timezone.utc)
    out = format_retained_turn("user", "Adam", "why <think>x</think>?", ts)
    assert out.endswith("Adam: why <think>x</think>?")


# ---------- _DocScopeStore ----------

def test_current_reports_session_start_and_sticky_taint(tmp_path):
    store = _DocScopeStore(str(tmp_path / "ds.db"))
    t0 = datetime(2026, 9, 27, 3, 45, tzinfo=timezone.utc)
    doc, _ = store.resolve("explr:web_ui", t0)
    store.resolve("explr:web_ui", t0 + timedelta(minutes=1), untrusted=True)
    store.resolve("explr:web_ui", t0 + timedelta(minutes=2), untrusted=False)
    assert store.current("explr:web_ui") == (doc, t0, True)
    assert store.current("other:scope") is None


def test_pre_dp409_store_gains_untrusted_column(tmp_path):
    path = tmp_path / "ds.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE Doc_Scope (scope_key TEXT PRIMARY KEY,"
                 " document_id TEXT NOT NULL, last_ts TEXT NOT NULL)")
    conn.execute("INSERT INTO Doc_Scope VALUES ('b:c', 'b:c:2026-09-27T00:00:00+00:00',"
                 " '2026-09-27T00:00:00+00:00')")
    conn.commit()
    conn.close()
    store = _DocScopeStore(str(path))
    doc, started, untrusted = store.current("b:c")
    assert doc == "b:c:2026-09-27T00:00:00+00:00"
    assert started == datetime(2026, 9, 27, tzinfo=timezone.utc)
    assert untrusted is False


# ---------- HindsightBackend.replace_session ----------

@pytest.mark.asyncio
async def test_replace_session_rewrites_the_open_document(backend):
    seen: List[Dict[str, Any]] = []

    async def fake_aretain(bank_id, items, async_=True):
        seen.extend(items)
        return {}

    now = datetime.now(timezone.utc)
    with patch.object(backend._get_client(), "aretain", side_effect=fake_aretain):
        await backend.retain_turn(
            "explr", "assistant", "attempt one", timestamp=now,
            scope_tags=["channel:web_ui"], source_persona="explr", untrusted=True,
        )
        assert await backend.session_start("explr", ["channel:web_ui"]) == now
        ok = await backend.replace_session(
            "explr", "canonical transcript",
            scope_tags=["channel:web_ui", "user:portal"], source_persona="explr",
            untrusted=False, timestamp=now + timedelta(minutes=1),
        )
        await backend.aclose()

    assert ok is True
    first, rebuilt = seen
    assert rebuilt["document_id"] == first["document_id"]
    assert rebuilt["update_mode"] == "replace"
    assert rebuilt["content"] == "canonical transcript"
    # A tainted turn earlier in the session keeps the rebuilt document tainted.
    assert UNTRUSTED_TAG in rebuilt["tags"] and TRUSTED_TAG not in rebuilt["tags"]


@pytest.mark.asyncio
async def test_replace_session_without_open_document_declines(backend):
    ok = await backend.replace_session(
        "explr", "x", scope_tags=["channel:web_ui"], source_persona="explr",
        untrusted=False, timestamp=datetime.now(timezone.utc),
    )
    assert ok is False
    assert await backend.session_start("explr", ["channel:web_ui"]) is None


# ---------- MemoryManager.get_session_turns ----------

def test_session_turns_span_mixed_timestamp_formats(mm):
    start = datetime(2026, 9, 27, 3, 45, tzinfo=timezone.utc)
    _log(mm, "user", "before the session", start - timedelta(hours=30))
    # Naive = host-local wall time, as `datetime.now()` writes it.
    u = _log(mm, "user", "first", start.astimezone().replace(tzinfo=None))
    a = _log(mm, "assistant", "reply", (start + timedelta(seconds=9)).astimezone().replace(tzinfo=None))
    # Aware UTC, as a platform timestamp is stored.
    u2 = _log(mm, "user", "second", start + timedelta(minutes=5))
    gone = _log(mm, "assistant", "deleted", start + timedelta(minutes=6))
    mm.suppress_interaction(gone)
    _log(mm, "user", "other channel", start + timedelta(minutes=7), channel="elsewhere")

    turns = mm.get_session_turns("explr", "web_ui", start)
    assert [t["interaction_id"] for t in turns] == [u, a, u2]
    assert all(t["timestamp"].tzinfo is not None for t in turns)


# ---------- TurnPersistence.rebuild_session_safe ----------

@pytest.mark.asyncio
async def test_rebuild_renders_canonical_session_with_the_new_attempt(mm):
    start = datetime(2026, 9, 27, 3, 45, tzinfo=timezone.utc)
    _log(mm, "user", "how do they look?", start)
    retried = _log(mm, "assistant", "new attempt\n\n*Ran out of tool steps*", start + timedelta(seconds=9))

    backend = AsyncMock(spec=MemoryBackend)
    backend.session_start.return_value = start
    backend.replace_session.return_value = True
    tp = TurnPersistence(mm, backend)
    await tp.rebuild_session_safe(
        persona_name="explr", channel="web_ui", user_identifier="portal",
        server_id=None, retried_id=retried, retried_text="new attempt",
        untrusted=False,
    )

    backend.retain_turn.assert_not_called()
    content = backend.replace_session.call_args.args[1]
    assert content == "[2026-09-26 23:45] Adam: how do they look?\n\nexplr: new attempt"
    assert backend.replace_session.call_args.kwargs["scope_tags"] == [
        "channel:web_ui", "user:portal",
    ]


@pytest.mark.asyncio
async def test_rebuild_without_session_falls_back_to_append(mm):
    backend = AsyncMock(spec=MemoryBackend)
    backend.session_start.return_value = None
    tp = TurnPersistence(mm, backend)
    await tp.rebuild_session_safe(
        persona_name="explr", channel="web_ui", user_identifier="portal",
        server_id=None, retried_id=7, retried_text="new attempt", untrusted=True,
    )
    backend.replace_session.assert_not_called()
    kwargs = backend.retain_turn.call_args.kwargs
    assert kwargs["role"] == "assistant" and kwargs["untrusted"] is True
    assert kwargs["content"] == "explr: new attempt"
