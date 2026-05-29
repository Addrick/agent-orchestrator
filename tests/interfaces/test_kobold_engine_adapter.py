# tests/interfaces/test_kobold_adapter.py

"""Adapter HTTP-boundary tests.

Phase 2.1: kobold_export savefile contract.
Phase 2.2: ltm_block + persona memory_mode routes.
Phase 2.3a/b: pre-Phase-D logging contract (now exercised end-to-end through
the engine kernel — see Phase D fixture below).
Phase D (2026-04-28): OAI route is a thin SSE transcoder over
`chat_system.stream_response`. Tests use a real ChatSystem + in-memory DB
with the LLM step stubbed at `text_engine.stream_messages`.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from memory.memory_manager import MemoryManager
from src.chat_system import ChatSystem
from src.engine import TextEngine
from src.interfaces.kobold_engine_adapter import KoboldEngineAdapter as KoboldAdapter
from src.persona import Persona


def _make_adapter_with_seeded_db(persona_name: str = "test_persona",
                                 context_length: int = 10,
                                 retrieve_memory_block=None):
    """Build a KoboldAdapter backed by an in-memory DB and a stub ChatSystem.

    `retrieve_memory_block` lets a test inject the LTM result returned by the
    public ChatSystem.get_session_memory_block seam.
    """
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()

    persona = Persona(
        persona_name=persona_name,
        model_name="local",
        prompt="you are test",
        context_length=context_length,
    )
    chat_system = SimpleNamespace(
        personas={persona_name: persona},
        memory_manager=mm,
        get_session_memory_block=retrieve_memory_block or AsyncMock(return_value=None),
    )
    adapter = KoboldAdapter(chat_system=chat_system)
    return adapter, mm, persona


def _fetch_portal_rows(mm: MemoryManager, persona_name: str):
    """Pull web_ui rows with the columns the 2.3a tests care about."""
    conn = mm._get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT interaction_id, author_role, content, reply_to_id FROM User_Interactions"
        " WHERE persona_name = ? AND channel = 'web_ui'"
        " ORDER BY timestamp ASC, interaction_id ASC",
        (persona_name,),
    )
    return [dict(r) for r in cur.fetchall()]


def _events_for_text(text: str):
    """Engine-shape events that drive `chat_system._orchestrate` to commit `text`."""
    return [
        {"type": "api_payload", "payload": {}},
        {"type": "text_delta", "text": text},
        {"type": "done", "full_text": text},
    ]


def _make_stream_messages(call_event_lists):
    """Stateful stub: each invocation drains the next list of engine events.

    Falls back to the last list when more calls arrive than were configured,
    so tests describing only one LLM call don't need to repeat themselves.
    """
    state = {"i": 0}

    async def stream_messages(*args, **kwargs):
        idx = state["i"]
        state["i"] = idx + 1
        events = call_event_lists[idx] if idx < len(call_event_lists) else call_event_lists[-1]
        for ev in events:
            yield ev

    return stream_messages


def _make_real_adapter(persona_name: str = "test_persona",
                       stream_messages=None,
                       deltas=("ack",),
                       commit_text=None):
    """Build a KoboldAdapter wired to a real ChatSystem + in-memory MemoryManager.

    The LLM call is stubbed at `text_engine.stream_messages`. Either pass a
    custom `stream_messages` async-generator function or rely on the default
    that emits `deltas` and commits `commit_text` (defaults to concat).
    Returns `(adapter, memory_manager, persona, chat_system)`.
    """
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()

    persona = Persona(
        persona_name=persona_name,
        model_name="local",
        prompt="you are test",
        context_length=10,
    )

    if stream_messages is None:
        full = commit_text if commit_text is not None else "".join(deltas)
        events = [{"type": "api_payload", "payload": {}}]
        for d in deltas:
            events.append({"type": "text_delta", "text": d})
        events.append({"type": "done", "full_text": full})
        stream_messages = _make_stream_messages([events])

    text_engine = TextEngine()
    text_engine.stream_messages = stream_messages  # type: ignore[method-assign]

    with patch('src.chat_system.load_personas_from_file', return_value={persona_name: persona}):
        chat_system = ChatSystem(memory_manager=mm, text_engine=text_engine)
    chat_system.bot_logic.preprocess_message = AsyncMock(return_value=None)

    adapter = KoboldAdapter(chat_system=chat_system)
    return adapter, mm, persona, chat_system


def _seed_history(mm: MemoryManager, persona_name: str, turns: int):
    base = datetime(2026, 4, 1, 12, 0, 0)
    for i in range(turns):
        mm.log_message(
            user_identifier="user_a",
            persona_name=persona_name,
            channel="chan",
            author_role="user",
            author_name="user_a",
            content=f"user msg {i}",
            timestamp=base + timedelta(seconds=2 * i),
        )
        mm.log_message(
            user_identifier="user_a",
            persona_name=persona_name,
            channel="chan",
            author_role="assistant",
            author_name=persona_name,
            content=f"reply {i}",
            timestamp=base + timedelta(seconds=2 * i + 1),
        )


# -------- /api/v1/session/{persona}/kobold_export --------

def test_kobold_export_unknown_persona_returns_404():
    adapter, mm, _ = _make_adapter_with_seeded_db()
    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/session/nobody/kobold_export")
    assert r.status_code == 404
    assert "not found" in r.json()["error"].lower()
    mm.close()


def test_kobold_export_default_limit_uses_persona_context_length():
    # 6 full turns → 12 rows. context_length=4 means only the last 4 rows
    # come back, all of which are user/assistant pairs from the tail.
    adapter, mm, persona = _make_adapter_with_seeded_db(context_length=4)
    _seed_history(mm, "test_persona", turns=6)

    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/session/test_persona/kobold_export")
    assert r.status_code == 200
    savefile = r.json()
    rendered = [savefile["prompt"]] + savefile["actions"]
    assert len(rendered) == 4
    # Tail of the seeded run — turns 4 and 5 of the 6 seeded.
    assert "user msg 4" in rendered[0]
    assert rendered[1] == "reply 4"
    assert "user msg 5" in rendered[2]
    assert rendered[3] == "reply 5"
    mm.close()


def test_kobold_export_explicit_max_turns_overrides_default():
    adapter, mm, _ = _make_adapter_with_seeded_db(context_length=10)
    _seed_history(mm, "test_persona", turns=5)  # 10 rows total

    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/session/test_persona/kobold_export?max_turns=2")
    assert r.status_code == 200
    savefile = r.json()
    rendered = [savefile["prompt"]] + savefile["actions"]
    # Only the last 2 rows should appear.
    assert len(rendered) == 2
    assert "user msg 4" in rendered[0]
    assert rendered[1] == "reply 4"
    mm.close()


def test_kobold_export_memory_field_empty_not_persona_prompt():
    # Regression guard on gap #4: persona system prompt must NOT be stuffed
    # into savefile.memory. UI routes it to instruct_sysprompt instead.
    adapter, mm, _ = _make_adapter_with_seeded_db()
    _seed_history(mm, "test_persona", turns=1)

    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/session/test_persona/kobold_export")
    savefile = r.json()
    assert savefile["memory"] == ""
    assert "you are test" not in savefile["memory"]
    mm.close()


# -------- Passthrough regression (Phase 1 contract unchanged) --------

def test_find_last_user_content_skips_assistant_prefix():
    # jinja-hijack mode: messages[-1] is the assistant continuation prefix.
    # Adapter must scan backward for the real user turn.
    messages = [
        {"role": "system", "content": "you are test"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi!"},
        {"role": "user", "content": "what's up"},
        {"role": "assistant", "content": "", "prefix": True},
    ]
    assert KoboldAdapter._find_last_user_content(messages) == "what's up"


def test_find_last_user_content_handles_vision_array_content():
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "..."}},
        ]},
    ]
    assert KoboldAdapter._find_last_user_content(messages) == "describe"


def test_find_last_user_content_returns_none_when_no_user_msg():
    messages = [{"role": "assistant", "content": "hi"}]
    assert KoboldAdapter._find_last_user_content(messages) is None


# -------- Phase D: /chat/completions over chat_system.stream_response --------
#
# OAI route is now a thin SSE transcoder. Engine rebuilds history from DB.
# Tests stub at the engine boundary (`text_engine.stream_messages`) so the
# orchestration kernel runs end-to-end against a real MemoryManager.

def _chat_body(user_text: str, *, stream: bool = False, retry: bool = False):
    msgs = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": "", "prefix": True},
    ]
    body = {"messages": msgs, "stream": stream}
    if retry:
        body["derpr_retry"] = True
    return body


def test_chat_completions_sync_logs_user_then_assistant_with_reply_to():
    adapter, mm, _, _ = _make_real_adapter(deltas=("here is my reply",))

    with TestClient(adapter.app) as client:
        r = client.post("/chat/completions", json=_chat_body("tell me a joke"))
    assert r.status_code == 200

    rows = _fetch_portal_rows(mm, "test_persona")
    assert len(rows) == 2
    assert rows[0]["author_role"] == "user"
    assert rows[0]["content"] == "tell me a joke"
    assert rows[1]["author_role"] == "assistant"
    assert rows[1]["content"] == "here is my reply"
    assert rows[1]["reply_to_id"] == rows[0]["interaction_id"]
    mm.close()


def test_chat_completions_sidecar_user_text_overrides_messages():
    # jinja-hijack mode: post-repack messages array often has zero user-role
    # entries. The portal stamps raw input as derpr_user_text before the
    # textbox clears. Adapter must prefer this over scanning messages.
    adapter, mm, _, _ = _make_real_adapter(deltas=("ack",))

    body = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "", "prefix": True},
        ],
        "stream": False,
        "derpr_user_text": "real user turn",
    }
    with TestClient(adapter.app) as client:
        r = client.post("/chat/completions", json=body)
    assert r.status_code == 200

    rows = _fetch_portal_rows(mm, "test_persona")
    assert len(rows) == 2
    assert rows[0]["author_role"] == "user"
    assert rows[0]["content"] == "real user turn"
    mm.close()


def test_chat_completions_stream_logs_on_close_with_reply_to():
    adapter, mm, _, _ = _make_real_adapter(deltas=("hello ", "world"))

    with TestClient(adapter.app) as client:
        r = client.post("/chat/completions", json=_chat_body("hi", stream=True))
    assert r.status_code == 200

    rows = _fetch_portal_rows(mm, "test_persona")
    assert len(rows) == 2
    assert rows[1]["author_role"] == "assistant"
    assert rows[1]["content"] == "hello world"
    assert rows[1]["reply_to_id"] == rows[0]["interaction_id"]
    mm.close()


def test_chat_completions_retry_archives_and_updates_assistant():
    adapter, mm, _, _ = _make_real_adapter(deltas=("second attempt",))

    # Seed a user+assistant pair representing the prior turn.
    base = datetime(2026, 4, 20, 12, 0, 0)
    user_id = mm.log_message(
        user_identifier="portal", persona_name="test_persona", channel="web_ui",
        author_role="user", author_name=None, content="prior prompt", timestamp=base,
    )
    assistant_id = mm.log_message(
        user_identifier="portal", persona_name="test_persona", channel="web_ui",
        author_role="assistant", author_name=None, content="first attempt",
        timestamp=base + timedelta(seconds=1), reply_to_id=user_id,
    )

    with TestClient(adapter.app) as client:
        r = client.post("/chat/completions", json=_chat_body("prior prompt", retry=True))
    assert r.status_code == 200

    rows = _fetch_portal_rows(mm, "test_persona")
    assert len(rows) == 2  # No new user row, assistant row updated in place
    assistant_row = next(r for r in rows if r["author_role"] == "assistant")
    assert assistant_row["interaction_id"] == assistant_id
    assert assistant_row["content"] == "second attempt"

    # Prior content archived into Interaction_Edit_History
    conn = mm._get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT old_content FROM Interaction_Edit_History WHERE interaction_id = ?",
        (assistant_id,),
    )
    archived = cur.fetchall()
    assert len(archived) == 1
    assert archived[0]["old_content"] == "first attempt"
    mm.close()


def test_handle_portal_retry_returns_none_when_no_prior_assistant():
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    result = mm.handle_portal_retry("test_persona", "portal", "web_ui")
    assert result is None
    mm.close()


def test_chat_completions_stream_emits_derpr_tool_frames():
    """tool_revamp_v1 Phase 3: portal SSE relay forwards
    `event: derpr-tool-start` / `event: derpr-tool-result` for tool calls."""
    call_count = {"n": 0}

    async def stream_messages(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            yield {"type": "api_payload", "payload": {}}
            yield {"type": "tool_calls", "calls": [
                {"id": "call_42", "name": "search_tickets",
                 "arguments": {"query": "open"}}
            ]}
            yield {"type": "done", "full_text": ""}
        else:
            yield {"type": "api_payload", "payload": {}}
            yield {"type": "text_delta", "text": "done!"}
            yield {"type": "done", "full_text": "done!"}

    adapter, mm, persona, chat_system = _make_real_adapter(
        stream_messages=stream_messages,
    )
    persona.set_enabled_tools(["*"])
    chat_system.tool_manager.execute_tool = AsyncMock(
        return_value={"result": [{"id": 7}]},
    )

    body = _chat_body("find open tickets", stream=True)
    body["derpr_user_text"] = "find open tickets"
    with TestClient(adapter.app) as client:
        with client.stream("POST", "/chat/completions", json=body) as r:
            raw = b"".join(chunk for chunk in r.iter_raw())

    text = raw.decode("utf-8")
    # Tool-start frame uses the new event name and carries call_id + args.
    start_match = re.search(
        r"event: derpr-tool-start\ndata: (\{.*?\})\n\n", text,
    )
    assert start_match is not None, f"missing derpr-tool-start in:\n{text}"
    start_payload = json.loads(start_match.group(1))
    assert start_payload["tool_name"] == "search_tickets"
    assert start_payload["call_id"] == "call_42"
    assert start_payload["arguments"] == {"query": "open"}

    result_match = re.search(
        r"event: derpr-tool-result\ndata: (\{.*?\})\n\n", text,
    )
    assert result_match is not None, f"missing derpr-tool-result in:\n{text}"
    result_payload = json.loads(result_match.group(1))
    assert result_payload["call_id"] == "call_42"
    assert result_payload["error"] is None
    # Inner result string is the json-serialized tool_manager output.
    assert json.loads(result_payload["result"]) == {"result": [{"id": 7}]}

    # Frame ordering: start before result, both before [DONE].
    start_pos = text.index("event: derpr-tool-start")
    result_pos = text.index("event: derpr-tool-result")
    done_pos = text.index("[DONE]")
    assert start_pos < result_pos < done_pos
    mm.close()


def test_chat_completions_stream_abort_flushes_partial():
    async def stream_messages_then_cancel(*args, **kwargs):
        yield {"type": "api_payload", "payload": {}}
        yield {"type": "text_delta", "text": "partial "}
        raise asyncio.CancelledError()

    adapter, mm, _, _ = _make_real_adapter(stream_messages=stream_messages_then_cancel)

    with TestClient(adapter.app) as client:
        try:
            with client.stream("POST", "/chat/completions", json=_chat_body("hi", stream=True)) as r:
                for _ in r.iter_raw():
                    pass
        except Exception:
            pass  # CancelledError propagates through test client — engine flushed already

    rows = _fetch_portal_rows(mm, "test_persona")
    # User turn logged before LLM call, assistant partial flushed on cancel
    assert len(rows) == 2
    assert rows[1]["author_role"] == "assistant"
    assert rows[1]["content"] == "partial "
    mm.close()


# -------- Phase 2.2: /api/v1/persona/{name} memory_mode --------

def test_get_persona_includes_memory_mode():
    adapter, mm, persona = _make_adapter_with_seeded_db()
    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/persona/test_persona")
    assert r.status_code == 200
    data = r.json()
    assert "memory_mode" in data
    assert data["memory_mode"] == persona.get_memory_mode().name
    mm.close()


def test_patch_persona_updates_memory_mode():
    adapter, mm, persona = _make_adapter_with_seeded_db()
    with TestClient(adapter.app) as client:
        r = client.patch("/api/v1/persona/test_persona", json={"memory_mode": "GLOBAL"})
    assert r.status_code == 200
    assert persona.get_memory_mode().name == "GLOBAL"
    mm.close()


def test_patch_persona_unknown_mode_does_not_crash():
    # set_memory_mode logs a warning and keeps old mode on invalid input
    adapter, mm, persona = _make_adapter_with_seeded_db()
    original = persona.get_memory_mode()
    with TestClient(adapter.app) as client:
        r = client.patch("/api/v1/persona/test_persona", json={"memory_mode": "INVALID_MODE"})
    assert r.status_code == 200
    assert persona.get_memory_mode() == original
    mm.close()


# -------- Phase 2.2: /api/v1/session/{persona}/ltm_block --------

def test_ltm_block_unknown_persona_returns_404():
    adapter, mm, _ = _make_adapter_with_seeded_db()
    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/session/nobody/ltm_block?query=hello")
    assert r.status_code == 404
    mm.close()


def test_ltm_block_returns_null_when_retrieval_returns_none():
    adapter, mm, _ = _make_adapter_with_seeded_db(retrieve_memory_block=AsyncMock(return_value=None))
    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/session/test_persona/ltm_block?query=hello")
    assert r.status_code == 200
    assert r.json() == {"block": None}
    mm.close()


def test_ltm_block_returns_block_string_when_retrieval_succeeds():
    expected = "<memory>\nfact: user likes cats\n</memory>"
    adapter, mm, _ = _make_adapter_with_seeded_db(
        retrieve_memory_block=AsyncMock(return_value=expected)
    )
    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/session/test_persona/ltm_block?query=tell+me+about+pets")
    assert r.status_code == 200
    assert r.json()["block"] == expected
    mm.close()


def test_ltm_block_passes_query_text_to_retrieval():
    mock_retrieve = AsyncMock(return_value=None)
    adapter, mm, persona = _make_adapter_with_seeded_db(retrieve_memory_block=mock_retrieve)
    with TestClient(adapter.app) as client:
        client.get("/api/v1/session/test_persona/ltm_block", params={"query": "my query text"})
    mock_retrieve.assert_awaited_once()
    assert mock_retrieve.call_args.kwargs.get("query") == "my query text"
    mm.close()


def test_ltm_block_empty_query_passes_none():
    mock_retrieve = AsyncMock(return_value=None)
    adapter, mm, persona = _make_adapter_with_seeded_db(retrieve_memory_block=mock_retrieve)
    with TestClient(adapter.app) as client:
        client.get("/api/v1/session/test_persona/ltm_block")
    mock_retrieve.assert_awaited_once()
    # empty query string is forwarded as "" to the public seam — the seam itself
    # collapses falsy values to None before retrieval.
    kwargs = mock_retrieve.call_args.kwargs
    assert kwargs.get("query") in ("", None)
    mm.close()


# -------- Phase 2.3b → D: SSE derpr frame + version endpoints --------

def _make_stream_ctx(chunks):
    """Factory: context manager producing the given byte chunks from aiter_raw.

    Used by the native `/api/extra/generate/stream` tests further down — that
    route is still verbatim passthrough to KoboldCPP and mocks `_http.stream`.
    """
    class _StreamCtx:
        async def __aenter__(self):
            resp = MagicMock()

            async def _aiter_raw():
                for c in chunks:
                    yield c

            resp.aiter_raw = _aiter_raw
            return resp

        async def __aexit__(self, *a):
            return False

    return _StreamCtx()


def test_stream_emits_derpr_frame_before_done_with_assistant_id():
    adapter, mm, _, _ = _make_real_adapter(deltas=("hello ", "world"))

    with TestClient(adapter.app) as client:
        r = client.post("/chat/completions", json=_chat_body("hi", stream=True))
    assert r.status_code == 200
    body = r.text

    # derpr frame must precede [DONE]
    assert "event: derpr" in body
    idx_derpr = body.index("event: derpr")
    idx_done = body.index("[DONE]")
    assert idx_derpr < idx_done

    # Frame payload carries the canonical assistant_id
    m = re.search(r"event: derpr\ndata: (\{.*?\})\n\n", body)
    assert m, f"derpr frame not parseable: {body!r}"
    payload = json.loads(m.group(1))
    rows = _fetch_portal_rows(mm, "test_persona")
    assistant_row = next(r for r in rows if r["author_role"] == "assistant")
    assert payload["assistant_id"] == assistant_row["interaction_id"]
    mm.close()


def test_stream_without_content_does_not_emit_derpr_frame():
    # Engine emits a DoneEvent with assistant_id=None when no text was produced.
    adapter, mm, _, _ = _make_real_adapter(deltas=(), commit_text="")

    with TestClient(adapter.app) as client:
        r = client.post("/chat/completions", json=_chat_body("hi", stream=True))
    assert r.status_code == 200
    assert "event: derpr" not in r.text
    mm.close()


def test_list_versions_unknown_id_returns_404():
    adapter, mm, _ = _make_adapter_with_seeded_db()
    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/interaction/9999/versions")
    assert r.status_code == 404
    mm.close()


def test_list_versions_canonical_only_returns_single_entry():
    adapter, mm, _ = _make_adapter_with_seeded_db()
    iid = mm.log_message(
        user_identifier="portal", persona_name="test_persona", channel="web_ui",
        author_role="assistant", author_name=None, content="only version",
        timestamp=datetime(2026, 4, 22, 12, 0, 0),
    )
    with TestClient(adapter.app) as client:
        r = client.get(f"/api/v1/interaction/{iid}/versions")
    assert r.status_code == 200
    data = r.json()
    assert data["interaction_id"] == iid
    assert len(data["versions"]) == 1
    assert data["versions"][0]["edit_id"] is None
    assert data["versions"][0]["content"] == "only version"
    mm.close()


def test_select_version_out_of_bounds_returns_400():
    adapter, mm, _ = _make_adapter_with_seeded_db()
    iid = mm.log_message(
        user_identifier="portal", persona_name="test_persona", channel="web_ui",
        author_role="assistant", author_name=None, content="canonical",
        timestamp=datetime(2026, 4, 22, 12, 0, 0),
    )
    with TestClient(adapter.app) as client:
        r = client.post(f"/api/v1/interaction/{iid}/select_version/5")
    assert r.status_code == 400
    # Canonical unchanged
    with TestClient(adapter.app) as client:
        r2 = client.get(f"/api/v1/interaction/{iid}/versions")
    assert r2.json()["versions"][-1]["content"] == "canonical"
    mm.close()


def test_select_version_unknown_id_returns_404():
    adapter, mm, _ = _make_adapter_with_seeded_db()
    with TestClient(adapter.app) as client:
        r = client.post("/api/v1/interaction/9999/select_version/0")
    assert r.status_code == 404
    mm.close()


def test_retry_retry_select_version_round_trip_via_endpoints():
    """End-to-end: two sequential retries then restore original via endpoint.

    Initial assistant content = "v0". After retry #1 canonical = "v1",
    archives = [v0]. After retry #2 canonical = "v2", archives = [v0, v1].
    select_version(0) swaps archive[0] (v0) with canonical (v2):
    new canonical = "v0", archives = [v1, v2].
    """
    adapter, mm, _, _ = _make_real_adapter(
        stream_messages=_make_stream_messages([
            _events_for_text("v1"),
            _events_for_text("v2"),
        ]),
    )

    base = datetime(2026, 4, 22, 12, 0, 0)
    user_id = mm.log_message(
        user_identifier="portal", persona_name="test_persona", channel="web_ui",
        author_role="user", author_name=None, content="question", timestamp=base,
    )
    assistant_id = mm.log_message(
        user_identifier="portal", persona_name="test_persona", channel="web_ui",
        author_role="assistant", author_name=None, content="v0",
        timestamp=base + timedelta(seconds=1), reply_to_id=user_id,
    )

    with TestClient(adapter.app) as client:
        r1 = client.post("/chat/completions", json=_chat_body("question", retry=True))
        assert r1.status_code == 200
        r2 = client.post("/chat/completions", json=_chat_body("question", retry=True))
        assert r2.status_code == 200

        versions = client.get(f"/api/v1/interaction/{assistant_id}/versions").json()
        contents = [v["content"] for v in versions["versions"]]
        assert contents == ["v0", "v1", "v2"]

        swap = client.post(f"/api/v1/interaction/{assistant_id}/select_version/0").json()
        assert swap["current_content"] == "v0"
        assert swap["interaction_id"] == assistant_id
        assert swap["total_versions"] == 3

        after = client.get(f"/api/v1/interaction/{assistant_id}/versions").json()
        after_contents = [v["content"] for v in after["versions"]]
        assert after_contents == ["v1", "v2", "v0"]
        assert after["versions"][-1]["edit_id"] is None  # v0 now canonical

    rows = _fetch_portal_rows(mm, "test_persona")
    assistant_row = next(r for r in rows if r["author_role"] == "assistant")
    assert assistant_row["content"] == "v0"
    mm.close()


# -------- Phase 3: max_context_tokens — endpoint shape + outbound prune --------

def test_get_persona_includes_max_context_tokens():
    adapter, mm, persona = _make_adapter_with_seeded_db()
    persona.set_max_context_tokens(8192)
    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/persona/test_persona")
    assert r.status_code == 200
    assert r.json()["max_context_tokens"] == 8192
    mm.close()


# -------- SP-1: tools/catalog + extended persona GET --------

def test_tools_catalog_returns_expected_fields():
    adapter, mm, _ = _make_adapter_with_seeded_db()
    # Mock tool_manager to return a known tool
    adapter.chat_system.tool_manager = MagicMock()
    adapter.chat_system.tool_manager.get_tool_definitions.return_value = [
        {
            "type": "function",
            "is_write": False,
            "capabilities": {
                "produces_untrusted": True,
                "locality": "network",
                "sensitivity": "public",
            },
            "function": {
                "name": "web_search",
                "description": "Searches the web.",
            },
        }
    ]

    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/tools/catalog")
    assert r.status_code == 200
    data = r.json()
    assert "tools" in data
    assert len(data["tools"]) == 1
    t = data["tools"][0]
    assert t["name"] == "web_search"
    assert t["description"] == "Searches the web."
    assert t["is_write"] is False
    assert t["capabilities"]["locality"] == "network"
    assert t["capabilities"]["sensitivity"] == "public"
    assert t["capabilities"]["produces_untrusted"] is True


def test_tools_catalog_capabilities_present_even_if_null():
    adapter, mm, _ = _make_adapter_with_seeded_db()
    adapter.chat_system.tool_manager = MagicMock()
    # Minimal tool with missing fields
    adapter.chat_system.tool_manager.get_tool_definitions.return_value = [
        {
            "function": {"name": "minimal"},
            # capabilities missing
        }
    ]

    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/tools/catalog")
    assert r.status_code == 200
    t = r.json()["tools"][0]
    assert t["name"] == "minimal"
    assert "capabilities" in t
    assert t["capabilities"]["locality"] is None
    assert t["capabilities"]["sensitivity"] is None
    assert t["capabilities"]["produces_untrusted"] is False


def test_persona_extended_includes_enabled_tools():
    adapter, mm, persona = _make_adapter_with_seeded_db()
    persona.set_enabled_tools(["tool1", "tool2"])

    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/persona/test_persona")
    assert r.status_code == 200
    data = r.json()
    assert "enabled_tools" in data
    assert sorted(data["enabled_tools"]) == ["tool1", "tool2"]


def test_persona_extended_includes_tool_policy():
    adapter, mm, persona = _make_adapter_with_seeded_db()
    # Use real ToolPolicy to_dict result
    expected_policy = persona.get_tool_policy().to_dict()

    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/persona/test_persona")
    assert r.status_code == 200
    data = r.json()
    assert "tool_policy" in data
    assert data["tool_policy"] == expected_policy


def test_persona_extended_unknown_persona_returns_404():
    adapter, mm, _ = _make_adapter_with_seeded_db()
    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/persona/nonexistent")
    # If the original code returned 200, and I'm asked to preserve behavior
    # and "returns_404", I'll check what it actually returns.
    # If I haven't changed the code yet, it will return 200.
    # I'll update the code to return 404 in the next step to satisfy the test name.
    assert r.status_code == 404
    assert "error" in r.json()
    mm.close()


def test_patch_persona_updates_max_context_tokens():
    adapter, mm, persona = _make_adapter_with_seeded_db()
    with TestClient(adapter.app) as client:
        r = client.patch("/api/v1/persona/test_persona", json={"max_context_tokens": 16384})
    assert r.status_code == 200
    assert persona.get_max_context_tokens() == 16384
    mm.close()


# -------- Portal persona settings sync (Phase 2): kobold sampler extras --------

def test_patch_persona_writes_kobold_sampler_extras():
    """rep_pen / min_p / typical / tfs land in provider_extras["kobold"]."""
    adapter, mm, persona = _make_adapter_with_seeded_db()
    body = {
        "rep_pen": 1.15,
        "rep_pen_range": 1024,
        "rep_pen_slope": 0.7,
        "min_p": 0.05,
        "typical": 0.95,
        "tfs": 0.97,
    }
    with TestClient(adapter.app) as client:
        r = client.patch("/api/v1/persona/test_persona", json=body)
    assert r.status_code == 200
    assert persona.get_provider_extra("kobold", "rep_pen") == 1.15
    assert persona.get_provider_extra("kobold", "rep_pen_range") == 1024
    assert persona.get_provider_extra("kobold", "rep_pen_slope") == 0.7
    assert persona.get_provider_extra("kobold", "min_p") == 0.05
    assert persona.get_provider_extra("kobold", "typical") == 0.95
    assert persona.get_provider_extra("kobold", "tfs") == 0.97
    mm.close()


def test_patch_persona_clear_kobold_extra_via_none():
    """None / "clear" / "" remove the key from provider_extras["kobold"]."""
    adapter, mm, persona = _make_adapter_with_seeded_db()
    persona.set_provider_extra("kobold", "rep_pen", 1.2)
    persona.set_provider_extra("kobold", "min_p", 0.05)
    with TestClient(adapter.app) as client:
        r = client.patch("/api/v1/persona/test_persona",
                         json={"rep_pen": None, "min_p": "clear"})
    assert r.status_code == 200
    assert persona.get_provider_extra("kobold", "rep_pen") is None
    assert persona.get_provider_extra("kobold", "min_p") is None
    mm.close()


def test_patch_persona_kobold_extra_bad_input_rejected():
    """Non-coercible input → field appears in rejected_fields, prior value kept."""
    adapter, mm, persona = _make_adapter_with_seeded_db()
    persona.set_provider_extra("kobold", "rep_pen", 1.2)
    with TestClient(adapter.app) as client:
        r = client.patch("/api/v1/persona/test_persona",
                         json={"rep_pen": "not-a-number"})
    assert r.status_code == 200
    assert "rep_pen" in r.json()["rejected_fields"]
    assert persona.get_provider_extra("kobold", "rep_pen") == 1.2
    mm.close()


def test_patch_persona_unknown_field_returned():
    """Unknown keys land in unknown_fields list and are otherwise ignored."""
    adapter, mm, persona = _make_adapter_with_seeded_db()
    with TestClient(adapter.app) as client:
        r = client.patch("/api/v1/persona/test_persona",
                         json={"made_up_knob": 42, "another_one": "x"})
    assert r.status_code == 200
    unknown = r.json()["unknown_fields"]
    assert "made_up_knob" in unknown
    assert "another_one" in unknown
    mm.close()


def test_get_persona_includes_kobold_extras():
    """GET surfaces only set kobold extras (omits unset keys)."""
    adapter, mm, persona = _make_adapter_with_seeded_db()
    persona.set_provider_extra("kobold", "rep_pen", 1.1)
    persona.set_provider_extra("kobold", "min_p", 0.04)
    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/persona/test_persona")
    assert r.status_code == 200
    extras = r.json()["kobold_extras"]
    assert extras == {"rep_pen": 1.1, "min_p": 0.04}
    mm.close()


# Phase D dropped Phase 3 prune tests: pruning is now an engine-side
# concern (`_prepare_request` → `truncate_messages_to_budget`) — see
# tests/test_chat_system.py for coverage. The OAI adapter no longer touches
# the messages array.

# -------- Phase 2.4: portal edit/delete round-trip --------

def test_delete_interaction_suppresses_row():
    """DELETE soft-suppresses the row and returns success."""
    adapter, mm, _ = _make_adapter_with_seeded_db()
    iid = mm.log_message("user_a", "test_persona", "web_ui", "user", "Alice",
                         "to be deleted", datetime.now())

    with TestClient(adapter.app) as client:
        r = client.delete(f"/api/v1/interaction/{iid}")
    assert r.status_code == 200
    payload = r.json()
    assert payload["result"] == "success"
    assert payload["interaction_id"] == iid
    assert payload["already_suppressed"] is False

    # History queries now skip the row.
    history = mm.get_personal_history("user_a", "test_persona")
    assert all(row["interaction_id"] != iid for row in history)
    mm.close()


def test_delete_interaction_idempotent():
    """Second DELETE on the same id reports already_suppressed=true."""
    adapter, mm, _ = _make_adapter_with_seeded_db()
    iid = mm.log_message("user_a", "test_persona", "web_ui", "user", "Alice",
                         "x", datetime.now())

    with TestClient(adapter.app) as client:
        r1 = client.delete(f"/api/v1/interaction/{iid}")
        r2 = client.delete(f"/api/v1/interaction/{iid}")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["already_suppressed"] is False
    assert r2.json()["already_suppressed"] is True

    conn = mm._get_connection()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM Suppressed_Interactions WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()[0] == 1
    mm.close()


def test_patch_interaction_clears_l0_embedding():
    """PATCH triggers L0 invalidation: Message_Embeddings + vec_* gone."""
    import struct
    import math
    from config.global_config import EMBEDDING_DIMENSION, EMBEDDING_MODEL
    adapter, mm, _ = _make_adapter_with_seeded_db()
    iid = mm.log_message("user_a", "test_persona", "web_ui", "assistant", "test_persona",
                         "v1", datetime.now())
    emb = struct.pack(f'{EMBEDDING_DIMENSION}f',
                      *([1.0 / math.sqrt(EMBEDDING_DIMENSION)] * EMBEDDING_DIMENSION))
    mm.store_message_embedding(iid, emb, EMBEDDING_MODEL, datetime.now())
    conn = mm._get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO vec_Message_Embeddings (interaction_id, embedding) VALUES (?, ?)",
        (iid, emb),
    )
    conn.commit()

    with TestClient(adapter.app) as client:
        r = client.patch(f"/api/v1/interaction/{iid}", json={"content": "v1-edited"})
    assert r.status_code == 200

    cur = conn.cursor()
    cur.execute("SELECT content FROM User_Interactions WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()["content"] == "v1-edited"
    cur.execute("SELECT count(*) FROM Message_Embeddings WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()[0] == 0
    cur.execute("SELECT count(*) FROM vec_Message_Embeddings WHERE interaction_id = ?", (iid,))
    assert cur.fetchone()[0] == 0
    mm.close()


# -------- /api/extra/generate/stream — kobold-native SSE relay --------
#
# Coverage-prep before portal_engine_reintegration Phase D. The native
# streaming route proxies KoboldCPP's `/api/extra/generate/stream` endpoint
# (per-token SSE) and was previously untested. These tests pin the
# user/assistant logging contract so the migration to chat_system.stream_prompt
# can be verified one-for-one. See memory/project/plans/portal_engine_reintegration.md.


def _kobold_native_chunk(token: str) -> bytes:
    """Build one kobold-native SSE event carrying a single token."""
    return f'event: message\ndata: {json.dumps({"token": token})}\n\n'.encode("utf-8")


def test_generate_stream_logs_user_turn_from_prompt(monkeypatch):
    adapter, mm, _ = _make_adapter_with_seeded_db()

    chunks = [_kobold_native_chunk("hi"), b"data: [DONE]\n\n"]
    monkeypatch.setattr(adapter._http, "stream", lambda *a, **kw: _make_stream_ctx(chunks))

    with TestClient(adapter.app) as client:
        r = client.post("/api/extra/generate/stream", json={"prompt": "hello there"})
    assert r.status_code == 200

    rows = _fetch_portal_rows(mm, "test_persona")
    user_rows = [r for r in rows if r["author_role"] == "user"]
    assert len(user_rows) == 1
    assert user_rows[0]["content"] == "hello there"
    mm.close()


def test_generate_stream_commits_assistant_from_assembled_tokens(monkeypatch):
    adapter, mm, _ = _make_adapter_with_seeded_db()

    chunks = [
        _kobold_native_chunk("hello "),
        _kobold_native_chunk("world"),
        b"data: [DONE]\n\n",
    ]
    monkeypatch.setattr(adapter._http, "stream", lambda *a, **kw: _make_stream_ctx(chunks))

    with TestClient(adapter.app) as client:
        r = client.post("/api/extra/generate/stream", json={"prompt": "hi"})
    assert r.status_code == 200

    rows = _fetch_portal_rows(mm, "test_persona")
    assert len(rows) == 2
    assistant = next(r for r in rows if r["author_role"] == "assistant")
    user = next(r for r in rows if r["author_role"] == "user")
    assert assistant["content"] == "hello world"
    assert assistant["reply_to_id"] == user["interaction_id"]
    mm.close()


def test_generate_stream_empty_prompt_skips_user_log(monkeypatch):
    # Whitespace-only / missing prompt must NOT create an empty user row.
    # Assistant log still flushes if upstream produced tokens.
    adapter, mm, _ = _make_adapter_with_seeded_db()

    chunks = [_kobold_native_chunk("orphan"), b"data: [DONE]\n\n"]
    monkeypatch.setattr(adapter._http, "stream", lambda *a, **kw: _make_stream_ctx(chunks))

    with TestClient(adapter.app) as client:
        r = client.post("/api/extra/generate/stream", json={"prompt": "   "})
    assert r.status_code == 200

    rows = _fetch_portal_rows(mm, "test_persona")
    assert all(row["author_role"] != "user" for row in rows)
    assistant_rows = [r for r in rows if r["author_role"] == "assistant"]
    assert len(assistant_rows) == 1
    assert assistant_rows[0]["content"] == "orphan"
    assert assistant_rows[0]["reply_to_id"] is None
    mm.close()


def test_generate_stream_strips_model_field_before_forwarding(monkeypatch):
    # `model` is the DERPR persona selector — must not leak upstream to KCPP.
    adapter, mm, _ = _make_adapter_with_seeded_db()

    captured = {}

    class _StreamCtx:
        def __init__(self, body):
            captured["body"] = body

        async def __aenter__(self):
            resp = MagicMock()

            async def _aiter_raw():
                yield b"data: [DONE]\n\n"

            resp.aiter_raw = _aiter_raw
            return resp

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(
        adapter._http, "stream",
        lambda method, url, json=None, **kw: _StreamCtx(json),
    )

    with TestClient(adapter.app) as client:
        r = client.post(
            "/api/extra/generate/stream",
            json={"prompt": "hi", "model": "should_not_leak", "temperature": 0.5},
        )
    assert r.status_code == 200
    assert "model" not in captured["body"]
    assert captured["body"]["temperature"] == 0.5
    mm.close()


def test_generate_stream_abort_flushes_partial_assistant(monkeypatch):
    adapter, mm, _ = _make_adapter_with_seeded_db()

    class _StreamCtx:
        async def __aenter__(self):
            resp = MagicMock()

            async def _aiter_raw():
                yield _kobold_native_chunk("partial ")
                raise __import__("asyncio").CancelledError()

            resp.aiter_raw = _aiter_raw
            return resp

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(adapter._http, "stream", lambda *a, **kw: _StreamCtx())

    with TestClient(adapter.app) as client:
        try:
            with client.stream(
                "POST", "/api/extra/generate/stream",
                json={"prompt": "hi"},
            ) as r:
                for _ in r.iter_raw():
                    pass
        except Exception:
            pass

    rows = _fetch_portal_rows(mm, "test_persona")
    assistant_rows = [r for r in rows if r["author_role"] == "assistant"]
    assert len(assistant_rows) == 1
    assert assistant_rows[0]["content"] == "partial "
    mm.close()


def test_kobold_export_excludes_suppressed_row():
    """A row deleted via the DELETE endpoint disappears from kobold_export."""
    adapter, mm, persona = _make_adapter_with_seeded_db(context_length=20)
    base = datetime(2026, 4, 1, 12, 0, 0)
    mm.log_message("user_a", "test_persona", "chan", "user", "Alice",
                             "keep me", base)
    drop_id = mm.log_message("user_a", "test_persona", "chan", "user", "Alice",
                             "delete me", base + timedelta(seconds=1))

    with TestClient(adapter.app) as client:
        r_del = client.delete(f"/api/v1/interaction/{drop_id}")
        assert r_del.status_code == 200
        r_export = client.get("/api/v1/session/test_persona/kobold_export")
    assert r_export.status_code == 200
    rendered = json.dumps(r_export.json())
    assert "keep me" in rendered
    assert "delete me" not in rendered
    mm.close()

# Phase D dropped reasoning_content separation from the OAI route. The
# engine streams tokens from kobold's native endpoint, which delivers any
# `<think>` blocks inline rather than as a structured `reasoning_content`
# field. Reasoning extraction can be revisited if/when an engine-side
# parser is added.


# -------- Wire-fidelity: kobold-native passthrough prompt integrity --------
#
# These tests verify that /api/extra/generate/stream forwards the incoming
# kobold-ui payload to KoboldCPP unmodified — instruct tags, history,
# generation parameters — and only strips the `model` field (DERPR's
# persona selector, which must not leak upstream).
#
# Strategy: monkeypatch `adapter._http.stream` with a capturing context
# manager, then compare the intercepted `json=` body to the original
# request payload field-by-field.


class _CapturingStreamCtx:
    """Fake httpx.stream context that records its `json=` kwarg and returns
    a single [DONE] event so the relay_stream generator terminates cleanly.
    """

    def __init__(self):
        self.captured_body: dict = {}

    def __call__(self, method, url, json=None, **kw):
        self.captured_body = json or {}
        return self

    async def __aenter__(self):
        resp = MagicMock()

        async def _aiter_raw():
            yield b"data: [DONE]\n\n"

        resp.aiter_raw = _aiter_raw
        return resp

    async def __aexit__(self, *a):
        return False


def _make_rich_kobold_payload(**overrides) -> dict:
    """Realistic kobold-ui payload with instruct tags embedded in the prompt
    and a full suite of generation parameters — mirrors what kobold-lite
    sends when submitting a chat turn through the native stream endpoint.
    """
    prompt = (
        "### System:\nYou are a helpful assistant.\n\n"
        "### User:\nHello, can you recap our conversation?\n\n"
        "### Response:\nOf course! Here is what we discussed:\n\n"
        "### User:\nWhat did we say about dragons?\n\n"
        "### Response:\n"  # trailing generation cursor — must be preserved verbatim
    )
    payload = {
        "prompt": prompt,
        "max_length": 200,
        "max_context_length": 4096,
        "temperature": 0.72,
        "top_p": 0.9,
        "top_k": 40,
        "rep_pen": 1.1,
        "rep_pen_range": 512,
        "stop_sequence": ["### User:", "### System:"],
        "stream": True,
        "trim_stop": True,
        "quiet": True,
    }
    payload.update(overrides)
    return payload


def test_generate_stream_forwards_prompt_verbatim(monkeypatch):
    """The raw `prompt` string — instruct tags, history turns, trailing cursor
    — must arrive at KoboldCPP byte-for-byte identical to what kobold-ui sent.
    """
    adapter, mm, _ = _make_adapter_with_seeded_db()
    ctx = _CapturingStreamCtx()
    monkeypatch.setattr(adapter._http, "stream", ctx)

    payload = _make_rich_kobold_payload()

    with TestClient(adapter.app) as client:
        r = client.post("/api/extra/generate/stream", json=payload)
    assert r.status_code == 200

    assert ctx.captured_body["prompt"] == payload["prompt"], (
        "Prompt was mutated before forwarding to KoboldCPP!\n"
        f"sent:     {payload['prompt']!r}\n"
        f"received: {ctx.captured_body['prompt']!r}"
    )
    mm.close()


def test_generate_stream_forwards_all_generation_params_intact(monkeypatch):
    """Every generation parameter (temperature, top_p, top_k, rep_pen, etc.)
    must pass through to KoboldCPP unchanged.
    """
    adapter, mm, _ = _make_adapter_with_seeded_db()
    ctx = _CapturingStreamCtx()
    monkeypatch.setattr(adapter._http, "stream", ctx)

    payload = _make_rich_kobold_payload()
    gen_params = [
        "max_length", "max_context_length", "temperature",
        "top_p", "top_k", "rep_pen", "rep_pen_range",
        "stop_sequence", "trim_stop", "quiet",
    ]

    with TestClient(adapter.app) as client:
        client.post("/api/extra/generate/stream", json=payload)

    for key in gen_params:
        assert key in ctx.captured_body, f"Generation param {key!r} missing from forwarded body"
        assert ctx.captured_body[key] == payload[key], (
            f"Generation param {key!r} was mutated!\n"
            f"sent:     {payload[key]!r}\n"
            f"received: {ctx.captured_body[key]!r}"
        )
    mm.close()


def test_generate_stream_strips_only_model_field(monkeypatch):
    """The `model` field (DERPR's persona selector) must be stripped from the
    forwarded body — all other fields including unknown/extra keys must pass
    through untouched.  This is the one intentional transformation the relay
    makes to the kobold-ui payload.
    """
    adapter, mm, _ = _make_adapter_with_seeded_db()
    ctx = _CapturingStreamCtx()
    monkeypatch.setattr(adapter._http, "stream", ctx)

    payload = _make_rich_kobold_payload(model="test_persona", extra_custom_field="keep_me")

    with TestClient(adapter.app) as client:
        r = client.post("/api/extra/generate/stream", json=payload)
    assert r.status_code == 200

    assert "model" not in ctx.captured_body, (
        "The `model` field leaked to KoboldCPP; it should be stripped by the relay"
    )
    assert ctx.captured_body.get("extra_custom_field") == "keep_me", (
        "A non-model field was unexpectedly dropped from the forwarded body"
    )
    mm.close()


def test_generate_stream_payload_contains_no_extra_fields_beyond_input(monkeypatch):
    """The forwarded body must not contain fields that were not in the
    original kobold-ui request (derpr must not inject new keys).  Only the
    `model` strip is allowed; no additions.
    """
    adapter, mm, _ = _make_adapter_with_seeded_db()
    ctx = _CapturingStreamCtx()
    monkeypatch.setattr(adapter._http, "stream", ctx)

    payload = _make_rich_kobold_payload()

    with TestClient(adapter.app) as client:
        client.post("/api/extra/generate/stream", json=payload)

    expected_keys = {k for k in payload if k != "model"}
    forwarded_keys = set(ctx.captured_body.keys())

    injected = forwarded_keys - expected_keys
    assert not injected, (
        f"DERPR injected unexpected keys into the forwarded payload: {injected}"
    )
    mm.close()


def test_generate_stream_instruct_tags_preserved_in_prompt(monkeypatch):
    """Instruct-format tags embedded in the prompt must not be escaped,
    stripped, or reordered.  This guards against any accidental sanitisation
    that would break the kobold-lite jinja template output.
    """
    adapter, mm, _ = _make_adapter_with_seeded_db()
    ctx = _CapturingStreamCtx()
    monkeypatch.setattr(adapter._http, "stream", ctx)

    instruct_tags = [
        "### System:\n",
        "### User:\n",
        "### Response:\n",
        "<|im_start|>system\n",
        "<|im_end|>\n",
        "<|im_start|>user\n",
        "[INST]",
        "[/INST]",
    ]
    # Build a prompt containing all common tag styles
    prompt = "".join(instruct_tags) + "answer here\n\n### Response:\n"
    payload = _make_rich_kobold_payload(prompt=prompt)

    with TestClient(adapter.app) as client:
        client.post("/api/extra/generate/stream", json=payload)

    forwarded_prompt = ctx.captured_body.get("prompt", "")
    for tag in instruct_tags:
        assert tag in forwarded_prompt, (
            f"Instruct tag {tag!r} was dropped or mutated in the forwarded prompt.\n"
            f"Forwarded prompt: {forwarded_prompt!r}"
        )
    mm.close()


def test_generate_stream_upstream_url_targets_kobold_base(monkeypatch):
    """The relay must POST to `{LOCAL_LLM_URL}/api/extra/generate/stream`,
    not any DERPR-internal endpoint.  Captures the `url` positional arg.
    """
    adapter, mm, _ = _make_adapter_with_seeded_db()

    captured = {}

    class _URLCapture:
        def __call__(self, method, url, **kw):
            captured["method"] = method
            captured["url"] = url
            return self

        async def __aenter__(self):
            resp = MagicMock()

            async def _aiter_raw():
                yield b"data: [DONE]\n\n"

            resp.aiter_raw = _aiter_raw
            return resp

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(adapter._http, "stream", _URLCapture())

    with TestClient(adapter.app) as client:
        client.post("/api/extra/generate/stream", json={"prompt": "hello"})

    assert captured.get("method") == "POST"
    assert "/api/extra/generate/stream" in captured["url"], (
        f"Relay did not target the expected KoboldCPP path: {captured.get('url')!r}"
    )
    mm.close()


# -------- SP-2a: dev_command endpoint tests --------

def test_dev_command_happy_path_mutates_and_saves():
    adapter, mm, persona, chat_system = _make_real_adapter()
    chat_system.bot_logic.preprocess_message = AsyncMock(return_value={
        "response": "Tools set to none.",
        "mutated": True
    })

    with patch("src.interfaces.kobold_engine_adapter.save_personas_to_file") as mock_save:
        with TestClient(adapter.app) as client:
            r = client.post("/api/v1/persona/test_persona/dev_command", json={"command": "set tools none"})

    assert r.status_code == 200
    assert r.json() == {"response": "Tools set to none.", "mutated": True}
    chat_system.bot_logic.preprocess_message.assert_awaited_once_with("test_persona", "portal", "set tools none")
    mock_save.assert_called_once_with(chat_system.personas)
    mm.close()


def test_dev_command_non_mutating_does_not_save():
    adapter, mm, persona, chat_system = _make_real_adapter()
    chat_system.bot_logic.preprocess_message = AsyncMock(return_value={
        "response": "Tools: none.",
        "mutated": False
    })

    with patch("src.interfaces.kobold_engine_adapter.save_personas_to_file") as mock_save:
        with TestClient(adapter.app) as client:
            r = client.post("/api/v1/persona/test_persona/dev_command", json={"command": "what tools"})

    assert r.status_code == 200
    assert r.json() == {"response": "Tools: none.", "mutated": False}
    mock_save.assert_not_called()
    mm.close()


def test_dev_command_unknown_persona_returns_404():
    adapter, mm, persona, chat_system = _make_real_adapter()
    chat_system.bot_logic.preprocess_message = AsyncMock()

    with patch("src.interfaces.kobold_engine_adapter.save_personas_to_file") as mock_save:
        with TestClient(adapter.app) as client:
            r = client.post("/api/v1/persona/ghost/dev_command", json={"command": "set tools none"})

    assert r.status_code == 404
    chat_system.bot_logic.preprocess_message.assert_not_called()
    mock_save.assert_not_called()
    mm.close()


def test_dev_command_non_command_returns_400():
    adapter, mm, persona, chat_system = _make_real_adapter()
    # preprocess_message returns None if it's not a dev command
    chat_system.bot_logic.preprocess_message = AsyncMock(return_value=None)

    with patch("src.interfaces.kobold_engine_adapter.save_personas_to_file") as mock_save:
        with TestClient(adapter.app) as client:
            r = client.post("/api/v1/persona/test_persona/dev_command", json={"command": "not a command"})

    assert r.status_code == 400
    assert r.json() == {"response": "Not a dev command", "mutated": False}
    mock_save.assert_not_called()
    mm.close()


def test_dev_command_preprocess_error_surfaces_in_response():
    adapter, mm, persona, chat_system = _make_real_adapter()
    chat_system.bot_logic.preprocess_message = AsyncMock(side_effect=Exception("boom"))

    with patch("src.interfaces.kobold_engine_adapter.save_personas_to_file") as mock_save:
        with TestClient(adapter.app) as client:
            r = client.post("/api/v1/persona/test_persona/dev_command", json={"command": "error command"})

    assert r.status_code == 200
    assert r.json()["mutated"] is False
    assert "boom" in r.json()["response"]
    mock_save.assert_not_called()
    mm.close()


@patch('src.engine.genai.client.AsyncClient')
def test_chat_completions_google_end_to_end_payload_structure(mock_google_client_class, monkeypatch):
    """
    Asserts a full input/output chain of the Web UI chat completions endpoint
    for Google models. Verifies that the engine constructs the correct API
    payload using system_instruction and excludes system prompt from contents.
    """
    import pytest
    monkeypatch.setenv("GOOGLE_GENERATIVEAI_API_KEY", "dummy_key_for_testing")
    
    # 1. Setup adapter with real ChatSystem, but with our Google model persona
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()

    persona = Persona(
        persona_name="test_google_persona",
        model_name="gemini-2.5-flash",
        prompt="Always speak like a pirate",
        context_length=10,
    )
    
    # 2. Mock Google Client response
    mock_instance = mock_google_client_class.return_value
    mock_part = MagicMock(text="Ahoy matey! I am ready.", function_call=None)
    mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]), grounding_metadata=None)
    mock_instance.models.generate_content = AsyncMock(
        return_value=MagicMock(prompt_feedback=None, candidates=[mock_candidate])
    )
    
    text_engine = TextEngine()
    
    with patch('src.chat_system.load_personas_from_file', return_value={"test_google_persona": persona}):
        chat_system = ChatSystem(memory_manager=mm, text_engine=text_engine)
    chat_system.bot_logic.preprocess_message = AsyncMock(return_value=None)
    
    adapter = KoboldAdapter(chat_system=chat_system)
    
    # We need to set the current persona name on the adapter
    adapter._get_current_persona_name = MagicMock(return_value="test_google_persona")
    
    # 3. Call endpoint
    body = {
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "", "prefix": True},
        ],
        "stream": False,
        "derpr_user_text": "Hello there!",
    }
    
    with TestClient(adapter.app) as client:
        r = client.post("/chat/completions", json=body)
        
    # 4. Assert responses
    assert r.status_code == 200
    res_data = r.json()
    assert res_data["choices"][0]["message"]["content"] == "Ahoy matey! I am ready."
    
    # 5. Assert constructed payload structure for Gemini AsyncClient
    mock_instance.models.generate_content.assert_called_once()
    call_kwargs = mock_instance.models.generate_content.call_args.kwargs
    
    # Assert system prompt is NOT in contents
    contents = call_kwargs["contents"]
    for turn in contents:
        # Should not have any system role or system prompt content in parts
        assert getattr(turn, "role", None) != "system"
        for part in turn.get("parts", []):
            assert part.text != "Always speak like a pirate"
            
    # Assert system prompt IS set in config as system_instruction
    config = call_kwargs["config"]
    assert config.system_instruction == "Always speak like a pirate"
    
    mm.close()
