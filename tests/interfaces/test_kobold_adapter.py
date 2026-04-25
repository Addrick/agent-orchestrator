# tests/interfaces/test_kobold_adapter.py

"""Adapter HTTP-boundary tests.

Phase 2.1: kobold_export savefile contract.
Phase 2.2: ltm_block + persona memory_mode routes.
Phase 2.3a: /chat/completions logging — user-turn detection in jinja-hijack
mode (messages[-1] is the assistant prefix, not the user), reply_to_id
threading, abort partial-buffer flush, and derpr_retry archive+update path.
Phase 2.3b: SSE `event: derpr` frame carrying assistant_id, and
/api/v1/interaction/{id}/versions + select_version/{k} endpoints.
"""

import json
import re
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from memory.memory_manager import MemoryManager
from src.interfaces.kobold_adapter import KoboldAdapter
from src.persona import Persona


def _make_adapter_with_seeded_db(persona_name: str = "test_persona",
                                 context_length: int = 10,
                                 retrieve_memory_block=None):
    """Build a KoboldAdapter backed by an in-memory DB and a stub ChatSystem."""
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
        _retrieve_memory_block=retrieve_memory_block or AsyncMock(return_value=None),
        _build_conversation_history=MagicMock(return_value=([], None)),
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

def test_strip_envelope_drops_derpr_only_fields():
    # _strip_envelope is the gatekeeper that stops DERPR-internal fields from
    # leaking upstream. Top-level and params-nested history_override must both
    # be dropped; other fields pass through.
    payload = {
        "prompt": "hello",
        "model": "test_persona",
        "history_override": True,
        "params": {"temperature": 0.7, "history_override": True},
    }
    stripped = KoboldAdapter._strip_envelope(payload)
    assert "model" not in stripped
    assert "history_override" not in stripped
    assert "history_override" not in stripped["params"]
    assert stripped["prompt"] == "hello"
    assert stripped["params"]["temperature"] == 0.7


def test_strip_envelope_drops_derpr_retry():
    payload = {"messages": [], "derpr_retry": True, "stream": True}
    stripped = KoboldAdapter._strip_envelope(payload)
    assert "derpr_retry" not in stripped
    assert stripped["stream"] is True


def test_strip_envelope_drops_derpr_user_text():
    # Sidecar field carries raw user input for logging; must not leak upstream.
    payload = {"messages": [], "derpr_user_text": "hello", "stream": False}
    stripped = KoboldAdapter._strip_envelope(payload)
    assert "derpr_user_text" not in stripped


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


# -------- Phase 2.3a: /chat/completions logging --------

def _chat_body(user_text: str, *, stream: bool = False, retry: bool = False):
    msgs = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": "", "prefix": True},
    ]
    body = {"messages": msgs, "stream": stream}
    if retry:
        body["derpr_retry"] = True
    return body


class _SyncChatResp:
    def __init__(self, content: str):
        self.status_code = 200
        self._payload = {"choices": [{"message": {"content": content}}]}
        self.content = b"x"

    def json(self):
        return self._payload


def test_chat_completions_sync_logs_user_then_assistant_with_reply_to(monkeypatch):
    adapter, mm, _ = _make_adapter_with_seeded_db()

    async def _fake_post(url, json=None, **kwargs):
        return _SyncChatResp("here is my reply")

    monkeypatch.setattr(adapter._http, "post", _fake_post)

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


def test_chat_completions_sidecar_user_text_overrides_messages(monkeypatch):
    # jinja-hijack mode: post-repack messages array often has zero user-role
    # entries. The portal stamps raw input as derpr_user_text before the
    # textbox clears. Adapter must prefer this over scanning messages.
    adapter, mm, _ = _make_adapter_with_seeded_db()

    async def _fake_post(url, json=None, **kwargs):
        # Forwarded body must NOT contain derpr_user_text.
        assert "derpr_user_text" not in (json or {})
        return _SyncChatResp("ack")

    monkeypatch.setattr(adapter._http, "post", _fake_post)

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


def test_chat_completions_stream_logs_on_close_with_reply_to(monkeypatch):
    adapter, mm, _ = _make_adapter_with_seeded_db()

    class _StreamCtx:
        async def __aenter__(self):
            resp = MagicMock()

            async def _aiter_raw():
                yield b'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n'
                yield b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
                yield b"data: [DONE]\n\n"

            resp.aiter_raw = _aiter_raw
            return resp

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(adapter._http, "stream", lambda *a, **kw: _StreamCtx())

    with TestClient(adapter.app) as client:
        r = client.post("/chat/completions", json=_chat_body("hi", stream=True))
    assert r.status_code == 200

    rows = _fetch_portal_rows(mm, "test_persona")
    assert len(rows) == 2
    assert rows[1]["author_role"] == "assistant"
    assert rows[1]["content"] == "hello world"
    assert rows[1]["reply_to_id"] == rows[0]["interaction_id"]
    mm.close()


def test_chat_completions_retry_archives_and_updates_assistant(monkeypatch):
    adapter, mm, _ = _make_adapter_with_seeded_db()

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

    async def _fake_post(url, json=None, **kwargs):
        return _SyncChatResp("second attempt")

    monkeypatch.setattr(adapter._http, "post", _fake_post)

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
    adapter, mm, _ = _make_adapter_with_seeded_db()
    result = mm.handle_portal_retry("test_persona", "portal", "web_ui")
    assert result is None
    mm.close()


def test_chat_completions_stream_abort_flushes_partial(monkeypatch):
    adapter, mm, _ = _make_adapter_with_seeded_db()

    class _StreamCtx:
        async def __aenter__(self):
            resp = MagicMock()

            async def _aiter_raw():
                yield b'data: {"choices":[{"delta":{"content":"partial "}}]}\n\n'
                raise __import__("asyncio").CancelledError()

            resp.aiter_raw = _aiter_raw
            return resp

        async def __aexit__(self, *a):
            return False

    async def _fake_abort(url, json=None, **kwargs):
        resp = MagicMock()
        resp.content = b""
        resp.json = lambda: {}
        resp.status_code = 200
        return resp

    monkeypatch.setattr(adapter._http, "stream", lambda *a, **kw: _StreamCtx())
    monkeypatch.setattr(adapter._http, "post", _fake_abort)

    with TestClient(adapter.app) as client:
        try:
            with client.stream("POST", "/chat/completions", json=_chat_body("hi", stream=True)) as r:
                for _ in r.iter_raw():
                    pass
        except Exception:
            pass  # CancelledError propagates through test client — flush still happened

    rows = _fetch_portal_rows(mm, "test_persona")
    # User turn logged, assistant partial flushed on cancel
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
    assert mock_retrieve.call_args.kwargs.get("current_message") == "my query text"
    mm.close()


def test_ltm_block_empty_query_passes_none():
    mock_retrieve = AsyncMock(return_value=None)
    adapter, mm, persona = _make_adapter_with_seeded_db(retrieve_memory_block=mock_retrieve)
    with TestClient(adapter.app) as client:
        client.get("/api/v1/session/test_persona/ltm_block")
    mock_retrieve.assert_awaited_once()
    # empty query string → current_message=None (falsy branch in endpoint)
    kwargs = mock_retrieve.call_args.kwargs
    assert kwargs.get("current_message") is None
    mm.close()


# -------- Phase 2.3b: SSE derpr frame + version endpoints --------

def _make_stream_ctx(chunks):
    """Factory: context manager producing the given byte chunks from aiter_raw."""
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


def test_stream_emits_derpr_frame_before_done_with_assistant_id(monkeypatch):
    adapter, mm, _ = _make_adapter_with_seeded_db()

    chunks = [
        b'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    monkeypatch.setattr(adapter._http, "stream", lambda *a, **kw: _make_stream_ctx(chunks))

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


def test_stream_without_content_does_not_emit_derpr_frame(monkeypatch):
    # Empty upstream response → nothing to commit → no derpr frame.
    adapter, mm, _ = _make_adapter_with_seeded_db()

    chunks = [b"data: [DONE]\n\n"]
    monkeypatch.setattr(adapter._http, "stream", lambda *a, **kw: _make_stream_ctx(chunks))

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


def test_retry_retry_select_version_round_trip_via_endpoints(monkeypatch):
    """End-to-end: two sequential retries then restore original via endpoint.

    Initial assistant content = "v0". After retry #1 canonical = "v1",
    archives = [v0]. After retry #2 canonical = "v2", archives = [v0, v1].
    select_version(0) swaps archive[0] (v0) with canonical (v2):
    new canonical = "v0", archives = [v1, v2].
    """
    adapter, mm, _ = _make_adapter_with_seeded_db()

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

    responses = iter(["v1", "v2"])

    async def _fake_post(url, json=None, **kwargs):
        return _SyncChatResp(next(responses))

    monkeypatch.setattr(adapter._http, "post", _fake_post)

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


def test_patch_persona_updates_max_context_tokens():
    adapter, mm, persona = _make_adapter_with_seeded_db()
    with TestClient(adapter.app) as client:
        r = client.patch("/api/v1/persona/test_persona", json={"max_context_tokens": 16384})
    assert r.status_code == 200
    assert persona.get_max_context_tokens() == 16384
    mm.close()


def test_chat_completions_prunes_oversized_messages(monkeypatch):
    """Outbound prune drops oldest non-system messages to fit budget."""
    adapter, mm, persona = _make_adapter_with_seeded_db()
    # Tight budget: ctx 200 - response 100 → prompt budget 100 tokens (= 400 chars char/4).
    persona.set_response_token_limit(100)
    persona.set_max_context_tokens(200)

    forwarded = {}

    async def _fake_post(url, json=None, **kwargs):
        forwarded["body"] = json
        return _SyncChatResp("ack")

    monkeypatch.setattr(adapter._http, "post", _fake_post)

    big = "x" * 600  # 150 tokens each — 4 of these blow the budget.
    body = {
        "messages": [
            {"role": "system", "content": "sys note"},
            {"role": "user", "content": big},
            {"role": "assistant", "content": big},
            {"role": "user", "content": big},
            {"role": "assistant", "content": big},
            {"role": "user", "content": "latest"},
        ],
        "stream": False,
        "derpr_user_text": "latest",
    }
    with TestClient(adapter.app) as client:
        r = client.post("/chat/completions", json=body)
    assert r.status_code == 200

    fwd_msgs = forwarded["body"]["messages"]
    # System + last user always preserved.
    assert any(m["role"] == "system" and m["content"] == "sys note" for m in fwd_msgs)
    assert fwd_msgs[-1]["content"] == "latest"
    assert len(fwd_msgs) < len(body["messages"])
    mm.close()


def test_chat_completions_under_budget_no_prune(monkeypatch):
    adapter, mm, persona = _make_adapter_with_seeded_db()
    persona.set_response_token_limit(100)
    persona.set_max_context_tokens(131072)  # comfortable

    forwarded = {}

    async def _fake_post(url, json=None, **kwargs):
        forwarded["body"] = json
        return _SyncChatResp("ack")

    monkeypatch.setattr(adapter._http, "post", _fake_post)

    body = _chat_body("hi there")
    with TestClient(adapter.app) as client:
        r = client.post("/chat/completions", json=body)
    assert r.status_code == 200
    assert len(forwarded["body"]["messages"]) == len(body["messages"])
    mm.close()
