# tests/interfaces/test_kobold_adapter.py

"""Phase 2.1 route tests + regression passthrough test for KoboldAdapter.

Covers the adapter HTTP boundary, not just the exporter function:
  - /api/v1/session/{persona}/kobold_export returns a valid savefile
  - Unknown persona → 404
  - ?max_turns=N overrides the default, default uses persona.context_length
  - Passthrough regression: /api/extra/generate/stream forwards request body
    verbatim (minus DERPR-only envelope fields)
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from src.database.memory_manager import MemoryManager
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


def test_generate_stream_passthrough_forwards_body_verbatim(monkeypatch):
    adapter, mm, _ = _make_adapter_with_seeded_db()

    captured = {}

    class _FakeStreamCtx:
        def __init__(self, body):
            captured["body"] = body

        async def __aenter__(self):
            resp = MagicMock()

            async def _aiter_raw():
                yield b"event: message\ndata: {\"token\":\"ok\"}\n\n"

            resp.aiter_raw = _aiter_raw
            return resp

        async def __aexit__(self, *a):
            return False

    def _fake_stream(method, url, json=None, **kwargs):
        return _FakeStreamCtx(json)

    monkeypatch.setattr(adapter._http, "stream", _fake_stream)

    body = {
        "prompt": "hi there",
        "model": "test_persona",
        "history_override": True,
        "params": {"temperature": 0.5, "history_override": True},
        "max_length": 64,
    }
    with TestClient(adapter.app) as client:
        r = client.post("/api/extra/generate/stream", json=body)
    assert r.status_code == 200
    # DERPR-only fields stripped; kobold params survive intact.
    assert "model" not in captured["body"]
    assert "history_override" not in captured["body"]
    assert "history_override" not in captured["body"]["params"]
    assert captured["body"]["prompt"] == "hi there"
    assert captured["body"]["params"]["temperature"] == 0.5
    assert captured["body"]["max_length"] == 64
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
    from src.persona import MemoryMode
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
