# tests/security/test_portal_auth.py
"""DP-277 Phase 3/4 — portal control-plane authorization + network hardening.

The kobold engine adapter is the capability control plane. Every non-GET route
outside the data-plane allowlist requires the static operator token
(DERPR_CONTROL_TOKEN); reads and generation stay open. Deny-by-default: a new
mutating route is gated unless explicitly added to DATA_PLANE_POST_PATHS.

These tests use the REAL token check (no bypass fixture), unlike
test_kobold_engine_adapter.py.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from config import global_config
from memory.memory_manager import MemoryManager
from src.interfaces.kobold_engine_adapter import KoboldEngineAdapter as KoboldAdapter
from src.persona import Persona


TOKEN = "s3cr3t-operator-token"


def _make_adapter():
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    persona = Persona(persona_name="p", model_name="local", prompt="x")
    chat_system = SimpleNamespace(
        personas={"p": persona},
        memory_manager=mm,
        system_persona_names=set(),
        get_session_memory_block=AsyncMock(return_value=None),
        get_view_history=lambda *a, **k: ([], "global"),
        confirmations=SimpleNamespace(pending={}),
        bot_logic=SimpleNamespace(preprocess_message=AsyncMock(return_value={"response": "ok", "mutated": False})),
    )
    return KoboldAdapter(chat_system=chat_system), mm, persona


@pytest.fixture
def token_set(monkeypatch):
    monkeypatch.setattr(global_config, "DERPR_CONTROL_TOKEN", TOKEN, raising=False)


@pytest.fixture
def token_unset(monkeypatch):
    monkeypatch.setattr(global_config, "DERPR_CONTROL_TOKEN", "", raising=False)


def _auth(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Loopback + docs hardening (Phase 4)
# ---------------------------------------------------------------------------

def test_default_bind_is_loopback(monkeypatch):
    monkeypatch.setattr(global_config, "KOBOLD_ADAPTER_HOST", "127.0.0.1", raising=False)
    adapter, _, _ = _make_adapter()
    assert adapter.host == "127.0.0.1"


def test_openapi_docs_disabled():
    adapter, _, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/docs").status_code == 404


# ---------------------------------------------------------------------------
# Control-plane routes require the token (Phase 3)
# ---------------------------------------------------------------------------

CONTROL_ROUTES = [
    ("PATCH", "/api/v1/persona/p", {"prompt": "pwned"}),
    ("POST", "/api/v1/personas", {"name": "evil"}),
    ("POST", "/api/v1/persona/p/dev_command", {"command": "set tools all"}),
    ("POST", "/api/v1/persona/p/confirm", {"approved": True}),
    ("POST", "/api/v1/persona/p/reset", {}),
    ("PUT", "/api/v1/model", {"model": "p"}),
    ("PATCH", "/api/v1/interaction/1", {"content": "x"}),
    ("DELETE", "/api/v1/interaction/1", None),
    ("POST", "/api/v1/interaction/1/select_version/0", None),
]


@pytest.mark.parametrize("method,path,body", CONTROL_ROUTES)
def test_control_route_rejects_without_token(token_set, method, path, body):
    adapter, _, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        r = client.request(method, path, json=body)
    assert r.status_code == 401, f"{method} {path} must require the operator token"


@pytest.mark.parametrize("method,path,body", CONTROL_ROUTES)
def test_control_route_rejects_wrong_token(token_set, method, path, body):
    adapter, _, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        r = client.request(method, path, json=body, headers=_auth("wrong"))
    assert r.status_code == 401


def test_confirm_endpoint_gated(token_set):
    """The most direct 'seize the reins' path: /confirm releases a parked
    write. Unauthenticated → 401, parked write stays parked."""
    adapter, _, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        r = client.post("/api/v1/persona/p/confirm", json={"approved": True})
    assert r.status_code == 401


def test_valid_token_passes_gate(token_set):
    """A correct token clears the middleware (dev_command reaches bot_logic)."""
    adapter, _, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        r = client.post(
            "/api/v1/persona/p/dev_command",
            json={"command": "what prompt"},
            headers=_auth(),
        )
    assert r.status_code == 200
    adapter.chat_system.bot_logic.preprocess_message.assert_awaited()


def test_x_derpr_token_header_accepted(token_set):
    adapter, _, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        r = client.post(
            "/api/v1/persona/p/dev_command",
            json={"command": "what prompt"},
            headers={"X-Derpr-Token": TOKEN},
        )
    assert r.status_code == 200


def test_empty_configured_token_locks_control_plane(token_unset):
    """Fail closed: no token configured → every control route 401 even if the
    caller sends something."""
    adapter, _, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        r = client.patch("/api/v1/persona/p", json={"prompt": "x"}, headers=_auth("anything"))
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Data-plane routes stay open (no token)
# ---------------------------------------------------------------------------

def test_reads_stay_open_without_token(token_set):
    adapter, _, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        assert client.get("/api/v1/persona/p").status_code == 200
        assert client.get("/api/v1/model").status_code == 200


def test_data_plane_post_paths_not_gated():
    """Allowlist is exactly the generation/abort/tokencount surface — the
    drift guard: anything else non-GET is gated by construction."""
    assert KoboldAdapter.DATA_PLANE_POST_PATHS == frozenset({
        "/api/v1/generate",
        "/api/extra/generate/stream",
        "/api/extra/generate/check",
        "/api/v1/abort",
        "/api/extra/abort",
        "/api/extra/tokencount",
        "/chat/completions",
        "/v1/chat/completions",
    })


def test_abort_open_without_token(token_set):
    adapter, _, _ = _make_adapter()
    # abort proxies upstream — stub the client so we test the gate decision,
    # not a real kobold round-trip.
    adapter._http.post = AsyncMock(return_value=SimpleNamespace(
        status_code=200, content=b"{}", json=lambda: {}
    ))
    with TestClient(adapter.app) as client:
        r = client.post("/api/v1/abort")
    assert r.status_code != 401


# ---------------------------------------------------------------------------
# CORS: credentials must be off with wildcard origins
# ---------------------------------------------------------------------------

def test_cors_credentials_disabled():
    adapter, _, _ = _make_adapter()
    for m in adapter.app.user_middleware:
        if "CORSMiddleware" in str(m.cls):
            assert m.kwargs.get("allow_credentials") is False
            return
    pytest.fail("CORS middleware not found")
