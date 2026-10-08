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

def test_bind_host_from_config():
    """The adapter binds KOBOLD_ADAPTER_HOST. Default is 0.0.0.0 because the
    prod deploy reaches it via Docker port publishing + the Caddy TLS front
    (the token gate, not the bind, is the security boundary) — see
    global_config. A loopback deploy can still override the env var."""
    adapter, _, _ = _make_adapter()
    assert adapter.host == global_config.KOBOLD_ADAPTER_HOST


def test_bind_host_override(monkeypatch):
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
    """Allowlist is exactly the generation/abort/voice-STT surface — the drift
    guard: anything else non-GET is gated by construction."""
    assert KoboldAdapter.DATA_PLANE_POST_PATHS == frozenset({
        "/chat/completions",
        "/v1/chat/completions",
        "/voice/transcribe",
        "/voice/utterance",
    })


def test_voice_stt_uploads_not_gated(token_set):
    """The voice STT uploads (mounted on this app by register_voice_web) are
    data plane — the SPA mic and the /voice PTT page send no token. Routes
    aren't registered on this bare adapter, so the middleware must let the
    request through to a 404 rather than answer 401 itself."""
    adapter, _, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        assert client.post("/voice/transcribe", content=b"\x00\x00").status_code == 404
        assert client.post("/voice/utterance", content=b"\x00\x00").status_code == 404


# ---------------------------------------------------------------------------
# CORS: none — every browser client is same-origin (DP-365)
# ---------------------------------------------------------------------------

def test_no_cors_middleware():
    adapter, _, _ = _make_adapter()
    assert not any("CORSMiddleware" in str(m.cls) for m in adapter.app.user_middleware)


def test_foreign_origin_gets_no_cors_grant(token_set):
    """The GET reads are unauthenticated (DP-333), so the browser's
    same-origin policy is what stops a foreign page from reading them. That
    only holds while no Access-Control-Allow-Origin header is sent."""
    adapter, _, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/persona/p", headers={"Origin": "https://other.example"})
        pre = client.options(
            "/api/v1/persona/p",
            headers={"Origin": "https://other.example",
                     "Access-Control-Request-Method": "PATCH"},
        )
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers
    assert "access-control-allow-origin" not in pre.headers


def test_non_ascii_token_rejected_not_500(token_set):
    """A latin-1 (non-ASCII) supplied token must fail as 401, not crash
    compare_digest with a TypeError (str comparison requires ASCII)."""
    adapter, _, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        r = client.patch(
            "/api/v1/persona/p", json={"prompt": "x"},
            headers={"Authorization": "Bearer s\xe9cret".encode("latin-1")},
        )
    assert r.status_code == 401


def test_x_derpr_token_whitespace_stripped(token_set):
    """A pasted token with stray whitespace still authenticates via the
    X-Derpr-Token header (parity with the Bearer path, which strips)."""
    adapter, _, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        r = client.post(
            "/api/v1/persona/p/dev_command",
            json={"command": "what prompt"},
            headers={"X-Derpr-Token": f" {TOKEN} "},
        )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# DP-343/DP-355 — the node's job-completion ping is a third principal with NO
# credential
# ---------------------------------------------------------------------------
#
# A bash script on the model host, holding nothing. It is exempt from the
# operator gate (like the MCP bridge) but, unlike the bridge, runs no check of
# its own: the ping carries a job id and nothing else, and derpr re-reads the
# job over SSH before it reports anything, so there is nothing for a credential
# to protect. DP-343 shipped a MODEL_JOB_CALLBACK_TOKEN here anyway, argued for
# on a forged-ping threat that re-read already kills, and DP-355 deleted it.
#
# These tests therefore pin the two things that DO still hold: the exemption is
# scoped to this ONE path (it is not a way into the rest of the control plane),
# and it exists only while a handler is wired (an instance that never deployed
# the node half grows no unauthenticated surface).

JOB_PATH = global_config.MODEL_JOB_CALLBACK_PATH


def _make_adapter_with_job_handler(calls=None):
    adapter, mm, persona = _make_adapter()
    recorded = calls if calls is not None else []

    async def _handle(job_id):
        recorded.append(job_id)
        return {"status": "ok", "woke": True, "job_id": job_id}

    adapter._job_completion = _handle
    return adapter, recorded


def test_job_callback_route_absent_until_wired(token_set):
    """No handler wired = no exemption and no unauthenticated surface.

    The adapter is built without a job handler here, exactly as an instance that
    never deployed the node half. The path must not become a standing hole in
    the control plane just because the code that can answer it exists.
    """
    adapter, _, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        r = client.post(JOB_PATH, json={"job_id": "x-1"})
    # The operator middleware answers: without a wired handler nothing exempts
    # the path from it, and this caller holds no operator token.
    assert r.status_code == 401


def test_job_callback_accepts_an_unauthenticated_ping(token_set):
    """DP-355 — no credential, by design.

    The ping carries no facts, so a caller who reaches this route gains only an
    SSH status read of a job document that says what it already said.
    """
    adapter, calls = _make_adapter_with_job_handler()
    with TestClient(adapter.app) as client:
        r = client.post(JOB_PATH, json={"job_id": "newmodel-abc123"})
    assert r.status_code == 200
    assert calls == ["newmodel-abc123"]


def test_job_callback_ignores_any_bearer_it_is_given(token_set):
    """A stray or wrong Authorization header changes nothing, in either direction.

    Pinned because the deleted check made a wrong token a 401 and the operator
    token a 401 too. Nothing branches on the header here any more.
    """
    adapter, calls = _make_adapter_with_job_handler()
    with TestClient(adapter.app) as client:
        wrong = client.post(JOB_PATH, json={"job_id": "j-1"},
                            headers=_auth("wrong"))
        operator = client.post(JOB_PATH, json={"job_id": "j-2"}, headers=_auth())
    assert (wrong.status_code, operator.status_code) == (200, 200)
    assert calls == ["j-1", "j-2"]


def test_job_callback_exemption_does_not_leak_to_other_routes(token_set):
    """The carve-out is one path, not a hole in the operator gate.

    This is the property the deleted node token was mostly credited with, and it
    was never the token's doing: the exemption is keyed on the path, so every
    other control-plane POST still answers 401 to the same unauthenticated
    caller this route accepts.
    """
    adapter, calls = _make_adapter_with_job_handler()
    with TestClient(adapter.app) as client:
        allowed = client.post(JOB_PATH, json={"job_id": "j-1"})
        confirm = client.post("/api/v1/persona/p/confirm", json={"approved": True})
        reset = client.post("/api/v1/persona/p/reset")
    assert allowed.status_code == 200
    assert confirm.status_code == 401
    assert reset.status_code == 401
    assert calls == ["j-1"]


def test_job_callback_works_without_an_operator_token_configured(token_unset):
    """The node must be able to ring the doorbell on an instance whose control
    plane is locked — this route does not depend on the operator credential."""
    adapter, calls = _make_adapter_with_job_handler()
    with TestClient(adapter.app) as client:
        r = client.post(JOB_PATH, json={"job_id": "j-1"})
    assert r.status_code == 200
    assert calls == ["j-1"]


@pytest.mark.parametrize("body", [None, {}, {"job_id": ""}, {"job_id": 7},
                                  {"job_id": "j" * 200}])
def test_job_callback_rejects_a_malformed_body(token_set, body):
    adapter, calls = _make_adapter_with_job_handler()
    with TestClient(adapter.app) as client:
        r = client.post(JOB_PATH, json=body)
    assert r.status_code == 400
    assert calls == []


def test_job_callback_forwards_only_the_job_id(token_set):
    """Anything else in the body is ignored by construction.

    The handler's whole contract is that facts come from the SSH status read, so
    a POST claiming `state: done` for a job that failed must reach the bridge as
    a job id and nothing more. That is what makes the route safe to leave open.
    """
    adapter, calls = _make_adapter_with_job_handler()
    with TestClient(adapter.app) as client:
        r = client.post(
            JOB_PATH,
            json={"job_id": " j-2 ", "state": "done", "name": "evil"},
        )
    assert r.status_code == 200
    assert calls == ["j-2"]
