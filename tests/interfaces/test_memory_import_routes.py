"""DP-292 memory import-panel route tests.

HTTP-boundary coverage for `/api/v1/memory/*`: query-param plumbing, the
`type`→`op_type` mapping, path-converter document ids, upload md/txt gating,
URL/path ingest, and backend-error → HTTP-status mapping (SQLite 501,
HindsightAPIError passthrough).

The backend is stubbed with AsyncMocks so these assert route wiring, not
Hindsight behavior (that lives in tests/memory/test_hindsight_backend.py).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from config import global_config
from src.interfaces.kobold_engine_adapter import KoboldEngineAdapter as KoboldAdapter
from src.memory.backend.hindsight import HindsightAPIError
from tests.interfaces.test_kobold_engine_adapter import _make_adapter_with_seeded_db


@pytest.fixture(autouse=True)
def _bypass_control_plane_auth(monkeypatch):
    # Mutations (DELETE/POST) pass the operator gate — the gate itself is
    # covered in tests/security/test_portal_auth.py.
    monkeypatch.setattr(KoboldAdapter, "_valid_control_token", lambda self, tok: True)
    monkeypatch.setattr(global_config, "DERPR_CONTROL_TOKEN", "test-token", raising=False)


def _adapter_with_backend():
    """Adapter whose memory_backend is an AsyncMock stub."""
    adapter, mm, _ = _make_adapter_with_seeded_db()
    backend = MagicMock()
    backend.list_banks = AsyncMock(return_value=[{"bank_id": "alice", "fact_count": 3}])
    backend.list_documents = AsyncMock(return_value={"items": [], "total": 0})
    backend.list_operations = AsyncMock(return_value={"bank_id": "alice", "operations": []})
    backend.delete_document = AsyncMock(return_value={"success": True, "document_id": "d"})
    backend.retain_document = AsyncMock(return_value=None)
    adapter.chat_system.memory_backend = backend
    return adapter, mm, backend


# ---------- reads ----------

def test_list_banks_returns_backend_list():
    adapter, mm, backend = _adapter_with_backend()
    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/memory/banks")
    assert r.status_code == 200
    assert r.json() == [{"bank_id": "alice", "fact_count": 3}]
    mm.close()


def test_list_documents_splits_tags_and_forwards_params():
    adapter, mm, backend = _adapter_with_backend()
    with TestClient(adapter.app) as client:
        r = client.get(
            "/api/v1/memory/banks/alice/documents",
            params={"q": "phish", "tags": "ingest,notes", "limit": 5, "offset": 10},
        )
    assert r.status_code == 200
    backend.list_documents.assert_awaited_once_with(
        "alice", q="phish", tags=["ingest", "notes"], limit=5, offset=10,
    )
    mm.close()


def test_list_documents_no_tags_forwards_none():
    adapter, mm, backend = _adapter_with_backend()
    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/memory/banks/alice/documents")
    assert r.status_code == 200
    _, kwargs = backend.list_documents.call_args
    assert kwargs["tags"] is None
    mm.close()


def test_list_operations_maps_type_to_op_type():
    adapter, mm, backend = _adapter_with_backend()
    with TestClient(adapter.app) as client:
        r = client.get(
            "/api/v1/memory/banks/alice/operations",
            params={"status": "pending", "type": "retain"},
        )
    assert r.status_code == 200
    backend.list_operations.assert_awaited_once_with(
        "alice", status="pending", op_type="retain", limit=None, offset=None,
    )
    mm.close()


def test_delete_document_path_converter_captures_slashes():
    adapter, mm, backend = _adapter_with_backend()
    with TestClient(adapter.app) as client:
        r = client.delete("/api/v1/memory/banks/alice/documents/sub/dir/file.md")
    assert r.status_code == 200
    backend.delete_document.assert_awaited_once_with("alice", "sub/dir/file.md")
    mm.close()


# ---------- upload ----------

def test_upload_md_accepted_and_retained():
    adapter, mm, backend = _adapter_with_backend()
    with TestClient(adapter.app) as client:
        r = client.post(
            "/api/v1/memory/banks/alice/upload",
            files={"files": ("note.md", b"# hello", "text/markdown")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["status"] == "accepted"
    assert body["results"][0]["document_id"] == "note.md"
    _, kwargs = backend.retain_document.call_args
    assert kwargs["document_id"] == "note.md"
    assert kwargs["content"] == "# hello"
    assert kwargs["metadata"]["untrusted"] == "false"
    mm.close()


def test_upload_rejects_non_md_txt():
    adapter, mm, backend = _adapter_with_backend()
    with TestClient(adapter.app) as client:
        r = client.post(
            "/api/v1/memory/banks/alice/upload",
            files={"files": ("evil.pdf", b"%PDF-1.4", "application/pdf")},
        )
    assert r.status_code == 200
    assert r.json()["results"][0]["status"] == "rejected"
    backend.retain_document.assert_not_awaited()
    mm.close()


def test_upload_rejects_non_utf8():
    adapter, mm, backend = _adapter_with_backend()
    with TestClient(adapter.app) as client:
        r = client.post(
            "/api/v1/memory/banks/alice/upload",
            files={"files": ("bad.txt", b"\xff\xfe\x00bad", "text/plain")},
        )
    assert r.status_code == 200
    assert r.json()["results"][0]["reason"] == "not valid utf-8"
    backend.retain_document.assert_not_awaited()
    mm.close()


# ---------- URL ingest ----------

def test_ingest_url_requires_url():
    adapter, mm, backend = _adapter_with_backend()
    with TestClient(adapter.app) as client:
        r = client.post("/api/v1/memory/banks/alice/ingest_url", json={})
    assert r.status_code == 400
    mm.close()


def test_ingest_url_fetches_and_retains():
    adapter, mm, backend = _adapter_with_backend()
    fake_resp = MagicMock()
    fake_resp.text = "fetched body"
    fake_resp.raise_for_status = MagicMock()
    with patch.object(adapter._http, "get", AsyncMock(return_value=fake_resp)):
        with TestClient(adapter.app) as client:
            r = client.post(
                "/api/v1/memory/banks/alice/ingest_url",
                json={"url": "https://example.com/doc.md"},
            )
    assert r.status_code == 200
    assert r.json()["status"] == "accepted"
    _, kwargs = backend.retain_document.call_args
    assert kwargs["document_id"] == "https://example.com/doc.md"
    assert kwargs["content"] == "fetched body"
    mm.close()


# ---------- path ingest ----------

def test_ingest_path_requires_path():
    adapter, mm, backend = _adapter_with_backend()
    with TestClient(adapter.app) as client:
        r = client.post("/api/v1/memory/banks/alice/ingest_path", json={})
    assert r.status_code == 400
    mm.close()


def test_ingest_path_delegates_to_ingest_root():
    adapter, mm, backend = _adapter_with_backend()
    canned = {"status": "ok", "bank": "alice", "ingested": 2, "skipped": 0, "failed": 0}
    with patch("src.tools.ingest_path.IngestPathHandler.ingest_root",
               AsyncMock(return_value=canned)) as m:
        with TestClient(adapter.app) as client:
            r = client.post(
                "/api/v1/memory/banks/alice/ingest_path",
                json={"path": "/notes", "glob": "**/*.md", "force": True},
            )
    assert r.status_code == 200
    assert r.json() == canned
    # bank_id + glob + force forwarded; root is a resolved Path.
    args, _ = m.call_args
    assert args[0] == "alice" and args[2] == "**/*.md" and args[3] is True
    mm.close()


# ---------- backend-error → HTTP-status mapping ----------

def test_sqlite_backend_maps_to_501():
    adapter, mm, backend = _adapter_with_backend()
    backend.list_banks = AsyncMock(side_effect=NotImplementedError("no list_banks"))
    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/memory/banks")
    assert r.status_code == 501
    assert "hindsight" in r.json()["error"].lower()
    mm.close()


def test_hindsight_api_error_passthrough():
    adapter, mm, backend = _adapter_with_backend()
    backend.list_documents = AsyncMock(side_effect=HindsightAPIError(404, "bank not found"))
    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/memory/banks/ghost/documents")
    assert r.status_code == 404
    assert r.json()["error"] == "bank not found"
    mm.close()
