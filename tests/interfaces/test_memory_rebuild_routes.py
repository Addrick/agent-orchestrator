"""DP-506: the Imports panel's purge-and-rebuild routes.

The destructive POST must need the operator token *and* the typed phrase, run
only on Hindsight, and run one job at a time. The rebuild itself is
`ChatSystem.rebuild_persona_memory` (tests/integration/test_memory_rebuild.py);
here it is a fake that records who it was asked to rebuild.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from config import global_config
from memory.memory_manager import MemoryManager
from src.interfaces.kobold_engine_adapter import KoboldEngineAdapter
from src.interfaces.memory_rebuild import MemoryRebuildJob
from src.memory.backend.hindsight import HindsightBackend
from src.persona import Persona
from src.tools.tool_manager import ToolManager

TOKEN = "rebuild-operator-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
GO = {"confirm": "PURGE"}


class _Hindsight(HindsightBackend):
    def __init__(self) -> None:  # no client, no sibling stores
        pass


@pytest.fixture(autouse=True)
def token_set(monkeypatch):
    monkeypatch.setattr(global_config, "DERPR_CONTROL_TOKEN", TOKEN, raising=False)


def _make_adapter(backend: Any = None):
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    mm.backend = backend if backend is not None else _Hindsight()
    calls: List[str] = []

    async def rebuild(name: str, progress=None) -> Dict[str, int]:
        calls.append(name)
        return {"blocks": 2, "queued": 0, "adopted": 1, "pending": 3, "total": 2}

    chat_system = SimpleNamespace(
        personas={"p": Persona(persona_name="p", model_name="local", prompt="x")},
        memory_manager=mm,
        memory_backend=mm.backend,
        system_persona_names=set(),
        tool_manager=ToolManager(),
        get_session_memory_block=AsyncMock(return_value=None),
        get_view_history=lambda *a, **k: ([], "global"),
        confirmations=SimpleNamespace(pending={}),
        bot_logic=SimpleNamespace(preprocess_message=AsyncMock()),
        memory_persona_names=lambda: ["p", "q"],
        rebuild_persona_memory=rebuild,
    )
    return KoboldEngineAdapter(chat_system=chat_system), calls, mm


def _wait_finished(client: TestClient) -> Dict[str, Any]:
    for _ in range(100):
        body = client.get("/api/v1/memory/rebuild").json()
        if body["status"] == "finished":
            return body
    raise AssertionError("rebuild never finished")


def test_status_is_open_and_lists_targets():
    adapter, calls, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        body = client.get("/api/v1/memory/rebuild").json()
    assert body["status"] == "idle" and body["targets"] == ["p", "q"]
    assert calls == []


def test_start_needs_the_operator_token():
    adapter, calls, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        assert client.post("/api/v1/memory/rebuild", json=GO).status_code == 401
    assert calls == []


@pytest.mark.parametrize("body", [{}, {"confirm": "purge"}, {"confirm": True}, ["PURGE"]])
def test_start_needs_the_exact_phrase(body):
    adapter, calls, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        r = client.post("/api/v1/memory/rebuild", json=body, headers=AUTH)
    assert r.status_code == 400 and "PURGE" in r.json()["error"]
    assert calls == []


def test_start_refuses_a_non_hindsight_backend():
    adapter, calls, _ = _make_adapter(backend=SimpleNamespace())
    with TestClient(adapter.app) as client:
        r = client.post("/api/v1/memory/rebuild", json=GO, headers=AUTH)
    assert r.status_code == 501
    assert calls == []


def test_confirmed_start_rebuilds_every_target_and_audits():
    adapter, calls, mm = _make_adapter()
    with TestClient(adapter.app) as client:
        r = client.post("/api/v1/memory/rebuild", json=GO, headers=AUTH)
        assert r.status_code == 202
        body = _wait_finished(client)
    assert calls == ["p", "q"]
    assert [p["state"] for p in body["personas"]] == ["done", "done"]
    assert body["personas"][0]["counts"]["adopted"] == 1
    rows = mm._get_connection().execute(
        "SELECT event_type FROM Audit_Log WHERE event_type = 'memory_rebuild'").fetchall()
    assert len(rows) == 1


def test_a_failing_persona_stops_the_run_before_the_next_bank_is_deleted():
    adapter, calls, _ = _make_adapter()

    async def rebuild(name: str, progress=None) -> Dict[str, int]:
        calls.append(name)
        if name == "p":
            raise RuntimeError("Hindsight stopped answering")
        return {}

    adapter.chat_system.rebuild_persona_memory = rebuild
    with TestClient(adapter.app) as client:
        client.post("/api/v1/memory/rebuild", json=GO, headers=AUTH)
        body = _wait_finished(client)
    assert calls == ["p"]
    assert body["personas"][0]["state"] == "error"
    assert "stopped answering" in body["personas"][0]["error"]
    assert body["personas"][1]["state"] == "skipped"


@pytest.mark.asyncio
async def test_a_second_start_is_refused_while_one_runs():
    release = asyncio.Event()

    async def rebuild(name: str, progress=None) -> Dict[str, int]:
        await release.wait()
        return {}

    job = MemoryRebuildJob()
    job.start(rebuild, ["p"])
    await asyncio.sleep(0)
    assert job.running and job.snapshot()["personas"][0]["state"] == "running"
    adapter, calls, _ = _make_adapter()
    adapter._memory_rebuild = job
    with TestClient(adapter.app) as client:
        r = client.post("/api/v1/memory/rebuild", json=GO, headers=AUTH)
    assert r.status_code == 409
    release.set()
    await asyncio.sleep(0.01)
    assert job.snapshot()["status"] == "finished"
