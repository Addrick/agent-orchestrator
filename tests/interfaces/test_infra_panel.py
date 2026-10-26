"""DP-439: portal INFRA panel routes.

The panel is a second caller of the registered proxmox tools, so these tests
drive the real ToolManager with fake handlers registered under the real tool
names, and the real control-plane token check (no bypass fixture).
"""

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from config import global_config
from memory.memory_manager import MemoryManager
from src.interfaces import infra_panel
from src.interfaces.kobold_engine_adapter import KoboldEngineAdapter
from src.persona import Persona
from src.tools.tool_manager import ToolManager

TOKEN = "infra-operator-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class FakeProxmox:
    """Fake handlers under the real proxmox tool names; records every call."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def register(self, tm: ToolManager) -> None:
        for name in sorted(infra_panel.INFRA_TOOLS):
            tm.register(name, self._make(name))
        # A non-infra tool on the same manager: the panel must not reach it.
        tm.register("web_search", self._make("web_search"))

    def _make(self, name: str) -> Any:
        async def handler(**kwargs: Any) -> Dict[str, Any]:
            self.calls.append((name, kwargs))
            if name == "pve_status":
                return {"status": "ok", "uptime": "up 3 days", "guests": [
                    {"vmid": "100", "name": "derpr", "kind": "ct", "status": "running", "lock": ""},
                ]}
            if name == "gpu_status":
                return {"status": "ok", "gpus": [
                    {"card": "card1", "vram_total_mib": 32624, "vram_used_mib": 28601, "vram_free_mib": 4022},
                ]}
            if name == "list_models":
                raise RuntimeError("ssh exploded")
            return {"status": "ok", "target": {"vmid": kwargs.get("vmid"), "kind": kwargs.get("kind")}}
        return handler


@pytest.fixture
def token_set(monkeypatch):
    monkeypatch.setattr(global_config, "DERPR_CONTROL_TOKEN", TOKEN, raising=False)


def _make_adapter():
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    tm = ToolManager()
    fake = FakeProxmox()
    fake.register(tm)
    chat_system = SimpleNamespace(
        personas={"p": Persona(persona_name="p", model_name="local", prompt="x")},
        memory_manager=mm,
        system_persona_names=set(),
        tool_manager=tm,
        get_session_memory_block=AsyncMock(return_value=None),
        get_view_history=lambda *a, **k: ([], "global"),
        confirmations=SimpleNamespace(pending={}),
        bot_logic=SimpleNamespace(preprocess_message=AsyncMock()),
    )
    return KoboldEngineAdapter(chat_system=chat_system), fake, mm


def test_status_reads_each_section_independently(token_set):
    adapter, fake, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        r = client.get("/api/v1/infra/status")
    assert r.status_code == 200
    body = r.json()
    assert body["pve"]["guests"][0]["name"] == "derpr"
    assert body["gpu"]["gpus"][0]["vram_free_mib"] == 4022
    # One failing read does not sink the other two.
    assert body["models"]["status"] == "error"
    assert "ssh exploded" in body["models"]["message"]
    assert sorted(n for n, _ in fake.calls) == [
        "gpu_status", "list_models", "pve_status", "sandbox_net_status"]
    assert body["net"]["status"] == "ok"


def test_status_reports_enabled_flag(token_set, monkeypatch):
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", False, raising=False)
    adapter, _, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        assert client.get("/api/v1/infra/status").json()["enabled"] is False


def test_action_requires_operator_token(token_set):
    adapter, fake, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        r = client.post("/api/v1/infra/action", json={"action": "reboot_node"})
        assert r.status_code == 401
        r = client.post("/api/v1/infra/action", json={"action": "reboot_node"},
                        headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401
    assert fake.calls == []


def test_action_runs_handler_and_audits(token_set):
    adapter, fake, mm = _make_adapter()
    with TestClient(adapter.app) as client:
        r = client.post("/api/v1/infra/action", headers=AUTH,
                        json={"action": "stop_guest", "vmid": 102, "kind": "ct"})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert fake.calls == [("stop_guest", {"vmid": "102", "kind": "ct"})]
    rows = mm._get_connection().execute(
        "SELECT event_type, new_state FROM Audit_Log WHERE event_type = 'infra_action'"
    ).fetchall()
    assert len(rows) == 1
    assert "stop_guest" in rows[0][1]


def test_a_net_switch_passes_confirm_through_and_audits_the_box(token_set):
    """DP-465: `confirm` is the one boolean the panel sends; it reaches the
    handler as a bool, and the audit row names the box."""
    adapter, fake, mm = _make_adapter()
    with TestClient(adapter.app) as client:
        r = client.post("/api/v1/infra/action", headers=AUTH, json={
            "action": "set_sandbox_net", "box": "stagr", "zone": "wan", "state": "off", "confirm": True})
    assert r.status_code == 200
    assert fake.calls == [("set_sandbox_net", {"box": "stagr", "zone": "wan", "state": "off", "confirm": True})]
    rows = mm._get_connection().execute(
        "SELECT metadata FROM Audit_Log WHERE event_type = 'infra_action'").fetchall()
    assert rows and '"persona": "stagr"' in rows[0][0]


@pytest.mark.parametrize("body,fragment", [
    ({"action": "web_search", "query": "x"}, "action must be one of"),
    ({"action": "stop_guest", "vmid": "1", "kind": "ct", "argv": "rm"}, "does not accept"),
    ({"action": "set_active_model"}, "requires name"),
    ({"action": "start_guest"}, "requires vmid"),
    ({"action": "set_active_model", "name": ["a"]}, "must be a string"),
    ({"action": "set_sandbox_net", "box": "codr", "zone": "lan"}, "requires box, zone and state"),
    ({"action": "set_sandbox_net", "box": "codr", "zone": "lan", "state": True}, "must be a string"),
    (["reboot_node"], "JSON object"),
])
def test_action_rejects_bad_bodies_without_executing(token_set, body, fragment):
    adapter, fake, _ = _make_adapter()
    with TestClient(adapter.app) as client:
        r = client.post("/api/v1/infra/action", headers=AUTH, json=body)
    assert r.status_code == 400
    assert fragment in r.json()["error"]
    assert fake.calls == []


@pytest.mark.asyncio
async def test_unregistered_proxmox_is_an_error_not_a_crash():
    """An instance without ProxmoxIntegration registered: every section
    reports an error rather than raising."""
    out = await infra_panel.read_status(ToolManager())
    assert all(out[k]["status"] == "error" for k in ("pve", "gpu", "models", "net"))
