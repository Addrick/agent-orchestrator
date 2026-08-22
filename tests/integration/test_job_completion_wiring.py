"""DP-343 — the node ping wired end to end, plus the composition root that wires it.

The two halves either side of this feature are covered elsewhere
(tests/services/test_job_completion_ping.py is the bash that sends the ping;
tests/huggingface/test_job_completion.py is the bridge). What neither can catch
is the wiring going missing: an integration built without a ChatSystem, or an
adapter built without the handler, produces a feature that is fully implemented
and never runs — the DP-332 failure mode, where every unit test passed while the
deployed thing was dead.

So this file drives the real path — HTTP POST → adapter route → bridge →
handler → (fake) SSH → persona turn — and then pins main.py's own wiring.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from config import global_config
from src.huggingface import HuggingFaceIntegration
from src.interfaces.kobold_engine_adapter import create_kobold_engine_adapter
from src.proxmox.ssh import SSHResult

pytestmark = pytest.mark.integration

NODE_TOKEN = "n0de-callback-token"
JOB = {
    "job_id": "newmodel-abc123",
    "state": "done",
    "step": "installed",
    "reason": "",
    "repo": "bartowski/Qwen3-32B-GGUF",
    "file": "Qwen3-32B-Q4_K_M.gguf",
    "name": "newmodel",
    "unit": "koboldcpp-newmodel.service",
    "size_bytes": 20_000_000_000,
    "downloaded_bytes": 20_000_000_000,
    "contextsize": 8192,
    "sha256": "a" * 64,
}


class _FakeRunner:
    """The SSH boundary: records argv, answers with the node's job document."""

    def __init__(self, payload=None):
        self.calls = []
        self._payload = json.dumps(payload if payload is not None else JOB)

    async def run(self, argv):
        self.calls.append(list(argv))
        return SSHResult(0, self._payload, "")


@pytest.fixture
def wake_configured(monkeypatch):
    monkeypatch.setattr(global_config, "HF_TOOLS_ENABLED", True, raising=False)
    monkeypatch.setattr(global_config, "MODEL_JOB_CALLBACK_TOKEN", NODE_TOKEN,
                        raising=False)
    monkeypatch.setattr(global_config, "MODEL_JOB_WAKE_PERSONA", "hypr", raising=False)
    monkeypatch.setattr(global_config, "MODEL_JOB_WAKE_CHANNEL", "infra", raising=False)
    monkeypatch.setattr(global_config, "MODEL_JOB_WAKE_USER", "42", raising=False)
    monkeypatch.setattr(global_config, "MODEL_JOB_ALERT_CHANNEL_ID", "777",
                        raising=False)
    monkeypatch.setattr(global_config, "DERPR_CONTROL_TOKEN", "operator-token",
                        raising=False)


def _wired(runner=None):
    """A HuggingFaceIntegration + adapter wired the way main() wires them."""
    chat_system = SimpleNamespace(
        generate_response=AsyncMock(return_value=("newmodel landed", "chat", 1, 2)),
        personas={},
        visible_personas={},
        system_persona_names=set(),
        memory_manager=None,
        get_view_history=lambda *a, **k: ([], "global"),
        confirmations=SimpleNamespace(pending={}),
    )
    notifier = SimpleNamespace(send=AsyncMock(return_value=True))
    integration = HuggingFaceIntegration(
        client=SimpleNamespace(),
        runner=runner or _FakeRunner(),
        chat_system=chat_system,
        notification_router=notifier,
    )
    assert integration.completion_bridge is not None
    adapter = create_kobold_engine_adapter(
        chat_system, job_completion=integration.completion_bridge.handle)
    return adapter, chat_system, notifier


def test_ping_reaches_the_persona_through_the_real_route(wake_configured):
    """One POST from the node; a persona turn and a Discord post come out."""
    runner = _FakeRunner()
    adapter, chat_system, notifier = _wired(runner)

    with TestClient(adapter.app) as client:
        r = client.post(
            global_config.MODEL_JOB_CALLBACK_PATH,
            json={"job_id": "newmodel-abc123"},
            headers={"Authorization": f"Bearer {NODE_TOKEN}"},
        )

    assert r.status_code == 200
    assert r.json()["woke"] is True

    # The facts came from the node over SSH, not from the POST body.
    assert runner.calls == [["/usr/local/sbin/derpr-model-install", "status",
                            "newmodel-abc123"]]
    msg = chat_system.generate_response.await_args.kwargs["message"]
    assert "koboldcpp-newmodel.service" in msg
    notifier.send.assert_awaited_once()
    assert notifier.send.await_args.kwargs["body"] == "newmodel landed"


def test_integration_without_a_chat_system_has_no_bridge():
    """The tool half must keep working standalone.

    Nothing to wake means no bridge — and then the adapter installs no auth
    exemption for the callback path at all, so an instance that never deployed
    the node half grows no new surface.
    """
    integration = HuggingFaceIntegration()
    assert integration.completion_bridge is None


def test_unwired_adapter_leaves_the_callback_path_gated(wake_configured):
    chat_system = SimpleNamespace(
        personas={}, visible_personas={}, system_persona_names=set(),
        memory_manager=None, get_view_history=lambda *a, **k: ([], "global"),
        confirmations=SimpleNamespace(pending={}),
    )
    adapter = create_kobold_engine_adapter(chat_system)

    with TestClient(adapter.app) as client:
        r = client.post(
            global_config.MODEL_JOB_CALLBACK_PATH,
            json={"job_id": "x"},
            headers={"Authorization": f"Bearer {NODE_TOKEN}"},
        )

    # The operator gate answers, because no exemption was installed.
    assert r.status_code == 401


def test_main_wires_the_bridge_into_the_adapter():
    """Pin the composition root itself.

    Every other test here builds the wiring the way main() does; this is the one
    that fails if main() stops doing it. A feature that is only reachable when a
    composition root passes two arguments needs the passing of those arguments
    under test — `config/optional_personas` and the DP-332 wrapper both shipped
    'working' code that nothing deployed reached.
    """
    src = (Path(__file__).resolve().parents[2] / "src" / "main.py").read_text(
        encoding="utf-8")
    assert "HuggingFaceIntegration(" in src
    assert "chat_system=bot" in src, \
        "main() must give HuggingFaceIntegration a ChatSystem or no ping can wake anyone"
    assert "notification_router=notification_router" in src
    assert "completion_bridge.handle" in src, \
        "main() must hand the bridge's handler to the adapter route"
    assert "job_completion=job_completion" in src, \
        "_register_interfaces must forward the handler to create_kobold_engine_adapter"
