# tests/integration/test_node_transport_gate.py
#
# DP-348 — the pve node has exactly ONE door, and both handlers go through it.
#
# `PVE_TOOLS_ENABLED` and the SSH in-flight cap are properties of the key and of
# the node's sshd, not of any one handler. They used to live inside
# `ProxmoxToolHandler._run`; when the HuggingFace tools grew their own copy of
# that method (DP-265) they inherited neither, so `install_model` shelled out to
# the box while every proxmox tool correctly reported itself disabled, and HF
# connections were invisible to the cap that keeps the node under sshd's
# MaxStartups (10:30:100).
#
# Nothing in the unit suites could see it: each handler's tests asserted that
# handler's `_run` behaved the way that handler's tests expected. This file
# asserts the *seam* instead — the same question asked of every handler that
# reaches the node — in the spirit of test_startup_wiring.py, which asserts a
# registration contract no single component's own tests can see either.

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, List, Sequence

import pytest

from config import global_config
from src.huggingface import handler as hf_handler
from src.huggingface.handler import HuggingFaceToolHandler
from src.proxmox import handler as proxmox_handler
from src.proxmox.handler import ProxmoxToolHandler
from src.proxmox.ssh import SSHResult, inflight_semaphore

pytestmark = pytest.mark.integration

SHA = "b" * 64
SIZE = 8_000_000_000


class RecordingRunner:
    """Stand-in for SSHRunner that records every argv crossing the boundary.

    An empty ``calls`` list is the assertion that matters here: a refusal that
    still opened the connection is not a refusal.
    """

    def __init__(self, stdout: str = "{}") -> None:
        self.calls: List[List[str]] = []
        self._stdout = stdout

    async def run(self, argv: Sequence[str]) -> SSHResult:
        self.calls.append(list(argv))
        return SSHResult(0, self._stdout, "")


class FailingRunner(RecordingRunner):
    """Records the argv, then reports a non-zero remote exit."""

    async def run(self, argv: Sequence[str]) -> SSHResult:
        self.calls.append(list(argv))
        return SSHResult(2, "", "boom")


class StubHF:
    """Enough of HFClient for install_model to reach the transport."""

    async def find_gguf_file(self, repo: str, file_path: str) -> Any:
        from src.huggingface.client import HFFile
        return HFFile(path=file_path, size_bytes=SIZE, sha256=SHA)


def _hf(runner: RecordingRunner) -> HuggingFaceToolHandler:
    return HuggingFaceToolHandler(StubHF(), runner)  # type: ignore[arg-type]


def _pve(runner: RecordingRunner) -> ProxmoxToolHandler:
    return ProxmoxToolHandler(runner)  # type: ignore[arg-type]


@pytest.fixture
def hf_feature_on(monkeypatch):
    """The HF *feature* switch on, so only the transport switch is in play."""
    monkeypatch.setattr(global_config, "HF_TOOLS_ENABLED", True)


# -- the assertion whose absence let this ship -------------------------------

@pytest.mark.asyncio
async def test_install_model_refuses_when_pve_tools_disabled(hf_feature_on, monkeypatch):
    """install_model must not reach the node with the node's own switch off.

    This is the whole ticket. ``HF_TOOLS_ENABLED`` being true says the operator
    wants the four HF tools; it says nothing about whether this instance may
    drive the pve node, which is what ``PVE_TOOLS_ENABLED`` and the mounted key
    answer. Before DP-348 this call downloaded a multi-GB file onto the box and
    templated a systemd unit on it while ``pve_status`` on the same persona, in
    the same turn, reported "Proxmox tools are disabled".
    """
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", False)
    runner = RecordingRunner()

    res = await _hf(runner)._install_model(
        "owner/model-GGUF", "model-Q6_K.gguf", "newmodel", kv_precision="q8"
    )

    assert res["status"] == "error"
    assert "PVE_TOOLS_ENABLED" in res["message"]
    assert runner.calls == [], "refused, but still opened an SSH connection"


@pytest.mark.asyncio
async def test_install_status_refuses_when_pve_tools_disabled(hf_feature_on, monkeypatch):
    """The read half of the HF/node pair is gated identically.

    Polling is the cheaper call, which is exactly why it would be the one left
    ungated: it looks harmless. It is still the same key against the same box.
    """
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", False)
    runner = RecordingRunner()

    res = await _hf(runner).job_status("newmodel-abc123")

    assert res["status"] == "error"
    assert "PVE_TOOLS_ENABLED" in res["message"]
    assert runner.calls == []


@pytest.mark.asyncio
async def test_both_handlers_report_the_same_disabled_state(hf_feature_on, monkeypatch):
    """One node, one answer about whether it is reachable.

    The defect's visible signature was disagreement: two tools on the *same*
    persona giving opposite accounts of the same box.
    """
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", False)
    hf_runner, pve_runner = RecordingRunner(), RecordingRunner()

    hf_res = await _hf(hf_runner).job_status("newmodel-abc123")
    pve_res = await _pve(pve_runner)._pve_status()

    assert hf_res["status"] == pve_res["status"] == "error"
    assert "PVE_TOOLS_ENABLED" in hf_res["message"]
    assert "PVE_TOOLS_ENABLED" in pve_res["message"]
    assert hf_runner.calls == pve_runner.calls == []


# -- the cap is one cap ------------------------------------------------------

@pytest.mark.asyncio
async def test_the_inflight_cap_is_shared_across_handlers(hf_feature_on, monkeypatch):
    """Two handlers, two runners, one semaphore.

    The cap protects the node's sshd, so it has to count *every* connection to
    the box. It was a per-instance ``asyncio.Semaphore`` on ProxmoxToolHandler,
    which meant two proxmox handlers would not have shared it either — the HF
    handler only made an existing scoping bug visible.
    """
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", True)
    hf, pve = _hf(RecordingRunner()), _pve(RecordingRunner())

    await hf.job_status("newmodel-abc123")
    await pve._pve_status()

    assert inflight_semaphore() is inflight_semaphore()
    # Nothing left held: every acquire released, on both paths.
    assert not inflight_semaphore().locked()


def test_the_cap_is_keyed_by_loop_not_a_module_global():
    """A second event loop gets its own semaphore.

    ``asyncio.Semaphore`` binds to the first loop that awaits it and raises on a
    second, so one module-level instance would make every test after the first
    fail — and pytest-asyncio gives each test its own loop. That is why the cap
    lives in a WeakKeyDictionary keyed by the running loop rather than in a
    module global, which is the obvious shape and the broken one.
    """
    seen = []

    async def grab():
        sem = inflight_semaphore()
        async with sem:
            pass
        seen.append(sem)

    for _ in range(2):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(grab())
        finally:
            loop.close()

    assert len(seen) == 2
    assert seen[0] is not seen[1]


@pytest.mark.asyncio
async def test_each_host_has_its_own_cap():
    """Each sshd has its own MaxStartups (DP-465 added host derpr), so a call
    to one host must not spend another's slots."""
    assert inflight_semaphore("10.0.0.71") is inflight_semaphore("10.0.0.71")
    assert inflight_semaphore("10.0.0.71") is not inflight_semaphore("10.0.0.74")


# -- no third door -----------------------------------------------------------

def test_no_handler_calls_the_ssh_runner_outside_the_gate():
    """Only ``ssh.run_node_command`` may call ``SSHRunner.run``.

    The structural half of the fix. Both handlers passing their own tests is
    what let two implementations coexist; this asserts the shape instead, so a
    third handler that reaches the node on its own terms fails here rather than
    in production with the switch off.
    """
    offenders = []
    for module in (proxmox_handler, hf_handler):
        source = inspect.getsource(module)
        for lineno, line in enumerate(source.splitlines(), start=1):
            if "._ssh.run(" in line:
                offenders.append(f"{module.__name__}:{lineno}: {line.strip()}")
    assert not offenders, (
        "these bypass ssh.run_node_command and are therefore exempt from "
        "PVE_TOOLS_ENABLED and the in-flight cap: " + "; ".join(offenders)
    )


def test_every_node_reaching_handler_routes_through_the_gate():
    """Both ``_run`` implementations are one call into the shared gate."""
    for owner in (ProxmoxToolHandler, HuggingFaceToolHandler):
        source = inspect.getsource(owner._run)
        assert "run_node_command(" in source, (
            f"{owner.__name__}._run does not use the shared node gate"
        )


@pytest.mark.asyncio
async def test_the_gate_labels_which_thing_exited(monkeypatch):
    """A non-zero remote exit still names the caller's own verb.

    Collapsing two implementations into one must not flatten their messages:
    "node script exited 2" and "remote command exited 2" are different
    diagnoses, and tests/test_confirmations.py reads the proxmox wording.
    """
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", True)
    monkeypatch.setattr(global_config, "HF_TOOLS_ENABLED", True)

    hf_res: Dict[str, Any] = await _hf(FailingRunner())._run(["/usr/local/sbin/x"])
    pve_res: Dict[str, Any] = await _pve(FailingRunner())._run(["uptime"])

    assert hf_res["message"] == "node script exited 2"
    assert pve_res["message"] == "remote command exited 2"
    assert hf_res["stderr"] == pve_res["stderr"] == "boom"
