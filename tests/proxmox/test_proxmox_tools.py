"""Unit tests for the Proxmox management tools (DP-262).

No network: a FakeRunner records the argv each tool sends and returns canned
results, so we assert on the exact commands + the disabled/validation guards.
"""

from __future__ import annotations

from typing import List, Sequence

import pytest

from config import global_config
from src.proxmox.handler import ProxmoxToolHandler
from src.proxmox.ssh import SSHError, SSHResult, SSHRunner, _reject_bad_args


class FakeRunner:
    """Stand-in for SSHRunner: records calls, returns queued/canned results."""

    def __init__(
        self,
        result: SSHResult | None = None,
        present_units: set[str] | None = None,
    ) -> None:
        self.calls: List[List[str]] = []
        self._result = result or SSHResult(0, "ok-stdout", "")
        # None => every unit's model file "exists"; else only these unit names.
        self._present = present_units

    async def run(self, argv: Sequence[str]) -> SSHResult:
        a = list(argv)
        self.calls.append(a)
        # `systemctl show <unit> --property=ExecStart --value` → synth a line that
        # embeds a --model path derived from the unit name.
        if "show" in a and "--property=ExecStart" in a:
            unit = next((x for x in a if x.endswith(".service")), "u.service")
            return SSHResult(0, f"argv[]=/opt/kcpp --model /models/{unit}.gguf ;", "")
        # `test -f /models/<unit>.gguf` → present per self._present.
        if len(a) >= 2 and a[-2] == "-f":
            unit = a[-1].split("/")[-1][:-5]  # strip ".gguf"
            ok = self._present is None or unit in self._present
            return SSHResult(0 if ok else 1, "", "" if ok else "No such file")
        return self._result


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", True)
    monkeypatch.setattr(global_config, "PVE_MODEL_HOST_VMID", "101")
    monkeypatch.setattr(
        global_config, "PVE_MODEL_UNITS",
        {"fable": "koboldcpp.service", "gemma": "gemma.service"},
    )


# -- disabled guard ----------------------------------------------------------

@pytest.mark.asyncio
async def test_tools_disabled_short_circuit(monkeypatch):
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", False)
    runner = FakeRunner()
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    for coro in (h._pve_status(), h._reboot_node(), h._list_models(),
                 h._reboot_guest("100", "ct"), h._set_active_model("fable")):
        res = await coro
        assert res["status"] == "error"
        assert "disabled" in res["message"]
    assert runner.calls == []  # never attempted SSH


# -- read tools --------------------------------------------------------------

@pytest.mark.asyncio
async def test_pve_status_runs_three_reads(enabled):
    runner = FakeRunner()
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._pve_status()
    assert res["status"] == "ok"
    assert ["uptime"] in runner.calls
    assert ["pct", "list"] in runner.calls
    assert ["qm", "list"] in runner.calls


@pytest.mark.asyncio
async def test_list_models_reports_active_state(enabled):
    runner = FakeRunner(SSHResult(0, "active", ""))
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._list_models()
    assert res["status"] == "ok"
    names = {m["name"] for m in res["models"]}
    assert names == {"fable", "gemma"}
    # each model queried via `pct exec 101 -- systemctl is-active <unit>`
    assert ["pct", "exec", "101", "--", "systemctl", "is-active", "koboldcpp.service"] in runner.calls


# -- write tools -------------------------------------------------------------

@pytest.mark.asyncio
async def test_reboot_node(enabled):
    runner = FakeRunner()
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._reboot_node()
    assert res["status"] == "ok"
    assert runner.calls == [["reboot"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,cli", [("ct", "pct"), ("vm", "qm")])
async def test_guest_actions_pick_correct_cli(enabled, kind, cli):
    runner = FakeRunner()
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    await h._reboot_guest("100", kind)
    await h._start_guest("100", kind)
    await h._stop_guest("100", kind)
    assert runner.calls == [
        [cli, "reboot", "100"], [cli, "start", "100"], [cli, "stop", "100"],
    ]


@pytest.mark.asyncio
async def test_guest_rejects_bad_vmid(enabled):
    runner = FakeRunner()
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._reboot_guest("100; rm -rf /", "ct")
    assert res["status"] == "error"
    assert "integer" in res["message"]
    assert runner.calls == []


@pytest.mark.asyncio
async def test_guest_rejects_bad_kind(enabled):
    runner = FakeRunner()
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._reboot_guest("100", "container")
    assert res["status"] == "error"
    assert runner.calls == []


@pytest.mark.asyncio
async def test_set_active_model_disables_others_then_enables_target(enabled):
    runner = FakeRunner()
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._set_active_model("fable")
    assert res["status"] == "ok"
    assert res["unit"] == "koboldcpp.service"
    # gemma disabled, fable enabled
    assert ["pct", "exec", "101", "--", "systemctl", "disable", "--now", "gemma.service"] in runner.calls
    assert ["pct", "exec", "101", "--", "systemctl", "enable", "--now", "koboldcpp.service"] in runner.calls


@pytest.mark.asyncio
async def test_set_active_model_unknown_name(enabled):
    runner = FakeRunner()
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._set_active_model("nope")
    assert res["status"] == "error"
    assert "unknown model" in res["message"]
    assert runner.calls == []


@pytest.mark.asyncio
async def test_remote_nonzero_exit_surfaced(enabled):
    runner = FakeRunner(SSHResult(1, "", "boom"))
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._reboot_node()
    assert res["status"] == "error"
    assert res["stderr"] == "boom"


@pytest.mark.asyncio
async def test_ssh_transport_error_mapped(enabled):
    class Boom:
        async def run(self, argv):
            raise SSHError("no route to host")

    h = ProxmoxToolHandler(Boom())  # type: ignore[arg-type]
    res = await h._reboot_node()
    assert res["status"] == "error"
    assert "no route to host" in res["message"]


# -- ssh runner guard --------------------------------------------------------

def test_reject_bad_args_blocks_metacharacters():
    with pytest.raises(SSHError):
        _reject_bad_args(["reboot", "; rm -rf /"])
    with pytest.raises(SSHError):
        _reject_bad_args(["$(whoami)"])
    # clean argv passes
    _reject_bad_args(["pct", "reboot", "100"])


def test_ssh_runner_config_defaults(monkeypatch):
    monkeypatch.setattr(global_config, "PVE_SSH_HOST", "1.2.3.4")
    monkeypatch.setattr(global_config, "PVE_SSH_USER", "root")
    monkeypatch.setattr(global_config, "PVE_SSH_KEY", "/k")
    monkeypatch.setattr(global_config, "PVE_SSH_TIMEOUT", 9.0)
    r = SSHRunner()
    assert r._host == "1.2.3.4" and r._user == "root" and r._key == "/k" and r._timeout == 9.0


# -- DP-264: model availability filter + swap guard --------------------------

@pytest.mark.asyncio
async def test_list_models_omits_units_with_missing_model_file(enabled):
    """Only units whose gguf is on disk are listed (koboldcpp present, gemma not)."""
    runner = FakeRunner(present_units={"koboldcpp.service"})
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._list_models()
    assert res["status"] == "ok"
    names = {m["name"] for m in res["models"]}
    assert names == {"fable"}  # gemma omitted (no file)


@pytest.mark.asyncio
async def test_set_active_model_refuses_when_target_file_missing(enabled):
    """Swap to a unit with no gguf is refused; current model left untouched
    (no disable/enable issued)."""
    # target for "fable" is koboldcpp.service; mark only gemma present.
    runner = FakeRunner(present_units={"gemma.service"})
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._set_active_model("fable")
    assert res["status"] == "error"
    assert "not on disk" in res["message"]
    # crucially: nothing was disabled or enabled
    assert not any("disable" in c or "enable" in c for c in runner.calls)
