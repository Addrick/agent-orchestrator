"""The second node key's wrapper must admit exactly what derpr emits (DP-465).

`services/sandbox-net/derpr-net-wrapper` gates derpr's key on host `derpr`,
where the sandbox boxes' LAN switch lives. Same method as test_pve_wrapper.py:
drive the real handlers against a recording runner and feed every argv they
send to that host through the real wrapper under bash.
"""

from __future__ import annotations

import asyncio
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

import pytest

from config import global_config
from src.proxmox.artifacts import NET_NODE_ARTIFACTS, NET_SERVICES_DIR, check_node_artifacts
from src.proxmox.ssh import SSHResult
from src.stagr.handler import StagrToolHandler

_BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(_BASH is None, reason="needs bash to run the wrapper")

_WRAPPER = NET_SERVICES_DIR / "derpr-net-wrapper"


@pytest.fixture(scope="module")
def wrapper(tmp_path_factory) -> Path:
    """The real wrapper with `allow` printing a marker and `log` silenced."""
    source = _WRAPPER.read_text(encoding="utf-8")
    stubbed = source.replace(
        'allow() { log "ALLOW: $cmd"; exec "${a[@]}"; }',
        'allow() { echo "__ALLOW__"; exit 0; }',
    ).replace(
        'log()  { logger -t derpr-net -- "$1"; }',
        'log()  { :; }',
    )
    assert "__ALLOW__" in stubbed and "logger" not in stubbed, "stubs did not apply — wrapper changed shape"
    path = tmp_path_factory.mktemp("net") / "wrapper.sh"
    path.write_text(stubbed, encoding="utf-8", newline="\n")
    return path


def _verdict(wrapper: Path, argv: Sequence[str]) -> str:
    proc = subprocess.run(
        [str(_BASH), str(wrapper)],
        env={"SSH_ORIGINAL_COMMAND": shlex.join(argv), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True,
    )
    return "ALLOW" if "__ALLOW__" in proc.stdout else "DENY"


class NetRecorder:
    """Host derpr: a LAN switch that remembers, so on and off are both emitted."""

    def __init__(self) -> None:
        self.calls: List[List[str]] = []
        self.on: Dict[str, bool] = {}

    async def run(self, argv: Sequence[str]) -> SSHResult:
        a = list(argv)
        self.calls.append(a)
        if a[0] == "sha256sum":
            return SSHResult(0, "", "")
        if a[3] in ("on", "off"):
            self.on[a[1]] = a[3] == "on"
        if a[3] == "conns":
            return SSHResult(0, "", "")
        return SSHResult(0, f"lan: {'on' if self.on.get(a[1]) else 'off'}", "")


class WanRecorder:
    async def run(self, argv: Sequence[str]) -> SSHResult:
        return SSHResult(0, "wan: off", "")


@pytest.fixture
def emitted(monkeypatch) -> List[List[str]]:
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", True)
    net = NetRecorder()
    handler = StagrToolHandler(WanRecorder(), net)  # type: ignore[arg-type]

    async def drive() -> None:
        await handler._sandbox_net_status()
        await handler._list_web_access()
        for box in ("codr", "stagr"):
            await handler._set_sandbox_net(box, "lan", "on", confirm=True)
            await handler._set_sandbox_net(box, "lan", "off", confirm=True)
        await handler._request_network("lan", "x")
        await check_node_artifacts(net, NET_SERVICES_DIR, NET_NODE_ARTIFACTS)  # type: ignore[arg-type]

    asyncio.run(drive())
    return net.calls


def test_every_argv_sent_to_host_derpr_is_allowlisted(wrapper, emitted):
    refused = [shlex.join(a) for a in emitted if _verdict(wrapper, a) != "ALLOW"]
    assert not refused, f"emitted but DENIED by derpr-net-wrapper (would fail only in production): {refused}"


def test_the_lan_shapes_are_covered(emitted):
    """Named, so the parity check cannot pass vacuously."""
    for box in ("codr", "stagr"):
        for verb in ("status", "conns", "on", "off"):
            assert ["/usr/local/sbin/agent-net", box, "lan", verb] in emitted, (box, verb)
    assert ["sha256sum", *(p for _, p in NET_NODE_ARTIFACTS)] in emitted


@pytest.mark.parametrize("cmd", [
    "id",
    "bash",
    "pct list",
    "qm stop 200",
    "cat /etc/pve/firewall/200.fw",
    "/usr/local/sbin/agent-net stagr lan",
    "/usr/local/sbin/agent-net stagr wan on",
    "/usr/local/sbin/agent-net dt21 lan on",
    "/usr/local/sbin/agent-net stagr lan on extra",
    # The hookscript form is Proxmox's, never sshd's.
    "/usr/local/sbin/agent-net 201 pre-start",
    "/usr/local/sbin/derpr-egress stagr on",
    "sha256sum /etc/shadow",
    "sha256sum /usr/local/sbin/agent-net /usr/local/bin/derpr-net-wrapper",
])
def test_hostile_shapes_are_refused(wrapper, cmd):
    assert _verdict(wrapper, shlex.split(cmd)) == "DENY"


def test_the_shipped_artifacts_exist():
    for repo_name, _ in NET_NODE_ARTIFACTS:
        assert (NET_SERVICES_DIR / repo_name).is_file(), repo_name
