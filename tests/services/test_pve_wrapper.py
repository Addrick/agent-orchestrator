"""The forced-command wrapper must admit exactly what the handlers emit (DP-265).

`services/pve/derpr-pve-wrapper` is what sshd runs instead of whatever derpr
asked for. It is the *second half* of every proxmox and HuggingFace tool, and it
is the half no other test touched — which is how DP-332 shipped `list_models` and
`gpu_status` with three new command shapes and no wrapper update. Unit tests,
mypy and a live read-only smoke test all passed; both tools were dead on the
deployed container, because a developer's own node key is unrestricted and a
workstation smoke test therefore exercises neither the wrapper nor the deployed
path.

So this drives the **real handlers** against a recording runner, takes the argv
they actually send, and feeds each one through the **real wrapper** under bash.
A new command shape that nobody allowlisted fails here instead of in production.

Skipped where bash is unavailable; CI is ubuntu, and dev boxes have git-bash.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pytest

from config import global_config
from src.huggingface.client import HFFile
from src.huggingface.handler import HuggingFaceToolHandler
from src.proxmox.handler import ProxmoxToolHandler
from src.proxmox.ssh import SSHResult

_BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(_BASH is None, reason="needs bash to run the wrapper")

_WRAPPER = Path(__file__).resolve().parents[2] / "services" / "pve" / "derpr-pve-wrapper"

SHA = "a" * 64
SIZE = 24_000_000_000
TIER = "/usr/local/sbin/derpr-model-tier"


@pytest.fixture(scope="module")
def wrapper(tmp_path_factory):
    """The real wrapper with `allow` and `log` stubbed.

    Only two lines change: `allow` prints a marker instead of `exec`-ing the
    command (there is no `pct` here, and running one would be the wrong test),
    and `log` drops its argument (`logger` is a syslog binary, not a portable
    one). Every regex, every arity check and the whole `case` structure — the
    parts that decide ALLOW from DENY — are exercised as deployed.
    """
    source = _WRAPPER.read_text(encoding="utf-8")
    stubbed = source.replace(
        'allow() { log "ALLOW: $cmd"; exec "${a[@]}"; }',
        'allow() { echo "__ALLOW__"; exit 0; }',
    ).replace(
        'log()  { logger -t derpr-pve -- "$1"; }',
        'log()  { :; }',
    )
    assert "__ALLOW__" in stubbed, "allow() stub did not apply — wrapper changed shape"
    assert "logger" not in stubbed, "log() stub did not apply — wrapper changed shape"
    path = tmp_path_factory.mktemp("pve") / "wrapper.sh"
    path.write_text(stubbed, encoding="utf-8")
    return path


def _verdict(wrapper: Path, argv: Sequence[str]) -> str:
    """ALLOW or DENY for one argv, decided by the actual wrapper."""
    proc = subprocess.run(
        [str(_BASH), str(wrapper)],
        env={"SSH_ORIGINAL_COMMAND": shlex.join(argv), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True,
    )
    return "ALLOW" if "__ALLOW__" in proc.stdout else "DENY"


class Recorder:
    """Captures every argv a handler sends, answering plausibly enough to run on."""

    def __init__(self) -> None:
        self.calls: List[List[str]] = []

    async def run(self, argv: Sequence[str]) -> SSHResult:
        a = list(argv)
        self.calls.append(a)
        if "list-unit-files" in a:
            return SSHResult(
                0,
                "koboldcpp-fable.service  disabled  enabled\n"
                "koboldcpp-archived.service  disabled  enabled",
                "",
            )
        # DP-340: the tier inventory. `fable` is hot, `archived` is cold, so
        # driving set_active_model over both exercises the straight swap AND
        # the promotion path — and therefore emits both argv families.
        if a[:2] == [TIER, "list"]:
            return SSHResult(0, json.dumps({
                "status": "ok",
                "hot_dir": "/srv/models",
                "archive_dir": "/srv/archive/models",
                "hot_free_bytes": 40_000_000_000,
                "active": "fable.gguf",
                "models": [
                    {"file": "fable.gguf", "tier": "hot",
                     "size_bytes": SIZE, "pinned": False,
                     "last_served": 1, "archived": True},
                    {"file": "archived.gguf", "tier": "cold",
                     "size_bytes": SIZE, "pinned": False,
                     "last_served": 0, "archived": True},
                ],
            }), "")
        if "show" in a and "--property=ExecStart" in a:
            unit = next((x for x in a if x.endswith(".service")), "")
            stem = unit[len("koboldcpp-"):-len(".service")] if unit else "fable"
            return SSHResult(
                0, "argv[]=/opt/koboldcpp/koboldcpp --model "
                   f"/opt/koboldcpp/models/{stem}.gguf --port 5001 ;", "")
        if a[-2:] == ["ls", "/sys/class/drm"]:
            return SSHResult(0, "card1\ncard1-DP-1\nrenderD128", "")
        if len(a) >= 3 and a[-3] == "cat":
            return SSHResult(0, "34208743424\n29990813696", "")
        if a[:2] == ["pct", "list"]:
            return SSHResult(0, "VMID       Status     Lock         Name\n"
                                "101        running                 gpu", "")
        if a[:2] == ["qm", "list"]:
            return SSHResult(0, "      VMID NAME    STATUS", "")
        return SSHResult(0, "ok", "")


class StubHF:
    async def find_gguf_file(self, repo: str, file_path: str) -> HFFile:
        return HFFile(path=file_path, size_bytes=SIZE, sha256=SHA)

    async def search_models(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        return []

    async def list_gguf_files(self, repo: str, revision: str = "main") -> List[HFFile]:
        return []


@pytest.fixture
def emitted(monkeypatch) -> List[List[str]]:
    """Every argv the two handlers send across their whole tool surface."""
    import asyncio

    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", True)
    monkeypatch.setattr(global_config, "HF_TOOLS_ENABLED", True)
    monkeypatch.setattr(global_config, "PVE_MODEL_HOST_VMID", "101")

    recorder = Recorder()
    pve = ProxmoxToolHandler(recorder)  # type: ignore[arg-type]
    hf = HuggingFaceToolHandler(StubHF(), recorder)  # type: ignore[arg-type]

    async def drive() -> None:
        await pve._pve_status()
        await pve._list_models()
        await pve._gpu_status()
        await pve._reboot_node()
        await pve._reboot_guest(vmid="101", kind="ct")
        await pve._start_guest(vmid="101", kind="ct")
        await pve._stop_guest(vmid="101", kind="ct")
        await pve._set_active_model("fable")
        # DP-340: the cold target promotes instead of swapping, which is a
        # different argv family and therefore a different wrapper entry.
        await pve._set_active_model("archived")
        await hf._install_model("owner/m-GGUF", "model-Q6_K.gguf", "newmodel")
        await hf.job_status("newmodel-abc123def456")

    asyncio.run(drive())
    assert recorder.calls
    return recorder.calls


def test_every_argv_the_handlers_emit_is_allowlisted(wrapper, emitted):
    """The parity check. A command shape added to a handler without a matching
    wrapper entry is dead on the deployed container and green everywhere else —
    this is the one place that notices."""
    refused = [shlex.join(a) for a in emitted if _verdict(wrapper, a) != "ALLOW"]
    assert not refused, (
        "these argvs are emitted by a handler but DENIED by the forced-command "
        f"wrapper — they would fail only in production: {refused}"
    )


def test_the_dp332_shapes_are_covered(wrapper, emitted):
    """Named explicitly, because these are the three that were missing. A
    regression here is the exact outage that happened."""
    joined = [shlex.join(a) for a in emitted]
    assert any("list-unit-files" in c for c in joined)
    assert any("/sys/class/drm" in c and " ls " in f" {c} " for c in joined)
    assert any("mem_info_vram_total" in c for c in joined)


def test_the_dp340_tier_shapes_are_covered(wrapper, emitted):
    """Named explicitly so the parity check above cannot pass vacuously.

    If the cold-tier branch stopped firing, `test_every_argv...` would still be
    green — it only checks the argvs that *were* emitted. These assert the new
    shapes actually reach the wrapper at all.
    """
    joined = [shlex.join(a) for a in emitted]
    assert any(c.endswith("derpr-model-tier list") for c in joined),         "list_models/set_active_model never asked the node for its tiers"
    assert any("derpr-model-tier promote archived.gguf" in c for c in joined),         "a cold model did not produce a promotion argv"


def test_a_cold_target_promotes_and_touches_nothing_else(emitted):
    """The safety property: promoting must not disable the running model.

    :5001 keeps serving until a *second* set_active_model swaps to the promoted
    file, so a promotion that fails costs time and nothing else. A `disable`
    emitted for the cold target's sake would break that.
    """
    joined = [shlex.join(a) for a in emitted]
    promote_at = next(i for i, c in enumerate(joined) if "tier promote" in c)
    after = joined[promote_at:]
    assert not any("systemctl disable" in c for c in after), (
        "set_active_model disabled a unit while promoting a cold model — "
        f":5001 would go dark for the length of the copy: {after}"
    )


@pytest.mark.parametrize("cmd", [
    "id",
    "bash",
    "cat /root/.ssh/authorized_keys",
    "pct exec 101 -- rm -rf /",
    "pct exec 101 -- cat /etc/shadow",
    "pct push 101 /tmp/x /etc/systemd/system/evil.service",
    # `run` is systemd's entry point, not sshd's: admitting it would let a
    # caller skip the free-space precheck and the existing-unit refusal.
    "/usr/local/sbin/derpr-model-install run owner/m model.gguf n 8192 1 "
    + "a" * 64 + " job1",
    # Arity and charset gates on the install verb.
    "/usr/local/sbin/derpr-model-install install owner/m model.gguf n 8192",
    "/usr/local/sbin/derpr-model-install install ../../etc model.gguf n 8192 1 "
    + "a" * 64 + " job1",
    "/usr/local/sbin/derpr-model-install install owner/m model.gguf UPPER 8192 1 "
    + "a" * 64 + " job1",
    "/usr/local/sbin/derpr-model-install install owner/m model.gguf n 8192 1 "
    "nothexdigest job1",
    "/usr/local/sbin/derpr-model-install status ../../etc/passwd",
    # DP-340 tiering. `run-promote` is systemd's local entry point: admitting it
    # over ssh would let a caller skip the job-record and duplicate-job gates.
    "/usr/local/sbin/derpr-model-tier run-promote model.gguf job1",
    # Path traversal and charset gates on the tier verbs.
    "/usr/local/sbin/derpr-model-tier promote ../../etc/passwd job1",
    "/usr/local/sbin/derpr-model-tier promote ../evil.gguf job1",
    "/usr/local/sbin/derpr-model-tier promote /srv/models/m.gguf job1",
    "/usr/local/sbin/derpr-model-tier promote model.txt job1",
    "/usr/local/sbin/derpr-model-tier promote model.gguf",
    "/usr/local/sbin/derpr-model-tier pin ../../root/.ssh/id_rsa",
    "/usr/local/sbin/derpr-model-tier pin",
    "/usr/local/sbin/derpr-model-tier list extra-arg",
    "/usr/local/sbin/derpr-model-tier evict model.gguf",
    # A sysfs read outside the VRAM counters.
    "pct exec 101 -- cat /sys/class/drm/card1/device/mem_info_vram_total /etc/shadow",
    # A directory listing that is not the DRM one.
    "pct exec 101 -- ls /root",
])
def test_hostile_shapes_are_refused(wrapper, cmd):
    assert _verdict(wrapper, shlex.split(cmd)) == "DENY"


def test_the_install_verb_is_admitted_in_full(wrapper):
    argv = [
        "/usr/local/sbin/derpr-model-install", "install",
        "unsloth/gemma-4-31b-it-GGUF", "gemma-4-31b-it-Q4_K_M.gguf",
        "gemma31b", "8192", str(SIZE), SHA, "gemma31b-0123456789ab",
    ]
    assert _verdict(wrapper, argv) == "ALLOW"
