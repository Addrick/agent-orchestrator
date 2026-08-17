"""Guest-by-name lookup and the structured inventory (DP-327).

No network: an InventoryRunner returns canned `pct list` / `qm list` text in the
real fixed-width shapes Proxmox emits, so the parser is exercised against output
that actually has the traps — a blank `Lock` column in `pct list`, a right-aligned
`VMID` in `qm list`, and two different column orders.

The load-bearing assertion in here is ``test_name_never_crosses_the_ssh_boundary``:
name resolution is only a safe feature because the hostname is matched locally and
never becomes an argv element.
"""

from __future__ import annotations

from typing import List, Sequence

import pytest

from config import global_config
from src.proxmox.handler import ProxmoxToolHandler, _parse_table
from src.proxmox.ssh import SSHResult

# Real `pct list` shape: Lock sits between Status and Name and is usually blank,
# which is exactly what breaks a naive whitespace split (Name lands in Lock).
PCT_LIST = (
    "VMID       Status     Lock         Name\n"
    "100        running                 docker\n"
    "101        running                 gpu\n"
    "102        stopped                 rocm\n"
    "103        running    backup       cuda\n"
)

# Real `qm list` shape: different column order, VMID right-aligned under an
# indented header.
QM_LIST = (
    "      VMID NAME                 STATUS     MEM(MB)    BOOTDISK(GB) PID\n"
    "       111 win-3080             stopped    16384             64.00 0\n"
    "       112 sandbox              running     8192             32.00 4242\n"
)


class InventoryRunner:
    """Serves the canned listings; records every argv for boundary assertions."""

    def __init__(self, pct: str = PCT_LIST, qm: str = QM_LIST) -> None:
        self.calls: List[List[str]] = []
        self._pct = pct
        self._qm = qm

    async def run(self, argv: Sequence[str]) -> SSHResult:
        a = list(argv)
        self.calls.append(a)
        if a == ["pct", "list"]:
            return SSHResult(0, self._pct, "")
        if a == ["qm", "list"]:
            return SSHResult(0, self._qm, "")
        if a == ["uptime"]:
            return SSHResult(0, " 04:12:01 up 9 days,  1:22,  0 users", "")
        return SSHResult(0, "", "")


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", True)


@pytest.fixture
def handler(enabled):
    runner = InventoryRunner()
    return ProxmoxToolHandler(runner), runner  # type: ignore[arg-type]


# -- table parser ------------------------------------------------------------

def test_parse_pct_list_keeps_name_out_of_the_blank_lock_column():
    rows = _parse_table(PCT_LIST)
    assert [r["vmid"] for r in rows] == ["100", "101", "102", "103"]
    assert [r["name"] for r in rows] == ["docker", "gpu", "rocm", "cuda"]
    assert [r["status"] for r in rows] == ["running", "running", "stopped", "running"]
    # the one row that *does* carry a lock keeps it in the right field
    assert rows[3]["lock"] == "backup"
    assert rows[0]["lock"] == ""


def test_parse_qm_list_handles_right_aligned_vmid_and_other_column_order():
    rows = _parse_table(QM_LIST)
    assert [r["vmid"] for r in rows] == ["111", "112"]
    assert [r["name"] for r in rows] == ["win-3080", "sandbox"]
    assert [r["status"] for r in rows] == ["stopped", "running"]


def test_parse_table_captures_a_vmid_wider_than_its_header():
    """`qm list` right-aligns VMID, so a 6-digit id starts left of the header."""
    text = (
        "      VMID NAME                 STATUS\n"
        "    100000 bigid                running\n"
    )
    assert _parse_table(text) == [
        {"vmid": "100000", "name": "bigid", "status": "running"}
    ]


@pytest.mark.parametrize("text", ["", "   \n\n", "VMID Status Lock Name\n"])
def test_parse_table_tolerates_empty_and_headerless_output(text):
    assert _parse_table(text) == []


def test_parse_table_drops_rows_without_a_numeric_vmid():
    """A trailing note or a wrapped line is dropped, not guessed at."""
    text = PCT_LIST + "no containers configured\n"
    assert len(_parse_table(text)) == 4


# -- structured inventory on pve_status --------------------------------------

@pytest.mark.asyncio
async def test_pve_status_returns_structured_guests(handler):
    h, _ = handler
    res = await h._pve_status()
    assert res["status"] == "ok"
    by_name = {g["name"]: g for g in res["guests"]}
    assert by_name["gpu"] == {
        "vmid": "101", "name": "gpu", "kind": "ct",
        "status": "running", "lock": "",
    }
    assert by_name["win-3080"]["kind"] == "vm"
    assert by_name["win-3080"]["vmid"] == "111"
    # raw listings are still there for anything the parse drops
    assert "MEM(MB)" in res["vms"]
    assert "inventory_errors" not in res


@pytest.mark.asyncio
async def test_pve_status_reports_a_half_failed_inventory_without_hiding_the_rest(enabled):
    """A dead `qm list` must not take the containers down with it."""

    class HalfBroken(InventoryRunner):
        async def run(self, argv):
            a = list(argv)
            if a == ["qm", "list"]:
                self.calls.append(a)
                return SSHResult(1, "", "qm: command not found")
            return await super().run(argv)

    h = ProxmoxToolHandler(HalfBroken())  # type: ignore[arg-type]
    res = await h._pve_status()
    assert res["status"] == "ok"
    assert {g["name"] for g in res["guests"]} == {"docker", "gpu", "rocm", "cuda"}
    assert any("qm: command not found" in e for e in res["inventory_errors"])


# -- resolution by name ------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("name,cli,vmid", [
    ("gpu", "pct", "101"),
    ("rocm", "pct", "102"),
    ("win-3080", "qm", "111"),
])
async def test_start_guest_by_name_picks_the_right_cli_and_id(handler, name, cli, vmid):
    h, runner = handler
    res = await h._start_guest(name=name)
    assert res["status"] == "ok"
    assert [cli, "start", vmid] in runner.calls


@pytest.mark.asyncio
async def test_name_match_is_case_insensitive(handler):
    h, runner = handler
    res = await h._stop_guest(name="WIN-3080")
    assert res["status"] == "ok"
    assert ["qm", "stop", "111"] in runner.calls


@pytest.mark.asyncio
async def test_result_echoes_the_resolved_target(handler):
    """The audit record should show what was acted on, not just what was typed."""
    h, _ = handler
    res = await h._reboot_guest(name="gpu")
    assert res["target"] == {"vmid": "101", "kind": "ct", "name": "gpu"}


@pytest.mark.asyncio
async def test_name_never_crosses_the_ssh_boundary(handler):
    """Only the resolved digits are ever sent — the hostname stays local.

    This is the property the whole transport's safety argument rests on: ssh.py's
    metacharacter guard is allowed to be the *second* line of defence precisely
    because no model-supplied string reaches it.
    """
    h, runner = handler
    await h._stop_guest(name="win-3080")
    flat = [arg for call in runner.calls for arg in call]
    assert "win-3080" not in flat
    assert ["qm", "stop", "111"] in runner.calls


@pytest.mark.asyncio
async def test_unknown_name_is_refused_and_lists_what_exists(handler):
    h, runner = handler
    res = await h._stop_guest(name="gpuu")
    assert res["status"] == "error"
    assert "no guest named 'gpuu'" in res["message"]
    assert "docker" in res["message"] and "win-3080" in res["message"]
    # the listing ran, but no power command did
    assert not any(c[1:2] in (["stop"], ["start"], ["reboot"]) for c in runner.calls)


@pytest.mark.asyncio
async def test_ambiguous_name_is_refused_until_kind_disambiguates(enabled):
    """A name held by both a CT and a VM must not be guessed at."""
    pct = "VMID       Status     Lock         Name\n200        running                 twin\n"
    qm = (
        "      VMID NAME                 STATUS\n"
        "       201 twin                 running\n"
    )
    runner = InventoryRunner(pct=pct, qm=qm)
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]

    res = await h._stop_guest(name="twin")
    assert res["status"] == "error"
    assert "ambiguous" in res["message"]
    assert "ct 200" in res["message"] and "vm 201" in res["message"]
    assert not any("stop" in c for c in runner.calls)

    res = await h._stop_guest(name="twin", kind="vm")
    assert res["status"] == "ok"
    assert ["qm", "stop", "201"] in runner.calls


@pytest.mark.asyncio
async def test_name_with_wrong_kind_is_refused(handler):
    h, _ = handler
    res = await h._stop_guest(name="gpu", kind="vm")
    assert res["status"] == "error"
    assert "no guest named 'gpu' of kind 'vm'" in res["message"]


# -- resolution by vmid (back-compat + the kind requirement) -----------------

@pytest.mark.asyncio
async def test_bare_vmid_still_works_when_kind_is_given(handler):
    h, runner = handler
    res = await h._reboot_guest("100", "ct")
    assert res["status"] == "ok"
    # resolving by id needs no inventory lookup at all
    assert runner.calls == [["pct", "reboot", "100"]]


@pytest.mark.asyncio
async def test_vmid_without_kind_is_refused_rather_than_guessed(handler):
    h, runner = handler
    res = await h._stop_guest(vmid="100")
    assert res["status"] == "error"
    assert "kind is required" in res["message"]
    assert runner.calls == []


@pytest.mark.asyncio
async def test_neither_name_nor_vmid_is_refused(handler):
    h, runner = handler
    res = await h._stop_guest()
    assert res["status"] == "error"
    assert "either a guest name or a numeric vmid" in res["message"]
    assert runner.calls == []


@pytest.mark.asyncio
async def test_lookup_is_skipped_entirely_when_tools_are_disabled(monkeypatch):
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", False)
    runner = InventoryRunner()
    h = ProxmoxToolHandler(runner)  # type: ignore[arg-type]
    res = await h._stop_guest(name="gpu")
    assert res["status"] == "error"
    assert "disabled" in res["message"]
    assert runner.calls == []
