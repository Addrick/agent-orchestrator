"""Tool handlers for the Proxmox management service (DP-262).

Seven tools behind the ``proxmox`` service binding:

- ``pve_status``      (read):  node uptime + `pct list` + `qm list`, both raw and
  parsed into a structured guest inventory (DP-327).
- ``list_models``     (read):  configured unit map + which is active on the GPU CT.
- ``reboot_node``     (WRITE, irreversible → parked): reboot the metal.
- ``reboot_guest``    (WRITE → parked): reboot one VM/CT.
- ``start_guest``     (WRITE → parked): start one VM/CT.
- ``stop_guest``      (WRITE → parked): stop one VM/CT.
- ``set_active_model``(WRITE → parked): swap the enabled koboldcpp unit on :5001.

Every handler returns a JSON-able dict. Transport failures and disabled state are
returned as ``{"status": "error", ...}`` rather than raised, so the model gets a
clean message instead of a tool crash.

DP-327 — guests are addressable by ``name`` as well as ``vmid``. The lookup is
deliberately **local**: the inventory comes from parsing `pct list` / `qm list`
output, the name is matched here, and only the resolved digits are ever handed to
``SSHRunner``. A model-supplied hostname therefore never crosses the SSH boundary,
which keeps ``ssh.py``'s metacharacter guard seeing nothing but integers and
config-pinned unit names — the property the whole transport's safety argument
rests on.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from config import global_config
from src.proxmox.ssh import SSHError, SSHRunner

if TYPE_CHECKING:
    from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)

#: Accepted guest kinds → the Proxmox CLI that manages them.
_GUEST_CLI = {"ct": "pct", "vm": "qm"}


def _err(message: str) -> Dict[str, Any]:
    return {"status": "error", "message": message}


def _validate_vmid(vmid: str) -> str:
    """Proxmox vmids are positive integers. Reject anything else early."""
    s = str(vmid).strip()
    if not s.isdigit():
        raise ValueError(f"vmid must be a positive integer, got {vmid!r}")
    return s


def _parse_table(text: str) -> List[Dict[str, str]]:
    """Parse one of Proxmox's fixed-width CLI tables into row dicts.

    `pct list` and `qm list` disagree on both column order and column set, and
    `pct list`'s ``Lock`` column is usually blank — so whitespace-splitting a data
    row silently shifts ``Name`` into ``Lock``'s slot. Instead we read the real
    header line and slice each row at the header tokens' own column offsets, which
    is correct by construction for fixed-width output and tolerates a PVE version
    that adds or reorders a column.

    Keys are the lowercased header tokens (``vmid``, ``name``, ``status``, ``lock``
    …). Rows whose ``vmid`` is not numeric are dropped as unparseable rather than
    guessed at.
    """
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0]
    # (start column, lowercased key) for every header token, in order.
    spans: List[Tuple[int, str]] = []
    pos = 0
    for token in header.split():
        start = header.index(token, pos)
        pos = start + len(token)
        spans.append((start, token.lower()))
    if not spans:
        return []
    # `qm list` right-aligns VMID under a header that is indented, so a vmid with
    # more digits than the header token is wide starts to the LEFT of the header's
    # column. Nothing precedes the first field, so anchor it at 0 and the overflow
    # is captured instead of truncated.
    spans[0] = (0, spans[0][1])

    rows: List[Dict[str, str]] = []
    for line in lines[1:]:
        row: Dict[str, str] = {}
        for i, (start, key) in enumerate(spans):
            end = spans[i + 1][0] if i + 1 < len(spans) else len(line)
            row[key] = line[start:end].strip()
        if row.get("vmid", "").isdigit():
            rows.append(row)
    return rows


class ProxmoxToolHandler:
    def __init__(self, runner: SSHRunner | None = None) -> None:
        self._ssh = runner or SSHRunner()

    def register(self, manager: "ToolManager") -> None:
        manager.register("pve_status", self._pve_status)
        manager.register("list_models", self._list_models)
        manager.register("reboot_node", self._reboot_node)
        manager.register("reboot_guest", self._reboot_guest)
        manager.register("start_guest", self._start_guest)
        manager.register("stop_guest", self._stop_guest)
        manager.register("set_active_model", self._set_active_model)

    # -- guards --------------------------------------------------------------

    def _enabled(self) -> bool:
        return bool(global_config.PVE_TOOLS_ENABLED)

    async def _run(self, argv: List[str]) -> Dict[str, Any]:
        """Run one remote argv, mapping transport/exit errors to result dicts."""
        if not self._enabled():
            return _err(
                "Proxmox tools are disabled (set PVE_TOOLS_ENABLED=true and mount "
                "the pve SSH key to enable)."
            )
        try:
            res = await self._ssh.run(argv)
        except SSHError as e:
            return _err(f"ssh failed: {e}")
        if res.returncode != 0:
            return {
                "status": "error",
                "message": f"remote command exited {res.returncode}",
                "stderr": res.stderr,
                "stdout": res.stdout,
            }
        return {"status": "ok", "stdout": res.stdout, "stderr": res.stderr}

    # -- guest inventory / name resolution (DP-327) ---------------------------

    async def _guest_inventory(self) -> Dict[str, Any]:
        """Every guest on the node as ``{vmid, name, kind, status, lock}`` dicts.

        One `pct list` and one `qm list`, gathered concurrently and parsed. A
        listing that fails is reported in ``errors`` rather than aborting the
        whole inventory — a dead `qm list` should not hide the containers.
        """
        cts, vms = await asyncio.gather(
            self._run(["pct", "list"]),
            self._run(["qm", "list"]),
        )
        guests: List[Dict[str, str]] = []
        errors: List[str] = []
        for res, kind in ((cts, "ct"), (vms, "vm")):
            if res.get("status") != "ok":
                # Keep both: `message` says how it failed ("exited 1"), `stderr`
                # says why. Reporting only the first is what makes an inventory
                # failure look like a mystery in the logs.
                detail = " — ".join(
                    p for p in (res.get("message"), res.get("stderr")) if p
                ) or "unknown error"
                errors.append(f"{kind}: {detail}")
                continue
            for row in _parse_table(res.get("stdout") or ""):
                guests.append({
                    "vmid": row.get("vmid", ""),
                    "name": row.get("name", ""),
                    "kind": kind,
                    "status": row.get("status", ""),
                    "lock": row.get("lock", ""),
                })
        return {"guests": guests, "errors": errors, "raw": {"ct": cts, "vm": vms}}

    async def _resolve_guest(
        self,
        vmid: Optional[str],
        name: Optional[str],
        kind: Optional[str],
    ) -> Dict[str, Any]:
        """Turn (vmid | name) [+ kind] into a concrete ``{vmid, kind, name}``.

        Errors are returned, never raised, so a bad target reads as a tool result.
        A bare ``vmid`` still requires ``kind``: Proxmox ids are unique per guest,
        but *we* cannot tell which CLI owns one without asking, and guessing wrong
        on a power-off is not a mistake worth risking. Addressing by name needs no
        ``kind`` — the lookup supplies it.
        """
        kind_in = (str(kind).strip().lower() or None) if kind else None
        if kind_in is not None and kind_in not in _GUEST_CLI:
            return _err(f"kind must be one of {sorted(_GUEST_CLI)}, got {kind!r}")

        vmid_in = str(vmid).strip() if vmid is not None else ""
        name_in = str(name).strip() if name is not None else ""

        if vmid_in:
            try:
                resolved_vmid = _validate_vmid(vmid_in)
            except ValueError as e:
                return _err(str(e))
            if kind_in is None:
                return _err(
                    "kind is required when addressing a guest by vmid "
                    f"(got vmid={resolved_vmid!r}); pass kind=\"ct\" or \"vm\", or "
                    "address the guest by name instead."
                )
            return {"status": "ok", "vmid": resolved_vmid, "kind": kind_in, "name": name_in or None}

        if not name_in:
            return _err("pass either a guest name or a numeric vmid.")
        return await self._resolve_by_name(name_in, kind_in)

    async def _resolve_by_name(self, name: str, kind: Optional[str]) -> Dict[str, Any]:
        """Match a hostname against the live inventory. Exact, case-insensitive.

        Deliberately **not** fuzzy: a near-miss is refused with the list of names
        that do exist, because the caller of this is about to power something off
        and the friendly behaviour — picking the closest match — is how you stop
        the wrong guest.
        """
        inventory = await self._guest_inventory()
        candidates = [
            g for g in inventory["guests"]
            if g["name"].lower() == name.lower()
            and (kind is None or g["kind"] == kind)
        ]
        if len(candidates) == 1:
            hit = candidates[0]
            return {"status": "ok", "vmid": hit["vmid"], "kind": hit["kind"], "name": hit["name"]}
        if candidates:
            where = ", ".join(f"{c['kind']} {c['vmid']}" for c in candidates)
            return _err(
                f"{name!r} is ambiguous — it matches {where}. Re-issue with "
                'kind="ct" or kind="vm".'
            )
        known = sorted(g["name"] for g in inventory["guests"] if g["name"])
        msg = f"no guest named {name!r}"
        if kind is not None:
            msg += f" of kind {kind!r}"
        msg += f"; known guests: {known}" if known else "; the node reported no guests"
        if inventory["errors"]:
            msg += f" (listing errors: {inventory['errors']})"
        return _err(msg)

    # -- model availability helpers ------------------------------------------

    async def _model_path(self, vmid: str, unit: str) -> Optional[str]:
        """Resolve a unit's ``--model`` gguf path from its ExecStart, or None."""
        res = await self._run([
            "pct", "exec", vmid, "--",
            "systemctl", "show", unit, "--property=ExecStart", "--value",
        ])
        toks = (res.get("stdout") or "").split()
        for i, t in enumerate(toks):
            if t == "--model" and i + 1 < len(toks):
                return toks[i + 1]
        return None

    async def _model_present(self, vmid: str, unit: str) -> bool:
        """True when the unit's model gguf exists on disk (immediately loadable)."""
        path = await self._model_path(vmid, unit)
        if not path:
            return False
        res = await self._run(["pct", "exec", vmid, "--", "test", "-f", path])
        return res.get("status") == "ok"

    # -- read tools ----------------------------------------------------------

    async def _pve_status(self) -> Dict[str, Any]:
        logger.info("Tool pve_status")
        if not self._enabled():
            return _err("Proxmox tools are disabled (set PVE_TOOLS_ENABLED=true).")
        # Metacharacter-free argv reads — no remote shell string is ever built
        # (the SSH runner rejects shell metacharacters), so these run as separate
        # round trips gathered concurrently.
        uptime, inventory = await asyncio.gather(
            self._run(["uptime"]),
            self._guest_inventory(),
        )
        cts = inventory["raw"]["ct"]
        vms = inventory["raw"]["vm"]
        result: Dict[str, Any] = {
            "status": "ok",
            "uptime": uptime.get("stdout") or uptime.get("message"),
            # Structured inventory (DP-327) — this is what makes pve_status a
            # topology audit rather than two blobs of text to re-read every turn.
            "guests": inventory["guests"],
            # Raw listings kept alongside: they carry columns the parse drops
            # (memory, bootdisk, pid) and are the ground truth if a PVE version
            # ever formats a table the parser cannot read.
            "containers": cts.get("stdout") or cts.get("message"),
            "vms": vms.get("stdout") or vms.get("message"),
        }
        if inventory["errors"]:
            result["inventory_errors"] = inventory["errors"]
        return result

    async def _list_models(self) -> Dict[str, Any]:
        logger.info("Tool list_models")
        units = global_config.PVE_MODEL_UNITS
        vmid = global_config.PVE_MODEL_HOST_VMID
        if not self._enabled():
            return _err("Proxmox tools are disabled (set PVE_TOOLS_ENABLED=true).")
        # Only surface models whose gguf is actually on disk (immediately
        # loadable). Units whose model file is missing are omitted — enabling one
        # would fail to start and take :5001 down. (A separate future tool will
        # download+deploy ggufs from HF — see DP-265 note.)
        models: List[Dict[str, Any]] = []
        for name, unit in units.items():
            if not await self._model_present(vmid, unit):
                continue
            state_res = await self._run(
                ["pct", "exec", vmid, "--", "systemctl", "is-active", unit]
            )
            state = (state_res.get("stdout") or state_res.get("message") or "unknown").strip()
            models.append({"name": name, "unit": unit, "state": state})
        return {"status": "ok", "host_vmid": vmid, "models": models}

    # -- write tools (parked for confirmation) -------------------------------

    async def _reboot_node(self) -> Dict[str, Any]:
        logger.info("Tool reboot_node")
        return await self._run(["reboot"])

    async def _guest_action(
        self,
        action: str,
        vmid: Optional[str] = None,
        kind: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve the target, then run ``<pct|qm> <action> <vmid>`` on it.

        Resolution happens here, at execution — i.e. *after* a human approved the
        park — so the guest acted on is the one that exists now, and the returned
        target is echoed back into the audit record rather than only the argument
        the model typed.
        """
        target = await self._resolve_guest(vmid, name, kind)
        if target.get("status") != "ok":
            return target
        cli = _GUEST_CLI[target["kind"]]
        res = await self._run([cli, action, target["vmid"]])
        res["target"] = {
            "vmid": target["vmid"],
            "kind": target["kind"],
            "name": target.get("name"),
        }
        return res

    async def _reboot_guest(
        self, vmid: Optional[str] = None, kind: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info("Tool reboot_guest: kind=%s vmid=%s name=%s", kind, vmid, name)
        return await self._guest_action("reboot", vmid, kind, name)

    async def _start_guest(
        self, vmid: Optional[str] = None, kind: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info("Tool start_guest: kind=%s vmid=%s name=%s", kind, vmid, name)
        return await self._guest_action("start", vmid, kind, name)

    async def _stop_guest(
        self, vmid: Optional[str] = None, kind: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info("Tool stop_guest: kind=%s vmid=%s name=%s", kind, vmid, name)
        return await self._guest_action("stop", vmid, kind, name)

    async def _set_active_model(self, name: str) -> Dict[str, Any]:
        logger.info("Tool set_active_model: %s", name)
        units = global_config.PVE_MODEL_UNITS
        vmid = global_config.PVE_MODEL_HOST_VMID
        target = units.get(name)
        if target is None:
            return _err(
                f"unknown model {name!r}; configured: {sorted(units)}"
            )
        if not self._enabled():
            return _err("Proxmox tools are disabled (set PVE_TOOLS_ENABLED=true).")
        # Pre-flight: never disable the running model to enable a unit that can't
        # start. If the target's gguf isn't on disk, refuse and leave :5001 as-is.
        if not await self._model_present(vmid, target):
            return _err(
                f"model file for {name!r} (unit {target}) is not on disk; "
                "not switching — current model left running."
            )
        # Disable every other configured unit (all bind :5001 — only one may run),
        # then enable+start the target. Idempotent: re-selecting the active model
        # just re-enables it.
        for other_name, other_unit in units.items():
            if other_unit == target:
                continue
            await self._run([
                "pct", "exec", vmid, "--",
                "systemctl", "disable", "--now", other_unit,
            ])
        res = await self._run([
            "pct", "exec", vmid, "--",
            "systemctl", "enable", "--now", target,
        ])
        if res.get("status") != "ok":
            return res
        return {"status": "ok", "active_model": name, "unit": target, "host_vmid": vmid}
