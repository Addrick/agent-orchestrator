"""Tool handlers for the Proxmox management service (DP-262).

Seven tools behind the ``proxmox`` service binding:

- ``pve_status``      (read):  node uptime + `pct list` + `qm list`.
- ``list_models``     (read):  configured unit map + which is active on the GPU CT.
- ``reboot_node``     (WRITE, irreversible → parked): reboot the metal.
- ``reboot_guest``    (WRITE → parked): reboot one VM/CT.
- ``start_guest``     (WRITE → parked): start one VM/CT.
- ``stop_guest``      (WRITE → parked): stop one VM/CT.
- ``set_active_model``(WRITE → parked): swap the enabled koboldcpp unit on :5001.

Every handler returns a JSON-able dict. Transport failures and disabled state are
returned as ``{"status": "error", ...}`` rather than raised, so the model gets a
clean message instead of a tool crash.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

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
        # Three metacharacter-free argv reads — no remote shell string is ever
        # built (the SSH runner rejects shell metacharacters), so these run as
        # separate round trips gathered concurrently.
        uptime, cts, vms = await asyncio.gather(
            self._run(["uptime"]),
            self._run(["pct", "list"]),
            self._run(["qm", "list"]),
        )
        return {
            "status": "ok",
            "uptime": uptime.get("stdout") or uptime.get("message"),
            "containers": cts.get("stdout") or cts.get("message"),
            "vms": vms.get("stdout") or vms.get("message"),
        }

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

    async def _guest_action(self, vmid: str, kind: str, action: str) -> Dict[str, Any]:
        cli = _GUEST_CLI.get(str(kind).lower())
        if cli is None:
            return _err(f"kind must be one of {sorted(_GUEST_CLI)}, got {kind!r}")
        try:
            vmid = _validate_vmid(vmid)
        except ValueError as e:
            return _err(str(e))
        return await self._run([cli, action, vmid])

    async def _reboot_guest(self, vmid: str, kind: str) -> Dict[str, Any]:
        logger.info("Tool reboot_guest: %s %s", kind, vmid)
        return await self._guest_action(vmid, kind, "reboot")

    async def _start_guest(self, vmid: str, kind: str) -> Dict[str, Any]:
        logger.info("Tool start_guest: %s %s", kind, vmid)
        return await self._guest_action(vmid, kind, "start")

    async def _stop_guest(self, vmid: str, kind: str) -> Dict[str, Any]:
        logger.info("Tool stop_guest: %s %s", kind, vmid)
        return await self._guest_action(vmid, kind, "stop")

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
