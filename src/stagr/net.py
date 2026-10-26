"""The sandbox boxes' network switches (DP-465).

Each box (codr-vm, stagr-vm) has two switches, both off by default:

- **LAN**: the box's firewall group on host `derpr`, flipped by `agent-net`
  over the *second* node key (``PVE_NET_SSH_*``), whose wrapper admits nothing
  else. Off cuts the box's open LAN flows.
- **WAN**: the box's own tinyproxy instance in CT110, started and stopped by
  `derpr-egress` over the first node key. Stopping it cuts every tunnel and
  forgets the hosts approved for it.

Both go through ``run_node_command``, so ``PVE_TOOLS_ENABLED`` and the SSH cap
apply. This module only talks to the two node verbs; who may flip what (the
operator's panel, stagr's parked ``request_network``) is the caller's business.

Turning a switch off while connections are open, or turning the second switch
of a box on, needs ``confirm``: without it the call answers ``status:
"confirm"`` with what would be cut or the warning, and changes nothing.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional

from src.proxmox.ssh import SSHRunner, run_node_command

EGRESS_VERB = "/usr/local/sbin/derpr-egress"
AGENT_NET = "/usr/local/sbin/agent-net"
BOXES = ("codr", "stagr")
ZONES = ("lan", "wan")
# The same gate as the node verb and egress-ctl. The last label starts with a
# letter, so no IP literal is accepted.
HOST_RE = re.compile(r"^([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]([a-z0-9-]{0,61}[a-z0-9])?$")
_SWITCH_RE = re.compile(r"^(lan|wan): (on|off)$", re.M)


def _error(message: str) -> Dict[str, Any]:
    return {"status": "error", "message": message}


def normalize_host(host: str) -> Optional[str]:
    host = (host or "").strip().lower().rstrip(".")
    return host if len(host) <= 253 and HOST_RE.match(host) else None


def _lines(result: Dict[str, Any]) -> List[str]:
    return [ln for ln in str(result.get("stdout") or "").splitlines() if ln.strip()]


class SandboxNet:
    def __init__(self, pve: SSHRunner | None = None, net: SSHRunner | None = None) -> None:
        # Built lazily from global_config when not injected, so a test that
        # patches the config after import still gets its own host.
        self._pve = pve
        self._net = net

    def _pve_runner(self) -> SSHRunner:
        return self._pve or SSHRunner()

    def _net_runner(self) -> SSHRunner:
        if self._net is None:
            from config import global_config
            return SSHRunner(host=global_config.PVE_NET_SSH_HOST, key_path=global_config.PVE_NET_SSH_KEY)
        return self._net

    async def _egress(self, box: str, *args: str) -> Dict[str, Any]:
        return await run_node_command(self._pve_runner(), [EGRESS_VERB, box, *args], exit_label="derpr-egress")

    async def _agent_net(self, box: str, verb: str) -> Dict[str, Any]:
        return await run_node_command(self._net_runner(), [AGENT_NET, box, "lan", verb], exit_label="agent-net")

    async def _switch(self, box: str, zone: str) -> Dict[str, Any]:
        """``{"status": "ok", "on": bool}`` or the error."""
        res = await (self._agent_net(box, "status") if zone == "lan" else self._egress(box, "status"))
        if res.get("status") == "error":
            return res
        m = _SWITCH_RE.search(str(res.get("stdout") or ""))
        if not m or m.group(1) != zone:
            return _error(f"unexpected {zone} status from the node: {res.get('stdout')!r}")
        return {"status": "ok", "on": m.group(2) == "on"}

    async def _conns(self, box: str, zone: str) -> Dict[str, Any]:
        res = await (self._agent_net(box, "conns") if zone == "lan" else self._egress(box, "conns"))
        if res.get("status") == "error":
            return res
        return {"status": "ok", "connections": _lines(res)}

    # -- reads -------------------------------------------------------------------

    async def status(self, box: str, *, detail: bool = True) -> Dict[str, Any]:
        """Both switches of one box, its allowlist, and (``detail``) its live
        connections and recent proxy log. Each part fails on its own."""
        if box not in BOXES:
            return _error(f"no such box: {box!r} (one of {', '.join(BOXES)}).")
        lan, wan, allow = await asyncio.gather(
            self._switch(box, "lan"), self._switch(box, "wan"), self._egress(box, "list"))
        out: Dict[str, Any] = {
            "status": "ok",
            "box": box,
            "lan": lan.get("on") if lan.get("status") == "ok" else None,
            "wan": wan.get("on") if wan.get("status") == "ok" else None,
            "allowlist": str(allow.get("stdout") or "").strip() if allow.get("status") == "ok" else None,
        }
        errors = {k: r.get("message") for k, r in (("lan", lan), ("wan", wan), ("allowlist", allow))
                  if r.get("status") == "error"}
        if detail:
            lan_c, wan_c, log = await asyncio.gather(
                self._conns(box, "lan"), self._conns(box, "wan"), self._egress(box, "log"))
            out["lan_connections"] = lan_c.get("connections", [])
            out["wan_connections"] = wan_c.get("connections", [])
            out["log"] = _lines(log) if log.get("status") == "ok" else []
            errors.update({k: r.get("message") for k, r in (("lan_connections", lan_c),
                           ("wan_connections", wan_c), ("log", log)) if r.get("status") == "error"})
        if errors:
            out["errors"] = errors
        return out

    # -- writes ------------------------------------------------------------------

    async def _turn_off_check(self, box: str, zone: str) -> Optional[Dict[str, Any]]:
        """The confirm answer (or error) when going off would cut connections."""
        conns = await self._conns(box, zone)
        if conns.get("status") == "error":
            return conns
        if not conns["connections"]:
            return None
        return {"status": "confirm", "box": box, "zone": zone, "connections": conns["connections"],
                "message": f"{box} has {len(conns['connections'])} open {zone.upper()} "
                           "connection(s); turning it off cuts them."}

    async def _turn_on_check(self, box: str, zone: str) -> Optional[Dict[str, Any]]:
        """The confirm answer (or error) when going on would put LAN and WAN on together."""
        other = "wan" if zone == "lan" else "lan"
        sibling = await self._switch(box, other)
        if sibling.get("status") == "error":
            return sibling
        if not sibling["on"]:
            return None
        return {"status": "confirm", "box": box, "zone": zone,
                "message": f"{box}'s {other.upper()} is on: with both on, the box can carry "
                           "anything from the home network to the internet."}

    async def set_switch(self, box: str, zone: str, on: bool, *, confirm: bool = False) -> Dict[str, Any]:
        if box not in BOXES:
            return _error(f"no such box: {box!r} (one of {', '.join(BOXES)}).")
        if zone not in ZONES:
            return _error(f"no such switch: {zone!r} (lan or wan).")
        current = await self._switch(box, zone)
        if current.get("status") == "error":
            return current
        if current["on"] == on:
            return {"status": "ok", "box": box, zone: on, "changed": False}
        if not confirm:
            held = await (self._turn_on_check(box, zone) if on else self._turn_off_check(box, zone))
            if held is not None:
                return held
        verb = "on" if on else "off"
        res = await (self._agent_net(box, verb) if zone == "lan" else self._egress(box, verb))
        if res.get("status") == "error":
            return res
        out: Dict[str, Any] = {"status": "ok", "box": box, zone: on, "changed": True}
        closed = [ln[len("closed "):] for ln in _lines(res) if ln.startswith("closed ")]
        if closed:
            out["closed"] = closed
        return out

    async def allow(self, box: str, host: str) -> Dict[str, Any]:
        if box not in BOXES:
            return _error(f"no such box: {box!r} (one of {', '.join(BOXES)}).")
        h = normalize_host(host)
        if h is None:
            return _error(f"not a host name: {host!r} (no IP addresses, no scheme, port or path).")
        res = await self._egress(box, "allow", h)
        if res.get("status") == "error":
            return res
        return {"status": "ok", "allowed": str(res.get("stdout") or "").strip()}
