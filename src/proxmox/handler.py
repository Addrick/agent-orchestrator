"""Tool handlers for the Proxmox management service (DP-262).

Eight tools behind the ``proxmox`` service binding:

- ``pve_status``      (read):  node uptime + `pct list` + `qm list`, both raw and
  parsed into a structured guest inventory (DP-327).
- ``list_models``     (read):  koboldcpp units discovered on the GPU CT + which
  one is active (DP-332).
- ``gpu_status``      (read):  live VRAM total/used/free per card, straight from
  the GPU CT's sysfs (DP-332).
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
unit names matching ``_UNIT_RE`` — the property the whole transport's safety
argument rests on.

DP-332 — the koboldcpp units are **discovered**, not configured. The box is the
authority on what it contains; the old ``PVE_MODEL_UNITS`` map asserted it and
drifted silently in both directions (DP-329: the unit actually holding :5001 was
unmapped, so ``list_models`` reported every model inactive and no swap could
succeed). Discovery costs one property the map gave for free — the unit names
are now *remote* values that get sent back over SSH — which is why every
discovered name must match ``_UNIT_RE`` before it is used for anything.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from config import global_config
from src.proxmox.ssh import SSHError, SSHRunner

if TYPE_CHECKING:
    from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)

#: Accepted guest kinds → the Proxmox CLI that manages them.
_GUEST_CLI = {"ct": "pct", "vm": "qm"}

#: A koboldcpp model unit on the GPU CT, and the friendly name inside it
#: (``koboldcpp-<name>.service`` → ``<name>``). Two jobs, both load-bearing:
#:
#: 1. It is the anchor for ``set_active_model``'s "disable every *other* unit".
#:    That set used to be a config map and is now whatever the box reports, so a
#:    loose match here would disable something unrelated on the CT. This is the
#:    one place where discovery is more dangerous than the map was.
#: 2. Its character class is deliberately narrower than ``ssh._FORBIDDEN``
#:    permits, because a discovered unit name is a remote value that we then send
#:    back over SSH as an argv element.
_UNIT_RE = re.compile(r"koboldcpp-([A-Za-z0-9._-]+)\.service")

#: A DRM card directory under /sys/class/drm — ``card1``, not the connector
#: nodes (``card1-DP-1``) or ``renderD128`` that sit beside it.
_CARD_RE = re.compile(r"card\d+")

_MIB = 1024 * 1024


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
        manager.register("gpu_status", self._gpu_status)
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
        if not self._enabled():
            # Short-circuit before the inventory. Without this the two listings
            # each fail with the disabled error and resolution reports "no guest
            # named X" — a false statement about the node, on the path that is
            # about to power something off.
            return _err(
                "Proxmox tools are disabled (set PVE_TOOLS_ENABLED=true and mount "
                "the pve SSH key to enable)."
            )
        kind_in = (str(kind).strip().lower() or None) if kind else None
        if kind_in is not None and kind_in not in _GUEST_CLI:
            return _err(f"kind must be one of {sorted(_GUEST_CLI)}, got {kind!r}")

        vmid_in = str(vmid).strip() if vmid is not None else ""
        name_in = str(name).strip() if name is not None else ""

        if vmid_in:
            return await self._resolve_by_vmid(vmid_in, name_in, kind_in)

        if not name_in:
            return _err("pass either a guest name or a numeric vmid.")
        return await self._resolve_by_name(name_in, kind_in)

    async def _resolve_by_vmid(
        self, vmid: str, name: str, kind: Optional[str]
    ) -> Dict[str, Any]:
        """Address by numeric id, cross-checking a ``name`` if one came too."""
        try:
            resolved_vmid = _validate_vmid(vmid)
        except ValueError as e:
            return _err(str(e))
        if kind is None:
            return _err(
                "kind is required when addressing a guest by vmid "
                f"(got vmid={resolved_vmid!r}); pass kind=\"ct\" or \"vm\", or "
                "address the guest by name instead."
            )
        if not name:
            return {"status": "ok", "vmid": resolved_vmid, "kind": kind, "name": None}
        # Both address forms given. The vmid decides what runs, so an unchecked
        # `name` would be echoed into `target` — and into the approval prompt —
        # describing a guest we are not touching. That is the audit record lying,
        # which is worse than either address form being wrong on its own. Resolve
        # the name too and refuse unless the two agree.
        by_name = await self._resolve_by_name(name, kind)
        if by_name.get("status") != "ok":
            return by_name
        if by_name["vmid"] != resolved_vmid:
            return _err(
                f"name and vmid disagree: {name!r} is {by_name['kind']} "
                f"{by_name['vmid']}, not {kind} {resolved_vmid}. Re-issue with "
                "just one of them."
            )
        return by_name

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
        if inventory["errors"]:
            # A listing failed, so "not found" is not something we know — the
            # guest may be sitting in the half we could not read. On a power
            # path, unknown and absent are different answers and must not be
            # collapsed: reporting "no such guest" invites the model to go
            # looking for a near neighbour to act on instead.
            return _err(
                f"could not confirm whether a guest named {name!r} exists — the "
                f"node listing failed: {inventory['errors']}. Guests read "
                f"successfully: {known}. Fix the listing before acting."
            )
        msg = f"no guest named {name!r}"
        if kind is not None:
            msg += f" of kind {kind!r}"
        msg += f"; known guests: {known}" if known else "; the node reported no guests"
        return _err(msg)

    # -- model discovery (DP-332) --------------------------------------------

    async def _discover_units(self, vmid: str) -> Dict[str, Any]:
        """Every ``koboldcpp-<name>.service`` that exists on the GPU CT *now*.

        Returns ``{"status": "ok", "units": {name: unit}}`` or an error dict.

        The listing is deliberately **unfiltered**, with the ``koboldcpp-``
        anchoring done here in Python rather than as a ``'koboldcpp-*.service'``
        argument: ``ssh._reject_bad_args`` forbids ``*`` outright, and pushing
        the glob to the far side would mean trusting a remote shell to expand it
        the way we meant. A literal prefix/suffix match (``_UNIT_RE``) has no
        such ambiguity — which matters, because this set is exactly what
        ``set_active_model`` disables.

        A bare ``koboldcpp.service`` with no ``-<name>`` does not match, so it is
        never listed *and never disabled*. That is the safe direction: it cannot
        be selected, so it is never the thing we take ``:5001`` down for.
        """
        res = await self._run([
            "pct", "exec", vmid, "--",
            "systemctl", "list-unit-files", "--type=service",
            "--no-legend", "--no-pager",
        ])
        if res.get("status") != "ok":
            detail = " — ".join(
                p for p in (res.get("message"), res.get("stderr")) if p
            ) or "unknown error"
            return _err(f"could not list koboldcpp units on CT {vmid}: {detail}")
        units: Dict[str, str] = {}
        for line in (res.get("stdout") or "").splitlines():
            tokens = line.split()
            if not tokens:
                continue
            match = _UNIT_RE.fullmatch(tokens[0])
            if match:
                units[match.group(1)] = tokens[0]
        return {"status": "ok", "units": units}

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
        if not self._enabled():
            return _err("Proxmox tools are disabled (set PVE_TOOLS_ENABLED=true).")
        vmid = global_config.PVE_MODEL_HOST_VMID
        # Discovered, not configured (DP-332): a unit installed on the box since
        # the last deploy is listed, and a unit that has been removed simply is
        # not. Neither direction needs anyone to remember to edit config.
        discovered = await self._discover_units(vmid)
        if discovered.get("status") != "ok":
            return discovered
        # Only surface models whose gguf is actually on disk (immediately
        # loadable). Units whose model file is missing are omitted — enabling one
        # would fail to start and take :5001 down (DP-264).
        models: List[Dict[str, Any]] = []
        for name, unit in sorted(discovered["units"].items()):
            if not await self._model_present(vmid, unit):
                continue
            state_res = await self._run(
                ["pct", "exec", vmid, "--", "systemctl", "is-active", unit]
            )
            state = (state_res.get("stdout") or state_res.get("message") or "unknown").strip()
            models.append({"name": name, "unit": unit, "state": state})
        return {"status": "ok", "host_vmid": vmid, "models": models}

    async def _gpu_status(self) -> Dict[str, Any]:
        """VRAM total/used/free per card on the GPU CT, read live from sysfs.

        A read, not an assertion. A card's usable VRAM is site-specific and must
        not be baked into a persona prompt or into config (DP-328) — the budget
        *formula* is portable, the numbers are not. Reading them per call is also
        the only answer that stays true when another process is already holding
        VRAM, which is the case this exists to inform.

        The card index is discovered for the same reason the units are: this box
        exposes ``card1``, not ``card0``, so a hardcoded index reads a path that
        does not exist.

        Never raises: a dead CT, a missing sysfs path, or a non-amdgpu card all
        come back as an error dict or an entry in ``errors``.
        """
        logger.info("Tool gpu_status")
        if not self._enabled():
            return _err("Proxmox tools are disabled (set PVE_TOOLS_ENABLED=true).")
        vmid = global_config.PVE_MODEL_HOST_VMID
        listing = await self._run(["pct", "exec", vmid, "--", "ls", "/sys/class/drm"])
        if listing.get("status") != "ok":
            detail = " — ".join(
                p for p in (listing.get("message"), listing.get("stderr")) if p
            ) or "unknown error"
            return _err(f"could not read /sys/class/drm on CT {vmid}: {detail}")
        # /sys/class/drm also holds connector nodes (card1-DP-1) and renderD128;
        # only the bare cardN directories carry the amdgpu mem_info files.
        cards = sorted(
            n for n in (listing.get("stdout") or "").split() if _CARD_RE.fullmatch(n)
        )
        if not cards:
            return _err(
                f"no DRM card found on CT {vmid}: /sys/class/drm lists no cardN "
                "entry (is the GPU passed through to this container?)"
            )
        gpus: List[Dict[str, Any]] = []
        errors: List[str] = []
        for card in cards:
            base = f"/sys/class/drm/{card}/device"
            res = await self._run([
                "pct", "exec", vmid, "--", "cat",
                f"{base}/mem_info_vram_total", f"{base}/mem_info_vram_used",
            ])
            values = [ln.strip() for ln in (res.get("stdout") or "").splitlines()]
            if (
                res.get("status") != "ok"
                or len(values) != 2
                or not all(v.isdigit() for v in values)
            ):
                # Only amdgpu exposes mem_info_vram_*. Report the odd card rather
                # than failing the whole call — one unreadable device must not
                # hide the card we actually care about.
                detail = res.get("stderr") or res.get("message") or "unparseable output"
                errors.append(f"{card}: no readable mem_info_vram_* ({detail})")
                continue
            total, used = int(values[0]), int(values[1])
            # free is floored from the byte difference, not from
            # total_mib - used_mib, so the three can disagree by 1 MiB. That is
            # deliberate: overstating free VRAM is the direction that gets a
            # context size picked too large.
            gpus.append({
                "card": card,
                "vram_total_mib": total // _MIB,
                "vram_used_mib": used // _MIB,
                "vram_free_mib": (total - used) // _MIB,
            })
        if not gpus:
            return _err(f"no VRAM readable on CT {vmid}: {errors}")
        result: Dict[str, Any] = {"status": "ok", "host_vmid": vmid, "gpus": gpus}
        if errors:
            result["errors"] = errors
        return result

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
        if not self._enabled():
            return _err("Proxmox tools are disabled (set PVE_TOOLS_ENABLED=true).")
        vmid = global_config.PVE_MODEL_HOST_VMID
        discovered = await self._discover_units(vmid)
        if discovered.get("status") != "ok":
            return discovered
        units = discovered["units"]
        # `name` is a lookup key and nothing else — what reaches SSH is the
        # discovered unit, which matched `_UNIT_RE`. Nothing the model typed ever
        # becomes an argv element, which is what keeps this tool's
        # `exfil_capable: False` claim true now that the units are discovered
        # rather than pinned in config.
        target = units.get(name)
        if target is None:
            return _err(
                f"unknown model {name!r}; CT {vmid} has: {sorted(units)}"
            )
        # Pre-flight: never disable the running model to enable a unit that can't
        # start. If the target's gguf isn't on disk, refuse and leave :5001 as-is.
        if not await self._model_present(vmid, target):
            return _err(
                f"model file for {name!r} (unit {target}) is not on disk; "
                "not switching — current model left running."
            )
        # Disable every other *discovered* koboldcpp unit (all bind :5001 — only
        # one may run), then enable+start the target. Idempotent: re-selecting
        # the active model just re-enables it. The set is bounded by `_UNIT_RE`,
        # so nothing outside `koboldcpp-<name>.service` can land in this loop —
        # that anchoring is the entire safety argument for disabling a discovered
        # set rather than a config-pinned one.
        for other_name, other_unit in sorted(units.items()):
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
