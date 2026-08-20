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
succeed). Discovery costs two properties the map gave for free:

1. The unit names are now *remote* values that get sent back over SSH, which is
   why every discovered name must match ``_UNIT_RE`` before it is used as an
   argv element. (A discovered ``--model`` path is a remote value too, and is
   not pattern-checked — ``ssh.py``'s metacharacter guard is what stands behind
   that one.)
2. The set is no longer curated, so "every koboldcpp unit on the box" is not the
   same set as "every unit competing for :5001". ``set_active_model`` disables
   by parsed ``--port``, not by name — see ``_port_contenders``.
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
#: 1. It is the outer bound on ``set_active_model``'s "disable every *other*
#:    unit". That set used to be a config map and is now whatever the box
#:    reports, so a loose match here would disable something unrelated on the
#:    CT. It is only the outer bound, though: a genuine ``koboldcpp-<name>``
#:    unit serving some other port matches this perfectly and still must not be
#:    touched, which is ``_port_contenders``' job.
#: 2. Its character class is deliberately narrower than ``ssh._FORBIDDEN``
#:    permits, because a discovered unit name is a remote value that we then send
#:    back over SSH as an argv element.
_UNIT_RE = re.compile(r"koboldcpp-([A-Za-z0-9._-]+)\.service")

#: koboldcpp's own default listen port. A unit whose ExecStart names no
#: ``--port`` binds this one, so the default carries as much weight as a parsed
#: value: "no --port" means "contends for :5001", not "harmless".
_KCPP_PORT = "5001"

#: A DRM card directory under /sys/class/drm — ``card1``, not the connector
#: nodes (``card1-DP-1``) or ``renderD128`` that sit beside it.
_CARD_RE = re.compile(r"card\d+")

#: Cap on SSH calls in flight at once. Every ``_run`` is a fresh ``ssh`` process
#: doing a full auth handshake — there is no ControlMaster here — and sshd's
#: default MaxStartups (10:30:100) begins randomly dropping connections past ten
#: unauthenticated ones. The per-unit probes fan out over however many units the
#: box holds, a number DP-332 deliberately stopped bounding in config, so the
#: cap belongs at the transport rather than at each call site.
_MAX_INFLIGHT_SSH = 4

_MIB = 1024 * 1024


def _err(message: str) -> Dict[str, Any]:
    return {"status": "error", "message": message}


def _detail(res: Dict[str, Any]) -> str:
    """The most useful failure text a ``_run`` result carries."""
    return " — ".join(
        p for p in (res.get("message"), res.get("stderr")) if p
    ) or "unknown error"


def _flag_value(tokens: List[str], flag: str) -> Optional[str]:
    """``--flag value`` or ``--flag=value`` from an argv token list, else None.

    systemd hands ExecStart back verbatim, so both spellings turn up on a box
    where anyone has hand-written a unit — and a unit file is exactly what
    DP-332 invites people to hand-write.
    """
    prefix = f"{flag}="
    for i, token in enumerate(tokens):
        if token == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith(prefix):
            return token[len(prefix):]
    return None


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
        self._inflight = asyncio.Semaphore(_MAX_INFLIGHT_SSH)

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
            async with self._inflight:
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
                # `_detail` keeps both halves: `message` says how it failed
                # ("exited 1"), `stderr` says why. Reporting only the first is
                # what makes an inventory failure a mystery in the logs.
                errors.append(f"{kind}: {_detail(res)}")
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
            return _err(
                f"could not list koboldcpp units on CT {vmid}: {_detail(res)}"
            )
        units: Dict[str, str] = {}
        for line in (res.get("stdout") or "").splitlines():
            tokens = line.split()
            if not tokens:
                continue
            match = _UNIT_RE.fullmatch(tokens[0])
            if match:
                units[match.group(1)] = tokens[0]
        return {"status": "ok", "units": units}

    # -- unit inspection helpers ---------------------------------------------

    async def _unit_spec(self, vmid: str, unit: str) -> Dict[str, Any]:
        """One unit's ExecStart, parsed into what the tools act on.

        - ``model``: the ``--model`` gguf path, or None when it names none.
        - ``port``: the ``--port`` it binds, defaulting to ``_KCPP_PORT``.
        - ``readable``: False when the ExecStart could not be read at all.

        ``readable`` exists because "this unit binds 5002" and "we do not know
        what this unit binds" are opposite answers for ``set_active_model`` and
        must not collapse into one another.
        """
        res = await self._run([
            "pct", "exec", vmid, "--",
            "systemctl", "show", unit, "--property=ExecStart", "--value",
        ])
        if res.get("status") != "ok":
            return {"model": None, "port": None, "readable": False}
        toks = (res.get("stdout") or "").split()
        return {
            "model": _flag_value(toks, "--model"),
            "port": _flag_value(toks, "--port") or _KCPP_PORT,
            "readable": True,
        }

    async def _unit_specs(
        self, vmid: str, units: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """``_unit_spec`` for every discovered unit, keyed by model name.

        The reads are independent round trips, so they go out concurrently:
        serialising them makes a call's latency scale with however many units
        the box happens to hold, and DP-332 removed the config map that used to
        bound that number. ``_run``'s semaphore keeps the fan-out inside what
        sshd will accept at once.
        """
        names = sorted(units)
        specs = await asyncio.gather(
            *(self._unit_spec(vmid, units[name]) for name in names)
        )
        return dict(zip(names, specs))

    async def _model_present(
        self, vmid: str, unit: str, spec: Optional[Dict[str, Any]] = None
    ) -> bool:
        """True when the unit's model gguf exists on disk (immediately loadable)."""
        if spec is None:
            spec = await self._unit_spec(vmid, unit)
        path = spec.get("model")
        if not path:
            return False
        res = await self._run(["pct", "exec", vmid, "--", "test", "-f", path])
        return res.get("status") == "ok"

    async def _model_row(
        self, vmid: str, name: str, unit: str, spec: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """One ``list_models`` row, or None when the unit is not a swappable model.

        Two omissions, both deliberate:

        - a unit binding anything other than ``_KCPP_PORT`` is not a model this
          tool can swap to — it is something else the box happens to call
          ``koboldcpp-<name>`` (an embedding or draft server, say). An
          unreadable ExecStart lands here too, with ``port`` None;
        - a unit whose gguf is not on disk would fail to start and take :5001
          down with it (DP-264).
        """
        if spec["port"] != _KCPP_PORT:
            return None
        present, state_res = await asyncio.gather(
            self._model_present(vmid, unit, spec),
            self._run(["pct", "exec", vmid, "--", "systemctl", "is-active", unit]),
        )
        if not present:
            return None
        state = (
            state_res.get("stdout") or state_res.get("message") or "unknown"
        ).strip()
        return {"name": name, "unit": unit, "state": state}

    @staticmethod
    def _port_contenders(
        units: Dict[str, str],
        specs: Dict[str, Dict[str, Any]],
        target: str,
    ) -> Tuple[List[str], List[str]]:
        """The units that must stop before ``target`` can bind :5001.

        Returns ``(contenders, unreadable)`` — the second list is a subset of
        the first, carried separately so the caller can say what it guessed at.

        ``_UNIT_RE`` bounds the discovered set to real koboldcpp units, but that
        was never the dangerous half: a *different* koboldcpp unit on the same
        box — an embedding or draft server on another port, a helper — matches
        the pattern perfectly and has nothing to do with :5001. The config map
        DP-332 deleted could not name one; discovery can, and ``list_models``
        does not show it, so disabling it would be an invisible side effect of
        every swap. The port is what separates them, not the name.

        A unit whose ExecStart could not be read counts as contending. It may be
        the unit holding :5001 right now, and skipping it is how a swap
        "succeeds" while the old model keeps answering (DP-329); stopping a unit
        that turns out to have been irrelevant is the recoverable direction.
        """
        contenders: List[str] = []
        unreadable: List[str] = []
        for other_name, other_unit in sorted(units.items()):
            if other_unit == target:
                continue
            spec = specs[other_name]
            if not spec["readable"]:
                unreadable.append(other_unit)
                contenders.append(other_unit)
            elif spec["port"] == _KCPP_PORT:
                contenders.append(other_unit)
        return contenders, unreadable

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
        # Surface only what `set_active_model` could actually switch to: a unit
        # that binds :5001 and whose gguf is on disk. `_model_row` carries the
        # reasoning for each omission.
        units: Dict[str, str] = discovered["units"]
        specs = await self._unit_specs(vmid, units)
        rows = await asyncio.gather(*(
            self._model_row(vmid, name, units[name], specs[name])
            for name in sorted(units)
        ))
        models = [row for row in rows if row is not None]
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
            return _err(
                f"could not read /sys/class/drm on CT {vmid}: {_detail(listing)}"
            )
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
        specs = await self._unit_specs(vmid, units)
        # Pre-flight: never disable the running model to enable a unit that can't
        # start. If the target's gguf isn't on disk, refuse and leave :5001 as-is.
        if not await self._model_present(vmid, target, specs[name]):
            return _err(
                f"model file for {name!r} (unit {target}) is not on disk; "
                "not switching — current model left running."
            )
        # Stop whatever else is competing for :5001, then enable+start the
        # target. Idempotent: re-selecting the active model just re-enables it.
        contenders, unreadable = self._port_contenders(units, specs, target)
        # Every disable is checked. `systemctl disable --now` over `pct exec` can
        # sit on a D-Bus job for minutes and get SIGTERM'd by PVE_SSH_TIMEOUT
        # first; the unit then keeps :5001, and the follow-up
        # `systemctl enable --now` on the target *still* exits 0, because a
        # Type=simple unit counts as started even though koboldcpp never managed
        # to bind. Returning ok there is exactly the DP-329 silent failure this
        # ticket exists to remove, so a failed disable aborts before the enable.
        failures: List[str] = []
        for other_unit in contenders:
            res = await self._run([
                "pct", "exec", vmid, "--",
                "systemctl", "disable", "--now", other_unit,
            ])
            if res.get("status") != "ok":
                failures.append(f"{other_unit}: {_detail(res)}")
        if failures:
            return _err(
                f"could not stop {len(failures)} unit(s) competing for :5001; not "
                f"switching to {name!r}. Units stopped before the failure stay "
                f"stopped, so :5001 may be serving nothing until a swap "
                f"succeeds. Failures: {'; '.join(failures)}"
            )
        res = await self._run([
            "pct", "exec", vmid, "--",
            "systemctl", "enable", "--now", target,
        ])
        if res.get("status") != "ok":
            return res
        result: Dict[str, Any] = {
            "status": "ok",
            "active_model": name,
            "unit": target,
            "host_vmid": vmid,
        }
        if unreadable:
            result["warnings"] = [
                f"could not read ExecStart for {unit}; stopped it anyway in case "
                "it was holding :5001"
                for unit in unreadable
            ]
        return result
