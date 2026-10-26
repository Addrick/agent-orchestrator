# src/interfaces/infra_panel.py
"""Portal INFRA panel: operator access to the Proxmox tools (DP-439).

The panel is a second *caller* of the existing ``proxmox`` tools, not a second
implementation of them. Every read and every action goes through
``ToolManager.execute_tool`` against the handlers ``ProxmoxIntegration``
registered, so the panel and ``hypr`` share guest resolution, unit discovery,
the :5001 contender logic, the node SSH gate — and its failure modes.

What differs is who approves. A ``hypr`` write is parked by ConfirmationManager
for a human; a panel action *is* the human, authenticated by the operator token
(the control-plane middleware gates every POST). Parking it would ask the
operator to approve their own click. The audit row the adapter writes for each
action is what keeps it on the record.

DP-465 adds the sandbox boxes' network switches the same way: the ``stagr``
tools ``sandbox_net_status`` (read) and ``set_sandbox_net`` (action). A switch
that would cut live connections, or put a box's LAN and WAN on together,
answers ``status: "confirm"`` and changes nothing; the client shows why and
re-sends with ``confirm``.

``offered`` is the exposure policy for this caller (DP-404): the fixed set
below, intersected with what is actually registered. A tool outside it is
refused by ``execute_tool`` itself, so the route cannot be used to reach an
arbitrary handler by naming it.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Optional, Tuple

if TYPE_CHECKING:
    from src.tools.tool_manager import ToolManager

#: Reads the status route gathers. Ungated tools — the same ones a persona may
#: call without a park.
INFRA_READS: Tuple[str, ...] = ("pve_status", "gpu_status", "list_models", "sandbox_net_status")

#: Action → the request fields it accepts. Anything else in the body is refused
#: rather than ignored, so a client bug cannot quietly act on the wrong guest.
#: Guests are addressed by vmid + kind, both read from ``pve_status`` by the
#: client; the handler re-resolves them on the node before acting.
INFRA_ACTIONS: Dict[str, FrozenSet[str]] = {
    "start_guest": frozenset({"vmid", "kind", "name"}),
    "stop_guest": frozenset({"vmid", "kind", "name"}),
    "reboot_guest": frozenset({"vmid", "kind", "name"}),
    "set_active_model": frozenset({"name"}),
    "reboot_node": frozenset(),
    "set_sandbox_net": frozenset({"box", "zone", "state", "confirm"}),
}

INFRA_TOOLS: FrozenSet[str] = frozenset(INFRA_READS) | frozenset(INFRA_ACTIONS)


def _offered(tool_manager: "ToolManager") -> FrozenSet[str]:
    registered = {
        t.get("function", {}).get("name")
        for t in tool_manager.get_tool_definitions()
    }
    return INFRA_TOOLS & frozenset(n for n in registered if n)


def _unwrap(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten execute_tool's ``{'result': …}`` / ``{'error': …}`` envelope into
    the handler's own ``{'status': 'ok'|'error', …}`` shape."""
    if "error" in envelope:
        return {"status": "error", "message": str(envelope["error"])}
    result = envelope.get("result")
    if isinstance(result, dict):
        return result
    return {"status": "error", "message": f"unexpected tool result: {result!r}"}


async def read_status(tool_manager: "ToolManager") -> Dict[str, Any]:
    """Run the reads concurrently; each section succeeds or fails alone."""
    offered = _offered(tool_manager)
    envelopes = await asyncio.gather(*(
        tool_manager.execute_tool(name, offered) for name in INFRA_READS
    ))
    pve, gpu, models, net = (_unwrap(e) for e in envelopes)
    return {"pve": pve, "gpu": gpu, "models": models, "net": net}


def validate_action(body: Any) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    """``(action, args, error)`` for a POST body. Exactly one of action/error is
    set. Values are coerced to strings; the handler validates their content."""
    if not isinstance(body, dict):
        return None, {}, "body must be a JSON object"
    action = body.get("action")
    if not isinstance(action, str) or action not in INFRA_ACTIONS:
        return None, {}, f"action must be one of {sorted(INFRA_ACTIONS)}"
    allowed = INFRA_ACTIONS[action]
    extra = sorted(set(body) - allowed - {"action"})
    if extra:
        return None, {}, f"{action} does not accept {extra}"
    args: Dict[str, Any] = {}
    for key in allowed:
        value = body.get(key)
        if value is None:
            continue
        if key == "confirm" and isinstance(value, bool):
            args[key] = value
            continue
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            return None, {}, f"{key} must be a string"
        args[key] = str(value)
    missing = _missing(action, args)
    if missing:
        return None, {}, missing
    return action, args, None


def _missing(action: str, args: Dict[str, Any]) -> Optional[str]:
    """What a well-typed body still lacks for its action, or None."""
    if action == "set_active_model" and not args.get("name"):
        return "set_active_model requires name"
    if action.endswith("_guest") and not (args.get("vmid") or args.get("name")):
        return f"{action} requires vmid (with kind) or name"
    if action == "set_sandbox_net" and not all(args.get(k) for k in ("box", "zone", "state")):
        return "set_sandbox_net requires box, zone and state"
    return None


async def run_action(
    tool_manager: "ToolManager", action: str, args: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute one validated action through the registered handler."""
    return _unwrap(await tool_manager.execute_tool(action, _offered(tool_manager), **args))
