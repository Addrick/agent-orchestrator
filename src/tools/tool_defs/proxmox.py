"""Proxmox management tools (service_binding: proxmox, DP-262).

Node/guest power ops + koboldcpp model swap on :5001, executed over SSH to the
pve node. Destructive tools are ``is_write: True`` so the ConfirmationManager
parks them for human approval regardless of persona execution mode;
``reboot_node`` is additionally ``irreversible``. Read tools (`pve_status`,
`list_models`) are ungated.

All results originate from infra we control (not attacker text) →
``produces_untrusted: False``; ``locality: "network"`` (SSH to the node);
``sensitivity: "internal"``.
"""

from typing import Any, Dict, List


def _caps(*, irreversible: bool = False, exfil_capable: bool = True) -> Dict[str, Any]:
    caps: Dict[str, Any] = {
        "produces_untrusted": False,
        "irreversible": irreversible,
        "locality": "network",
        "sensitivity": "internal",
    }
    if not exfil_capable:
        caps["exfil_capable"] = False
    return caps


_GUEST_PARAMS = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": (
                "Guest hostname as pve_status reports it (e.g. \"docker\", \"gpu\"), "
                "case-insensitive. Preferred over vmid — it needs no kind and reads "
                "back clearly in the approval prompt."
            ),
        },
        "vmid": {
            "type": "string",
            "description": (
                "Numeric Proxmox guest id (e.g. \"100\", \"101\"). Alternative to "
                "name; requires kind."
            ),
        },
        "kind": {
            "type": "string",
            "enum": ["ct", "vm"],
            "description": (
                "\"ct\" for an LXC container (pct) or \"vm\" for a QEMU VM (qm). "
                "Required with vmid. Optional with name — omit it unless the same "
                "name exists as both a container and a VM."
            ),
        },
    },
    # Neither is individually required, but one of them is: enforced in the
    # handler rather than the schema, because providers vary in how (and whether)
    # they honour anyOf/oneOf in a function schema, and a constraint the provider
    # silently drops is worse than no constraint at all — it reads as enforced.
    "required": [],
}


PROXMOX_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "is_write": False,
        "service_binding": "proxmox",
        "capabilities": _caps(),
        "function": {
            "name": "pve_status",
            "description": (
                "Audit the Proxmox node: uptime plus every guest as structured "
                "data (vmid, name, kind ct/vm, status, lock), with the raw "
                "pct list / qm list text alongside. Use this to answer what is "
                "running and to find a guest's name or id before acting."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "is_write": False,
        "service_binding": "proxmox",
        "capabilities": _caps(),
        "function": {
            "name": "list_models",
            "description": (
                "List the koboldcpp models configured for the GPU container's "
                ":5001 endpoint and which one is currently active. Only one model "
                "runs at a time. Use before set_active_model to see the choices."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "proxmox",
        "capabilities": _caps(irreversible=True),
        "function": {
            "name": "reboot_node",
            "description": (
                "Reboot the Proxmox HOST (the metal). This takes down every VM "
                "and container on it. Requires human approval. Use only when the "
                "node itself is wedged — reboot a single guest instead when you can."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "proxmox",
        "capabilities": _caps(),
        "function": {
            "name": "reboot_guest",
            "description": (
                "Reboot one VM or container. Address it by name (preferred) or by "
                "vmid + kind. Requires human approval. Get names/ids from pve_status."
            ),
            "parameters": _GUEST_PARAMS,
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "proxmox",
        "capabilities": _caps(),
        "function": {
            "name": "start_guest",
            "description": (
                "Start a stopped VM or container. Address it by name (preferred) "
                "or by vmid + kind. Requires human approval."
            ),
            "parameters": _GUEST_PARAMS,
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "proxmox",
        "capabilities": _caps(),
        "function": {
            "name": "stop_guest",
            "description": (
                "Stop a running VM or container. Address it by name (preferred) or "
                "by vmid + kind. Requires human approval. This is a hard stop (like "
                "power-off), not a graceful shutdown."
            ),
            "parameters": _GUEST_PARAMS,
        },
    },
    {
        "type": "function",
        "is_write": True,
        "service_binding": "proxmox",
        # exfil_capable=False: arg is a name from a fixed config map, no payload
        # can ride out over the SSH → not a data-exfil vector, so it must never
        # trip the exfil-composition rules. Destruction risk is nil (reversible
        # model swap) and any write still parks for confirmation.
        "capabilities": _caps(exfil_capable=False),
        "function": {
            "name": "set_active_model",
            "description": (
                "Swap which koboldcpp model serves :5001 on the GPU container. "
                "Disables the current model's service and enables+starts the "
                "target's (only one can run at a time). Requires human approval. "
                "Pass a name from list_models."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Friendly model name from list_models (e.g. \"fable\", \"gemma\").",
                    },
                },
                "required": ["name"],
            },
        },
    },
]
