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


def _caps(*, irreversible: bool = False) -> Dict[str, Any]:
    return {
        "produces_untrusted": False,
        "irreversible": irreversible,
        "locality": "network",
        "sensitivity": "internal",
    }


_GUEST_PARAMS = {
    "type": "object",
    "properties": {
        "vmid": {
            "type": "string",
            "description": "Numeric Proxmox guest id (e.g. \"100\", \"101\").",
        },
        "kind": {
            "type": "string",
            "enum": ["ct", "vm"],
            "description": "\"ct\" for an LXC container (pct) or \"vm\" for a QEMU VM (qm).",
        },
    },
    "required": ["vmid", "kind"],
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
                "Read the Proxmox node's health: uptime plus the list of LXC "
                "containers (pct list) and QEMU VMs (qm list) with their run "
                "state. Use this before acting to find guest ids and see what's up."
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
                "Reboot one VM or container by id. Requires human approval. Get "
                "the id/kind from pve_status."
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
                "Start a stopped VM or container by id. Requires human approval."
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
                "Stop a running VM or container by id. Requires human approval. "
                "This is a hard stop (like power-off), not a graceful shutdown."
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
