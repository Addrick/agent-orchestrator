"""Proxmox management tools (DP-262).

Bot-callable ServiceIntegration that drives the Proxmox node over SSH: node and
guest (VM/CT) power ops plus swapping the active koboldcpp model on the GPU
container's :5001. Destructive tools are ``is_write`` → parked for human
confirmation by the ConfirmationManager.
"""

from src.proxmox.integration import ProxmoxIntegration

__all__ = ["ProxmoxIntegration"]
