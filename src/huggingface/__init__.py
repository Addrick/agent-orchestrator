"""HuggingFace model provisioning (DP-265).

Read-only Hub API searches plus a parked ``install_model`` that provisions a
gguf onto the pve node so it *becomes* a valid ``set_active_model`` target.
"""

from src.huggingface.integration import HuggingFaceIntegration

__all__ = ["HuggingFaceIntegration"]
