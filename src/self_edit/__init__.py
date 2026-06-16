# src/self_edit/
"""Self-improvement (DP-227 "fixr") orchestration.

This package owns the logic that sits ABOVE the engine: it prepares a pristine,
per-run-refreshed clone of derpr's own repository so a `self_edit` persona can
diagnose -> edit -> test -> commit -> open a PR against the codebase itself.

`engine.py` stays transport-only; it merely honours a per-call workspace
override that this package injects into the engine `config` dict.
"""

from src.self_edit.clone_manager import (
    CloneManagerError,
    prepare_fixr_workspace,
)

__all__ = ["CloneManagerError", "prepare_fixr_workspace"]
