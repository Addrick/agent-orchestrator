# src/self_edit/
"""Self-improvement (DP-227 "fixr") orchestration.

This package owns the logic that sits ABOVE the engine. The "fixr" supervisor
dispatches one detached coding agent per bug; this package prepares a pristine
base clone of derpr's own repository and an isolated `git worktree` per dispatch
so each agent can diagnose -> edit -> test -> commit -> open a PR against the
codebase itself, without colliding with parallel dispatches.

`engine.py` stays transport-only; it merely honours a per-call workspace
override that the dispatch tool points at the agent's worktree.
"""

from src.self_edit.clone_manager import (
    CloneManagerError,
    create_worktree,
    prepare_base_clone,
    remove_worktree,
)

__all__ = [
    "CloneManagerError",
    "create_worktree",
    "prepare_base_clone",
    "remove_worktree",
]
