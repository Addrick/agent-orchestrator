# src/utils/git_support.py
"""Shared git plumbing for the repos derpr clones on an agent's behalf.

Extracted from `self_edit/clone_manager.py` (DP-314) when the notes repo became
a second clone needing exactly the same two things: a subprocess wrapper that
fails loudly, and a push credential that never lands on disk. Lives in `utils`
so both the self_edit clone manager and the engine's cc provider can reach it
without either importing the other (`utils` is the dependency leaf — see the
import-linter contracts in setup.cfg).

Pure `subprocess` + `git`. No engine coupling, no derpr imports beyond config.
"""

import logging
import subprocess
from typing import List, Optional

logger = logging.getLogger(__name__)

# Generous ceiling for a clone/fetch over the network.
GIT_TIMEOUT_SECONDS = 600

# git runs a `!`-prefixed helper via sh, calling `<helper> get`; it expects
# `username=`/`password=` on stdout. Reading $GH_TOKEN at call time keeps the
# token out of .git/config. Single-quoted so Python interpolates nothing.
_GH_CREDENTIAL_HELPER = (
    '!f() { test "$1" = get && '
    'printf "username=x-access-token\\npassword=%s\\n" "$GH_TOKEN"; }; f'
)


class GitError(RuntimeError):
    """Raised when a git invocation fails, times out, or git is missing."""


def run_git(args: List[str], cwd: Optional[str] = None) -> str:
    """Run a git command, returning stdout. Raises `GitError` on failure
    (non-zero exit, missing binary, or timeout) with a clear message."""
    cmd = ["git", *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as e:
        raise GitError("git binary not found on PATH.") from e
    except subprocess.TimeoutExpired as e:
        raise GitError(
            f"git {' '.join(args)} timed out after {GIT_TIMEOUT_SECONDS}s."
        ) from e
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise GitError(
            f"git {' '.join(args)} failed (exit {proc.returncode}): {stderr}"
        )
    return proc.stdout


def configure_push_auth(repo_dir: str) -> None:
    """Configure git so an agent's `git push` / `gh pr create` authenticates to
    GitHub without writing any secret to disk.

    We install an inline shell credential helper that emits ``x-access-token``
    plus ``$GH_TOKEN`` (read from the environment) on git's ``get`` request. The
    token therefore never lands in `.git/config`; only the helper *script* is
    stored. Shared `.git/config` => every worktree inherits it. Idempotent:
    re-running rewrites the same value.

    We deliberately do NOT use `gh auth git-credential`: the deployed chatbot
    container ships git but not the `gh` CLI, so that helper fails
    (`gh: not found`) and every push dies with "could not read Username for
    https://github.com". The inline helper needs only `git` + `sh` (both
    present) and the ``GH_TOKEN`` already inherited by the child.

    Non-fatal on failure: the agent can still diagnose/commit, just not push.
    """
    try:
        run_git(
            ["config", "credential.https://github.com.helper", _GH_CREDENTIAL_HELPER],
            cwd=repo_dir,
        )
    except GitError as e:
        logger.warning("Could not configure GitHub push credential helper: %s", e)
