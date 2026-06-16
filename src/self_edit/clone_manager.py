# src/self_edit/clone_manager.py
"""Prepare a pristine, per-run clone of derpr's own repo for the fixr persona.

DP-227. The "fixr" self-edit persona routes to the cc-* (Claude Code) engine and
must operate on a real checkout of derpr's source — not the live tree (the live
tree carries the venv-junction footgun and uncommitted WIP) and not an empty
scratch dir. So we keep ONE persistent clone and refresh it to a pristine,
up-to-date base ref at the start of every run:

    git fetch origin
    git reset --hard <base_ref>
    git clean -fdx

This is cheaper than re-cloning per bug while still guaranteeing a clean tree.
On first use (clone dir absent) we `git clone` once. Test-time data the suite
needs (`.env.test`, and any sample db / persona config) is seeded in afterwards
from a known source dir, since `git clean -fdx` wipes gitignored files.

Pure `subprocess` + `git` — no engine coupling. A module-level lock serializes
concurrent calls so two turns never fight over the same working tree. The
caller (orchestration, above engine.py) injects the returned path into the
engine `config` dict as `cc_workspace_override`.
"""

import logging
import os
import shutil
import subprocess
import threading
from typing import List, Optional

from config import global_config

logger = logging.getLogger(__name__)

# Files to seed into the fresh clone after refresh (git clean -fdx removes
# gitignored files, so these must be re-copied every run). Each entry is a
# path relative to the source checkout root; copied only if it exists.
_SEED_FILES = (".env.test", ".env")

# Serializes clone/refresh so concurrent fixr turns don't corrupt the tree.
_clone_lock = threading.Lock()

# Generous ceiling for a clone/fetch over the network.
_GIT_TIMEOUT_SECONDS = 600


class CloneManagerError(RuntimeError):
    """Raised when seeding the fixr workspace fails (clone/fetch/reset/clean)."""


def _run_git(args: List[str], cwd: Optional[str] = None) -> str:
    """Run a git command, returning stdout. Raises CloneManagerError on failure
    (non-zero exit, missing binary, or timeout) with a clear message."""
    cmd = ["git", *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as e:
        raise CloneManagerError("git binary not found on PATH.") from e
    except subprocess.TimeoutExpired as e:
        raise CloneManagerError(
            f"git {' '.join(args)} timed out after {_GIT_TIMEOUT_SECONDS}s."
        ) from e
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise CloneManagerError(
            f"git {' '.join(args)} failed (exit {proc.returncode}): {stderr}"
        )
    return proc.stdout


def _derive_repo_url(source_root: str) -> str:
    """Resolve the clone source URL: explicit config wins, else the running
    checkout's `origin` remote."""
    if global_config.CC_FIXR_REPO_URL:
        return str(global_config.CC_FIXR_REPO_URL)
    url = _run_git(["remote", "get-url", "origin"], cwd=source_root).strip()
    if not url:
        raise CloneManagerError(
            "Could not derive repo URL: `git remote get-url origin` was empty "
            "and CC_FIXR_REPO_URL is unset."
        )
    return url


def _seed_support_files(clone_dir: str, source_root: str) -> None:
    """Copy gitignored test-support files (.env.test, .env, ...) into the fresh
    clone. Missing source files are skipped silently — a live deploy may not
    have a local .env.test, and the suite auto-skips when creds are absent."""
    for rel in _SEED_FILES:
        src = os.path.join(source_root, rel)
        if os.path.isfile(src):
            dst = os.path.join(clone_dir, rel)
            os.makedirs(os.path.dirname(dst) or clone_dir, exist_ok=True)
            shutil.copy2(src, dst)
            logger.debug("Seeded %s into fixr clone.", rel)


def prepare_fixr_workspace(
    clone_dir: Optional[str] = None,
    *,
    repo_url: Optional[str] = None,
    base_ref: Optional[str] = None,
    source_root: Optional[str] = None,
) -> str:
    """Ensure a pristine, up-to-date clone of derpr exists at `clone_dir` and
    return its absolute path.

    - If the clone dir is absent (no `.git`), clone the repo there.
    - Otherwise refresh it in place: `git fetch origin`,
      `git reset --hard <base_ref>`, `git clean -fdx`.
    Then seed gitignored test-support files (.env.test, ...).

    Defaults are read from `global_config.CC_FIXR_*`. `source_root` is the
    running checkout used to derive the origin URL and to copy seed files from
    (defaults to PROJECT_ROOT). Serialized by a module-level lock so concurrent
    turns never share a half-built tree.

    Raises `CloneManagerError` on any git failure.
    """
    clone_dir = clone_dir or global_config.CC_FIXR_CLONE_DIR
    base_ref = base_ref or global_config.CC_FIXR_BASE_REF
    src_root = source_root or str(global_config.PROJECT_ROOT)
    clone_dir = os.path.abspath(clone_dir)

    with _clone_lock:
        git_dir = os.path.join(clone_dir, ".git")
        if not os.path.isdir(git_dir):
            # Fresh clone. Resolve URL (explicit arg > config > origin).
            url = repo_url or _derive_repo_url(src_root)
            parent = os.path.dirname(clone_dir) or "."
            os.makedirs(parent, exist_ok=True)
            # Clean up any stale, non-repo dir at the target so clone succeeds.
            if os.path.exists(clone_dir):
                shutil.rmtree(clone_dir, ignore_errors=True)
            logger.info("Cloning fixr workspace from %s into %s", url, clone_dir)
            _run_git(["clone", url, clone_dir])
            _run_git(["fetch", "origin"], cwd=clone_dir)
            _run_git(["reset", "--hard", base_ref], cwd=clone_dir)
        else:
            # Refresh existing clone to a pristine base ref.
            logger.info("Refreshing fixr workspace at %s to %s", clone_dir, base_ref)
            _run_git(["fetch", "origin"], cwd=clone_dir)
            _run_git(["reset", "--hard", base_ref], cwd=clone_dir)
            _run_git(["clean", "-fdx"], cwd=clone_dir)

        _seed_support_files(clone_dir, src_root)
        return clone_dir
