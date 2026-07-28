# src/self_edit/clone_manager.py
"""Prepare per-bug git worktrees off a pristine base clone of derpr's own repo.

DP-227 (dispatcher model). The "fixr" supervisor dispatches one detached coding
agent per bug. Each agent must operate on a real, isolated checkout of derpr's
source — not the live tree (venv-junction footgun + uncommitted WIP) and not an
empty scratch dir. The model:

    1. ONE persistent BASE CLONE, kept PRISTINE — we only ever `git fetch` it.
       It is never `reset --hard`/`clean`ed while worktrees are attached, so
       parallel dispatches can't corrupt each other.
    2. Per dispatch: `git worktree add worktrees/<bug-id> -b <branch> <base_ref>`
       off the base clone. The worktree is the agent's isolated sandbox.
    3. Each worktree links its `.venv` to the base clone's `.venv` (same pattern
       as the main repo's worktrees) so `pytest` runs without a per-worktree
       install. The base clone carries the venv; worktrees just point at it.

Test-time data the suite needs (`.env.test`) is seeded into each fresh worktree,
since a fresh checkout doesn't carry gitignored files.

Pure `subprocess` + `git` — no engine coupling. A module-level lock serializes
base-clone creation/fetch so concurrent dispatches never race the base tree.
`git worktree add`/`remove` are themselves safe to run concurrently against a
shared base (git locks internally), but we hold the lock across fetch+add to
keep the base ref stable for the duration of a single dispatch.

The caller (the dispatch tool, above engine.py) passes the returned worktree
path into the engine `config` dict as `cc_workspace_override`.
"""

import logging
import os
import shutil
import subprocess
import threading
from typing import List, Optional

from config import global_config
from src.utils.git_support import GitError, configure_push_auth, run_git
from src.utils.notes_workspace import (
    NOTES_LINK_NAME,
    link_notes_into,
    prepare_notes_clone,
)

logger = logging.getLogger(__name__)

# Files to seed into each fresh worktree (a clean checkout doesn't carry
# gitignored files). Each entry is a path relative to the source checkout root;
# copied only if it exists.
#
# Deliberately ONLY the test-creds file (.env.test). The live `.env` holds
# production secrets (provider API keys, the Zammad token, GH_TOKEN); copying
# it into the cc-* (yolo) worktree would let the dispatched claude read it
# straight off disk — exfiltrating those secrets into the model context,
# *outside* derpr's egress scrubber. GH_TOKEN is supplied to the child via the
# inherited environment, not via a seeded file (see CC_FIXR_* in global_config).
_SEED_FILES = (".env.test",)

# Serializes base clone creation/fetch so concurrent dispatches don't race.
_clone_lock = threading.Lock()


class CloneManagerError(RuntimeError):
    """Raised when preparing the base clone or a worktree fails."""


def _run_git(args: List[str], cwd: Optional[str] = None) -> str:
    """Thin adapter over `utils.git_support.run_git`, re-raising as this
    module's error type. The plumbing moved to `utils` in DP-314 so the notes
    clone could share it; callers here still see `CloneManagerError`."""
    try:
        return run_git(args, cwd=cwd)
    except GitError as e:
        raise CloneManagerError(str(e)) from e


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


def _seed_support_files(target_dir: str, source_root: str) -> None:
    """Copy gitignored test-support files (.env.test, ...) into a fresh worktree.
    Never the live .env — see _SEED_FILES. Missing source files are skipped
    silently — a live deploy may not have a local .env.test, and the suite
    auto-skips when creds are absent."""
    for rel in _SEED_FILES:
        src = os.path.join(source_root, rel)
        if os.path.isfile(src):
            dst = os.path.join(target_dir, rel)
            os.makedirs(os.path.dirname(dst) or target_dir, exist_ok=True)
            shutil.copy2(src, dst)
            logger.debug("Seeded %s into worktree.", rel)


def _refuses_live_tree(target: str, src_abs: str) -> bool:
    """True if `target` is, or contains, the live source tree `src_abs`."""
    try:
        # ValueError when on different drives/roots (Windows) → no containment.
        common = os.path.commonpath([target, src_abs])
    except ValueError:
        return False
    return target == src_abs or common == target


def _link_venv(worktree_dir: str, base_clone_dir: str) -> None:
    """Point the worktree's `.venv` at the base clone's `.venv` so `pytest` runs
    without a per-worktree install (same pattern as the main repo's worktrees).
    No-op when the base clone has no `.venv` yet — the agent just won't be able
    to run the suite, which is advisory anyway (CI re-runs it)."""
    base_venv = os.path.join(base_clone_dir, ".venv")
    if not os.path.isdir(base_venv):
        logger.debug("Base clone has no .venv; skipping worktree venv link.")
        return
    link = os.path.join(worktree_dir, ".venv")
    if os.path.exists(link) or os.path.islink(link):
        return
    try:
        if os.name == "nt":
            # Windows: directory junction (symlink needs elevation).
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", link, base_venv],
                check=True, capture_output=True, text=True,
            )
        else:
            os.symlink(base_venv, link, target_is_directory=True)
    except (OSError, subprocess.CalledProcessError) as e:
        # Non-fatal: the agent can still edit + commit, just not run the suite.
        logger.warning("Could not link worktree .venv -> %s: %s", base_venv, e)


def prepare_base_clone(
    clone_dir: Optional[str] = None,
    *,
    repo_url: Optional[str] = None,
    source_root: Optional[str] = None,
) -> str:
    """Ensure the pristine base clone exists at `clone_dir` and is fetched
    up-to-date. Returns its absolute path.

    - First use (no `.git`): `git clone` then `git fetch origin`.
    - Subsequent: `git fetch origin` ONLY. The base clone is never
      reset/cleaned here — worktrees branch off it and own their own state, so
      mutating the base would corrupt in-flight dispatches.

    Defaults from `global_config.CC_FIXR_*`. Serialized by a module-level lock.
    Raises `CloneManagerError` on any git failure.
    """
    clone_dir = clone_dir or global_config.CC_FIXR_CLONE_DIR
    src_root = source_root or str(global_config.PROJECT_ROOT)
    clone_dir = os.path.abspath(clone_dir)
    src_abs = os.path.abspath(src_root)

    if _refuses_live_tree(clone_dir, src_abs):
        raise CloneManagerError(
            f"Refusing to use {clone_dir} as the base clone dir: it is, or "
            f"contains, the live source tree {src_abs}. Point CC_FIXR_CLONE_DIR "
            "at a dedicated path."
        )

    with _clone_lock:
        git_dir = os.path.join(clone_dir, ".git")
        if not os.path.isdir(git_dir):
            url = repo_url or _derive_repo_url(src_root)
            parent = os.path.dirname(clone_dir) or "."
            os.makedirs(parent, exist_ok=True)
            # Clean up any stale, non-repo dir at the target so clone succeeds.
            if os.path.exists(clone_dir):
                shutil.rmtree(clone_dir, ignore_errors=True)
            logger.info("Cloning fixr base from %s into %s", url, clone_dir)
            _run_git(["clone", url, clone_dir])
        logger.debug("Fetching origin in base clone %s", clone_dir)
        _run_git(["fetch", "origin"], cwd=clone_dir)
        configure_push_auth(clone_dir)
        return clone_dir


def create_worktree(
    bug_id: str,
    *,
    branch: Optional[str] = None,
    base_ref: Optional[str] = None,
    clone_dir: Optional[str] = None,
    source_root: Optional[str] = None,
) -> str:
    """Create an isolated worktree for one dispatched agent and return its
    absolute path.

    Ensures the base clone exists/fetched (`prepare_base_clone`), then runs
    `git worktree add <clone>/worktrees/<bug_id> -b <branch> <base_ref>` off it,
    seeds `.env.test`, and links `.venv` to the base clone's venv.

    - `branch` defaults to `bugfix/<bug_id>-fix`.
    - `base_ref` defaults to `global_config.CC_FIXR_BASE_REF` (origin/master).

    Raises `CloneManagerError` on any git failure (including a worktree/branch
    that already exists — the caller owns unique bug ids).
    """
    base = prepare_base_clone(clone_dir=clone_dir, source_root=source_root)
    base_ref = base_ref or global_config.CC_FIXR_BASE_REF
    branch = branch or f"bugfix/{bug_id}-fix"
    src_root = source_root or str(global_config.PROJECT_ROOT)

    worktrees_root = os.path.join(base, "worktrees")
    os.makedirs(worktrees_root, exist_ok=True)
    worktree_dir = os.path.join(worktrees_root, bug_id)

    if os.path.exists(worktree_dir):
        raise CloneManagerError(
            f"Worktree path already exists: {worktree_dir}. Each dispatch needs "
            "a unique bug id; remove the stale worktree first."
        )

    logger.info(
        "Adding worktree %s (branch %s off %s)", worktree_dir, branch, base_ref
    )
    _run_git(
        ["worktree", "add", worktree_dir, "-b", branch, base_ref],
        cwd=base,
    )
    _seed_support_files(worktree_dir, src_root)
    _link_venv(worktree_dir, base)
    _link_notes(worktree_dir, src_root)
    return os.path.abspath(worktree_dir)


def _drop_shared_links(worktree_dir: str) -> None:
    """Unlink everything in the worktree that points at SHARED state, before any
    recursive removal touches it.

    Both the `.venv` and the `memory/` notes clone live outside the worktree and
    are used by every other dispatch. A recursive delete through a junction
    destroys the target rather than the link — the footgun that repeatedly
    destroyed the shared venv before DP-227, and which would now also take out
    every agent's memory.
    """
    for name in (".venv", NOTES_LINK_NAME):
        link = os.path.join(worktree_dir, name)
        if not (os.path.islink(link) or os.path.exists(link)):
            continue
        try:
            if os.path.islink(link):
                os.unlink(link)
            elif os.name == "nt":
                # Windows junction: rmdir removes the link, not the target.
                os.rmdir(link)
        except OSError as e:
            logger.warning("Could not drop worktree %s link %s: %s", name, link, e)


def _link_notes(worktree_dir: str, source_root: str) -> None:
    """Link the shared notes clone into the worktree as `memory/` (DP-314).

    The worktree already carries CLAUDE.md as a tracked file, so only the notes
    tree is missing — `memory/` is gitignored and therefore in no checkout. Best
    effort: an agent with no memory is the pre-DP-314 agent, which is worse but
    not broken, whereas failing the dispatch over a notes repo would be.
    """
    notes = prepare_notes_clone(source_root=source_root)
    if not notes:
        return
    if not link_notes_into(worktree_dir, notes):
        logger.warning(
            "Worktree %s has no memory/ link; the agent runs without notes.",
            worktree_dir,
        )


def remove_worktree(
    bug_id: str,
    *,
    clone_dir: Optional[str] = None,
    force: bool = False,
) -> None:
    """Tear down a dispatched agent's worktree. Drops the `.venv` link FIRST (so
    `git worktree remove` never recurses into the shared base venv), then
    `git worktree remove` + prune. Idempotent: a missing worktree is a no-op."""
    base = os.path.abspath(clone_dir or global_config.CC_FIXR_CLONE_DIR)
    worktree_dir = os.path.join(base, "worktrees", bug_id)
    if not os.path.exists(worktree_dir):
        return

    _drop_shared_links(worktree_dir)

    args = ["worktree", "remove", worktree_dir]
    if force:
        args.append("--force")
    try:
        _run_git(args, cwd=base)
    except CloneManagerError as e:
        logger.warning("git worktree remove failed for %s: %s", worktree_dir, e)
    _run_git(["worktree", "prune"], cwd=base)
