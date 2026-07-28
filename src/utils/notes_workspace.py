# src/utils/notes_workspace.py
"""Give a cc-* workspace the two things CLAUDE.md assumes are there (DP-314).

CLAUDE.md is read automatically by any `claude` process whose cwd contains it —
including under `--system-prompt`, which replaces the system prompt but does not
suppress project instruction files (measured, claude 2.1.220). What CLAUDE.md
then tells the reader to do is navigate and *update* `memory/`, the Viking
L0/L1/L2 notes tree. Neither half held for cc-* instances:

- a fixr worktree is a real checkout, so it always had CLAUDE.md — but `memory/`
  is gitignored, so no checkout anywhere (including the CI-built prod image) has
  ever carried it;
- a persona workspace (`data/workspaces/cc_<persona>`) is a bare directory that
  is not a repo at all, so it had neither.

This module supplies both: ONE shared clone of the notes repo, symlinked in as
`memory/`, plus a copied CLAUDE.md for the workspaces that aren't checkouts.

**The link points out of the workspace, and that matters.** Claude Code's
sandbox confines writes to the cwd, so a symlink to a shared clone would fail
with a bare EACCES unless the clone's real path is also in the sandbox's
`filesystem.allowWrite`. `notes_allow_write_paths()` exists for exactly that,
and `utils/cc_sandbox.py` is the only thing that should call it.

Everything here is best-effort and loud: a missing notes clone degrades an agent
to how it behaved before this module existed, which is survivable, whereas
failing a dispatch because a *notes* repo was unreachable is not. Every
degradation logs at WARNING and is reported through the return value, so the
caller can record that the agent ran without memory rather than assume it didn't.
"""

import logging
import os
import shutil
import subprocess
import threading
from typing import List, Optional

from config import global_config
from src.utils.git_support import GitError, configure_push_auth, run_git

logger = logging.getLogger(__name__)

#: Name the notes tree is linked in as. CLAUDE.md refers to it by this path, and
#: the code repo gitignores `/memory/`, so the link is never committed.
NOTES_LINK_NAME = "memory"

# Serializes clone creation/fetch so concurrent dispatches don't race it. Mirrors
# the base-clone lock in clone_manager; they guard different directories.
_notes_lock = threading.Lock()


def notes_clone_dir() -> str:
    """Absolute path of the one shared notes clone."""
    return os.path.abspath(global_config.CC_NOTES_DIR)


def _derive_notes_url(source_root: str) -> Optional[str]:
    """Resolve the notes repo URL: explicit config wins, else read `origin` from
    the running checkout's own `memory/`. Returns None when neither is available
    — the container has no `memory/`, so a deploy must set CC_NOTES_REPO_URL."""
    if global_config.CC_NOTES_REPO_URL:
        return str(global_config.CC_NOTES_REPO_URL)
    local_notes = os.path.join(source_root, NOTES_LINK_NAME)
    if not os.path.isdir(os.path.join(local_notes, ".git")):
        return None
    try:
        url = run_git(["remote", "get-url", "origin"], cwd=local_notes).strip()
    except GitError as e:
        logger.warning("Could not read notes remote from %s: %s", local_notes, e)
        return None
    return url or None


def prepare_notes_clone(
    clone_dir: Optional[str] = None,
    *,
    repo_url: Optional[str] = None,
    source_root: Optional[str] = None,
) -> Optional[str]:
    """Ensure the shared notes clone exists and is current; return its absolute
    path, or None when notes are disabled or unavailable.

    First use (no `.git`): `git clone` then configure push auth. Subsequent:
    `git pull --ff-only` on the tracked branch — unlike the fixr BASE clone,
    which is only ever fetched because worktrees hang off it, this clone IS the
    working copy every agent edits, so it has to actually advance.

    A dirty tree (an earlier agent committed but could not push, or left edits)
    makes the pull fail; that is logged and the existing clone is handed back
    anyway, because stale notes beat no notes.
    """
    if not global_config.CC_NOTES_ENABLED:
        return None

    target = os.path.abspath(clone_dir or global_config.CC_NOTES_DIR)
    src_root = source_root or str(global_config.PROJECT_ROOT)
    branch = global_config.CC_NOTES_BRANCH

    with _notes_lock:
        if os.path.isdir(os.path.join(target, ".git")):
            try:
                run_git(["pull", "--ff-only", "origin", branch], cwd=target)
            except GitError as e:
                logger.warning(
                    "Notes clone %s could not fast-forward (%s); using it as-is.",
                    target, e,
                )
            configure_push_auth(target)
            return target

        url = repo_url or _derive_notes_url(src_root)
        if not url:
            logger.warning(
                "Notes repo unavailable: no CC_NOTES_REPO_URL and no git remote in "
                "%s/%s. cc-* instances will run without memory.",
                src_root, NOTES_LINK_NAME,
            )
            return None

        parent = os.path.dirname(target) or "."
        os.makedirs(parent, exist_ok=True)
        # Clear any stale non-repo dir at the target so the clone can succeed.
        if os.path.exists(target):
            shutil.rmtree(target, ignore_errors=True)
        logger.info("Cloning notes repo into %s", target)
        try:
            run_git(["clone", "--branch", branch, url, target])
        except GitError as e:
            logger.warning(
                "Could not clone the notes repo (%s). cc-* instances will run "
                "without memory.", e,
            )
            return None
        configure_push_auth(target)
        return target


def link_notes_into(workspace_dir: str, notes_dir: Optional[str] = None) -> Optional[str]:
    """Link the shared notes clone into `workspace_dir` as `memory/`.

    Returns the link path on success, or None when notes are disabled,
    unavailable, or the workspace already has a real `memory/` of its own (which
    is the case when CC_WORKSPACE_DIR points at a developer checkout — that tree
    already carries the notes repo and must not be shadowed).

    Symlink on POSIX, directory junction on Windows (a symlink needs elevation).
    Idempotent: an existing link is left alone.
    """
    if not global_config.CC_NOTES_ENABLED:
        return None
    notes = notes_dir or notes_clone_dir()
    if not os.path.isdir(notes):
        logger.warning("Notes clone %s missing; not linking into %s.", notes, workspace_dir)
        return None

    link = os.path.join(workspace_dir, NOTES_LINK_NAME)
    if os.path.islink(link):
        return link
    if os.path.exists(link):
        # A real directory — either a previous junction or a checkout's own
        # memory/. Either way it already answers the need; don't clobber it.
        logger.debug("%s already exists; leaving it in place.", link)
        return link

    try:
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", link, notes],
                check=True, capture_output=True, text=True,
            )
        else:
            os.symlink(notes, link, target_is_directory=True)
    except (OSError, subprocess.CalledProcessError) as e:
        logger.warning("Could not link %s -> %s: %s", link, notes, e)
        return None
    return link


def seed_claude_md(workspace_dir: str, source_root: Optional[str] = None) -> Optional[str]:
    """Copy CLAUDE.md into a workspace that is not a checkout.

    A fixr worktree needs nothing here — it has the tracked file. A persona
    workspace is a bare directory, so without this the instructions the whole
    memory protocol depends on are simply absent. Copied, not linked: the file
    is small, and a stale copy is more debuggable than a dangling link.

    Refreshed on every call so an edited CLAUDE.md reaches long-lived persona
    workspaces. No-op (returns None) when the workspace already tracks its own.
    """
    src = os.path.join(source_root or str(global_config.PROJECT_ROOT), "CLAUDE.md")
    if not os.path.isfile(src):
        return None
    if os.path.isdir(os.path.join(workspace_dir, ".git")):
        return None  # a checkout carries its own
    dst = os.path.join(workspace_dir, "CLAUDE.md")
    try:
        if os.path.abspath(src) == os.path.abspath(dst):
            return dst
        shutil.copy2(src, dst)
    except OSError as e:
        logger.warning("Could not seed CLAUDE.md into %s: %s", workspace_dir, e)
        return None
    return dst


def prepare_workspace_notes(
    workspace_dir: str, source_root: Optional[str] = None
) -> Optional[str]:
    """Full treatment for one workspace: refresh the shared clone, link it in as
    `memory/`, and seed CLAUDE.md if the workspace isn't a checkout. Returns the
    link path, or None if the workspace ran without notes."""
    if not global_config.CC_NOTES_ENABLED:
        return None
    seed_claude_md(workspace_dir, source_root)
    notes = prepare_notes_clone(source_root=source_root)
    if not notes:
        return None
    return link_notes_into(workspace_dir, notes)


def notes_allow_write_paths() -> List[str]:
    """Paths the sandbox must grant write access to for the linked notes tree to
    be writable. Empty when notes are disabled or the clone doesn't exist yet —
    never return a path we haven't confirmed, since a bogus `allowWrite` entry
    widens the sandbox for nothing."""
    if not global_config.CC_NOTES_ENABLED:
        return []
    notes = notes_clone_dir()
    return [notes] if os.path.isdir(notes) else []
