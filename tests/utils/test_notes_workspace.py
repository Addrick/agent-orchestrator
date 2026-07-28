# tests/utils/test_notes_workspace.py
"""Unit tests for the shared notes clone linked into cc-* workspaces (DP-314).

All git interaction is mocked — these never touch the network or a real repo.
Cover: the disabled switch, URL derivation and its fallbacks, clone-then-pull
(the notes clone ADVANCES, unlike the pristine fixr base clone), linking and its
refusal to clobber, CLAUDE.md seeding, and that every failure degrades to
"agent runs without memory" rather than raising.
"""

import os
import subprocess

import pytest

import src.utils.notes_workspace as nw
from src.utils.notes_workspace import (
    link_notes_into,
    notes_allow_write_paths,
    prepare_notes_clone,
    prepare_workspace_notes,
    seed_claude_md,
)


def _ok(stdout=""):
    return subprocess.CompletedProcess(args=["git"], returncode=0, stdout=stdout, stderr="")


@pytest.fixture
def notes_on(monkeypatch, tmp_path):
    monkeypatch.setattr(nw.global_config, "CC_NOTES_ENABLED", True)
    monkeypatch.setattr(nw.global_config, "CC_NOTES_BRANCH", "main")
    monkeypatch.setattr(nw.global_config, "CC_NOTES_REPO_URL", None)
    monkeypatch.setattr(nw.global_config, "CC_NOTES_DIR", str(tmp_path / "notes"))


# --- the master switch -------------------------------------------------------


def test_everything_is_inert_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(nw.global_config, "CC_NOTES_ENABLED", False)
    ran = {"git": False}
    monkeypatch.setattr(
        nw.subprocess, "run", lambda *a, **k: (ran.__setitem__("git", True), _ok())[1]
    )

    assert prepare_notes_clone(str(tmp_path / "notes")) is None
    assert link_notes_into(str(tmp_path)) is None
    assert prepare_workspace_notes(str(tmp_path)) is None
    assert notes_allow_write_paths() == []
    assert ran["git"] is False


# --- resolving where to clone from -------------------------------------------


def test_url_derived_from_the_checkouts_own_notes_remote(notes_on, tmp_path, monkeypatch):
    source_root = tmp_path / "checkout"
    (source_root / "memory" / ".git").mkdir(parents=True)
    target = tmp_path / "notes"

    cmds = []

    def fake_run(cmd, **k):
        cmds.append(cmd)
        if cmd[1:3] == ["remote", "get-url"]:
            return _ok("https://example.test/private-notes.git\n")
        if cmd[1] == "clone":
            os.makedirs(cmd[-1], exist_ok=True)
        return _ok("")

    monkeypatch.setattr(nw.subprocess, "run", fake_run)

    assert prepare_notes_clone(str(target), source_root=str(source_root)) == str(target)
    clone = next(c for c in cmds if c[1] == "clone")
    assert "https://example.test/private-notes.git" in clone


def test_explicit_repo_url_wins_over_the_local_remote(notes_on, tmp_path, monkeypatch):
    """The container has no memory/ to read a remote from, so the deploy sets
    this — and when both exist, config is the authority."""
    monkeypatch.setattr(nw.global_config, "CC_NOTES_REPO_URL", "https://cfg.test/n.git")
    source_root = tmp_path / "checkout"
    (source_root / "memory" / ".git").mkdir(parents=True)

    cmds = []

    def fake_run(cmd, **k):
        cmds.append(cmd)
        if cmd[1] == "clone":
            os.makedirs(cmd[-1], exist_ok=True)
        return _ok("")

    monkeypatch.setattr(nw.subprocess, "run", fake_run)

    prepare_notes_clone(str(tmp_path / "notes"), source_root=str(source_root))

    assert not any(c[1:3] == ["remote", "get-url"] for c in cmds)
    assert "https://cfg.test/n.git" in next(c for c in cmds if c[1] == "clone")


def test_no_url_anywhere_degrades_instead_of_raising(notes_on, tmp_path, monkeypatch):
    """The prod image has neither a memory/ nor (if unconfigured) a URL. That
    must leave the agent without memory, not blow up its dispatch."""
    source_root = tmp_path / "checkout"
    source_root.mkdir()
    monkeypatch.setattr(nw.subprocess, "run", lambda *a, **k: _ok(""))

    assert prepare_notes_clone(str(tmp_path / "notes"), source_root=str(source_root)) is None


def test_clone_failure_degrades_instead_of_raising(notes_on, tmp_path, monkeypatch):
    monkeypatch.setattr(nw.global_config, "CC_NOTES_REPO_URL", "https://cfg.test/n.git")

    def fake_run(cmd, **k):
        if cmd[1] == "clone":
            return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="auth failed")
        return _ok("")

    monkeypatch.setattr(nw.subprocess, "run", fake_run)

    assert prepare_notes_clone(str(tmp_path / "notes"), source_root=str(tmp_path)) is None


# --- refreshing an existing clone --------------------------------------------


def test_existing_clone_fast_forwards(notes_on, tmp_path, monkeypatch):
    """Unlike the pristine fixr BASE clone (fetch-only, never advanced because
    worktrees hang off it), this clone IS the working copy agents edit, so it
    must actually move."""
    target = tmp_path / "notes"
    (target / ".git").mkdir(parents=True)

    cmds = []
    monkeypatch.setattr(
        nw.subprocess, "run", lambda cmd, **k: (cmds.append(cmd), _ok(""))[1]
    )

    assert prepare_notes_clone(str(target)) == str(target)
    assert ["git", "pull", "--ff-only", "origin", "main"] in cmds
    assert not any(c[1] == "clone" for c in cmds)


def test_unpullable_clone_is_still_returned(notes_on, tmp_path, monkeypatch):
    """A previous agent committing without pushing leaves the tree ahead. Stale
    notes beat no notes."""
    target = tmp_path / "notes"
    (target / ".git").mkdir(parents=True)

    def fake_run(cmd, **k):
        if cmd[1] == "pull":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="diverged")
        return _ok("")

    monkeypatch.setattr(nw.subprocess, "run", fake_run)

    assert prepare_notes_clone(str(target)) == str(target)


def test_push_auth_is_configured_so_agents_can_push_memory(notes_on, tmp_path, monkeypatch):
    """Writable is the whole point; a clone that cannot push is write-only-local."""
    target = tmp_path / "notes"
    (target / ".git").mkdir(parents=True)

    cmds = []
    monkeypatch.setattr(
        nw.subprocess, "run", lambda cmd, **k: (cmds.append(cmd), _ok(""))[1]
    )

    prepare_notes_clone(str(target))

    helper = next(c for c in cmds if c[1] == "config")
    assert helper[2] == "credential.https://github.com.helper"
    # The token is read from $GH_TOKEN at call time, never written into config.
    assert "$GH_TOKEN" in helper[3]


# --- linking it into a workspace ---------------------------------------------


def test_link_created_and_idempotent(notes_on, tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    ws = tmp_path / "ws"
    ws.mkdir()

    first = link_notes_into(str(ws), str(notes))
    assert first is not None
    assert os.path.basename(first) == "memory"
    assert os.path.exists(os.path.join(str(ws), "memory"))
    # Second call must not raise or duplicate.
    assert link_notes_into(str(ws), str(notes)) == first


def test_link_refuses_to_shadow_a_real_memory_dir(notes_on, tmp_path):
    """CC_WORKSPACE_DIR can point at a developer checkout, which already carries
    the notes repo. Replacing it with a link to a different clone would silently
    split the developer's memory in two."""
    notes = tmp_path / "notes"
    notes.mkdir()
    ws = tmp_path / "checkout"
    (ws / "memory").mkdir(parents=True)
    (ws / "memory" / "MEMORY.md").write_text("the real index")

    link_notes_into(str(ws), str(notes))

    assert not os.path.islink(str(ws / "memory"))
    assert (ws / "memory" / "MEMORY.md").read_text() == "the real index"


def test_link_skipped_when_clone_missing(notes_on, tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    assert link_notes_into(str(ws), str(tmp_path / "absent")) is None
    assert not os.path.exists(os.path.join(str(ws), "memory"))


# --- CLAUDE.md ---------------------------------------------------------------


def test_claude_md_seeded_into_a_bare_workspace(notes_on, tmp_path):
    """A persona workspace is not a checkout, so nothing else puts CLAUDE.md
    there — and without it the memory protocol is never read at all."""
    source_root = tmp_path / "checkout"
    source_root.mkdir()
    (source_root / "CLAUDE.md").write_text("prefix is DP")
    ws = tmp_path / "cc_persona"
    ws.mkdir()

    assert seed_claude_md(str(ws), str(source_root)) is not None
    assert (ws / "CLAUDE.md").read_text() == "prefix is DP"


def test_claude_md_refreshed_on_later_calls(notes_on, tmp_path):
    """Persona workspaces are long-lived; an edited CLAUDE.md must reach them."""
    source_root = tmp_path / "checkout"
    source_root.mkdir()
    claude_md = source_root / "CLAUDE.md"
    claude_md.write_text("v1")
    ws = tmp_path / "cc_persona"
    ws.mkdir()

    seed_claude_md(str(ws), str(source_root))
    claude_md.write_text("v2")
    seed_claude_md(str(ws), str(source_root))

    assert (ws / "CLAUDE.md").read_text() == "v2"


def test_claude_md_not_seeded_into_a_checkout(notes_on, tmp_path):
    """A fixr worktree tracks its own; copying over it would let a stale host
    copy silently outrank the branch's version."""
    source_root = tmp_path / "checkout"
    source_root.mkdir()
    (source_root / "CLAUDE.md").write_text("host copy")
    ws = tmp_path / "worktree"
    (ws / ".git").mkdir(parents=True)
    (ws / "CLAUDE.md").write_text("branch copy")

    assert seed_claude_md(str(ws), str(source_root)) is None
    assert (ws / "CLAUDE.md").read_text() == "branch copy"


# --- what the sandbox is told ------------------------------------------------


def test_allow_write_paths_only_when_the_clone_exists(notes_on, tmp_path, monkeypatch):
    """Never widen the sandbox for a path we haven't confirmed."""
    assert notes_allow_write_paths() == []

    notes = tmp_path / "notes"
    notes.mkdir()
    monkeypatch.setattr(nw.global_config, "CC_NOTES_DIR", str(notes))
    assert notes_allow_write_paths() == [os.path.abspath(str(notes))]


def test_prepare_workspace_notes_does_both_halves(notes_on, tmp_path, monkeypatch):
    source_root = tmp_path / "checkout"
    source_root.mkdir()
    (source_root / "CLAUDE.md").write_text("prefix is DP")
    notes = tmp_path / "notes"
    (notes / ".git").mkdir(parents=True)
    monkeypatch.setattr(nw.global_config, "CC_NOTES_DIR", str(notes))
    cmds = []
    monkeypatch.setattr(
        nw.subprocess, "run", lambda cmd, **k: (cmds.append(cmd), _ok(""))[1]
    )
    ws = tmp_path / "cc_persona"
    ws.mkdir()

    link = prepare_workspace_notes(str(ws), str(source_root))

    assert (ws / "CLAUDE.md").is_file()
    assert link == os.path.join(str(ws), "memory")
    # git is mocked, and on Windows so is the junction call, so assert the link
    # was actually attempted rather than trusting the return value alone.
    if os.name == "nt":
        assert any(c[:3] == ["cmd", "/c", "mklink"] for c in cmds)
    else:
        assert os.path.islink(str(ws / "memory"))
