# tests/test_clone_manager.py
"""Unit tests for the DP-227 fixr clone manager (worktree dispatch model).

All `git`/`subprocess` interaction is mocked — these never touch the network or
a real repo. Cover: fresh-clone path, fetch-only refresh (base stays pristine —
no reset/clean), per-dispatch `worktree add`, seeding, venv link, removal drops
the venv link first, the live-tree guard, and that errors surface as
CloneManagerError.
"""

import os
import subprocess
import threading

import pytest

import src.self_edit.clone_manager as cm
from src.self_edit.clone_manager import (
    CloneManagerError,
    create_worktree,
    prepare_base_clone,
    remove_worktree,
)


def _ok(stdout=""):
    return subprocess.CompletedProcess(args=["git"], returncode=0, stdout=stdout, stderr="")


def test_fresh_clone_then_fetch(tmp_path, monkeypatch):
    """No .git at the base dir => clone, then fetch. No reset/clean of base."""
    clone_dir = tmp_path / "fixr_clone"  # does not exist yet
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()

    calls = []

    def fake_run(cmd, cwd=None, capture_output=True, text=True, timeout=None):
        calls.append(cmd)
        if cmd[:3] == ["git", "remote", "get-url"]:
            return _ok("https://example.test/derpr.git\n")
        return _ok("")

    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    result = prepare_base_clone(str(clone_dir), source_root=str(source_root))

    assert result == os.path.abspath(str(clone_dir))
    op_names = [c[1] for c in calls]
    assert "clone" in op_names
    assert "fetch" in op_names
    # Base clone is never mutated — pristine model.
    assert "reset" not in op_names
    assert "clean" not in op_names


def test_existing_base_fetch_only(tmp_path, monkeypatch):
    """An existing base clone (has .git) is only `git fetch`ed — never reset/cleaned."""
    clone_dir = tmp_path / "fixr_clone"
    (clone_dir / ".git").mkdir(parents=True)
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()

    calls = []
    monkeypatch.setattr(
        cm.subprocess, "run",
        lambda cmd, **k: (calls.append(cmd), _ok(""))[1],
    )

    prepare_base_clone(str(clone_dir), source_root=str(source_root))

    op_names = [c[1] for c in calls]
    # fetch refreshes the base; config sets the push credential helper.
    assert op_names == ["fetch", "config"]
    assert "clone" not in op_names
    assert "reset" not in op_names
    assert "clean" not in op_names


def test_configures_github_push_credential_helper(tmp_path, monkeypatch):
    """prepare_base_clone installs an inline git credential helper that emits
    $GH_TOKEN at push time, so the dispatched agent's `git push` authenticates.
    The helper must NOT invoke `gh` (absent in the deployed container) and must
    reference GH_TOKEN only by env-var name, never the secret's value."""
    clone_dir = tmp_path / "fixr_clone"
    (clone_dir / ".git").mkdir(parents=True)
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()

    monkeypatch.setenv("GH_TOKEN", "s3cr3t-token-value")
    calls = []
    monkeypatch.setattr(
        cm.subprocess, "run",
        lambda cmd, **k: (calls.append(cmd), _ok(""))[1],
    )

    prepare_base_clone(str(clone_dir), source_root=str(source_root))

    config_call = next(c for c in calls if c[1] == "config")
    assert config_call[2] == "credential.https://github.com.helper"
    helper = config_call[3]
    # Inline shell helper resolving the token from the env by NAME, not `gh`.
    assert helper.startswith("!")
    assert "gh auth git-credential" not in helper
    assert "$GH_TOKEN" in helper
    assert "x-access-token" in helper
    # The secret's VALUE never appears in any git argv (only the var name does).
    assert all("s3cr3t-token-value" not in str(arg) for c in calls for arg in c)


def test_push_auth_failure_is_nonfatal(tmp_path, monkeypatch):
    """A failed credential-helper config must not abort base-clone prep — the
    agent can still diagnose/commit, just not push."""
    clone_dir = tmp_path / "fixr_clone"
    (clone_dir / ".git").mkdir(parents=True)
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()

    def fake_run(cmd, **k):
        if cmd[1] == "config":
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="config failed"
            )
        return _ok("")

    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    # Does not raise despite the config op failing.
    result = prepare_base_clone(str(clone_dir), source_root=str(source_root))
    assert result == os.path.abspath(str(clone_dir))


def test_create_worktree_adds_branch_off_base_ref(tmp_path, monkeypatch):
    """create_worktree fetches the base then `git worktree add -b <branch> <ref>`."""
    clone_dir = tmp_path / "fixr_clone"
    (clone_dir / ".git").mkdir(parents=True)
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()

    calls = []
    monkeypatch.setattr(
        cm.subprocess, "run",
        lambda cmd, **k: (calls.append(cmd), _ok(""))[1],
    )

    result = create_worktree(
        "DP-999",
        base_ref="origin/master",
        clone_dir=str(clone_dir),
        source_root=str(source_root),
    )

    expected = os.path.abspath(str(clone_dir / "worktrees" / "DP-999"))
    assert result == expected
    add_call = next(c for c in calls if c[1] == "worktree" and c[2] == "add")
    assert add_call[3] == expected
    assert add_call[4] == "-b"
    assert add_call[5] == "bugfix/DP-999-fix"  # default branch
    assert add_call[6] == "origin/master"


def test_create_worktree_rejects_existing_path(tmp_path, monkeypatch):
    clone_dir = tmp_path / "fixr_clone"
    (clone_dir / ".git").mkdir(parents=True)
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()
    # Pre-create the target worktree dir to trigger the guard.
    (clone_dir / "worktrees" / "DP-999").mkdir(parents=True)

    monkeypatch.setattr(cm.subprocess, "run", lambda *a, **k: _ok(""))

    with pytest.raises(CloneManagerError, match="already exists"):
        create_worktree(
            "DP-999", clone_dir=str(clone_dir), source_root=str(source_root)
        )


def test_create_worktree_seeds_support_files(tmp_path, monkeypatch):
    """.env.test from the source checkout is copied into the fresh worktree."""
    clone_dir = tmp_path / "fixr_clone"
    (clone_dir / ".git").mkdir(parents=True)
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()
    (source_root / ".env.test").write_text("ZAMMAD_URL=http://test\n")

    # `git worktree add` is mocked, so create the dir the seeder writes into.
    def fake_run(cmd, **k):
        if cmd[1] == "worktree" and cmd[2] == "add":
            os.makedirs(cmd[3], exist_ok=True)
        return _ok("")

    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    wt = create_worktree(
        "DP-999", clone_dir=str(clone_dir), source_root=str(source_root)
    )

    seeded = os.path.join(wt, ".env.test")
    assert os.path.isfile(seeded)
    with open(seeded) as f:
        assert "ZAMMAD_URL" in f.read()


def test_never_seeds_live_env():
    """Guard: the live .env must never be in the seed set (secret-exfil risk)."""
    assert ".env" not in cm._SEED_FILES
    assert cm._SEED_FILES == (".env.test",)


def test_create_worktree_links_venv(tmp_path, monkeypatch):
    """When the base clone has a .venv, the worktree gets a link to it."""
    clone_dir = tmp_path / "fixr_clone"
    (clone_dir / ".git").mkdir(parents=True)
    (clone_dir / ".venv").mkdir()  # base carries the venv
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()

    linked = {}

    def fake_run(cmd, **k):
        if cmd[1] == "worktree" and cmd[2] == "add":
            os.makedirs(cmd[3], exist_ok=True)
        return _ok("")

    monkeypatch.setattr(cm.subprocess, "run", fake_run)
    monkeypatch.setattr(cm.os, "name", "posix")

    def fake_symlink(src, dst, target_is_directory=False):
        linked["src"], linked["dst"] = src, dst

    monkeypatch.setattr(cm.os, "symlink", fake_symlink)

    create_worktree(
        "DP-999", clone_dir=str(clone_dir), source_root=str(source_root)
    )

    assert linked["src"] == os.path.join(str(clone_dir), ".venv")
    assert linked["dst"].endswith(os.path.join("DP-999", ".venv"))


def test_remove_worktree_drops_venv_link_before_remove(tmp_path, monkeypatch):
    """remove_worktree unlinks .venv before `git worktree remove` so removal
    never recurses into the shared base venv."""
    clone_dir = tmp_path / "fixr_clone"
    wt = clone_dir / "worktrees" / "DP-999"
    wt.mkdir(parents=True)
    real_venv = clone_dir / ".venv"
    real_venv.mkdir()
    (real_venv / "marker").write_text("keep me")
    # Symlink .venv -> base venv (POSIX).
    if os.name != "nt":
        os.symlink(str(real_venv), str(wt / ".venv"), target_is_directory=True)

    order = []
    monkeypatch.setattr(
        cm.subprocess, "run",
        lambda cmd, **k: (order.append(cmd[1:3]), _ok(""))[1],
    )

    remove_worktree("DP-999", clone_dir=str(clone_dir))

    # The link is gone but the real base venv (and its content) survives.
    assert not os.path.islink(str(wt / ".venv"))
    assert (real_venv / "marker").read_text() == "keep me"
    assert ["worktree", "remove"] in order
    assert ["worktree", "prune"] in order


def test_remove_worktree_missing_is_noop(tmp_path, monkeypatch):
    clone_dir = tmp_path / "fixr_clone"
    ran = {"called": False}
    monkeypatch.setattr(
        cm.subprocess, "run",
        lambda *a, **k: (ran.__setitem__("called", True), _ok(""))[1],
    )
    remove_worktree("DP-NOPE", clone_dir=str(clone_dir))
    assert ran["called"] is False


def test_repo_url_explicit_config_overrides_origin(tmp_path, monkeypatch):
    """CC_FIXR_REPO_URL (via config) is used instead of `git remote get-url`."""
    clone_dir = tmp_path / "fixr_clone"
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()
    monkeypatch.setattr(cm.global_config, "CC_FIXR_REPO_URL", "https://cfg.test/r.git")

    cloned_from = {}

    def fake_run(cmd, **k):
        if cmd[1] == "clone":
            cloned_from["url"] = cmd[2]
        assert cmd[:3] != ["git", "remote", "get-url"]
        return _ok("")

    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    prepare_base_clone(str(clone_dir), source_root=str(source_root))
    assert cloned_from["url"] == "https://cfg.test/r.git"


def test_git_failure_surfaces_as_clone_manager_error(tmp_path, monkeypatch):
    clone_dir = tmp_path / "fixr_clone"
    (clone_dir / ".git").mkdir(parents=True)
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()

    monkeypatch.setattr(
        cm.subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="fatal: could not fetch"
        ),
    )

    with pytest.raises(CloneManagerError, match="could not fetch"):
        prepare_base_clone(str(clone_dir), source_root=str(source_root))


def test_missing_git_binary_surfaces_error(tmp_path, monkeypatch):
    clone_dir = tmp_path / "fixr_clone"
    (clone_dir / ".git").mkdir(parents=True)
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()

    def fake_run(*a, **k):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    with pytest.raises(CloneManagerError, match="git binary not found"):
        prepare_base_clone(str(clone_dir), source_root=str(source_root))


def test_timeout_surfaces_error(tmp_path, monkeypatch):
    clone_dir = tmp_path / "fixr_clone"
    (clone_dir / ".git").mkdir(parents=True)
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()

    def fake_run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="git fetch", timeout=600)

    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    with pytest.raises(CloneManagerError, match="timed out"):
        prepare_base_clone(str(clone_dir), source_root=str(source_root))


def test_refuses_base_dir_that_is_live_source(tmp_path, monkeypatch):
    """A base clone_dir equal to (or containing) the live source tree must be
    refused before any git op runs."""
    source_root = tmp_path / "repo"
    (source_root / ".git").mkdir(parents=True)

    ran = {"called": False}
    monkeypatch.setattr(
        cm.subprocess, "run",
        lambda *a, **k: (ran.__setitem__("called", True), _ok(""))[1],
    )

    with pytest.raises(CloneManagerError, match="live source tree"):
        prepare_base_clone(str(source_root), source_root=str(source_root))
    with pytest.raises(CloneManagerError, match="live source tree"):
        prepare_base_clone(str(tmp_path), source_root=str(source_root))

    assert ran["called"] is False


def test_lock_serializes_concurrent_base_prep(tmp_path, monkeypatch):
    """The module lock prevents two callers running base-clone git ops interleaved."""
    clone_dir = tmp_path / "fixr_clone"
    (clone_dir / ".git").mkdir(parents=True)
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()

    in_critical = {"count": 0, "max": 0}
    barrier_lock = threading.Lock()

    def fake_run(cmd, cwd=None, capture_output=True, text=True, timeout=None):
        with barrier_lock:
            in_critical["count"] += 1
            in_critical["max"] = max(in_critical["max"], in_critical["count"])
        import time
        time.sleep(0.01)
        with barrier_lock:
            in_critical["count"] -= 1
        return _ok("")

    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    def worker():
        prepare_base_clone(str(clone_dir), source_root=str(source_root))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert in_critical["max"] == 1
