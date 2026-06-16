# tests/test_clone_manager.py
"""Unit tests for the DP-227 fixr self-edit clone manager.

All `git`/`subprocess` interaction is mocked — these never touch the network or
a real repo. Cover: fresh-clone path, refresh path, seeding, lock serializes
concurrency, and that errors surface as CloneManagerError.
"""

import os
import subprocess
import threading

import pytest

import src.self_edit.clone_manager as cm
from src.self_edit.clone_manager import CloneManagerError, prepare_fixr_workspace


def _ok(stdout=""):
    return subprocess.CompletedProcess(args=["git"], returncode=0, stdout=stdout, stderr="")


def test_fresh_clone_path(tmp_path, monkeypatch):
    """When the clone dir has no .git, the manager clones then resets."""
    clone_dir = tmp_path / "fixr_clone"  # does not exist yet
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()

    calls = []

    def fake_run(cmd, cwd=None, capture_output=True, text=True, timeout=None):
        calls.append((cmd, cwd))
        if cmd[:3] == ["git", "remote", "get-url"]:
            return _ok("https://example.test/derpr.git\n")
        return _ok("")

    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    result = prepare_fixr_workspace(
        str(clone_dir), base_ref="origin/master", source_root=str(source_root)
    )

    assert result == os.path.abspath(str(clone_dir))
    # First real op is a clone (after deriving the origin url).
    op_names = [c[0][1] for c in calls]
    assert "clone" in op_names
    assert "reset" in op_names
    # No `clean` on a fresh clone (nothing to clean).
    assert "clean" not in op_names


def test_refresh_path_fetches_resets_cleans(tmp_path, monkeypatch):
    """An existing clone (has .git) is refreshed: fetch + reset --hard + clean."""
    clone_dir = tmp_path / "fixr_clone"
    (clone_dir / ".git").mkdir(parents=True)
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()

    calls = []

    def fake_run(cmd, cwd=None, capture_output=True, text=True, timeout=None):
        calls.append(cmd)
        return _ok("")

    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    result = prepare_fixr_workspace(
        str(clone_dir), base_ref="origin/master", source_root=str(source_root)
    )

    assert result == os.path.abspath(str(clone_dir))
    op_names = [c[1] for c in calls]
    assert op_names == ["fetch", "reset", "clean"]
    # No clone on refresh — origin url is never derived.
    assert "clone" not in op_names
    # reset targets the configured base ref
    reset_call = next(c for c in calls if c[1] == "reset")
    assert reset_call[-1] == "origin/master"


def test_seeds_support_files(tmp_path, monkeypatch):
    """.env.test from the source checkout is copied into the refreshed clone."""
    clone_dir = tmp_path / "fixr_clone"
    (clone_dir / ".git").mkdir(parents=True)
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()
    (source_root / ".env.test").write_text("ZAMMAD_URL=http://test\n")

    monkeypatch.setattr(
        cm.subprocess, "run",
        lambda *a, **k: _ok(""),
    )

    prepare_fixr_workspace(
        str(clone_dir), base_ref="origin/master", source_root=str(source_root)
    )

    seeded = clone_dir / ".env.test"
    assert seeded.is_file()
    assert "ZAMMAD_URL" in seeded.read_text()


def test_repo_url_explicit_config_overrides_origin(tmp_path, monkeypatch):
    """CC_FIXR_REPO_URL (via config) is used instead of `git remote get-url`."""
    clone_dir = tmp_path / "fixr_clone"
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()
    monkeypatch.setattr(cm.global_config, "CC_FIXR_REPO_URL", "https://cfg.test/r.git")

    cloned_from = {}

    def fake_run(cmd, cwd=None, capture_output=True, text=True, timeout=None):
        if cmd[1] == "clone":
            cloned_from["url"] = cmd[2]
        # remote get-url must NOT be called when config provides the url
        assert cmd[:3] != ["git", "remote", "get-url"]
        return _ok("")

    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    prepare_fixr_workspace(str(clone_dir), source_root=str(source_root))
    assert cloned_from["url"] == "https://cfg.test/r.git"


def test_git_failure_surfaces_as_clone_manager_error(tmp_path, monkeypatch):
    clone_dir = tmp_path / "fixr_clone"
    (clone_dir / ".git").mkdir(parents=True)
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()

    def fake_run(cmd, cwd=None, capture_output=True, text=True, timeout=None):
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="fatal: could not fetch"
        )

    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    with pytest.raises(CloneManagerError, match="could not fetch"):
        prepare_fixr_workspace(str(clone_dir), source_root=str(source_root))


def test_missing_git_binary_surfaces_error(tmp_path, monkeypatch):
    clone_dir = tmp_path / "fixr_clone"
    (clone_dir / ".git").mkdir(parents=True)
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()

    def fake_run(*a, **k):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    with pytest.raises(CloneManagerError, match="git binary not found"):
        prepare_fixr_workspace(str(clone_dir), source_root=str(source_root))


def test_timeout_surfaces_error(tmp_path, monkeypatch):
    clone_dir = tmp_path / "fixr_clone"
    (clone_dir / ".git").mkdir(parents=True)
    source_root = tmp_path / "src_checkout"
    source_root.mkdir()

    def fake_run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="git fetch", timeout=600)

    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    with pytest.raises(CloneManagerError, match="timed out"):
        prepare_fixr_workspace(str(clone_dir), source_root=str(source_root))


def test_refuses_clone_dir_that_is_live_source(tmp_path, monkeypatch):
    """A clone_dir equal to (or containing) the live source tree must be
    refused before any destructive git op runs — reset --hard/clean -fdx would
    wipe the running checkout."""
    source_root = tmp_path / "repo"
    (source_root / ".git").mkdir(parents=True)

    ran = {"called": False}

    def fake_run(*a, **k):
        ran["called"] = True
        return _ok("")

    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    # clone_dir == source tree
    with pytest.raises(CloneManagerError, match="live source tree"):
        prepare_fixr_workspace(str(source_root), source_root=str(source_root))
    # clone_dir is a parent of the source tree
    with pytest.raises(CloneManagerError, match="live source tree"):
        prepare_fixr_workspace(str(tmp_path), source_root=str(source_root))

    assert ran["called"] is False  # refused before any git ran


def test_lock_serializes_concurrent_callers(tmp_path, monkeypatch):
    """The module lock prevents two callers from running git ops interleaved
    against the same tree."""
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
        # Give the other thread a chance to (try to) enter.
        import time
        time.sleep(0.01)
        with barrier_lock:
            in_critical["count"] -= 1
        return _ok("")

    monkeypatch.setattr(cm.subprocess, "run", fake_run)

    def worker():
        prepare_fixr_workspace(str(clone_dir), source_root=str(source_root))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # If the lock works, only one git op is ever in flight at a time.
    assert in_critical["max"] == 1
