"""DP-418: the startup check that the node runs the services/pve/ copies the
image shipped with. The node half was five tickets stale for four weeks and no
gate could see it; this is what sees it."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import List, Sequence, Tuple

import pytest

from config import global_config
from src.proxmox import artifacts
from src.proxmox.artifacts import (
    NODE_ARTIFACTS,
    SERVICES_DIR,
    check_node_artifacts,
    local_digests,
    report_node_artifact_drift,
)
from src.proxmox.ssh import SSHResult


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", True)


@pytest.fixture
def shipped(tmp_path) -> Path:
    """A services/pve/ with LF content, standing in for the image's copy."""
    for repo_name, _ in NODE_ARTIFACTS:
        (tmp_path / repo_name).write_bytes(f"#!/bin/bash\n# {repo_name}\n".encode())
    return tmp_path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Node:
    """Answers sha256sum the way the real node does, from a dict of contents."""

    def __init__(self, files: dict, refuse: bool = False) -> None:
        self.files = files
        self.refuse = refuse
        self.calls: List[List[str]] = []

    async def run(self, argv: Sequence[str]) -> SSHResult:
        self.calls.append(list(argv))
        if self.refuse:
            return SSHResult(1, "", "derpr-pve: not allowed")
        out, err = [], []
        for p in argv[1:]:
            if p in self.files:
                out.append(f"{_sha(self.files[p])}  {p}")
            else:
                err.append(f"sha256sum: {p}: No such file or directory")
        return SSHResult(1 if err else 0, "\n".join(out), "\n".join(err))


def _deployed(shipped: Path) -> dict:
    return {node: (shipped / repo).read_bytes() for repo, node in NODE_ARTIFACTS}


def _run(coro):
    return asyncio.run(coro)


def test_every_listed_artifact_ships_in_the_repo():
    """The list and services/pve/ must not drift from each other either."""
    for repo_name, _ in NODE_ARTIFACTS:
        assert (SERVICES_DIR / repo_name).is_file(), repo_name


def test_matching_node_is_ok(shipped):
    report = _run(check_node_artifacts(Node(_deployed(shipped)), shipped))  # type: ignore[arg-type]
    assert report.ok


def test_a_changed_file_is_stale(shipped):
    files = _deployed(shipped)
    stale_path = NODE_ARTIFACTS[1][1]
    files[stale_path] = b"#!/bin/bash\n# older\n"
    report = _run(check_node_artifacts(Node(files), shipped))  # type: ignore[arg-type]
    assert report.stale == [stale_path] and not report.missing and not report.ok


def test_an_absent_file_is_missing_and_the_rest_still_compare(shipped):
    """sha256sum exits 1 on an absent file but still hashes the others."""
    files = _deployed(shipped)
    gone = NODE_ARTIFACTS[2][1]
    del files[gone]
    report = _run(check_node_artifacts(Node(files), shipped))  # type: ignore[arg-type]
    assert report.missing == [gone] and not report.stale and report.unverifiable is None


def test_a_refusing_wrapper_is_unverifiable_not_ok(shipped):
    """A pre-DP-418 wrapper refuses the hash; that is drift, not silence."""
    report = _run(check_node_artifacts(Node({}, refuse=True), shipped))  # type: ignore[arg-type]
    assert not report.ok and "not allowed" in (report.unverifiable or "")


def test_crlf_in_the_shipped_copy_does_not_read_as_drift(shipped):
    """A Windows autocrlf checkout must not report every file stale."""
    files = _deployed(shipped)
    for repo_name, _ in NODE_ARTIFACTS:
        p = shipped / repo_name
        p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    assert local_digests(shipped) == {n: _sha(b) for n, b in files.items()}
    assert _run(check_node_artifacts(Node(files), shipped)).ok  # type: ignore[arg-type]


def test_disabled_transport_is_unverifiable_without_ssh(shipped, monkeypatch):
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", False)
    node = Node(_deployed(shipped))
    report = _run(check_node_artifacts(node, shipped))  # type: ignore[arg-type]
    assert node.calls == [] and report.unverifiable


def _collect() -> Tuple[list, object]:
    sent: list = []

    async def send(subject: str, body: str) -> bool:
        sent.append((subject, body))
        return True
    return sent, send


def test_report_posts_only_on_drift(shipped):
    sent, send = _collect()
    _run(report_node_artifact_drift(Node(_deployed(shipped)), send, shipped))  # type: ignore[arg-type]
    assert sent == []

    files = _deployed(shipped)
    files[NODE_ARTIFACTS[0][1]] = b"old"
    _run(report_node_artifact_drift(Node(files), send, shipped))  # type: ignore[arg-type]
    assert len(sent) == 1
    assert NODE_ARTIFACTS[0][1] in sent[0][1] and "services/pve/README.md" in sent[0][1]


def test_report_never_raises(shipped, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(artifacts, "check_node_artifacts", boom)
    sent, send = _collect()
    report = _run(report_node_artifact_drift(Node({}), send, shipped))  # type: ignore[arg-type]
    assert not report.ok and sent == []
