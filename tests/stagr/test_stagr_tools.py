# tests/stagr/test_stagr_tools.py
#
# DP-456 — the stagr tools, against two fake `opencode serve`s (stagr and
# codr's sandbox) whose shell endpoints run the scripts in a real bash, each in
# its own folder. What matters is what lands where, so the transfers are run
# for real rather than asserted on as script text.

import json
import subprocess
from pathlib import Path

import pytest
from aiohttp.test_utils import TestServer

from config import global_config
from src.proxmox.ssh import SSHResult
from src.stagr.handler import StagrToolHandler
from src.tool_policy import ToolPolicy
from src.tools.definitions import ALL_TOOL_DEFINITIONS
from src.tools.tool_defs.stagr import STAGR_TOOLS
from tests.test_opencode_provider import FakeOpencode, real_bash

_STAGR_JSON = Path(__file__).resolve().parents[2] / "config" / "optional_personas" / "stagr.json"


@pytest.fixture
async def boxes(monkeypatch, tmp_path):
    """(stagr fake, codr fake, stagr workdir, codr workspace)."""
    stagr_dir, codr_dir = tmp_path / "stagr", tmp_path / "codr"
    stagr_dir.mkdir()
    codr_dir.mkdir()
    fakes, servers = [], []
    for folder in (stagr_dir, codr_dir):
        fake = FakeOpencode()
        fake.shell_output, fake.file_root = real_bash(folder), folder
        server = TestServer(fake.app())
        await server.start_server()
        fakes.append(fake)
        servers.append(server)
    monkeypatch.setattr(global_config, "STAGR_SERVER_URL", str(servers[0].make_url("")).rstrip("/"))
    monkeypatch.setattr(global_config, "STAGR_SERVER_PASSWORD", "spw")
    monkeypatch.setattr(global_config, "OPENCODE_SERVER_URL", str(servers[1].make_url("")).rstrip("/"))
    monkeypatch.setattr(global_config, "OPENCODE_SERVER_PASSWORD", "pw")
    yield fakes[0], fakes[1], stagr_dir, codr_dir
    for server in servers:
        await server.close()


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def _upstream_repo(path):
    """A repo whose history holds a 'fix' after the commit it is checked out at."""
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    (path / "lib.py").write_text("def add(a, b):\n    return a - b\n")
    _git(path, "add", "-A")
    _git(path, "-c", "user.name=u", "-c", "user.email=u@x", "commit", "-qm", "buggy")
    (path / "lib.py").write_text("def add(a, b):\n    return a + b\n")
    _git(path, "-c", "user.name=u", "-c", "user.email=u@x", "commit", "-qam", "THE FIX")
    _git(path, "checkout", "-q", "HEAD~1")


async def test_shell_reports_exit_code_and_output(boxes):
    result = await StagrToolHandler()._stagr_shell("echo hi; echo oops >&2; exit 3")
    assert result["status"] == "ok" and result["exit_code"] == 3
    assert "hi" in result["output"] and "oops" in result["output"]
    assert "__STAGR_EXIT" not in result["output"]


async def test_shell_cuts_the_middle_of_long_output(boxes, monkeypatch):
    monkeypatch.setattr(global_config, "STAGR_OUTPUT_CHARS", 1000)
    result = await StagrToolHandler()._stagr_shell("seq 1 20000")
    assert len(result["output"]) < 1200
    assert result["output"].startswith("1\n") and result["output"].rstrip().endswith("20000")
    assert "characters cut" in result["output"]


async def test_tools_refuse_when_stagr_is_not_configured(monkeypatch):
    monkeypatch.setattr(global_config, "STAGR_SERVER_URL", "")
    result = await StagrToolHandler()._stagr_shell("true")
    assert result["status"] == "error" and "not configured" in result["message"]


@pytest.mark.skipif(subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
                    reason="needs git")
async def test_delivery_leaves_upstream_history_behind(boxes):
    """A checkout at the pre-fix commit still carries the fix in .git; codr
    must get the files and a one-commit repo, never that history."""
    stagr, codr, stagr_dir, codr_dir = boxes
    _upstream_repo(stagr_dir / "repo")
    result = await StagrToolHandler()._deliver_to_codr("repo", "09-c1")
    assert result["status"] == "ok", result
    task = codr_dir / "tasks" / "09-c1"
    assert "a - b" in (task / "lib.py").read_text()
    assert _git(task, "rev-list", "--all", "--count").strip() == "1"
    assert "THE FIX" not in _git(task, "log", "--all", "--format=%s")
    assert [p.name for p in codr_dir.iterdir()] == ["tasks"]    # staged archive removed
    assert sorted(p.name for p in stagr_dir.iterdir()) == ["repo"]


async def test_delivery_never_replaces_an_existing_task_folder(boxes):
    stagr, codr, stagr_dir, codr_dir = boxes
    (stagr_dir / "repo").mkdir()
    (codr_dir / "tasks" / "09-c1").mkdir(parents=True)
    (codr_dir / "tasks" / "09-c1" / "work.py").write_text("codr's work\n")
    result = await StagrToolHandler()._deliver_to_codr("repo", "09-c1")
    assert result["status"] == "error" and "already exists" in result["message"]
    assert (codr_dir / "tasks" / "09-c1" / "work.py").read_text() == "codr's work\n"


async def test_delivery_waits_for_codr_to_be_idle(boxes):
    stagr, codr, stagr_dir, codr_dir = boxes
    (stagr_dir / "repo").mkdir()
    codr.busy.add("ses_1")
    result = await StagrToolHandler()._deliver_to_codr("repo", "09-c1")
    assert result["status"] == "error" and "idle" in result["message"]
    assert not (codr_dir / "tasks").exists()


@pytest.mark.parametrize("path", ["../etc", "/etc", "a/../../b", "a b", "$(id)"])
async def test_delivery_refuses_paths_outside_the_workdir(boxes, path):
    result = await StagrToolHandler()._deliver_to_codr(path, "09-c1")
    assert result["status"] == "error" and "relative" in result["message"]


async def test_retrieval_copies_a_codr_task_to_stagr(boxes):
    stagr, codr, stagr_dir, codr_dir = boxes
    (codr_dir / "tasks" / "08-cron" / "cronlib").mkdir(parents=True)
    (codr_dir / "tasks" / "08-cron" / "cronlib" / "cron.py").write_text("x = 1\n\n")
    result = await StagrToolHandler()._retrieve_from_codr("08-cron")
    assert result["status"] == "ok", result
    assert (stagr_dir / "from-codr" / "08-cron" / "cronlib" / "cron.py").read_text() == "x = 1\n\n"


class Recorder:
    def __init__(self):
        self.calls = []

    async def run(self, argv):
        self.calls.append(list(argv))
        return SSHResult(0, "allowed files.example.org until 2026-10-24 12:00 EDT\n", "")


async def test_web_access_sends_only_a_checked_domain_to_the_node(monkeypatch):
    monkeypatch.setattr(global_config, "PVE_TOOLS_ENABLED", True)
    recorder = Recorder()
    handler = StagrToolHandler(recorder)  # type: ignore[arg-type]
    ok = await handler._request_web_access("Files.Example.org.", 30, "docs")
    assert ok["status"] == "ok"
    assert recorder.calls == [["/usr/local/sbin/derpr-egress", "allow", "files.example.org", "30"]]
    for bad in ("10.0.0.70", "https://example.org", "example.org/x", "localhost", "a;b.com"):
        assert (await handler._request_web_access(bad, 30, "x"))["status"] == "error", bad
    assert (await handler._request_web_access("example.org", 5000, "x"))["status"] == "error"
    assert len(recorder.calls) == 1


def test_the_shipped_persona_validates_clean_and_ships_unrestricted():
    """stagr.json exposes exactly the five tools, all in the one `stagr`
    domain, so ToolPolicy has nothing to refuse and the persona is not
    quarantined at load. The file is public, so the operator sets
    origin_allowlist on their own instance (as for hypr)."""
    persona = json.loads(_STAGR_JSON.read_text(encoding="utf-8"))["personas"][0]
    policy = ToolPolicy(**persona["tool_policy"])
    exposed = policy.filter_tools([t for t in ALL_TOOL_DEFINITIONS
                                   if t.get("service_binding") in (None, *persona["service_bindings"])])
    names = {t["function"]["name"] for t in exposed}
    assert names == {t["function"]["name"] for t in STAGR_TOOLS} == set(persona["enabled_tools"])
    assert policy.explicit_overrides == [] and policy.validate_composition(exposed) == []
    assert {t["function"]["name"] for t in STAGR_TOOLS if t["is_write"]} == {
        "request_web_access", "deliver_to_codr", "retrieve_from_codr"}
    assert persona["origin_allowlist"] == []
