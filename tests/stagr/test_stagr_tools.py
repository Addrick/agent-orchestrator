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
from src.engine.providers.opencode import (
    AUTO_FIELD, session_title, workspace_directory, workspace_hold, workspace_lock,
)
from src.agents.codr_agent import load_tasks
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


def _codr_lock():
    return workspace_lock(workspace_directory("codr")[2])


async def test_codr_shell_runs_in_codr_workspace_and_holds_its_lock(boxes):
    stagr, codr, stagr_dir, codr_dir = boxes
    seen = []
    inner = codr.shell_output

    def shell(script):
        hold = workspace_hold("codr")
        seen.append((_codr_lock().locked(), hold and hold.holder, hold and hold.max_seconds))
        return inner(script)

    codr.shell_output = shell
    result = await StagrToolHandler()._codr_shell("touch made-here; echo hi; exit 4", 120)
    assert result["status"] == "ok" and result["exit_code"] == 4 and "hi" in result["output"]
    assert (codr_dir / "made-here").exists() and not (stagr_dir / "made-here").exists()
    # codr starts nothing while it runs, and a chat to codr is told why (DP-466)
    assert seen == [(True, "stagr's codr_shell", 180)]
    assert not _codr_lock().locked() and workspace_hold("codr") is None


async def test_codr_shell_is_refused_while_codr_works(boxes):
    stagr, codr, stagr_dir, codr_dir = boxes
    codr.busy.add("ses_1")
    result = await StagrToolHandler()._codr_shell("touch x")
    assert result["status"] == "error" and "idle" in result["message"]
    assert not (codr_dir / "x").exists()
    assert workspace_hold("codr") is None


async def test_codr_shell_is_refused_while_a_codr_turn_starts(boxes):
    stagr, codr, stagr_dir, codr_dir = boxes
    async with _codr_lock():
        result = await StagrToolHandler()._codr_shell("touch x")
    assert result["status"] == "error" and "idle" in result["message"]
    assert not (codr_dir / "x").exists()


async def test_codr_shell_waits_for_an_ended_task_turn_to_be_checked(boxes):
    """A task turn that has ended is idle, but its check runs on the agent's
    next tick: changing the box before then changes what is measured."""
    stagr, codr, stagr_dir, codr_dir = boxes
    sid = "ses_task"
    codr.sessions[sid] = {"id": sid, "directory": workspace_directory("codr")[2],
                          "title": session_title("codr"), "time": {"created": 1, "updated": 2}}
    codr.messages[sid] = []
    start = codr.add(sid, "user", [{"type": "text", "text": "do 01",
                                    "metadata": {AUTO_FIELD: {"task": "01", "attempt": 1}}}])
    codr.add(sid, "assistant", [{"type": "text", "text": "done"}],
             time={"created": 1, "completed": 2}, finish="stop")
    result = await StagrToolHandler()._codr_shell("touch x")
    assert result["status"] == "error" and "not checked" in result["message"]
    assert not (codr_dir / "x").exists()
    start["parts"][0]["metadata"][AUTO_FIELD]["result"] = "pass"      # the check ran
    result = await StagrToolHandler()._codr_shell("touch x")
    assert result["status"] == "ok" and (codr_dir / "x").exists()


async def test_a_git_folder_is_not_silently_dropped(boxes):
    stagr, codr, stagr_dir, codr_dir = boxes
    (stagr_dir / "repo" / ".git").mkdir(parents=True)
    result = await StagrToolHandler()._copy_to_codr("repo/.git", "tools/g")
    assert result["status"] == "error" and "keep_git" in result["message"]


@pytest.mark.skipif(subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
                    reason="needs git")
async def test_a_copy_to_codr_leaves_upstream_history_behind(boxes):
    """A checkout at the pre-fix commit still carries the fix in .git; codr
    must get the files and none of that history."""
    stagr, codr, stagr_dir, codr_dir = boxes
    _upstream_repo(stagr_dir / "repo")
    result = await StagrToolHandler()._copy_to_codr("repo", "tasks/09-c1")
    assert result["status"] == "ok", result
    task = codr_dir / "tasks" / "09-c1"
    assert "a - b" in (task / "lib.py").read_text()
    assert not (task / ".git").exists()
    assert [p.name for p in codr_dir.iterdir()] == ["tasks"]    # staged files removed
    assert sorted(p.name for p in stagr_dir.iterdir()) == ["repo"]


@pytest.mark.skipif(subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
                    reason="needs git")
async def test_keep_git_copies_the_history(boxes):
    stagr, codr, stagr_dir, codr_dir = boxes
    _upstream_repo(stagr_dir / "repo")
    result = await StagrToolHandler()._copy_to_codr("repo", "tools/repo", keep_git="true")
    assert result["status"] == "ok" and result["git"] == "kept", result
    assert "THE FIX" in _git(codr_dir / "tools" / "repo", "log", "--all", "--format=%s")


async def test_a_single_file_copies_to_a_new_name(boxes):
    stagr, codr, stagr_dir, codr_dir = boxes
    (stagr_dir / "wheels").mkdir()
    (stagr_dir / "wheels" / "a.whl").write_bytes(b"\x00wheel\n\n")
    result = await StagrToolHandler()._copy_to_codr("wheels/a.whl", "tools/w/b.whl")
    assert result["status"] == "ok", result
    assert (codr_dir / "tools" / "w" / "b.whl").read_bytes() == b"\x00wheel\n\n"


async def test_a_copy_to_codr_marks_codr_held_only_while_it_unpacks(boxes):
    """DP-466: the existence check runs before the lock; the unpack under it
    carries the mark a chat to codr is refused on."""
    stagr, codr, stagr_dir, codr_dir = boxes
    (stagr_dir / "repo").mkdir()
    (stagr_dir / "repo" / "a.txt").write_text("a\n")
    holders = []
    inner = codr.shell_output

    def shell(script):
        hold = workspace_hold("codr")
        holders.append(hold and hold.holder)
        return inner(script)

    codr.shell_output = shell
    result = await StagrToolHandler()._copy_to_codr("repo", "tasks/09-c1")
    assert result["status"] == "ok", result
    assert holders[0] is None and holders[-1] == "stagr's copy_to_codr"
    assert workspace_hold("codr") is None


async def test_a_copy_to_codr_never_replaces_what_is_there(boxes):
    stagr, codr, stagr_dir, codr_dir = boxes
    (stagr_dir / "repo").mkdir()
    (codr_dir / "tasks" / "09-c1").mkdir(parents=True)
    (codr_dir / "tasks" / "09-c1" / "work.py").write_text("codr's work\n")
    result = await StagrToolHandler()._copy_to_codr("repo", "tasks/09-c1")
    assert result["status"] == "error" and "already exists" in result["message"]
    assert [p.name for p in (codr_dir / "tasks" / "09-c1").iterdir()] == ["work.py"]


async def test_a_copy_to_codr_waits_for_codr_to_be_idle(boxes):
    stagr, codr, stagr_dir, codr_dir = boxes
    (stagr_dir / "repo").mkdir()
    codr.busy.add("ses_1")
    result = await StagrToolHandler()._copy_to_codr("repo", "tasks/09-c1")
    assert result["status"] == "error" and "idle" in result["message"]
    assert not (codr_dir / "tasks").exists()


@pytest.mark.parametrize("path", ["../etc", "/etc", "a/../../b", "a b", "$(id)", ".", ""])
async def test_copies_refuse_paths_outside_the_workdir(boxes, path):
    handler = StagrToolHandler()
    for result in (await handler._copy_to_codr(path, "tasks/x"), await handler._copy_to_codr("repo", path),
                   await handler._copy_from_codr(path, "x"), await handler._copy_from_codr("tasks/x", path)):
        assert result["status"] == "error" and "relative" in result["message"], (path, result)


async def test_a_copy_from_codr_lands_on_stagr_and_never_overwrites(boxes):
    stagr, codr, stagr_dir, codr_dir = boxes
    (codr_dir / "tasks" / "08-cron" / "cronlib").mkdir(parents=True)
    (codr_dir / "tasks" / "08-cron" / "cronlib" / "cron.py").write_text("x = 1\n\n")
    handler = StagrToolHandler()
    result = await handler._copy_from_codr("tasks/08-cron", "from-codr/08-cron")
    assert result["status"] == "ok", result
    assert (stagr_dir / "from-codr" / "08-cron" / "cronlib" / "cron.py").read_text() == "x = 1\n\n"
    again = await handler._copy_from_codr("tasks/08-cron", "from-codr/08-cron")
    assert again["status"] == "error" and "already exists" in again["message"]


async def test_a_missing_source_is_reported(boxes):
    result = await StagrToolHandler()._copy_from_codr("tasks/nope", "x")
    assert result["status"] == "error" and "could not pack tasks/nope" in result["message"]


def test_the_shipped_persona_validates_clean_and_ships_unrestricted():
    """stagr.json exposes exactly its eight tools, all in the one `stagr`
    domain, so ToolPolicy has nothing to refuse and the persona is not
    quarantined at load. The panel's two network tools (DP-465) are not among
    them: stagr asks for a switch, it never flips one. The file is public, so
    the operator sets origin_allowlist on their own instance (as for hypr)."""
    persona = json.loads(_STAGR_JSON.read_text(encoding="utf-8"))["personas"][0]
    policy = ToolPolicy(**persona["tool_policy"])
    exposed = policy.filter_tools([t for t in ALL_TOOL_DEFINITIONS
                                   if t.get("service_binding") in (None, *persona["service_bindings"])])
    names = {t["function"]["name"] for t in exposed}
    panel_only = {"sandbox_net_status", "set_sandbox_net"}
    assert names == {t["function"]["name"] for t in STAGR_TOOLS} - panel_only == set(persona["enabled_tools"])
    assert policy.explicit_overrides == [] and policy.validate_composition(exposed) == []
    assert {t["function"]["name"] for t in STAGR_TOOLS if t["is_write"]} == {
        "request_web_access", "request_network", "copy_from_codr", "set_sandbox_net", "queue_codr_task"}
    assert persona["origin_allowlist"] == []


# --- DP-481: the codr task queue, and no codr started behind DERPR's back -----


@pytest.mark.parametrize("command", [
    "opencode run --attach http://127.0.0.1:4096 --dir tasks/x 'Work the task'",
    "cd tasks/x && ~/.opencode/bin/opencode run hi",
    "nohup opencode serve &",
    "opencode",
])
async def test_codr_shell_refuses_to_run_opencode(boxes, command):
    _, codr, _, _ = boxes
    result = await StagrToolHandler()._codr_shell(command)
    assert result["status"] == "error" and "queue_codr_task" in result["message"]
    assert codr.scripts == []


async def test_codr_shell_still_reads_opencode_files(boxes):
    _, _, _, codr_dir = boxes
    (codr_dir / ".opencode").mkdir()
    (codr_dir / ".opencode" / "opencode.json").write_text("{}")
    result = await StagrToolHandler()._codr_shell("cat .opencode/opencode.json")
    assert result["status"] == "ok" and result["output"].strip() == "{}"


@pytest.fixture
def queue(monkeypatch, tmp_path):
    folder = tmp_path / "codr_tasks"
    monkeypatch.setattr(global_config, "CODR_TASKS_DIR", folder)
    return folder


async def test_a_queued_task_is_one_the_codr_agent_starts(queue):
    result = await StagrToolHandler()._queue_codr_task(
        "09-click", "# Fix the bug\nIn tasks/09-click.", "python3 -I -m unittest\n", max_attempts=1)
    assert result["status"] == "ok"
    [task] = load_tasks(queue, default_attempts=3)
    assert (task.id, task.text, task.check, task.max_attempts) == (
        "09-click", "# Fix the bug\nIn tasks/09-click.", "python3 -I -m unittest\n", 1)
    assert [p.name for p in queue.iterdir()] == ["09-click"]  # no staging folder left


async def test_queueing_never_replaces_a_task(queue):
    handler = StagrToolHandler()
    assert (await handler._queue_codr_task("t", "first", "true"))["status"] == "ok"
    result = await handler._queue_codr_task("t", "second", "false")
    assert result["status"] == "error" and "new name" in result["message"]
    assert (queue / "t" / "TASK.md").read_text() == "first\n"


@pytest.mark.parametrize("name", ["", ".hidden", "-x", "../out", "a/b", "a b"])
async def test_queueing_refuses_a_bad_name(queue, name):
    result = await StagrToolHandler()._queue_codr_task(name, "task", "true")
    assert result["status"] == "error"
    assert not queue.exists() or list(queue.iterdir()) == []


def test_load_tasks_skips_a_half_staged_task(tmp_path):
    staged = tmp_path / ".queue-1234"
    staged.mkdir()
    (staged / "TASK.md").write_text("x")
    (staged / "check.sh").write_text("true")
    assert load_tasks(tmp_path, default_attempts=3) == []
