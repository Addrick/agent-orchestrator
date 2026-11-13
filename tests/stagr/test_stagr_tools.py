# tests/stagr/test_stagr_tools.py
#
# DP-456 — the stagr tools, against two fake `opencode serve`s (stagr and
# codr's sandbox) whose shell endpoints run the scripts in a real bash, each in
# its own folder. What matters is what lands where, so the transfers are run
# for real rather than asserted on as script text.

import asyncio
import json
import shutil
import subprocess
from src.deferral_kinds import DEFERRAL_KIND_CODR_IDLE, DEFERRAL_KIND_CODR_TICKET, declared_deferral
from pathlib import Path

import pytest
from aiohttp.test_utils import TestServer

from config import global_config
from src.engine.providers.opencode import (
    AUTO_FIELD, session_title, workspace_directory, workspace_hold, workspace_lock,
)
from src.agents import codr_agent
from src.agents.codr_agent import CodrAgent, load_tasks
from src.memory.memory_manager import MemoryManager
from src.stagr.codr_ticket import ATTEMPT_ACTION as CODR_ATTEMPT_ACTION, CODR_AGENT_NAME
from src.stagr.handler import StagrToolHandler
from src.tool_policy import ToolPolicy
from src.tools.tool_loop import _render_confirmation_text
from src.tools.tool_manager import tool_error
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


def _waits_for_codr(result):
    """DP-517: a call that found codr busy is a wait, not an error: it
    declares a `codr_idle` deferral, so stagr is woken when codr is idle."""
    assert result["status"] == "waiting" and "re-invoked" in result["message"], result
    assert tool_error({"result": result}) is None          # parked, not reported as a failure
    kind, token = declared_deferral(result)
    assert kind == DEFERRAL_KIND_CODR_IDLE and token.startswith("codr-idle-")
    return token


async def test_codr_shell_waits_while_codr_works(boxes):
    stagr, codr, stagr_dir, codr_dir = boxes
    codr.busy.add("ses_1")
    result = await StagrToolHandler()._codr_shell("touch x")
    _waits_for_codr(result)
    assert not (codr_dir / "x").exists()
    assert workspace_hold("codr") is None


async def test_codr_shell_waits_while_a_codr_turn_starts(boxes):
    stagr, codr, stagr_dir, codr_dir = boxes
    async with _codr_lock():
        result = await StagrToolHandler()._codr_shell("touch x")
    _waits_for_codr(result)
    assert not (codr_dir / "x").exists()


async def test_each_wait_gets_its_own_token(boxes):
    stagr, codr, stagr_dir, codr_dir = boxes
    codr.busy.add("ses_1")
    handler = StagrToolHandler()
    first = _waits_for_codr(await handler._codr_shell("true"))
    assert _waits_for_codr(await handler._codr_shell("true")) != first


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
    _waits_for_codr(result)
    assert "not checked" in result["message"]
    assert not (codr_dir / "x").exists()
    start["parts"][0]["metadata"][AUTO_FIELD]["result"] = "pass"      # the check ran
    result = await StagrToolHandler()._codr_shell("touch x")
    assert result["status"] == "ok" and (codr_dir / "x").exists()


async def test_a_superseded_open_task_turn_does_not_hold_the_box(boxes):
    """DP-517: an older task turn left open in a session where the agent has
    since started a newer one will never be closed; it must not keep codr's
    box looking busy (CD-004, -007, -024, -025 on the live box)."""
    stagr, codr, stagr_dir, codr_dir = boxes
    sid = "ses_task"
    codr.sessions[sid] = {"id": sid, "directory": workspace_directory("codr")[2],
                          "title": session_title("codr") + ":CD-004", "time": {"created": 1, "updated": 2}}
    codr.messages[sid] = []
    codr.add(sid, "user", [{"type": "text", "text": "do CD-004",
                            "metadata": {AUTO_FIELD: {"task": "CD-004", "attempt": 1}}}])     # lost
    codr.add(sid, "assistant", [{"type": "text", "text": "done"}],
             time={"created": 1, "completed": 2}, finish="stop")
    codr.add(sid, "user", [{"type": "text", "text": "do CD-004",
                            "metadata": {AUTO_FIELD: {"task": "CD-004", "attempt": 1, "result": "pass"}}}])
    codr.add(sid, "assistant", [{"type": "text", "text": "done"}],
             time={"created": 3, "completed": 4}, finish="stop")
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
    _waits_for_codr(result)
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
    policy = ToolPolicy.from_dict(persona["tool_policy"])
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


# --- DP-489: stagr writes codr tickets, DERPR picks the CD-NNN ------------------


@pytest.fixture
def action_log():
    """A real action log, so the codr agent's own rows are what is read."""
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()
    yield mm
    mm.close()


def _attempted(mm, task_id, outcome="fail"):
    """An attempt row exactly as the codr agent writes it."""
    mm.log_agent_action(agent_name=CodrAgent.agent_name, action_type=codr_agent.ATTEMPT_ACTION,
                        trigger_context=f"task:{task_id}", outcome=outcome)


_TICKET = dict(title="Fix the --count flag", project="click-cli", spec="It must count.",
               plan="Edit cli.py; risk: none.", packages="none", check_sh="python3 -I -m unittest\n")


async def _queue(handler, **fields):
    return await handler._queue_codr_task(**{**_TICKET, **fields})


async def test_a_queued_ticket_declares_a_codr_ticket_deferral(queue, action_log):
    """DP-493: the ticket ID is the token the codr agent settles, so the
    conversation that queued it hears its final outcome."""
    result = await _queue(StagrToolHandler(memory_manager=action_log))
    assert declared_deferral(result) == (DEFERRAL_KIND_CODR_TICKET, "CD-001")


async def test_a_refused_ticket_declares_nothing(queue, action_log):
    result = await _queue(StagrToolHandler(memory_manager=action_log), project="Not A Project")
    assert result["status"] == "error" and declared_deferral(result) is None


def test_the_shared_attempt_row_names_match_the_codr_agent():
    """stagr reads the codr agent's attempt rows by these names; src.stagr
    cannot import them from src.agents, so they are pinned here instead."""
    assert (CODR_AGENT_NAME, CODR_ATTEMPT_ACTION) == (CodrAgent.agent_name, codr_agent.ATTEMPT_ACTION)


async def test_a_ticket_is_written_in_the_shared_format(queue, action_log):
    handler = StagrToolHandler(memory_manager=action_log)
    assert (await _queue(handler))["queued"] == "CD-001"
    result = await _queue(handler, title="Add --total", depends_on=["CD-001"], max_attempts=2,
                          max_seconds=1800,
                          packages="python3-click (apt), installed and imported on codr")
    assert result["status"] == "ok" and result["queued"] == "CD-002" and "CD-002" in result["note"]
    assert (queue / "CD-002" / "TASK.md").read_text(encoding="utf-8") == (
        "title: Add --total\n"
        "project: click-cli\n"
        "depends_on: CD-001\n"
        "max_attempts: 2\n"
        "max_seconds: 1800\n"
        "\n"
        "## Spec\n\nIt must count.\n\n"
        "## Plan\n\nEdit cli.py; risk: none.\n\n"
        "## Packages\n\npython3-click (apt), installed and imported on codr\n")
    assert (queue / "CD-001" / "TASK.md").read_text(encoding="utf-8").startswith(
        "title: Fix the --count flag\nproject: click-cli\n\n## Spec\n")
    assert (queue / "CD-002" / "check.sh").read_text(encoding="utf-8") == "python3 -I -m unittest\n"
    # The codr agent sees both, and no staging folder is left behind.
    tasks = load_tasks(queue, default_attempts=3)
    assert [t.id for t in tasks] == ["CD-001", "CD-002"]
    assert tasks[1].max_seconds == 1800
    assert sorted(p.name for p in queue.iterdir() if p.is_dir()) == ["CD-001", "CD-002"]


async def test_the_next_id_follows_the_highest_ever_seen(queue, action_log):
    """Gaps are not refilled, and a number lives on in the action log, in a
    folder being staged, and in the high-water mark after its folder is gone."""
    (queue / "CD-003").mkdir(parents=True)
    (queue / "01-battery").mkdir()
    handler = StagrToolHandler(memory_manager=action_log)
    assert (await _queue(handler))["queued"] == "CD-004"
    _attempted(action_log, "CD-011")
    assert (await _queue(handler))["queued"] == "CD-012"
    (queue / ".CD-020.queue-1a2b3c4d").mkdir()
    assert (await _queue(handler))["queued"] == "CD-021"
    for name in ("CD-003", "CD-004", "CD-012", "CD-021", ".CD-020.queue-1a2b3c4d"):
        shutil.rmtree(queue / name)
    assert (await _queue(handler))["queued"] == "CD-022"


async def test_ids_run_past_three_digits(queue, action_log):
    _attempted(action_log, "CD-999")
    assert (await _queue(StagrToolHandler(memory_manager=action_log)))["queued"] == "CD-1000"


async def test_two_queue_calls_at_once_get_two_ids(queue, action_log):
    """Allocation is under a lock: calls racing from several threads still
    get one ID each."""
    handler = StagrToolHandler(memory_manager=action_log)
    results = await asyncio.gather(*(asyncio.to_thread(asyncio.run, _queue(handler, title=f"t{i}"))
                                     for i in range(4)))
    assert sorted(r["queued"] for r in results) == ["CD-001", "CD-002", "CD-003", "CD-004"]


async def test_depends_on_must_name_a_known_ticket(queue, action_log):
    handler = StagrToolHandler(memory_manager=action_log)
    (queue / "03-cron").mkdir(parents=True)           # a hand-written ticket counts
    (queue / "03-cron" / "TASK.md").write_text("Build it.\n", encoding="utf-8")
    (queue / "03-cron" / "check.sh").write_text("exit 0\n", encoding="utf-8")
    (queue / "04-empty").mkdir()                      # a folder codr would not load does not
    _attempted(action_log, "CD-007", outcome="pass")  # so does one only in the log
    ok = await _queue(handler, depends_on="CD-007, 03-cron")
    assert ok["status"] == "ok"
    assert "depends_on: CD-007, 03-cron\n" in (queue / ok["queued"] / "TASK.md").read_text(encoding="utf-8")
    bad = await _queue(handler, depends_on=["CD-007", "CD-404"])
    assert bad["status"] == "error" and "CD-404" in bad["message"]
    empty = await _queue(handler, depends_on="04-empty")
    assert empty["status"] == "error" and "04-empty" in empty["message"]
    assert not (queue / "CD-009").exists()


@pytest.mark.parametrize("fields, word", [
    ({"project": "Click_CLI"}, "project"),
    ({"project": ""}, "project"),
    ({"project": "../x"}, "project"),
    ({"title": ""}, "title"),
    ({"title": "two\nlines"}, "title"),
    ({"spec": " "}, "spec"),
    ({"plan": ""}, "plan"),
    ({"packages": ""}, "packages"),
    ({"check_sh": ""}, "check_sh"),
    ({"depends_on": ["../x"]}, "depends_on"),
    ({"max_attempts": 0}, "max_attempts"),
    ({"max_attempts": "lots"}, "max_attempts"),
    ({"max_attempts": True}, "max_attempts"),
    ({"max_seconds": 0}, "max_seconds"),
    ({"max_seconds": "fast"}, "max_seconds"),
    ({"max_seconds": True}, "max_seconds"),
    ({"max_seconds": 1.5}, "max_seconds"),
    ({"max_seconds": float("inf")}, "max_seconds"),
])
async def test_a_bad_ticket_is_refused_and_writes_nothing(queue, action_log, fields, word):
    result = await _queue(StagrToolHandler(memory_manager=action_log), **fields)
    assert result["status"] == "error" and word in result["message"]
    assert not queue.exists() or list(queue.iterdir()) == []


async def test_without_the_action_log_nothing_is_queued(queue):
    """Without it an ID could repeat one the codr agent already used."""
    result = await _queue(StagrToolHandler())
    assert result["status"] == "error" and "action log" in result["message"]
    assert not any(p.is_dir() for p in queue.iterdir())


async def test_the_approval_prompt_shows_the_whole_ticket_and_its_id(queue, action_log):
    """The prompt prints every argument; the enrichment adds the ID it would
    get, or why it would be refused, before the operator approves."""
    handler = StagrToolHandler(memory_manager=action_log)
    _attempted(action_log, "CD-004")
    args = {**_TICKET, "depends_on": ["CD-004"]}
    enrichment = await handler._queue_preview(**args)
    assert enrichment == "new ticket CD-005, project click-cli"
    prompt = _render_confirmation_text(
        {"tool": "queue_codr_task", "arguments": args, "irreversible": False, "always_confirm": False,
         "service_binding": "stagr", "sensitivity": None, "enrichment": enrichment}, False, [])
    for value in ("CD-005", "Fix the --count flag", "click-cli", "CD-004", "It must count.",
                  "Edit cli.py; risk: none.", '"packages": "none"', "python3 -I -m unittest"):
        assert value in prompt
    refused = await handler._queue_preview(**{**args, "depends_on": ["CD-404"]})
    assert refused is not None and refused.startswith("⚠️") and "CD-404" in refused
    bad_project = await handler._queue_preview(**{**args, "project": "Bad"})
    assert bad_project is not None and bad_project.startswith("⚠️")


def test_the_tool_schema_takes_the_ticket_fields():
    [tool] = [t for t in STAGR_TOOLS if t["function"]["name"] == "queue_codr_task"]
    params = tool["function"]["parameters"]
    assert set(params["properties"]) == {"title", "project", "depends_on", "spec", "plan", "packages",
                                         "check_sh", "max_attempts", "max_seconds"}
    assert set(params["required"]) == {"title", "project", "spec", "plan", "packages", "check_sh"}
    assert tool["is_write"] is True


def test_the_stagr_prompt_carries_the_package_recipe():
    """DP-479: packages are resolved against codr's state, not stagr's."""
    prompt = json.loads(_STAGR_JSON.read_text(encoding="utf-8"))["personas"][0]["prompt"]
    for needle in ("Dir::State::status", "dpkg --print-foreign-architectures", "--only-binary=:all:",
                   "--python-version", "--implementation", "--abi", "--platform", "queue_codr_task",
                   "projects/<project>", "keep_git"):
        assert needle in prompt


def test_load_tasks_skips_a_half_staged_task(tmp_path):
    staged = tmp_path / ".CD-001.queue-1234"
    staged.mkdir()
    (staged / "TASK.md").write_text("x")
    (staged / "check.sh").write_text("true")
    assert load_tasks(tmp_path, default_attempts=3) == []
