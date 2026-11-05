# tests/test_codr_agent.py
#
# DP-449 — the codr agent, ticked against the fake `opencode serve` from the
# provider tests. Each test drives deploy() and plays the server's side of a
# turn by adding messages, the way opencode records them.

import json
import subprocess
import time
from types import SimpleNamespace

import pytest

from src.agents import codr_agent
from src.agents.codr_agent import (
    CodrAgent, detect_loop, load_tasks, parse_check, sync_script, validate_mcp, workspace_files,
)
from src.engine.providers import opencode
from src.generation_events import ResponseType
from src.turn_persistence import TurnPersistence
from tests import test_opencode_provider as provider_tests
from tests.test_opencode_provider import ROOT, real_bash

fake = provider_tests.fake  # the fake `opencode serve` fixture

AUTO = opencode.AUTO_FIELD


@pytest.fixture(autouse=True)
def _queue(monkeypatch, tmp_path):
    """Each test's task queue is its own tmp_path."""
    monkeypatch.setattr(codr_agent.global_config, "CODR_TASKS_DIR", tmp_path)


class FakeMemory:
    """The agent-action-log calls the agent makes."""

    def __init__(self):
        self.rows = []

    def log_agent_action(self, agent_name, action_type, trigger_context=None, action_payload=None,
                         outcome=None, outcome_payload=None, parent_id=None):
        self.rows.append({"id": len(self.rows) + 1, "agent_name": agent_name, "action_type": action_type,
                          "trigger_context": trigger_context, "action_payload": action_payload,
                          "outcome": outcome, "outcome_payload": outcome_payload})
        return len(self.rows)

    def update_agent_action_outcome(self, action_id, outcome, outcome_payload=None):
        self.rows[action_id - 1].update(outcome=outcome, outcome_payload=outcome_payload)

    def add_action_contexts(self, action_id, contexts):
        pass

    def get_agent_actions(self, agent_name, limit=20, action_type=None):
        rows = [r for r in self.rows if r["agent_name"] == agent_name
                and (action_type is None or r["action_type"] == action_type)]
        return list(reversed(rows))[:limit]          # newest first, like the DB

    # The interaction rows a ticket's channel gets (DP-484).
    interactions: list

    def _rows(self):
        if not hasattr(self, "interactions"):
            self.interactions = []
        return self.interactions

    def get_interaction_by_platform_id(self, pid):
        row = next((r for r in self._rows() if r["platform_message_id"] == pid), None)
        return None if row is None else {"interaction_id": row["interaction_id"], "content": row["content"],
                                         "reasoning_content": row["reasoning_content"],
                                         "tool_context": row["tool_context"]}

    def log_message(self, **kw):
        if kw.get("platform_message_id") and self.get_interaction_by_platform_id(kw["platform_message_id"]):
            raise RuntimeError("UNIQUE constraint failed: platform_message_id")
        row = {"interaction_id": len(self._rows()) + 1, "tool_context": None, "reasoning_content": None, **kw}
        self._rows().append(row)
        return row["interaction_id"]

    def update_interaction_content(self, interaction_id, content, reasoning_content=None, tool_context=None):
        row = self._rows()[interaction_id - 1]
        row.update(content=content, reasoning_content=reasoning_content, tool_context=tool_context)
        return True

    def channel(self, name):
        return [r for r in self._rows() if r["channel"] == name]


class Persistence:
    """The TurnPersistence calls a relayed reply makes; the keyed rows of a
    ticket's channel (DP-484) go through the real method into FakeMemory."""

    upsert_keyed_row = TurnPersistence.upsert_keyed_row

    def __init__(self, memory_manager=None):
        self.committed, self.queued = [], []
        self.memory_manager = memory_manager

    def commit_or_update_assistant(self, **kw):
        self.committed.append(kw)
        return 100 + len(self.committed)

    def queue_retain_safe(self, **kw):
        self.queued.append(kw)


class Router:
    def __init__(self):
        self.sent = []
        self.ok = True
        self.keys = {"123": ("codr-chat", "9")}     # notification recipient -> conversation

    async def send(self, channel, recipient, subject, body):
        if self.ok:
            self.sent.append((channel, recipient, body))
        return self.ok

    async def conversation_key(self, channel, recipient):
        return self.keys.get(recipient)


def _task(tmp_path, name, text="Build it.", check="exit 0", seed=None):
    folder = tmp_path / name
    folder.mkdir(parents=True)
    (folder / "TASK.md").write_text(text)
    (folder / "check.sh").write_text(check)
    for rel, content in (seed or {}).items():
        (folder / "seed" / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / "seed" / rel).write_bytes(content.encode())


def _agent(tmp_path, **cfg):
    persona = SimpleNamespace(get_prompt=lambda: "You are codr.", get_model_name=lambda: "opencode")
    memory = FakeMemory()
    chat = SimpleNamespace(text_engine=None, memory_manager=memory, personas={"codr": persona},
                           turn_persistence=Persistence(memory))
    router = Router()
    agent = CodrAgent(chat, router, {
        "_recipients": {},
        "notification_target": {"channel": "discord_channel", "recipient": "123"}, **cfg})
    return agent, router


def _shell(check_passes=True, merge="__DERPR_MAIN=main\n__DERPR_MERGED=abc1234\n"):
    """Each workspace script's success marker, standing in for codr's box."""
    def run(script):
        if "DERPR_EOF_0" in script:
            return "__DERPR_CHANGED=0\n"
        if "__DERPR_EXIT=" in script:
            return ("PASS\n__DERPR_EXIT=0\n" if check_passes
                    else "FAIL: csvstats.py missing\n__DERPR_EXIT=1\n")
        if "__DERPR_MERGED=" in script:
            return merge
        for mark in ("__DERPR_TICKET_READY", "__DERPR_MEMORY_READY", "__DERPR_NOTED", "__DERPR_WRITTEN"):
            if f"echo {mark}" in script:
                return mark + "\n"
        return ""
    return run


def _session(fake, ticket=None):
    """codr's own session, or a ticket's (DP-484)."""
    title = "derpr:codr" + (f":{ticket}" if ticket else "")
    (sid,) = [s for s in fake.sessions if fake.sessions[s]["title"] == title]
    return sid


def _all_auto(fake):
    """Every task prompt in any session, in the order they were posted."""
    msgs = [m for sid in fake.sessions for m in _auto_messages(fake, sid)]
    return sorted(msgs, key=lambda m: int(m["info"]["id"].split("_")[1]))


def _auto_messages(fake, sid):
    return [m for m in fake.messages[sid] if opencode.message_meta(m, AUTO)]


def _reply(fake, sid, text="done, tests pass"):
    fake.add(sid, "assistant", [{"type": "text", "text": text}],
             time={"created": 1, "completed": 2}, finish="stop")


def _tool_step(fake, sid, tool="bash", inp=None, reasoning=None):
    parts = [{"type": "reasoning", "text": reasoning}] if reasoning else []
    parts.append({"type": "tool", "tool": tool, "state": {"status": "completed", "input": inp or {"command": "ls"}}})
    fake.add(sid, "assistant", parts, time={"created": 1, "completed": 2}, finish="tool-calls")


# --- pure parts ----------------------------------------------------------------


def test_load_tasks_reads_the_attempts_header_and_seed(tmp_path):
    _task(tmp_path, "02-b", text="max_attempts: 5\nFix it.", seed={"shop/cart.py": "x = 1\n"})
    _task(tmp_path, "01-a")
    (tmp_path / "03-incomplete").mkdir()                       # no check.sh: not a task
    a, b = load_tasks(tmp_path, 3)
    assert (a.id, a.max_attempts, a.seed) == ("01-a", 3, ())
    assert (b.id, b.max_attempts, b.text) == ("02-b", 5, "Fix it.")
    assert b.seed == (("shop/cart.py", b"x = 1\n"),)


def test_check_script_keeps_stdin_away_from_the_check(tmp_path):
    # DP-463: a command that reads stdin must not swallow the rest of the check.
    (tmp_path / "worktrees" / "t").mkdir(parents=True)
    task = codr_agent.CodrTask(id="t", text="", check="cat\necho after\nexit 3", max_attempts=1)
    out = real_bash(tmp_path)(codr_agent.check_script(task, 30))
    assert parse_check(out) == (False, "after\n")
    assert out.endswith("__DERPR_EXIT=3\n")


def test_parse_check():
    assert parse_check("PASS\n__DERPR_EXIT=0\n") == (True, "PASS\n")
    assert parse_check("FAIL: x\n__DERPR_EXIT=1\n") == (False, "FAIL: x\n")
    assert parse_check("no tasks/x folder")[0] is False        # bailed before the marker


def test_detect_loop_sees_repeats_across_steps():
    def step(cmd, reasoning=None):
        parts = [{"type": "reasoning", "text": reasoning}] if reasoning else []
        parts.append({"type": "tool", "tool": "bash", "state": {"status": "completed", "input": {"command": cmd}}})
        return {"info": {"role": "assistant"}, "parts": parts}

    assert detect_loop([step("ls"), step("ls")]) is None
    assert "same `bash` call" in detect_loop([step("pwd"), step("ls"), step("ls"), step("ls")])
    assert detect_loop([step("ls"), step("pwd"), step("ls")]) is None
    same = "I should check the file again before editing it, carefully."
    assert "same reasoning" in detect_loop([step("a", same), step("b", same), step("c", same)])


def test_workspace_files_stop_the_offline_npm_install():
    files = workspace_files({**codr_agent.DEFAULTS})
    assert {".opencode/package.json", ".opencode/package-lock.json"} <= set(files)
    config = json.loads(files[".opencode/opencode.json"])
    assert config["provider"]["kcpp-pool"]["options"]["baseURL"] == codr_agent.DEFAULTS["pool_url"]
    assert config["provider"]["kcpp-pool"]["options"]["headerTimeout"] >= 30 * 60 * 1000  # router holds (DP-490)
    assert "model: kcpp-pool/local" in files[".opencode/agent/pool.md"]
    assert "mkdir -p .opencode .opencode/agent .opencode/node_modules .opencode/tool" in sync_script(files)


def test_workspace_files_carry_the_shipped_agents_md():
    """DP-488: codr's rules ship with DERPR and sync like `.opencode/`."""
    files = workspace_files({**codr_agent.DEFAULTS})
    assert files["AGENTS.md"] == codr_agent.AGENTS_MD.read_text(encoding="utf-8")
    for rule in ("worktrees/<ticket>/", "notify_adam", "memory/MEMORY.md", "staging files by path"):
        assert rule in files["AGENTS.md"]


@pytest.mark.parametrize("mcp, error", [
    ([], "must be an object"),
    ({"a": "x"}, "mcp.a must be an object"),
    ({"a": {"type": "local"}}, "needs `command`"),
    ({"a": {"type": "local", "command": []}}, "needs `command`"),
    ({"a": {"type": "local", "command": ["ok", 3]}}, "needs `command`"),
    ({"a": {"type": "remote"}}, "needs `url`"),
    ({"a": {"type": "remote", "url": "file:///etc/passwd"}}, "needs `url`"),
    ({"a": {"type": "stdio", "command": ["x"]}}, "`type` must be"),
    ({"a": {"enabled": "no"}}, "`type` must be"),
    ({"a": {"type": "local", "command": ["x"], "enabled": "yes"}}, "`enabled` must be"),
    ({"a": {"type": "local", "command": ["x"], "timeout": "5000"}}, "`timeout` must be"),
    ({"a": {"type": "local", "command": ["x"], "environment": {"K": 1}}}, "`environment` must map"),
    ({"a": {"type": "remote", "url": "https://h/m", "headers": ["x"]}}, "`headers` must map"),
])
def test_a_malformed_mcp_entry_is_refused(mcp, error):
    with pytest.raises(ValueError, match=error):
        validate_mcp(mcp)


def test_a_valid_mcp_setting_passes():
    validate_mcp(None)
    validate_mcp({})
    validate_mcp({"a": {"type": "local", "command": ["x"]}, "b": {"type": "remote", "url": "https://h/m"},
                  "c": {"enabled": False}})


async def test_an_mcp_change_rewrites_the_config_and_reloads_codr(fake, tmp_path):
    """The `mcp` block goes through the same changed -> dispose path as every
    other managed file, against a real workspace on disk."""
    fake.shell_output = real_bash(tmp_path)
    agent, _ = _agent(tmp_path, start_tasks=False)
    await agent.deploy()
    await agent.deploy()                    # the first sync's reload, while idle
    fake.calls.clear()
    agent, _ = _agent(tmp_path, start_tasks=False, mcp={"docs": {"type": "local", "command": ["docs-mcp"]}})
    await agent.deploy()
    written = json.loads((tmp_path / ".opencode" / "opencode.json").read_text())
    assert written["mcp"] == {"docs": {"type": "local", "command": ["docs-mcp"]}}
    assert [c for c in fake.calls if c[1] == "/instance/dispose"]


async def test_a_changed_agents_md_is_rewritten_and_reloads_codr(fake, tmp_path, monkeypatch):
    """AGENTS.md is a managed file like `.opencode/`: a new text on DERPR's
    side replaces codr's copy and reloads the opencode instance."""
    fake.shell_output = real_bash(tmp_path)
    agent, _ = _agent(tmp_path, start_tasks=False)
    await agent.deploy()
    await agent.deploy()
    assert (tmp_path / "AGENTS.md").read_text() == codr_agent.AGENTS_MD.read_text(encoding="utf-8")
    fake.calls.clear()
    rules = tmp_path / "rules.md"
    rules.write_text("# new rules\n")
    monkeypatch.setattr(codr_agent, "AGENTS_MD", rules)
    agent, _ = _agent(tmp_path, start_tasks=False)
    await agent.deploy()
    assert (tmp_path / "AGENTS.md").read_text() == "# new rules\n"
    assert [c for c in fake.calls if c[1] == "/instance/dispose"]


async def _prepare(task, work):
    from aiohttp import ClientSession
    async with ClientSession(headers=opencode.auth_headers()) as http:
        await codr_agent.prepare_ticket(opencode.OpencodeClient(http, f"{ROOT}/codr"), task)


async def test_a_seed_of_any_size_arrives_and_keeps_what_codr_changed(fake, tmp_path):
    """DP-455: a seed bigger than one shell command's argument cap still
    arrives whole, and files codr already edited are left alone."""
    seed = {f"pkg/mod{i}.py": "x = 1  # " + "y" * 20_000 + "\n" for i in range(10)}
    _task(tmp_path, "09-big", seed=seed)
    (task,) = load_tasks(tmp_path, 3)
    work = tmp_path / "work"
    work.mkdir()
    fake.shell_output = real_bash(work)
    await _prepare(task, work)
    wt = work / "worktrees/09-big"
    (wt / "pkg/mod0.py").write_text("edited by codr\n")
    (wt / "pkg/mod1.py").unlink()
    await _prepare(task, work)                                 # the retry
    assert (wt / "pkg/mod0.py").read_text() == "edited by codr\n"
    assert (wt / "pkg/mod1.py").read_text() == seed["pkg/mod1.py"]   # missing: seeded again
    assert (wt / "pkg/mod9.py").read_text() == seed["pkg/mod9.py"]
    assert sorted(p.name for p in work.iterdir()) == ["memory", "projects", "worktrees"]   # staging removed


# --- ticks ---------------------------------------------------------------------


async def test_no_ticket_starts_while_stagr_is_being_woken(fake, tmp_path):
    """DP-493: a ticket queued ahead waits for stagr's wake turn to end, so
    stagr can prepare or step in before codr starts it."""
    _task(tmp_path, "CD-002", text="Build csvstats.")
    fake.shell_output = _shell()
    agent, _ = _agent(tmp_path)
    agent._waking.add("CD-001")                                # stagr's turn about CD-001 is running
    await agent.deploy()
    assert not [s for s in fake.sessions.values() if s.get("title", "").endswith(":CD-002")]

    agent._waking.clear()                                      # the turn ended
    await agent.deploy()
    assert _session(fake, "CD-002")


async def test_first_tick_sets_up_the_workspace_and_starts_the_oldest_task(fake, tmp_path):
    _task(tmp_path, "01-a", text="Build csvstats.")
    fake.shell_output = _shell()
    agent, _ = _agent(tmp_path)
    await agent.deploy()

    sid = _session(fake, "01-a")                               # the ticket's own session
    assert fake.sessions[sid]["directory"] == f"{ROOT}/codr"
    assert not _auto_messages(fake, _session(fake))            # chat's session gets no task
    scripts = [s for _, s in fake.scripts]
    assert any("DERPR_EOF_0" in s for s in scripts)            # .opencode/ synced
    assert any("echo __DERPR_MEMORY_READY" in s for s in scripts)   # memory/ skeleton
    assert any("worktree add -q -b ticket/01-a" in s for s in scripts)   # worktree cut
    (start,) = _auto_messages(fake, sid)
    part = start["parts"][0]
    assert part["metadata"][AUTO]["task"] == "01-a" and part["metadata"][AUTO]["attempt"] == 1
    assert "Build csvstats." in part["text"] and "memory/tickets/01-a.md" in part["text"]
    assert "worktrees/01-a/" in part["text"] and "AGENTS.md" in part["text"]
    assert "attempts.md" not in part["text"]
    post = next(c for c in fake.calls if c[1].endswith("/prompt_async"))
    assert post[3]["system"] == "You are codr."

    await agent.deploy()                                       # still running: nothing new
    assert len(_auto_messages(fake, sid)) == 1


async def test_a_finished_turn_is_checked_recorded_and_posted_once(fake, tmp_path):
    _task(tmp_path, "01-a")
    fake.shell_output = _shell(check_passes=True)
    agent, router = _agent(tmp_path)
    await agent.deploy()
    sid = _session(fake, "01-a")
    _reply(fake, sid)

    await agent.deploy()
    (row,) = agent.memory_manager.rows
    assert (row["trigger_context"], row["outcome"]) == ("task:01-a", "pass")
    assert _auto_messages(fake, sid)[0]["parts"][0]["metadata"][AUTO]["result"] == "pass"
    ((channel, recipient, body),) = router.sent
    assert (channel, recipient) == ("discord_channel", "123")
    assert "`01-a` attempt 1/3: **PASS**" in body
    assert "done, tests pass" not in body      # the final reply is not posted; notify_adam is codr's voice
    assert "done, tests pass" in json.loads(row["outcome_payload"])["reply"]   # kept in the attempt record

    await agent.deploy()                                       # passed: no new attempt, no repost
    assert len(_auto_messages(fake, sid)) == 1 and len(router.sent) == 1


async def test_a_failed_check_retries_with_its_output_until_attempts_run_out(fake, tmp_path):
    _task(tmp_path, "01-a", text="max_attempts: 2\nBuild it.")
    fake.shell_output = _shell(check_passes=False)
    agent, router = _agent(tmp_path)
    sid = None
    for _ in range(2):
        await agent.deploy()                                   # start
        sid = _session(fake, "01-a")                           # retries continue it
        _reply(fake, sid)
        await agent.deploy()                                   # finish: fail
    second = _auto_messages(fake, sid)[1]["parts"][0]
    assert second["metadata"][AUTO]["attempt"] == 2
    assert "FAIL: csvstats.py missing" in second["text"]
    assert [r["outcome"] for r in agent.memory_manager.rows] == ["fail", "fail"]

    await agent.deploy()                                       # out of attempts
    assert len(_auto_messages(fake, sid)) == 2
    assert "attempt 2/2: **FAIL**" in router.sent[-1][2]


async def test_a_loop_is_nudged_once_then_aborted(fake, tmp_path):
    _task(tmp_path, "01-a")
    fake.shell_output = _shell()
    agent, router = _agent(tmp_path)
    await agent.deploy()
    sid = _session(fake, "01-a")
    for _ in range(3):
        _tool_step(fake, sid)
    await agent.deploy()

    first, nudge = _auto_messages(fake, sid)
    assert first["parts"][0]["metadata"][AUTO]["result"] == "nudged"
    assert nudge["parts"][0]["metadata"][AUTO]["nudges"] == 1
    assert "loop guard" in nudge["parts"][0]["text"] and "same `bash` call" in nudge["parts"][0]["text"]
    assert [c for c in fake.calls if c[1].endswith("/abort")]
    assert not router.sent

    for _ in range(3):
        _tool_step(fake, sid, inp={"command": "cat x"})
    await agent.deploy()
    assert _auto_messages(fake, sid)[1]["parts"][0]["metadata"][AUTO]["result"] == "stuck"
    (row,) = agent.memory_manager.rows
    assert row["outcome"] == "stuck"
    assert "**STUCK**" in router.sent[-1][2] and "same `bash` call" in router.sent[-1][2]
    # DP-482: stuck ends the ticket immediately; max_attempts is 3 but no second attempt starts.
    await agent.deploy()
    assert len(_auto_messages(fake, sid)) == 2


async def test_an_overrun_is_aborted_without_a_nudge(fake, tmp_path):
    _task(tmp_path, "01-a")
    fake.shell_output = _shell()
    agent, router = _agent(tmp_path, max_turn_seconds=0)
    await agent.deploy()
    sid = _session(fake, "01-a")
    _tool_step(fake, sid)
    await agent.deploy()
    assert len(_auto_messages(fake, sid)) == 1
    assert _auto_messages(fake, sid)[0]["parts"][0]["metadata"][AUTO]["result"] == "timed_out"
    assert agent.memory_manager.rows[0]["outcome"] == "timed_out"
    assert "**TIMED_OUT**" in router.sent[-1][2] and "ran past 0 s" in router.sent[-1][2]
    # DP-482: timed_out ends the ticket immediately; max_attempts is 3 but no second attempt starts.
    await agent.deploy()
    assert len(_auto_messages(fake, sid)) == 1


async def test_a_stall_counts_as_stuck(fake, tmp_path, monkeypatch):
    _task(tmp_path, "01-a")
    fake.shell_output = _shell()
    agent, _ = _agent(tmp_path, stall_seconds=100)
    clock = [1000.0]
    monkeypatch.setattr(codr_agent.time, "monotonic", lambda: clock[0])
    await agent.deploy()
    sid = _session(fake, "01-a")
    _tool_step(fake, sid)
    await agent.deploy()                                       # progress seen
    clock[0] += 50
    await agent.deploy()
    assert len(_auto_messages(fake, sid)) == 1                 # not yet
    clock[0] += 100
    await agent.deploy()
    nudge = _auto_messages(fake, sid)[1]["parts"][0]
    assert "nothing new for 100 s" in nudge["text"]


async def test_notify_adam_is_relayed_once_and_rate_limited(fake, tmp_path):
    fake.shell_output = _shell()
    agent, router = _agent(tmp_path, notify_per_hour=1)        # no tasks: relay only
    await agent.deploy()
    sid = _session(fake)
    for text in ("tests pass, starting auth", "stuck on sqlite locking"):
        fake.add(sid, "assistant", [{"type": "tool", "tool": "notify_adam",
                                     "state": {"status": "completed", "input": {"message": text}}}],
                 time={"created": 1, "completed": 2}, finish="tool-calls")
    await agent.deploy()
    await agent.deploy()
    assert [b for _, _, b in router.sent] == ["codr: tests pass, starting auth"]
    relays = [r for r in agent.memory_manager.rows if r["action_type"] == codr_agent.RELAY_ACTION]
    assert [r["outcome"] for r in relays] == ["sent", "dropped"]
    assert agent._dropped == 1
    # The posted one (only) is codr's reply, in the conversation it was posted to.
    persistence = agent.chat_system.turn_persistence
    (row,) = persistence.committed
    assert (row["channel"], row["server_id"], row["final_text"]) == ("codr-chat", "9", "tests pass, starting auth")
    assert row["response_type"] == ResponseType.LLM_GENERATION
    assert [q["interaction_id"] for q in persistence.queued] == [101]
    # The agent's own parts are never written to: opencode sends an assistant
    # part's metadata to the model, and a foreign key there breaks the session.
    assert not [c for c in fake.calls if c[0] == "PATCH"]


async def test_a_long_notify_adam_message_is_posted_and_stored_whole(fake, tmp_path):
    """DP-485: a message past the action log's cap was cut short in Discord and
    history; the Discord client splits long posts, so nothing should be cut."""
    fake.shell_output = _shell()
    agent, router = _agent(tmp_path)
    await agent.deploy()
    sid = _session(fake)
    text = "report " + "x" * 3000 + " END"
    fake.add(sid, "assistant", [{"type": "tool", "tool": "notify_adam",
                                 "state": {"status": "completed", "input": {"message": text}}}],
             time={"created": 1, "completed": 2}, finish="tool-calls")
    await agent.deploy()
    assert [b for _, _, b in router.sent] == [f"codr: {text}"]
    (row,) = agent.chat_system.turn_persistence.committed
    assert row["final_text"] == text


async def test_a_relay_to_a_target_with_no_conversation_is_posted_but_not_stored(fake, tmp_path):
    fake.shell_output = _shell()
    agent, router = _agent(tmp_path)
    router.keys = {}
    await agent.deploy()
    sid = _session(fake)
    fake.add(sid, "assistant", [{"type": "tool", "tool": "notify_adam",
                                 "state": {"status": "completed", "input": {"message": "hi"}}}],
             time={"created": 1, "completed": 2}, finish="tool-calls")
    await agent.deploy()
    assert [b for _, _, b in router.sent] == ["codr: hi"]
    assert agent.chat_system.turn_persistence.committed == []


async def test_a_relay_that_fails_to_post_is_marked_and_reported(fake, tmp_path):
    _task(tmp_path, "01-a")
    fake.shell_output = _shell()
    agent, router = _agent(tmp_path)
    await agent.deploy()                                       # starts 01-a
    sid = _session(fake, "01-a")
    fake.add(sid, "assistant", [{"type": "tool", "tool": "notify_adam",
                                 "state": {"status": "completed", "input": {"message": "done"}}}],
             time={"created": 1, "completed": 2}, finish="tool-calls")
    router.ok = False
    await agent.deploy()
    (relay,) = [r for r in agent.memory_manager.rows if r["action_type"] == codr_agent.RELAY_ACTION]
    assert relay["outcome"] == "failed"
    assert agent.chat_system.turn_persistence.committed == []
    router.ok = True
    _reply(fake, sid)
    await agent.deploy()                                       # finish: the result line says so
    assert "1 notify_adam message(s) failed to post" in router.sent[-1][2]


async def test_a_task_turn_followed_by_a_chat_turn_is_still_finished(fake, tmp_path):
    """A chat message sent after the task turn ended but before the next tick
    starts a turn of its own; the task turn must still be checked and recorded."""
    _task(tmp_path, "01-a")
    fake.shell_output = _shell(check_passes=True)
    agent, router = _agent(tmp_path)
    await agent.deploy()
    sid = _session(fake, "01-a")
    _reply(fake, sid, "task done")
    fake.add(sid, "user", [{"type": "text", "text": "User: hi"}])
    _reply(fake, sid, "hello")

    await agent.deploy()
    (row,) = [r for r in agent.memory_manager.rows if r["action_type"] == codr_agent.ATTEMPT_ACTION]
    assert row["outcome"] == "pass"
    assert json.loads(row["outcome_payload"])["reply"] == "task done"   # its own reply, not the chat's
    assert len(_auto_messages(fake, sid)) == 1                 # passed: no restart of the attempt


async def test_a_closed_task_turn_is_not_running_even_if_its_last_step_never_completed(fake, tmp_path):
    _task(tmp_path, "01-a")
    _task(tmp_path, "02-b")
    fake.shell_output = _shell()
    agent, _ = _agent(tmp_path)
    await agent.deploy()                                       # starts 01-a
    sid = _session(fake, "01-a")
    fake.add(sid, "assistant", [{"type": "text", "text": "half"}], time={"created": 1})   # never completed
    start = _auto_messages(fake, sid)[0]["parts"][0]
    start["metadata"][AUTO]["result"] = "aborted"
    msgs = fake.messages[sid]
    assert not opencode.auto_turn_running(msgs)                # a chat message is an ordinary turn
    await agent.deploy()
    assert len(_auto_messages(fake, sid)) == 2                 # the queue moves on


async def test_the_loop_guard_holds_the_workspace_lock_until_the_nudge_is_in(fake, tmp_path, monkeypatch):
    _task(tmp_path, "01-a")
    fake.shell_output = _shell()
    agent, _ = _agent(tmp_path, stall_seconds=0)
    await agent.deploy()
    sid = _session(fake, "01-a")
    fake.busy.add(sid)
    _tool_step(fake, sid)
    seen = []
    real_prompt = agent._prompt

    async def prompt(*a, **k):
        seen.append(opencode.workspace_lock(agent.directory).locked())
        await real_prompt(*a, **k)
    monkeypatch.setattr(agent, "_prompt", prompt)
    await agent.deploy()                                       # notes progress
    agent._progress = (agent._progress[0], 0.0)                # long ago
    await agent.deploy()                                       # stalled: abort + nudge
    assert seen == [True]


def test_a_long_check_output_still_stores_valid_json(tmp_path):
    agent, _ = _agent(tmp_path)
    detail = {"reply": "→ ✓ " * 300, "check_tail": "line\n" * 1000 + "FAIL: the end"}
    fitted = agent._fit_payload(detail)
    stored = agent._serialize_payload(fitted)
    assert len(stored) <= agent.MAX_PAYLOAD_CHARS
    assert json.loads(stored)["check_tail"].endswith("FAIL: the end")   # the end says what failed


async def test_no_task_starts_while_a_chat_turn_runs(fake, tmp_path):
    _task(tmp_path, "01-a")
    fake.shell_output = _shell()
    agent, _ = _agent(tmp_path)
    await agent.deploy()                                       # creates the session, starts 01-a
    _reply(fake, _session(fake, "01-a"))
    await agent.deploy()                                       # 01-a passes
    _task(tmp_path, "02-b")
    fake.busy.add(_session(fake))                              # a chat turn is running in codr's session
    await agent.deploy()
    assert _started(fake) == ["01-a"]


async def test_start_tasks_false_only_watches(fake, tmp_path):
    _task(tmp_path, "01-a")
    fake.shell_output = _shell()
    agent, _ = _agent(tmp_path, start_tasks=False)
    await agent.deploy()
    assert not _all_auto(fake)


async def test_changed_workspace_config_reloads_the_instance_when_idle(fake, tmp_path):
    fake.shell_output = lambda s: "__DERPR_CHANGED=1\n" if "DERPR_EOF_0" in s else _shell()(s)
    agent, _ = _agent(tmp_path)
    await agent.deploy()
    disposes = [c for c in fake.calls if c[1] == "/instance/dispose"]
    assert disposes and disposes[0][2] == f"{ROOT}/codr"


@pytest.mark.parametrize("bad", ["", "__DERPR_CHANGED=\n"])
async def test_a_failed_workspace_sync_raises(fake, tmp_path, bad):
    fake.shell_output = lambda s: bad
    agent, _ = _agent(tmp_path)
    with pytest.raises(RuntimeError, match="workspace setup failed"):
        await agent.deploy()


# --- DP-488: ticket header, queue order --------------------------------------


def test_a_ticket_header_is_parsed_and_hidden_from_codr(tmp_path):
    _task(tmp_path, "CD-007", text="title: notes auth\nproject: notes\ndepends_on: CD-005, CD-006\n"
                                   "max_attempts: 4\n\n## Spec\nAdd logins.")
    (task,) = load_tasks(tmp_path, 3)
    assert (task.title, task.project, task.depends_on, task.max_attempts) == (
        "notes auth", "notes", ("CD-005", "CD-006"), 4)
    assert task.text == "## Spec\nAdd logins."
    assert (task.project_dir, task.worktree, task.branch) == ("projects/notes", "worktrees/CD-007", "ticket/CD-007")


def test_a_ticket_without_a_header_is_its_own_project(tmp_path):
    _task(tmp_path, "CD-001", text="Note: this first line is text, not a header.\nBuild it.")
    (task,) = load_tasks(tmp_path, 3)
    assert task.text.startswith("Note: this first line")
    assert (task.project, task.project_name, task.depends_on) == ("", "cd-001", ())


@pytest.mark.parametrize("text, error", [
    ("priority: high\n\nBuild it.", "unknown header key(s): priority"),
    ("project: Notes_App\n\nBuild it.", "project name 'Notes_App'"),
    ("max_attempts: 0\n\nBuild it.", "`max_attempts:` must be"),
    ("title: a\ntitle: b\n\nBuild it.", "appears twice"),
    ("depends_on: t\n\nBuild it.", "cannot depend on itself"),
    ("max_seconds: 0\n\nBuild it.", "must be a whole number of at least 1"),
    ("max_seconds: -5\n\nBuild it.", "must be a whole number of at least 1"),
    ("max_seconds: abc\n\nBuild it.", "must be a whole number of at least 1"),
])
def test_a_bad_header_skips_the_ticket_with_the_reason(tmp_path, text, error):
    _task(tmp_path, "t", text=text)
    tasks, errors = codr_agent.load_tickets(tmp_path, 3)
    assert tasks == [] and error in errors["t"]


def test_a_folder_name_that_cannot_be_a_branch_is_skipped(tmp_path):
    _task(tmp_path, "a..b")
    assert "git branch name" in codr_agent.load_tickets(tmp_path, 3)[1]["a..b"]


def _states(deps, outcomes, attempts=2):
    tasks = [codr_agent.CodrTask(t, "", "", attempts, depends_on=d) for t, d in deps.items()]
    return {t: s[0] for t, s in codr_agent.queue_states(tasks, outcomes).items()}


def test_queue_states_follow_the_dependencies():
    deps = {"a": (), "b": ("a",), "c": ("b",), "d": ("x",), "e": ("f",), "f": ("e",)}
    assert _states(deps, {}) == {"a": "open", "b": "waiting", "c": "waiting", "d": "held",
                                 "e": "blocked", "f": "blocked"}
    assert _states(deps, {"a": ["fail", "pass"]})["b"] == "open"
    failed = _states(deps, {"a": ["fail", "aborted"]})
    assert (failed["a"], failed["b"], failed["c"]) == ("failed", "blocked", "blocked")
    timed_out = _states(deps, {"a": ["timed_out"]}, attempts=3)
    assert (timed_out["a"], timed_out["b"], timed_out["c"]) == ("failed", "blocked", "blocked")
    stuck = _states(deps, {"a": ["stuck"]}, attempts=3)
    assert (stuck["a"], stuck["b"], stuck["c"]) == ("failed", "blocked", "blocked")


def test_a_dependency_that_passed_and_left_the_queue_still_counts():
    """stagr accepts an attempted ticket as a dependency even once its folder is gone."""
    deps = {"b": ("a",)}
    assert _states(deps, {"a": ["fail", "pass"]}) == {"b": "open"}
    assert _states(deps, {"a": ["fail"]}) == {"b": "held"}


def _started(fake):
    return [m["parts"][0]["metadata"][AUTO]["task"] for m in _all_auto(fake)]


async def test_a_bad_header_is_posted_once_and_the_queue_moves_on(fake, tmp_path):
    _task(tmp_path, "01-a", text="priority: high\n\nBuild it.")
    _task(tmp_path, "02-b")
    fake.shell_output = _shell(check_passes=False)
    agent, router = _agent(tmp_path)
    await agent.deploy()
    assert _started(fake) == ["02-b"]
    _reply(fake, _session(fake, "02-b"))
    await agent.deploy()                                       # 02-b fails, its result is posted
    await agent.deploy()                                       # 02-b attempt 2 starts
    notices = [b for _, _, b in router.sent if "01-a" in b]
    assert notices == ["codr ticket `01-a`: skipped: unknown header key(s): priority "
                       "(known: title, project, depends_on, max_attempts, max_seconds)"]


async def test_a_ticket_whose_dependency_failed_is_blocked_posted_once_and_never_started(fake, tmp_path):
    _task(tmp_path, "01-a", text="max_attempts: 1\nBuild it.")
    _task(tmp_path, "02-b", text="depends_on: 01-a\n\nBuild on it.")
    _task(tmp_path, "03-c")
    fake.shell_output = _shell(check_passes=False)
    agent, router = _agent(tmp_path)
    await agent.deploy()                                       # starts 01-a
    _reply(fake, _session(fake, "01-a"))
    await agent.deploy()                                       # 01-a fails for good
    await agent.deploy()                                       # 02-b blocked; 03-c starts
    await agent.deploy()
    assert _started(fake) == ["01-a", "03-c"]
    assert [b for _, _, b in router.sent if "02-b" in b] == [
        "codr ticket `02-b`: blocked: it depends on `01-a`, which failed for good"]


async def test_a_dependent_ticket_starts_only_after_its_dependency_passed(fake, tmp_path):
    _task(tmp_path, "01-b", text="depends_on: 02-a\n\nBuild on it.")
    _task(tmp_path, "02-a")
    fake.shell_output = _shell(check_passes=True)
    agent, router = _agent(tmp_path)
    await agent.deploy()
    assert _started(fake) == ["02-a"]
    _reply(fake, _session(fake, "02-a"))
    await agent.deploy()                                       # 02-a passes and merges
    assert "merged into `02-a` main" in router.sent[-1][2]
    await agent.deploy()
    assert _started(fake) == ["02-a", "01-b"]


async def test_a_merge_conflict_fails_the_attempt_and_the_retry_says_to_merge_main(fake, tmp_path):
    _task(tmp_path, "01-a")
    fake.shell_output = _shell(merge="__DERPR_MAIN=main\n__DERPR_CONFLICT\napp.py\ntest_app.py\n")
    agent, router = _agent(tmp_path)
    await agent.deploy()
    sid = _session(fake, "01-a")
    _reply(fake, sid)
    await agent.deploy()                                       # check passes, merge conflicts
    (row,) = [r for r in agent.memory_manager.rows if r["action_type"] == codr_agent.ATTEMPT_ACTION]
    detail = json.loads(row["outcome_payload"])
    assert row["outcome"] == "fail" and detail["merge_conflict"] is True
    assert detail["check_tail"] == "merging into `main` conflicted in:\napp.py\ntest_app.py"
    notes = [s for _, s in fake.scripts if "__DERPR_NOTED" in s]
    assert "- attempt 1: fail: merging into `main` conflicted in:" in notes[-1]
    await agent.deploy()                                       # the retry
    retry = _auto_messages(fake, sid)[1]["parts"][0]["text"]
    assert "Merge `main` into your branch in `worktrees/01-a/`" in retry and "test_app.py" in retry


async def test_a_pass_notes_the_merge_commit_in_the_ticket_file(fake, tmp_path):
    _task(tmp_path, "01-a")
    fake.shell_output = _shell()
    agent, _ = _agent(tmp_path)
    await agent.deploy()
    _reply(fake, _session(fake, "01-a"))
    await agent.deploy()
    (note,) = [s for _, s in fake.scripts if "__DERPR_NOTED" in s]
    assert "- attempt 1: pass, merged into `main` as abc1234" in note
    assert "memory/tickets/01-a.md" in note


# --- DP-488: the scripts against a real bash and git ---------------------------


def _git(cwd, *args):
    done = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    return done.stdout.strip()


def _commit_all(cwd, message):
    _git(cwd, "add", "-A")
    _git(cwd, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", message)


def _run(work, script):
    return real_bash(work)(script)


async def test_a_ticket_gets_a_project_a_worktree_and_a_memory_record(fake, tmp_path):
    _task(tmp_path / "q", "01-a", text="title: first\nproject: notes\n\nBuild app.py.", seed={"README": "seed\n"})
    (task,) = load_tasks(tmp_path / "q", 3)
    work = tmp_path / "work"
    work.mkdir()
    fake.shell_output = real_bash(work)
    await _prepare(task, work)
    proj, wt = work / "projects/notes", work / "worktrees/01-a"
    assert _git(proj, "symbolic-ref", "--short", "HEAD") == "main"
    assert _git(proj, "log", "--format=%s") == "DERPR: start notes"
    assert _git(proj, "ls-files") == ".gitignore"             # the first commit ignores caches
    assert _git(wt, "rev-parse", "--abbrev-ref", "HEAD") == "ticket/01-a"
    assert (wt / "README").read_text() == "seed\n"
    record = (work / "memory/tickets/01-a.md").read_text()
    assert record.startswith("# 01-a: first\n") and "Build app.py." in record
    assert "title:" not in record                              # the header stays hidden
    assert _git(work / "memory", "status", "--porcelain") == ""        # all committed

    # Check -> merge: leftovers committed, merged with a merge commit, worktree gone, branch kept.
    (wt / "app.py").write_text("print('hi')\n")
    (wt / "__pycache__").mkdir()
    (wt / "__pycache__" / "app.cpython-312.pyc").write_bytes(b"x")
    (wt / "notes.db").write_bytes(b"x")                        # left by the check
    check =codr_agent.CodrTask("01-a", "", "test -f app.py && echo in-the-worktree", 1, project="notes")
    assert parse_check(_run(work, codr_agent.check_script(check, 30))) == (True, "in-the-worktree\n")
    merged, main, problem = codr_agent.parse_merge(_run(work, codr_agent.merge_script(task)))
    assert (main, problem) == ("main", "") and merged
    assert (proj / "app.py").read_text() == "print('hi')\n"
    assert _git(proj, "log", "-1", "--format=%s") == "01-a: merge ticket/01-a"
    assert len(_git(proj, "log", "-1", "--format=%p").split()) == 2          # a merge commit
    assert "01-a: uncommitted at check pass" in _git(proj, "log", "--format=%s")
    assert _git(proj, "ls-files").split() == [".gitignore", "README", "app.py"]   # no caches, no db
    assert not wt.exists() and _git(proj, "branch", "--list", "ticket/01-a")

    # A ticket depending on it is cut from the new main.
    _task(tmp_path / "q", "02-b", text="project: notes\ndepends_on: 01-a\n\nExtend it.")
    task2 = next(t for t in load_tasks(tmp_path / "q", 3) if t.id == "02-b")
    await _prepare(task2, work)
    assert (work / "worktrees/02-b/app.py").read_text() == "print('hi')\n"

    out = _run(work, codr_agent.ticket_note_script("01-a", "- attempt 1: pass, merged into `main` as x"))
    assert "__DERPR_NOTED" in out
    assert (work / "memory/tickets/01-a.md").read_text().endswith("- attempt 1: pass, merged into `main` as x\n")
    assert _git(work / "memory", "status", "--porcelain") == ""


async def test_a_conflicting_merge_is_aborted_and_the_worktree_stays(fake, tmp_path):
    _task(tmp_path / "q", "01-a", text="project: p\n\nx")
    (task,) = load_tasks(tmp_path / "q", 3)
    work = tmp_path / "work"
    work.mkdir()
    fake.shell_output = real_bash(work)
    await _prepare(task, work)
    proj, wt = work / "projects/p", work / "worktrees/01-a"
    (proj / "app.py").write_text("main side\n")
    _commit_all(proj, "main moved on")
    (wt / "app.py").write_text("ticket side\n")
    merged, main, problem = codr_agent.parse_merge(_run(work, codr_agent.merge_script(task)))
    assert merged is None and problem == "merging into `main` conflicted in:\napp.py"
    assert wt.is_dir() and (proj / "app.py").read_text() == "main side\n"
    assert _git(proj, "status", "--porcelain") == ""           # merge aborted, main clean


async def test_an_existing_repo_keeps_its_branch_and_old_task_files_move_in(fake, tmp_path):
    _task(tmp_path / "q", "04-x", text="project: shop\n\nx", seed={"a.py": "seed a\n", "b.py": "seed b\n"})
    (task,) = load_tasks(tmp_path / "q", 3)
    work = tmp_path / "work"
    proj = work / "projects/shop"
    proj.mkdir(parents=True)
    _git(proj, "init", "-q", "-b", "master")
    (proj / "lib.py").write_text("lib\n")
    _commit_all(proj, "imported")
    (work / "tasks/04-x").mkdir(parents=True)                  # pre-DP-488 layout, in flight
    (work / "tasks/04-x/a.py").write_text("codr's work\n")
    fake.shell_output = real_bash(work)
    await _prepare(task, work)
    wt = work / "worktrees/04-x"
    assert _git(proj, "log", "--format=%s") == "imported"      # no DERPR commit in a repo with history
    assert (wt / "lib.py").read_text() == "lib\n"
    assert (wt / "a.py").read_text() == "codr's work\n"       # old work wins over the seed
    assert (wt / "b.py").read_text() == "seed b\n"
    merged, main, _ = codr_agent.parse_merge(_run(work, codr_agent.merge_script(task)))
    assert main == "master" and merged


def test_the_memory_skeleton_never_touches_codrs_notes(tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory/MEMORY.md").write_text("codr's index\n")
    assert "__DERPR_MEMORY_READY" in _run(tmp_path, codr_agent.memory_skeleton_script())
    assert (tmp_path / "memory/MEMORY.md").read_text() == "codr's index\n"
    assert (tmp_path / "memory/tickets/_overview.md").is_file()
    assert _git(tmp_path / "memory", "log", "--format=%s").splitlines() == [
        "DERPR: add projects/_overview.md", "DERPR: add tickets/_overview.md"]
    assert _git(tmp_path / "memory", "status", "--porcelain") == "?? MEMORY.md"   # codr's, left to codr
    _run(tmp_path, codr_agent.memory_skeleton_script())       # idempotent
    assert len(_git(tmp_path / "memory", "log", "--format=%s").splitlines()) == 2


def test_the_shipped_battery_loads_and_04_builds_on_03():
    tasks, errors = codr_agent.load_tickets(codr_agent.global_config.CONFIG_DIR / "codr_tasks", 3)
    assert errors == {}
    by_id = {t.id: t for t in tasks}
    assert by_id["04-notes-auth"].depends_on == ("03-notes-api",)
    assert by_id["03-notes-api"].project_name == by_id["04-notes-auth"].project_name == "notes"
    assert "ticket/03-notes-api" in by_id["04-notes-auth"].check


async def test_a_ticket_that_cannot_be_prepared_is_posted_once_and_not_started(fake, tmp_path):
    _task(tmp_path, "01-a")
    ok = _shell()
    fake.shell_output = lambda s: "projects/01-a exists but is not a git repo\n" if "TICKET_READY" in s else ok(s)
    agent, router = _agent(tmp_path)
    await agent.deploy()
    await agent.deploy()
    assert not _all_auto(fake)
    assert [b for _, _, b in router.sent] == [
        "codr ticket `01-a`: could not start: codr: preparing ticket 01-a failed: "
        "projects/01-a exists but is not a git repo\n"]


def test_the_reader_round_trips_what_stagr_writes(tmp_path):
    """DP-489 writes TASK.md with `codr_ticket.render_task_md`; this side parses it."""
    from src.stagr.codr_ticket import render_task_md
    text = render_task_md(title="notes auth", project="notes", depends_on=["CD-001", "CD-002"], max_attempts=4,
                          max_seconds=1800,
                          spec="Add logins.", plan="Extend app.py.", packages="none")
    _task(tmp_path, "CD-003", text=text)
    (tmp_path / ".last-ticket-number").write_text("3\n")        # stagr's high-water mark: not a ticket
    tasks, errors = codr_agent.load_tickets(tmp_path, 3)
    assert errors == {}
    (task,) = tasks
    assert (task.title, task.project, task.depends_on, task.max_attempts, task.max_seconds) == (
        "notes auth", "notes", ("CD-001", "CD-002"), 4, 1800)
    assert task.text == "## Spec\n\nAdd logins.\n\n## Plan\n\nExtend app.py.\n\n## Packages\n\nnone"


async def test_task_max_seconds_overrides_agent_config(fake, tmp_path):
    """DP-482: max_seconds on a task overrides max_turn_seconds on the agent."""
    _task(tmp_path, "01-a", text="max_seconds: 10\n\nBuild it.")
    _task(tmp_path, "02-b", text="Build it.")
    tasks, errors = codr_agent.load_tickets(tmp_path, 3)
    assert errors == {}
    assert tasks[0].max_seconds == 10
    assert tasks[1].max_seconds is None

    fake.shell_output = _shell()
    agent, router = _agent(tmp_path, max_turn_seconds=604800)
    await agent.deploy()
    sid = _session(fake, "01-a")
    _tool_step(fake, sid)

    # Started 15s ago: past task's 10s max_seconds, well under agent's 604800s
    auto_msg = _auto_messages(fake, sid)[0]
    auto_part = agent._auto_part(auto_msg)
    auto_part["metadata"][AUTO]["started"] = time.time() - 15

    await agent.deploy()
    assert agent.memory_manager.rows[0]["outcome"] == "timed_out"
    assert "ran past 10 s" in router.sent[-1][2]


def test_max_seconds_single_line_legacy_header(tmp_path):
    """DP-482: single-line legacy form max_seconds: N works as header."""
    _task(tmp_path, "CD-001", text="max_seconds: 10\nBuild it.")
    tasks, errors = codr_agent.load_tickets(tmp_path, 3)
    assert errors == {}
    assert tasks[0].max_seconds == 10
    assert tasks[0].text == "Build it."


# --- DP-484: a session and a channel per ticket --------------------------------


def _context_shell(files):
    """`_shell`, plus codr's memory files for the ticket's first prompt."""
    base = _shell(check_passes=False)

    def run(script):
        if "__DERPR_FILE=" in script:
            return "".join(f"__DERPR_FILE={p}\n{body}\n__DERPR_FILE_END\n" for p, body in files.items())
        return base(script)
    return run


async def test_a_tickets_first_prompt_carries_codrs_memory_index_and_a_retry_does_not(fake, tmp_path):
    _task(tmp_path, "CD-001", text="title: csv\nproject: tools\n\nBuild csvstats.")
    fake.shell_output = _context_shell({"memory/MEMORY.md": "# Index\n- tickets/: one file per ticket",
                                        "memory/projects/tools.md": "tools: python, no deps"})
    agent, _ = _agent(tmp_path)
    await agent.deploy()
    sid = _session(fake, "CD-001")
    first = _auto_messages(fake, sid)[0]["parts"][0]["text"]
    assert "`memory/MEMORY.md`:\n```\n# Index\n- tickets/: one file per ticket\n```" in first
    assert "tools: python, no deps" in first and "memory/tickets/_overview.md`:" not in first
    context_reads = [s for _, s in fake.scripts if "__DERPR_FILE=" in s]
    assert "memory/projects/tools.md" in context_reads[0]
    _reply(fake, sid)
    await agent.deploy()                                       # fails
    await agent.deploy()                                       # the retry, same session
    retry = _auto_messages(fake, sid)[1]["parts"][0]["text"]
    assert "attempt 2 of 3" in retry and "# Index" not in retry
    assert len([s for _, s in fake.scripts if "__DERPR_FILE=" in s]) == 1


async def test_a_tickets_channel_holds_its_transcript_and_each_attempts_result(fake, tmp_path):
    _task(tmp_path, "CD-001", text="max_attempts: 1\n\nBuild it.")
    fake.shell_output = _shell(check_passes=False)
    agent, router = _agent(tmp_path)
    await agent.deploy()
    sid = _session(fake, "CD-001")
    fake.add(sid, "assistant", [{"type": "tool", "callID": "c1", "tool": "notify_adam",
                                 "state": {"status": "completed", "input": {"message": "halfway"}}}],
             time={"created": 1, "completed": 2}, finish="tool-calls")
    await agent.deploy()                                       # watched: relayed, mirrored
    assert router.sent[-1][2] == "codr [CD-001]: halfway"      # from a ticket: prefixed
    rows = agent.memory_manager.channel("task:CD-001")
    assert [(r["author_role"], r["author_name"]) for r in rows] == [("user", "DERPR"), ("assistant", "codr")]
    assert "Build it." in rows[0]["content"]
    assert json.loads(rows[1]["tool_context"])[0]["tool_calls"][0]["name"] == "notify_adam"

    _reply(fake, sid, "gave up")
    await agent.deploy()                                       # finished: its last step, then the result
    rows = agent.memory_manager.channel("task:CD-001")
    assert rows[1]["content"] == "gave up"
    note = rows[-1]
    assert (note["author_role"], note["author_name"]) == ("assistant", "DERPR")
    assert "attempt 1/1: **FAIL**" in note["content"] and "FAIL: csvstats.py missing" in note["content"]
    assert note["platform_message_id"] == "codr-attempt:CD-001:1"


async def test_a_restarted_agent_finds_the_running_ticket_in_its_own_session(fake, tmp_path):
    _task(tmp_path, "01-a")
    fake.shell_output = _shell()
    agent, _ = _agent(tmp_path)
    await agent.deploy()
    _reply(fake, _session(fake, "01-a"))
    restarted, router = _agent(tmp_path)
    restarted.memory_manager.rows = agent.memory_manager.rows
    await restarted.deploy()
    assert "`01-a` attempt 1/3: **PASS**" in router.sent[-1][2]


async def test_a_chat_turn_in_codrs_own_session_never_sees_a_ticket(fake, tmp_path):
    """Discord talks to `derpr:codr`; a ticket's turn runs in its own session."""
    _task(tmp_path, "01-a")
    fake.shell_output = _shell()
    agent, _ = _agent(tmp_path)
    await agent.deploy()
    own, ticket = _session(fake), _session(fake, "01-a")
    assert own != ticket and not fake.messages[own]
    assert opencode.auto_turn_running(fake.messages[ticket])


async def test_a_settled_session_is_not_read_again_until_it_changes(fake, tmp_path):
    fake.shell_output = _shell()
    agent, router = _agent(tmp_path)
    await agent.deploy()
    reads = lambda: len([c for c in fake.calls if c[0] == "GET" and c[1].endswith("/message")])  # noqa: E731
    before = reads()
    await agent.deploy()
    assert reads() == before                                   # idle, unchanged: skipped
    sid = _session(fake)
    fake.add(sid, "assistant", [{"type": "tool", "tool": "notify_adam",
                                 "state": {"status": "completed", "input": {"message": "hi"}}}],
             time={"created": 1, "completed": 2}, finish="tool-calls")
    await agent.deploy()                                       # changed: read, relayed
    assert [b for _, _, b in router.sent] == ["codr: hi"]


async def test_a_removed_tickets_aborted_turn_is_still_recorded(fake, tmp_path):
    """Delete the folder, then abort the turn: the next tick records it."""
    _task(tmp_path, "01-a")
    fake.shell_output = _shell()
    agent, router = _agent(tmp_path)
    await agent.deploy()
    sid = _session(fake, "01-a")
    (tmp_path / "01-a" / "TASK.md").unlink()
    fake.add(sid, "assistant", [{"type": "text", "text": "partial"}],
             time={"created": 1, "completed": 2}, finish="stop", error={"name": "MessageAbortedError"})
    await agent.deploy()
    assert "**ABORTED**" in router.sent[-1][2] and "no longer in the queue" in router.sent[-1][2]
