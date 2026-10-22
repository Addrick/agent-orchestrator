# tests/test_codr_agent.py
#
# DP-449 — the codr agent, ticked against the fake `opencode serve` from the
# provider tests. Each test drives deploy() and plays the server's side of a
# turn by adding messages, the way opencode records them.

import json
from types import SimpleNamespace

import pytest

from src.agents import codr_agent
from src.agents.codr_agent import (
    CodrAgent, detect_loop, load_tasks, parse_check, sync_script, workspace_files,
)
from src.engine.providers import opencode
from src.generation_events import ResponseType
from tests import test_opencode_provider as provider_tests
from tests.test_opencode_provider import ROOT

fake = provider_tests.fake  # the fake `opencode serve` fixture

AUTO = opencode.AUTO_FIELD


class FakeMemory:
    """The agent-action-log calls the agent makes."""

    def __init__(self):
        self.rows = []
        self.channels = [{"channel": "codr-chat", "server_id": "9"}]

    def get_distinct_channels(self, persona_name=None):
        return self.channels

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


class Persistence:
    """The TurnPersistence calls a relayed reply makes."""

    def __init__(self):
        self.committed, self.queued = [], []

    def commit_or_update_assistant(self, **kw):
        self.committed.append(kw)
        return 100 + len(self.committed)

    def queue_retain_safe(self, **kw):
        self.queued.append(kw)


class Router:
    def __init__(self):
        self.sent = []

    async def send(self, channel, recipient, subject, body):
        self.sent.append((channel, recipient, body))
        return True


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
    chat = SimpleNamespace(text_engine=None, memory_manager=FakeMemory(), personas={"codr": persona},
                           turn_persistence=Persistence())
    router = Router()
    agent = CodrAgent(chat, router, {
        "tasks_dir": str(tmp_path), "_recipients": {},
        "notification_target": {"channel": "discord_channel", "recipient": "123"}, **cfg})
    return agent, router


def _shell(check_passes=True):
    def run(script):
        if "DERPR_EOF_0" in script:
            return "__DERPR_CHANGED=0\n"
        if "DERPR_CHECK" in script:
            return ("PASS\n__DERPR_EXIT=0\n" if check_passes
                    else "FAIL: csvstats.py missing\n__DERPR_EXIT=1\n")
        return ""
    return run


def _session(fake):
    (sid,) = [s for s in fake.sessions if fake.sessions[s]["title"] == "derpr:codr"]
    return sid


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
    assert "model: kcpp-pool/local" in files[".opencode/agent/pool.md"]
    assert "mkdir -p .opencode/tool .opencode/agent .opencode/node_modules" in sync_script(files)


# --- ticks ---------------------------------------------------------------------


async def test_first_tick_sets_up_the_workspace_and_starts_the_oldest_task(fake, tmp_path):
    _task(tmp_path, "01-a", text="Build csvstats.")
    fake.shell_output = _shell()
    agent, _ = _agent(tmp_path)
    await agent.deploy()

    sid = _session(fake)
    assert fake.sessions[sid]["directory"] == f"{ROOT}/codr"
    scripts = [s for _, s in fake.scripts]
    assert any("DERPR_EOF_0" in s for s in scripts)            # .opencode/ synced
    assert "mkdir -p -- 'tasks/01-a'" in scripts[-1]           # task folder seeded
    (start,) = _auto_messages(fake, sid)
    part = start["parts"][0]
    assert part["metadata"][AUTO]["task"] == "01-a" and part["metadata"][AUTO]["attempt"] == 1
    assert "Build csvstats." in part["text"] and "tasks/01-a/attempts.md" in part["text"]
    post = next(c for c in fake.calls if c[1].endswith("/prompt_async"))
    assert post[3]["system"] == "You are codr."

    await agent.deploy()                                       # still running: nothing new
    assert len(_auto_messages(fake, sid)) == 1


async def test_a_finished_turn_is_checked_recorded_and_posted_once(fake, tmp_path):
    _task(tmp_path, "01-a")
    fake.shell_output = _shell(check_passes=True)
    agent, router = _agent(tmp_path)
    await agent.deploy()
    sid = _session(fake)
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
        sid = _session(fake)
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
    sid = _session(fake)
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
    assert _auto_messages(fake, sid)[1]["parts"][0]["metadata"][AUTO]["result"] == "aborted"
    (row,) = agent.memory_manager.rows
    assert row["outcome"] == "aborted"
    assert "**ABORTED**" in router.sent[-1][2] and "same `bash` call" in router.sent[-1][2]


async def test_an_overrun_is_aborted_without_a_nudge(fake, tmp_path):
    _task(tmp_path, "01-a")
    fake.shell_output = _shell()
    agent, router = _agent(tmp_path, max_turn_seconds=0)
    await agent.deploy()
    sid = _session(fake)
    _tool_step(fake, sid)
    await agent.deploy()
    assert len(_auto_messages(fake, sid)) == 1
    assert agent.memory_manager.rows[0]["outcome"] == "aborted"
    assert "ran past 0 s" in router.sent[-1][2]


async def test_a_stall_counts_as_stuck(fake, tmp_path, monkeypatch):
    _task(tmp_path, "01-a")
    fake.shell_output = _shell()
    agent, _ = _agent(tmp_path, stall_seconds=100)
    clock = [1000.0]
    monkeypatch.setattr(codr_agent.time, "monotonic", lambda: clock[0])
    await agent.deploy()
    sid = _session(fake)
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
    # The posted one (only) is codr's reply in its latest conversation.
    persistence = agent.chat_system.turn_persistence
    (row,) = persistence.committed
    assert (row["channel"], row["server_id"], row["final_text"]) == ("codr-chat", "9", "tests pass, starting auth")
    assert row["response_type"] == ResponseType.LLM_GENERATION
    assert [q["interaction_id"] for q in persistence.queued] == [101]
    # The agent's own parts are never written to: opencode sends an assistant
    # part's metadata to the model, and a foreign key there breaks the session.
    assert not [c for c in fake.calls if c[0] == "PATCH"]


async def test_a_relay_with_no_conversation_yet_is_posted_but_not_stored(fake, tmp_path):
    fake.shell_output = _shell()
    agent, router = _agent(tmp_path)
    agent.memory_manager.channels = []
    await agent.deploy()
    sid = _session(fake)
    fake.add(sid, "assistant", [{"type": "tool", "tool": "notify_adam",
                                 "state": {"status": "completed", "input": {"message": "hi"}}}],
             time={"created": 1, "completed": 2}, finish="tool-calls")
    await agent.deploy()
    assert [b for _, _, b in router.sent] == ["codr: hi"]
    assert agent.chat_system.turn_persistence.committed == []


async def test_no_task_starts_while_a_chat_turn_runs(fake, tmp_path):
    _task(tmp_path, "01-a")
    fake.shell_output = _shell()
    agent, _ = _agent(tmp_path)
    await agent.deploy()                                       # creates the session, starts 01-a
    sid = _session(fake)
    _reply(fake, sid)
    await agent.deploy()                                       # 01-a passes
    _task(tmp_path, "02-b")
    fake.busy.add(sid)                                         # a chat turn is running
    await agent.deploy()
    assert len(_auto_messages(fake, sid)) == 1


async def test_start_tasks_false_only_watches(fake, tmp_path):
    _task(tmp_path, "01-a")
    fake.shell_output = _shell()
    agent, _ = _agent(tmp_path, start_tasks=False)
    await agent.deploy()
    assert not _auto_messages(fake, _session(fake))


async def test_changed_workspace_config_reloads_the_instance_when_idle(fake, tmp_path):
    fake.shell_output = lambda s: "__DERPR_CHANGED=1\n" if "DERPR_EOF_0" in s else ""
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
