# tests/test_dispatcher.py
"""Unit tests for the DP-227 Dispatcher: the file-tail event bridge, dispatch
wiring, and resume guards. No real subprocess or git — spawn + clone_manager
are stubbed."""

import asyncio
import json
import os

import pytest

import src.self_edit.dispatcher as disp
from config import global_config
from src.self_edit import registry as reg
from src.self_edit.dispatcher import Dispatcher, DispatcherError, _read_from
from src.self_edit.events import DONE, ERROR, QUESTION
from src.self_edit.registry import AgentRecord, AgentRegistry

# asyncio_mode=auto handles async tests; this module mixes in one sync test.


class _FakeProc:
    def __init__(self, pid=4321, returncode=None):
        self.pid = pid
        self.returncode = returncode
        self.terminated = False

    def terminate(self):
        self.terminated = True


def _write_log(path, objs):
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o) + "\n")


def _record(tmp_path, agent_id="DP-9-1", bug_id="DP-9"):
    fixr = tmp_path / ".fixr"
    fixr.mkdir(exist_ok=True)
    return AgentRecord(
        agent_id=agent_id, bug_id=bug_id, description="bug",
        worktree=str(tmp_path), branch="bugfix/DP-9-fix",
        raw_log=str(fixr / "raw.jsonl"), events_log=str(fixr / "events.jsonl"),
        status=reg.RUNNING,
    )


def test_read_from_only_returns_complete_lines(tmp_path):
    p = tmp_path / "raw.jsonl"
    p.write_bytes(b'{"a":1}\n{"b":2}\n{"partial":')
    lines, offset = _read_from(str(p), 0)
    assert lines == ['{"a":1}', '{"b":2}']
    # offset stops at the last newline; the partial line is re-read next time.
    p.write_bytes(b'{"a":1}\n{"b":2}\n{"partial":3}\n')
    lines2, _ = _read_from(str(p), offset)
    assert lines2 == ['{"partial":3}']


async def test_bridge_done_fires_wake_and_updates_registry(tmp_path):
    registry = AgentRegistry()
    woken = []

    async def on_wake(record, event):
        woken.append((record.agent_id, event.type, event.payload.get("pr_url")))

    d = Dispatcher(registry, on_wake=on_wake)
    rec = _record(tmp_path)
    await registry.add(rec)
    d._procs[rec.agent_id] = _FakeProc(returncode=0)
    _write_log(rec.raw_log, [
        {"type": "system", "subtype": "init", "session_id": "sX"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "fixing"}]}},
        {"type": "result", "subtype": "success", "is_error": False,
         "result": "FIXR_DONE: https://github.com/x/y/pull/7 done"},
    ])

    from src.self_edit.events import ClaudeStreamAdapter
    await d._bridge(rec, ClaudeStreamAdapter(rec.agent_id), resume_tail=False)

    assert woken == [(rec.agent_id, DONE, "https://github.com/x/y/pull/7")]
    updated = await registry.get(rec.agent_id)
    assert updated.status == reg.DONE
    assert updated.pr_url == "https://github.com/x/y/pull/7"
    assert updated.session_id == "sX"
    # The common-schema audit log was written.
    with open(rec.events_log) as f:
        events = [json.loads(ln) for ln in f]
    assert events[-1]["type"] == DONE


async def test_bridge_question_parks_agent_and_stops(tmp_path):
    registry = AgentRegistry()
    woken = []

    async def on_wake(record, event):
        woken.append(event.type)

    d = Dispatcher(registry, on_wake=on_wake)
    rec = _record(tmp_path)
    await registry.add(rec)
    d._procs[rec.agent_id] = _FakeProc(returncode=0)
    _write_log(rec.raw_log, [
        {"type": "system", "subtype": "init", "session_id": "sQ"},
        {"type": "result", "subtype": "success",
         "result": "FIXR_QUESTION: which fix?"},
    ])

    from src.self_edit.events import ClaudeStreamAdapter
    await d._bridge(rec, ClaudeStreamAdapter(rec.agent_id), resume_tail=False)

    assert woken == [QUESTION]
    updated = await registry.get(rec.agent_id)
    assert updated.status == reg.WAITING
    assert updated.session_id == "sQ"


async def test_bridge_synthesizes_done_on_clean_exit_without_sentinel(tmp_path, monkeypatch):
    monkeypatch.setattr(disp, "_TAIL_POLL_SECONDS", 0.01)
    registry = AgentRegistry()
    woken = []

    async def on_wake(record, event):
        woken.append(event.type)

    d = Dispatcher(registry, on_wake=on_wake)
    rec = _record(tmp_path)
    await registry.add(rec)
    d._procs[rec.agent_id] = _FakeProc(returncode=0)
    # Only progress, no terminal result; the process is already exited (rc=0).
    _write_log(rec.raw_log, [
        {"type": "assistant", "message": {"content": "working"}},
    ])

    from src.self_edit.events import ClaudeStreamAdapter
    await d._bridge(rec, ClaudeStreamAdapter(rec.agent_id), resume_tail=False)

    assert woken == [DONE]
    assert (await registry.get(rec.agent_id)).status == reg.DONE


async def test_bridge_synthesizes_error_on_nonzero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(disp, "_TAIL_POLL_SECONDS", 0.01)
    registry = AgentRegistry()
    woken = []
    d = Dispatcher(registry, on_wake=lambda r, e: woken.append(e.type))
    rec = _record(tmp_path)
    await registry.add(rec)
    d._procs[rec.agent_id] = _FakeProc(returncode=1)
    _write_log(rec.raw_log, [])

    from src.self_edit.events import ClaudeStreamAdapter
    # on_wake above is sync; wrap to coroutine via adapter call path expects await
    async def on_wake(r, e):
        woken.append(e.type)
    d._on_wake = on_wake

    await d._bridge(rec, ClaudeStreamAdapter(rec.agent_id), resume_tail=False)
    assert woken == [ERROR]
    assert (await registry.get(rec.agent_id)).status == reg.ERROR


async def test_dispatch_creates_worktree_and_registers(tmp_path, monkeypatch):
    registry = AgentRegistry()
    d = Dispatcher(registry, on_wake=_noop_wake)

    wt = tmp_path / "worktrees" / "DP-50"
    wt.mkdir(parents=True)
    monkeypatch.setattr(
        disp.clone_manager, "create_worktree",
        lambda bug_id, clone_dir=None: str(wt),
    )

    started = {}

    async def fake_spawn(**kwargs):
        started.update(kwargs)
        return _FakeProc(pid=999)

    monkeypatch.setattr(d, "_spawn", fake_spawn)
    # Don't actually start a bridge task tailing a file in the test.
    monkeypatch.setattr(d, "_start_bridge", lambda record, **k: None)

    rec = await d.dispatch("DP-50", "thing is broken")
    assert rec.pid == 999
    assert rec.bug_id == "DP-50"
    assert rec.worktree == str(wt)
    assert (await registry.get(rec.agent_id)) is not None
    # The dispatched agent got the bug as its prompt.
    assert "DP-50" in started["prompt"]


async def test_dispatch_refuses_duplicate_active_bug(tmp_path):
    registry = AgentRegistry()
    d = Dispatcher(registry, on_wake=_noop_wake)
    await registry.add(AgentRecord(
        agent_id="DP-50-1", bug_id="DP-50", description="x", worktree="/w",
        branch="b", raw_log="/r", events_log="/e", status=reg.RUNNING))
    with pytest.raises(DispatcherError, match="already in flight"):
        await d.dispatch("DP-50", "again")


async def test_answer_agent_requires_session(tmp_path):
    registry = AgentRegistry()
    d = Dispatcher(registry, on_wake=_noop_wake)
    await registry.add(AgentRecord(
        agent_id="a1", bug_id="DP-1", description="x", worktree="/w",
        branch="b", raw_log="/r", events_log="/e", status=reg.WAITING,
        session_id=None))
    with pytest.raises(DispatcherError, match="no session_id"):
        await d.answer_agent("a1", "go with option B")

    with pytest.raises(DispatcherError, match="Unknown agent_id"):
        await d.answer_agent("ghost", "hi")


async def _add(registry, status, *, session_id="sid-1", agent_id="a1"):
    await registry.add(AgentRecord(
        agent_id=agent_id, bug_id="DP-1", description="x", worktree="/w",
        branch="b", raw_log="/r", events_log="/e", status=status,
        session_id=session_id))


def _stub_spawn(d, monkeypatch, proc=None, fail=False):
    """Replace _spawn + _start_bridge so answer_agent never touches a real
    subprocess or starts a background tail. Returns a dict flagging whether
    _spawn was invoked."""
    seen = {"spawned": False}
    proc = proc or _FakeProc(pid=99)

    async def fake_spawn(**kwargs):
        seen["spawned"] = True
        if fail:
            raise RuntimeError("spawn boom")
        return proc

    monkeypatch.setattr(d, "_spawn", fake_spawn)
    monkeypatch.setattr(d, "_start_bridge", lambda *a, **k: None)
    return seen


async def test_answer_agent_resumes_waiting(tmp_path, monkeypatch):
    registry = AgentRegistry()
    d = Dispatcher(registry, on_wake=_noop_wake)
    await _add(registry, reg.WAITING)
    seen = _stub_spawn(d, monkeypatch)

    rec = await d.answer_agent("a1", "use option B")

    assert seen["spawned"] is True
    assert rec.status == reg.RUNNING
    assert (await registry.get("a1")).pid == 99


@pytest.mark.parametrize("status", [reg.RUNNING, reg.DONE, reg.ERROR, reg.KILLED, reg.ORPHANED])
async def test_answer_agent_rejects_non_waiting(tmp_path, monkeypatch, status):
    """Only a WAITING agent is resumable — resuming a RUNNING agent would spawn
    a competing claude; resuming a terminal agent would resurrect it."""
    registry = AgentRegistry()
    d = Dispatcher(registry, on_wake=_noop_wake)
    await _add(registry, status)
    seen = _stub_spawn(d, monkeypatch)

    with pytest.raises(DispatcherError, match="not waiting"):
        await d.answer_agent("a1", "answer")

    assert seen["spawned"] is False  # never spawned
    assert (await registry.get("a1")).status == status  # status untouched


async def test_answer_agent_spawn_failure_reverts_to_waiting(tmp_path, monkeypatch):
    """If the resume spawn fails after the WAITING→RUNNING claim, the status is
    rolled back to WAITING so the answer can be retried."""
    registry = AgentRegistry()
    d = Dispatcher(registry, on_wake=_noop_wake)
    await _add(registry, reg.WAITING)
    _stub_spawn(d, monkeypatch, fail=True)

    with pytest.raises(RuntimeError, match="spawn boom"):
        await d.answer_agent("a1", "answer")

    assert (await registry.get("a1")).status == reg.WAITING  # reverted


async def test_compare_and_set_status_single_winner(tmp_path):
    """Concurrent CAS on one WAITING agent: exactly one transition wins."""
    registry = AgentRegistry()
    await _add(registry, reg.WAITING)

    results = await asyncio.gather(*[
        registry.compare_and_set_status("a1", reg.WAITING, reg.RUNNING)
        for _ in range(8)
    ])

    assert sum(results) == 1  # only one caller saw WAITING
    assert (await registry.get("a1")).status == reg.RUNNING


async def test_compare_and_set_status_rejects_wrong_state_or_missing(tmp_path):
    registry = AgentRegistry()
    await _add(registry, reg.RUNNING)
    assert await registry.compare_and_set_status("a1", reg.WAITING, reg.RUNNING) is False
    assert await registry.compare_and_set_status("ghost", reg.WAITING, reg.RUNNING) is False
    assert (await registry.get("a1")).status == reg.RUNNING  # unchanged


async def test_spawn_strips_api_key_from_child_env(tmp_path, monkeypatch):
    """DP-232: the dispatched `claude` must run on the subscription — `_spawn`
    must hand create_subprocess_exec an env with ANTHROPIC_API_KEY removed."""
    monkeypatch.setattr(global_config, "CC_USE_SUBSCRIPTION", True)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-keep")
    monkeypatch.setenv("CLAUDE_CLI_PATH", "claude")  # skip shutil.which lookup

    captured = {}

    async def fake_exec(*args, **kwargs):
        captured.update(kwargs)
        return _FakeProc(pid=7)

    monkeypatch.setattr(disp.asyncio, "create_subprocess_exec", fake_exec)

    d = Dispatcher(AgentRegistry(), on_wake=_noop_wake)
    await d._spawn(
        prompt="p", system_prompt="s", cwd=str(tmp_path),
        raw_log=str(tmp_path / "raw.jsonl"),
    )

    env = captured["env"]
    assert env is not None
    assert "ANTHROPIC_API_KEY" not in env
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-keep"


async def _noop_wake(record, event):
    return None
