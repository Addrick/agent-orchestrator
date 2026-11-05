# tests/test_stagr_codr_handoff.py
"""DP-493: stagr hears when codr finishes a ticket it queued, and may queue
the next without a prompt when its `auto_approve` switch is on.

The resume itself is the DP-345 deferral mechanism, pinned end to end in
tests/integration/test_auto_approve_deferral.py. Here: the pieces around it —
which tickets wake stagr, what the wake carries, the switch, the flag's guard
and the deferral's longer lifetime."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from config import global_config
from src.agents.codr_agent import wake_result
from src.confirmations import ParkedWrite, deferral_ttl
from src.deferral_kinds import DEFERRAL_KIND_APPROVAL, DEFERRAL_KIND_CODR_TICKET, DEFERRAL_KIND_NODE_JOB
from src.generation_events import DoneEvent, PendingConfirmationEvent, ResponseType
from src.persona import Persona
from src.persona_fields import PERSONA_FIELDS
from src.personas.store import to_dict
from src.stagr.codr_ticket import ATTEMPT_ACTION, attempt_context
from src.tools.definitions import is_auto_approvable, validate_tool_capabilities
from tests.test_codr_agent import _agent, _task


# --- the codr agent wakes stagr ---------------------------------------------

@pytest.fixture(autouse=True)
def _queue(monkeypatch, tmp_path):
    """Each test's task queue is its own tmp_path."""
    monkeypatch.setattr(global_config, "CODR_TASKS_DIR", tmp_path)


class FakeConfirmations:
    def __init__(self, *tokens, kind=DEFERRAL_KIND_CODR_TICKET):
        self.parks = [SimpleNamespace(token=t, kind=kind) for t in tokens]

    def pending_of_kind(self, kind):
        return [p for p in self.parks if p.kind == kind]


def _wired(tmp_path, *tickets, reply="Queued CD-002 next.", confirm=False):
    """A codr agent whose chat system holds deferrals for `tickets` and
    whose resume records each call and answers with `reply`."""
    agent, router = _agent(tmp_path)
    resolved = []

    async def resolve(token, *, kind, status, result, note=None):
        resolved.append((token, kind, status, result))
        if confirm:
            yield PendingConfirmationEvent(text="", write_calls=[], persona_name="stagr", token="t",
                                           audit_info={})
        yield DoneEvent(text=reply, response_type=ResponseType.LLM_GENERATION)

    agent.chat_system.confirmations = FakeConfirmations(*tickets)
    agent.chat_system.stream_resolve_deferral = resolve
    return agent, router, resolved


def _attempt(agent, task_id, outcome, **detail):
    agent.memory_manager.log_agent_action(
        "codr", ATTEMPT_ACTION, trigger_context=attempt_context(task_id),
        action_payload=json.dumps({"task": task_id}), outcome=outcome,
        outcome_payload=json.dumps(detail))


async def _tick(agent):
    agent._wake_waiting()
    await asyncio.gather(*list(agent._wake_runs))


@pytest.mark.asyncio
async def test_a_passed_ticket_wakes_stagr_once_and_posts_its_reply(tmp_path):
    _task(tmp_path, "CD-001", text="max_attempts: 2\n\nBuild it.")
    agent, router, resolved = _wired(tmp_path, "CD-001")
    _attempt(agent, "CD-001", "pass", merged=True, main_branch="main", seconds=60)

    await _tick(agent)

    assert [(t, k, s) for t, k, s, _ in resolved] == [("CD-001", DEFERRAL_KIND_CODR_TICKET, "passed")]
    assert resolved[0][3]["merged"] is True and "queue it now" in resolved[0][3]["instruction"]
    assert [body for _, _, body in router.sent] == ["stagr [CD-001]: Queued CD-002 next."]
    assert not agent._waking, "a finished wake must release its ticket"


@pytest.mark.asyncio
async def test_an_open_or_retrying_ticket_does_not_wake_stagr(tmp_path):
    _task(tmp_path, "CD-001", text="max_attempts: 3\n\nBuild it.")
    agent, router, resolved = _wired(tmp_path, "CD-001")
    _attempt(agent, "CD-001", "fail", check_tail="1 failed")     # 1 of 3: codr retries

    await _tick(agent)

    assert resolved == [] and router.sent == []


@pytest.mark.asyncio
async def test_a_ticket_out_of_attempts_wakes_stagr_with_the_check_output(tmp_path):
    _task(tmp_path, "CD-001", text="max_attempts: 1\n\nBuild it.")
    agent, _, resolved = _wired(tmp_path, "CD-001")
    _attempt(agent, "CD-001", "fail", check_tail="AssertionError: 2 != 3")

    await _tick(agent)

    (token, _, status, result), = resolved
    assert status == "failed" and result["check_tail"] == "AssertionError: 2 != 3"
    assert "Do not queue" in result["instruction"]


@pytest.mark.asyncio
async def test_a_removed_ticket_wakes_stagr_and_a_passed_one_still_counts_as_passed(tmp_path):
    agent, _, resolved = _wired(tmp_path, "CD-001", "CD-002")    # neither folder exists
    _attempt(agent, "CD-002", "pass")

    await _tick(agent)

    assert sorted((t, s) for t, _, s, _ in resolved) == [("CD-001", "removed"), ("CD-002", "passed")]


@pytest.mark.asyncio
async def test_a_skipped_ticket_is_not_taken_for_a_removed_one(tmp_path):
    """A ticket DERPR cannot read (a bad header) is still in the queue,
    waiting for a fix: stagr is not told it was removed."""
    _task(tmp_path, "CD-001", text="colour: blue\n\nBuild it.")
    agent, _, resolved = _wired(tmp_path, "CD-001")

    await _tick(agent)

    assert resolved == []


@pytest.mark.asyncio
async def test_a_wake_in_flight_is_not_started_twice(tmp_path):
    _task(tmp_path, "CD-001")
    agent, _, resolved = _wired(tmp_path, "CD-001")
    _attempt(agent, "CD-001", "pass")

    agent._wake_waiting()
    agent._wake_waiting()                         # the next tick, while the first wake runs
    await asyncio.gather(*list(agent._wake_runs))

    assert len(resolved) == 1


@pytest.mark.asyncio
async def test_a_wake_that_parks_for_approval_says_so(tmp_path):
    _task(tmp_path, "CD-001")
    agent, router, _ = _wired(tmp_path, "CD-001", reply="Proposed CD-002.", confirm=True)
    _attempt(agent, "CD-001", "pass")

    await _tick(agent)

    assert "approval" in router.sent[0][2]


@pytest.mark.asyncio
async def test_a_failing_resume_does_not_stop_the_agent(tmp_path):
    _task(tmp_path, "CD-001")
    agent, router, _ = _wired(tmp_path, "CD-001")
    _attempt(agent, "CD-001", "pass")

    async def boom(*a, **k):
        raise RuntimeError("model down")
        yield  # pragma: no cover

    agent.chat_system.stream_resolve_deferral = boom
    await _tick(agent)

    assert router.sent == [] and not agent._waking


def test_no_confirmations_means_nothing_to_wake(tmp_path):
    agent, _ = _agent(tmp_path)
    agent._wake_waiting()                         # the test chat system has no store
    assert not agent._wake_runs


def test_wake_result_tolerates_an_unreadable_payload():
    result = wake_result("CD-009", "passed", "", [{"outcome": "pass", "outcome_payload": "{not json"}])
    assert result["ticket"] == "CD-009" and result["last_outcome"] == "pass"


# --- the switch -------------------------------------------------------------

FIELDS_BY_NAME = {f.name: f for f in PERSONA_FIELDS}


def _persona():
    return Persona(persona_name="stagr", model_name="local", prompt="p")


def test_auto_approve_is_off_by_default_and_set_by_the_dev_command():
    persona = _persona()
    assert persona.get_auto_approve() is False
    reply, mutated = FIELDS_BY_NAME["auto_approve"].set_cli(["auto_approve", "on"], persona)
    assert persona.get_auto_approve() is True and mutated and "until the next restart" in reply


def test_auto_approve_is_never_saved_so_a_restart_turns_it_off():
    persona = _persona()
    persona.set_auto_approve(True)
    (saved,) = to_dict({"stagr": persona})
    assert not any("auto_approve" in key for key in saved)


def test_auto_approve_has_no_patch_key():
    """Portal PATCH cannot reach it: the toggle is a dev command only."""
    assert FIELDS_BY_NAME["auto_approve"].patch_key is None


# --- the flag ---------------------------------------------------------------

def _tool(**extra):
    return {"type": "function", "is_write": True, "function": {"name": "x"},
            "capabilities": {"produces_untrusted": False, "irreversible": False,
                             "locality": "network", "sensitivity": "internal"}, **extra}


def test_only_queue_codr_task_is_auto_approvable():
    assert is_auto_approvable("queue_codr_task")
    assert not is_auto_approvable("request_network")
    assert not is_auto_approvable("copy_from_codr")


@pytest.mark.parametrize("tool", [
    _tool(auto_approvable="yes"),
    _tool(auto_approvable=True, is_write=False),
    {**_tool(auto_approvable=True),
     "capabilities": {**_tool()["capabilities"], "irreversible": True}},
])
def test_auto_approvable_is_refused_where_it_could_skip_a_real_gate(tool):
    with pytest.raises(ValueError):
        validate_tool_capabilities(tool)


# --- the deferral -----------------------------------------------------------

def test_a_codr_ticket_deferral_outlives_an_approval():
    assert deferral_ttl(DEFERRAL_KIND_CODR_TICKET) == global_config.CODR_TICKET_DEFERRAL_TTL
    assert deferral_ttl(DEFERRAL_KIND_CODR_TICKET) > deferral_ttl(DEFERRAL_KIND_APPROVAL)
    assert deferral_ttl(DEFERRAL_KIND_NODE_JOB) == global_config.PENDING_ACTION_TTL


def _park(token, kind, age=0.0):
    return ParkedWrite(token=token, write_call={"id": "c", "name": "queue_codr_task", "arguments": {}},
                       audit_info={}, confirmation_text="", user_identifier="u", persona_name="stagr",
                       channel="portal", server_id=None, kind=kind, created_at=time.time() - age)


def test_pending_of_kind_never_hands_out_an_approval(mocked_confirmations):
    store = mocked_confirmations
    store.pending = {"a": _park("a", DEFERRAL_KIND_APPROVAL), "CD-001": _park("CD-001", DEFERRAL_KIND_CODR_TICKET)}
    assert [p.token for p in store.pending_of_kind(DEFERRAL_KIND_CODR_TICKET)] == ["CD-001"]
    assert store.pending_of_kind(DEFERRAL_KIND_APPROVAL) == []


def test_a_two_day_old_ticket_deferral_is_not_expired(mocked_confirmations):
    store = mocked_confirmations
    two_days = 2 * 24 * 60 * 60
    assert not store.is_expired(_park("CD-001", DEFERRAL_KIND_CODR_TICKET, age=two_days))
    assert store.is_expired(_park("a", DEFERRAL_KIND_APPROVAL, age=two_days))


@pytest.fixture
def mocked_confirmations():
    from src.confirmations import ConfirmationManager
    store = ConfirmationManager.__new__(ConfirmationManager)
    return store
