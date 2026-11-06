# tests/test_stagr_codr_handoff.py
"""DP-493: stagr hears when codr finishes a ticket it queued, and may queue
the next without a prompt when its tool policy lists `queue_codr_task` under
`allow` (DP-497: allow/ask decides for every tool, so the same holds for its
other three gated tools).

The resume itself is the DP-345 deferral mechanism, pinned end to end in
tests/integration/test_unprompted_write_deferral.py. Here: the pieces around
it — which tickets wake stagr, what the wake carries, the allow/ask gate and
the deferral's longer lifetime."""

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import global_config
from src.agents.codr_agent import wake_result
from src.confirmations import ParkedWrite, deferral_ttl
from src.deferral_kinds import DEFERRAL_KIND_APPROVAL, DEFERRAL_KIND_CODR_TICKET, DEFERRAL_KIND_NODE_JOB
from src.generation_events import DoneEvent, PendingConfirmationEvent, ResponseType
from src.persona import Persona
from src.persona_fields import PERSONA_FIELDS
from src.stagr.codr_ticket import ATTEMPT_ACTION, attempt_context
from src.tool_policy import ToolPolicy
from src.tools.definitions import ALL_TOOL_DEFINITIONS
from src.tools.tool_loop import _needs_approval
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
@pytest.mark.parametrize("outcome", ["timed_out", "stuck"])
async def test_a_timed_out_or_stuck_ticket_immediately_wakes_stagr(tmp_path, outcome):
    """DP-482: timed_out and stuck are not retried even with attempts left; they immediately wake stagr."""
    _task(tmp_path, "CD-001", text="max_attempts: 3\n\nBuild it.")
    agent, _, resolved = _wired(tmp_path, "CD-001")
    _attempt(agent, "CD-001", outcome, reason=f"the turn ended as {outcome}")

    await _tick(agent)

    (token, _, status, result), = resolved
    assert status == "failed"
    assert result["last_outcome"] == outcome
    assert result["reason"] == f"the turn ended as {outcome}"
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


# --- the allow/ask gate (DP-496, DP-497) -------------------------------------

# Every gated stagr tool ships under ask; since DP-497 any of them may be set
# to allow, like every other tool.
GATED = ("queue_codr_task", "copy_from_codr", "request_network", "request_web_access")
_STAGR_JSON = Path(__file__).resolve().parents[1] / "config" / "optional_personas" / "stagr.json"


def _persona(**policy):
    persona = Persona(persona_name="stagr", model_name="local", prompt="p")
    persona.set_tool_policy({"default": "deny", **policy})
    return persona


def _runs_unprompted(persona, name):
    return not _needs_approval(persona, name)


@pytest.mark.parametrize("name", GATED)
def test_a_gated_stagr_tool_under_allow_runs_unprompted(name):
    assert _runs_unprompted(_persona(allow=[name]), name)


@pytest.mark.parametrize("name", GATED)
def test_a_gated_stagr_tool_under_ask_parks(name):
    assert not _runs_unprompted(_persona(ask=[name]), name)


def test_listed_under_both_ask_wins():
    assert not _runs_unprompted(_persona(allow=["queue_codr_task"], ask=["queue_codr_task"]), "queue_codr_task")


@pytest.mark.parametrize("name", ["create_ticket", "set_sandbox_net", "merge_tickets"])
def test_any_other_write_follows_allow_too(name):
    """DP-497: no tool is outside the rule -- an irreversible one included
    (setting it to allow is warned about, not refused)."""
    assert _runs_unprompted(_persona(allow=[name]), name)
    assert not _runs_unprompted(_persona(ask=[name]), name)


def test_a_read_under_ask_parks():
    assert not _runs_unprompted(_persona(ask=["stagr_shell"]), "stagr_shell")
    assert _runs_unprompted(_persona(allow=["stagr_shell"]), "stagr_shell")


def test_a_stand_in_persona_gets_each_tools_default():
    """No ToolPolicy (a test double): a write asks, a read runs."""
    assert not _runs_unprompted(SimpleNamespace(), "queue_codr_task")
    assert not _runs_unprompted(SimpleNamespace(get_tool_policy=lambda: {"allow": list(GATED)}), "queue_codr_task")
    assert _runs_unprompted(SimpleNamespace(), "stagr_shell")


def test_the_shipped_stagr_lists_its_gated_tools_under_ask():
    """Default behaviour is DP-493's: every gated tool parks until Adam sets
    it to allow."""
    policy = ToolPolicy.from_dict(json.loads(_STAGR_JSON.read_text(encoding="utf-8"))["personas"][0]["tool_policy"])
    assert set(policy.ask) == set(GATED) and not set(policy.allow) & set(GATED)
    persona = _persona(**policy.to_dict())
    assert not any(_runs_unprompted(persona, name) for name in GATED)


def test_there_is_no_auto_approve_field_any_more():
    assert "auto_approve" not in {f.name for f in PERSONA_FIELDS}
    assert not hasattr(Persona, "set_auto_approve")


def test_no_definition_carries_the_removed_auto_approvable_key():
    assert not [t for t in ALL_TOOL_DEFINITIONS if "auto_approvable" in t]


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
