# tests/proposals/test_agent_call.py
"""DP-240: approved subagent tool calls, and the execute-time policy re-check.

A proposal row can sit in the queue indefinitely. The property under test is
that approval executes against the policy in force *now*, never the one that was
in force when the row was written.
"""

import pytest

from src.proposals.agent_call import AgentCallRunner
from src.proposals.executor import ProposalExecutor
from src.proposals.schemas import (
    AGENT_CALL_ACTIONS,
    EXECUTABLE_ACTIONS,
    PROPOSAL_ACTIONS,
    build_submission_tool_schema,
    validate_proposal_args,
)
from src.tool_policy import ToolPolicy
from src.tools.tool_manager import ToolManager


TOOL_DEF = {
    "type": "function",
    "is_write": True,
    "function": {"name": "restart_thing", "description": "d", "parameters": {}},
}


class _FakeToolManager(ToolManager):
    def __init__(self, definitions):
        super().__init__()
        self._definitions = definitions

    def get_tool_definitions(self):
        return list(self._definitions)


@pytest.fixture
def runner_and_calls():
    manager = _FakeToolManager([TOOL_DEF])
    calls = []

    async def _handler(**kwargs):
        calls.append(kwargs)
        return "restarted"

    manager.register("restart_thing", _handler)
    policy = ToolPolicy(default="deny", allow=["restart_thing"])
    return AgentCallRunner(lambda: manager, lambda: policy), calls, policy


# -- whitelist separation ----------------------------------------------------

def test_agent_call_action_is_not_offered_to_managr():
    """managr reads attacker-reachable ticket content. Its extraction schema must
    not contain a generic tool-call action, or a poisoned ticket could steer it
    into proposing arbitrary derpr tool calls."""
    enum = (build_submission_tool_schema()["function"]["parameters"]
            ["properties"]["proposals"]["items"]["properties"]["action_type"]["enum"])

    assert "call_derpr_tool" not in enum
    assert set(enum) == set(PROPOSAL_ACTIONS.keys())


def test_default_validation_scope_rejects_agent_calls():
    """Every pre-existing caller (managr's emission path included) passes no
    scope, so the safe whitelist must be the default."""
    errors = validate_proposal_args(
        "call_derpr_tool",
        {"tool_name": "restart_thing", "tool_args": {}, "agent_id": "a-1"},
    )
    assert errors == ["unknown action_type 'call_derpr_tool'"]


def test_executor_scope_accepts_agent_calls():
    errors = validate_proposal_args(
        "call_derpr_tool",
        {"tool_name": "restart_thing", "tool_args": {}, "agent_id": "a-1"},
        scope=EXECUTABLE_ACTIONS,
    )
    assert errors == []


def test_executable_actions_is_the_union():
    assert set(EXECUTABLE_ACTIONS) == set(PROPOSAL_ACTIONS) | set(AGENT_CALL_ACTIONS)


def test_agent_call_args_are_validated():
    scope = EXECUTABLE_ACTIONS
    assert "missing required argument 'tool_args'" in validate_proposal_args(
        "call_derpr_tool", {"tool_name": "x", "agent_id": "a"}, scope=scope)
    assert "argument 'tool_args' must be dict" in validate_proposal_args(
        "call_derpr_tool",
        {"tool_name": "x", "tool_args": "not-a-dict", "agent_id": "a"}, scope=scope)
    assert "unexpected argument 'sneaky'" in validate_proposal_args(
        "call_derpr_tool",
        {"tool_name": "x", "tool_args": {}, "agent_id": "a", "sneaky": 1}, scope=scope)


# -- execute-time re-check ---------------------------------------------------

@pytest.mark.asyncio
async def test_approved_call_executes(runner_and_calls):
    runner, calls, _ = runner_and_calls
    ok, message = await runner.run("restart_thing", {"name": "chatbot"})

    assert ok is True
    assert message == "restarted"
    assert calls == [{"name": "chatbot"}]


@pytest.mark.asyncio
async def test_call_refused_when_policy_narrowed_after_queueing(runner_and_calls):
    """The load-bearing case: the row was queued while the tool was exposed, and
    the operator narrowed the policy before approving."""
    runner, calls, policy = runner_and_calls
    policy.allow = []

    ok, message = await runner.run("restart_thing", {"name": "chatbot"})

    assert ok is False
    assert "not exposed by the current bridge policy" in message
    assert calls == []


@pytest.mark.asyncio
async def test_call_refused_when_tool_unregistered_after_queueing(runner_and_calls):
    runner, calls, _ = runner_and_calls
    runner._tool_manager_lookup()._definitions = []

    ok, message = await runner.run("restart_thing", {})

    assert ok is False
    assert "not exposed" in message
    assert calls == []


@pytest.mark.asyncio
async def test_tool_error_is_reported_as_failure(runner_and_calls):
    runner, _, _ = runner_and_calls

    async def _boom(**kwargs):
        raise RuntimeError("nope")

    runner._tool_manager_lookup().register("restart_thing", _boom)
    ok, message = await runner.run("restart_thing", {})

    assert ok is False
    assert "nope" in message


# -- executor integration ----------------------------------------------------

@pytest.mark.asyncio
async def test_executor_runs_agent_call(runner_and_calls):
    runner, calls, _ = runner_and_calls
    executor = ProposalExecutor(zammad_client=None, agent_call_runner=runner)

    ok, message = await executor.execute({
        "action_type": "call_derpr_tool",
        "action_args": {"tool_name": "restart_thing", "tool_args": {"name": "c"},
                        "agent_id": "a-1"},
    })

    assert ok is True
    assert message == "restarted"
    assert calls == [{"name": "c"}]


@pytest.mark.asyncio
async def test_executor_refuses_agent_call_without_a_runner():
    """Bridge not wired: the row must fail loudly rather than be marked executed."""
    executor = ProposalExecutor(zammad_client=None)

    ok, message = await executor.execute({
        "action_type": "call_derpr_tool",
        "action_args": {"tool_name": "restart_thing", "tool_args": {},
                        "agent_id": "a-1"},
    })

    assert ok is False
    assert "not wired" in message


@pytest.mark.asyncio
async def test_executor_refuses_ticket_actions_without_zammad(runner_and_calls):
    """Bridge-only deployment: the ticket half of the executor must refuse
    with a readable reason, not crash on a None client. This is what makes it
    safe to register the review surface without Zammad."""
    runner, _, _ = runner_and_calls
    executor = ProposalExecutor(zammad_client=None, agent_call_runner=runner)

    ok, message = await executor.execute({
        "action_type": "add_note",
        "action_args": {"ticket_number": 42, "body": "hello"},
    })

    assert ok is False
    assert "Zammad is not configured" in message


@pytest.mark.asyncio
async def test_executor_revalidates_agent_call_args(runner_and_calls):
    """A row tampered between review and execution still can't smuggle args."""
    runner, calls, _ = runner_and_calls
    executor = ProposalExecutor(zammad_client=None, agent_call_runner=runner)

    ok, message = await executor.execute({
        "action_type": "call_derpr_tool",
        "action_args": {"tool_name": "restart_thing"},
    })

    assert ok is False
    assert "failed re-validation" in message
    assert calls == []
