# tests/test_system_prompt_merge.py
#
# DP-317 — every exit from `history_object` merges a leading system turn onto
# the persona prompt; none of them substitutes for it.
#
# Three places split a `history_object` back into (system prompt, messages):
#
#   * `_shared.extract_system_prompt`      -> anthropic, google, agy, cc
#   * `openai.build_openai_params`         -> gpt-*
#   * `StreamEngine._build_messages`       -> local (koboldcpp)
#
# The latter two used to inline their own split that DISCARDED
# `persona_prompt` whenever the history opened with a system turn. That
# divergence was introduced by a single 2025-10-06 commit (`3921318`, subject:
# "reimplement history limit") which added the leading-system-turn branch to
# all three providers at once and transcribed it two different ways; before it,
# no provider had the branch at all. There is no design rationale to preserve.
#
# These are differential tests on purpose: the same `history_object` goes
# through every exit and the results must agree. A future exit that re-derives
# the split will fail here.

import pytest

from src.engine.providers._shared import extract_system_prompt
from src.engine.providers.openai import build_openai_params
from src.stream_engine import StreamEngine

PERSONA = "You are the persona. Follow these standing instructions."
INJECTED = "[Recent actions]\n- did a thing"


def _history_object(message_history):
    return {
        "persona_prompt": PERSONA,
        "message_history": list(message_history),
        "history": list(message_history),  # legacy alias
        "current_message": {"text": "", "image_url": None},
    }


def _openai_system(history_object):
    params = build_openai_params({"model_name": "gpt-4o"}, history_object)
    return params["messages"][0]["content"], params["messages"][1:]


def _local_system(history_object):
    messages = StreamEngine._build_messages(history_object)
    return messages[0]["content"], messages[1:]


def _shared_system(history_object):
    system_prompt, history = extract_system_prompt(history_object)
    return system_prompt, history


ALL_EXITS = [
    pytest.param(_shared_system, id="shared/anthropic+google+agy+cc"),
    pytest.param(_openai_system, id="openai"),
    pytest.param(_local_system, id="local/kobold"),
]


@pytest.mark.parametrize("exit_fn", ALL_EXITS)
def test_leading_system_turn_merges_onto_persona_prompt(exit_fn):
    """The regression: a leading system turn must not evict the persona prompt.

    `agents/base._build_history_object` is the one producer in the tree that
    sets `persona_prompt` AND prepends a system turn (its action-history
    block), so an agent on a gpt-* or local model used to lose its entire
    persona prompt with no error and no log line.
    """
    ho = _history_object([
        {"role": "system", "content": INJECTED},
        {"role": "user", "content": "hello"},
    ])

    system_prompt, rest = exit_fn(ho)

    assert PERSONA in system_prompt, "persona prompt was dropped"
    assert INJECTED in system_prompt, "injected system turn was dropped"
    # The system turn is consumed, not left in the message array to be sent twice.
    assert [m["role"] for m in rest] == ["user"]


@pytest.mark.parametrize("exit_fn", ALL_EXITS)
def test_no_leading_system_turn_uses_persona_prompt_alone(exit_fn):
    """The common case — every non-agent caller. Guards the DP-206 goldens:
    this path's wire bytes must not move."""
    ho = _history_object([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ])

    system_prompt, rest = exit_fn(ho)

    assert system_prompt == PERSONA
    assert [m["role"] for m in rest] == ["user", "assistant"]


def test_all_exits_agree():
    """Differential: the same history_object through every exit yields the
    same system prompt and the same remaining turns."""
    ho = _history_object([
        {"role": "system", "content": INJECTED},
        {"role": "user", "content": "hello"},
    ])

    results = [exit_fn.values[0](ho) for exit_fn in ALL_EXITS]
    systems = {system for system, _ in results}
    assert len(systems) == 1, f"exits disagree on the system prompt: {systems}"

    roles = {tuple(m["role"] for m in rest) for _, rest in results}
    assert len(roles) == 1, f"exits disagree on the remaining turns: {roles}"


def test_agent_history_object_shape_is_the_reachable_case():
    """Pin the producer, so this stays a regression test rather than a
    hypothetical: the agent base class really does emit both fields."""
    from src.agents.base import Agent

    class _Stub(Agent):
        agent_name = "stub"
        action_history_limit = 5

        async def deploy(self):  # abstract on Agent
            raise NotImplementedError

        def _get_action_history_message(self, task_data=None):
            return INJECTED

    class _Persona:
        def get_prompt(self):
            return PERSONA

    # __new__ without __init__: the builder only reads class attributes and
    # `_get_action_history_message`, and __init__ demands a live ChatSystem.
    agent = _Stub.__new__(_Stub)
    built = agent._build_history_object(_Persona(), "do the thing")

    assert built["persona_prompt"] == PERSONA
    assert built["message_history"][0]["role"] == "system"

    system_prompt, _ = extract_system_prompt(built)
    assert PERSONA in system_prompt
    assert INJECTED in system_prompt
