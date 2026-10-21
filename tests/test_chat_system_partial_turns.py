# tests/test_chat_system_partial_turns.py
"""DP-444: what a stopped or errored turn persists.

Both keep their tool steps and, as the reply, only the text written after the
last of them. Each step's own prose already rides on its assistant entry in
`tool_context` (DP-338), so saving every streamed delta as the reply sent the
model that prose twice. A step still in flight when the turn stops is kept,
each call with the outcome it actually had; gated writes stay approvable.
"""

import asyncio
import itertools
import json

import pytest

from src.chat_system import ErrorEvent
from src.engine import LLMCommunicationError
from tests.helpers import offer_tools

# Reuse the module's fixture rather than re-deriving the mocked ChatSystem.
from tests.test_chat_system import chat_system_with_mocks  # noqa: F401


def _script(text_engine, *steps):
    """Each step is one model call: a list of unified engine events, or an
    exception to raise once the listed events (if any) have streamed."""
    it = iter(steps)

    async def _stream(persona_config, history_object, tools=None,
                      local_inference_config=None):
        step = next(it)
        events, exc = (step, None) if isinstance(step, list) else step
        yield {"type": "api_payload", "payload": {}}
        for ev in events:
            yield ev
        if exc is not None:
            raise exc

    text_engine._stream_response = _stream


def _say(text):
    return {"type": "text_delta", "text": text}


def _calls(*calls):
    return [{"type": "tool_calls", "calls": list(calls)},
            {"type": "done", "full_text": ""}]


def _call(cid, name, **args):
    return {"id": cid, "name": name, "arguments": args}


def _setup(chat_system_with_mocks):
    system, memory_mock, text_engine, persona, tool_manager = chat_system_with_mocks
    memory_mock.get_channel_history.return_value = []
    ids = itertools.count(10)
    memory_mock.log_message.side_effect = lambda *a, **k: next(ids)
    offer_tools(system, "test_persona", "search_tickets", "update_ticket")
    persona.set_enabled_tools(["*"])
    tool_manager.execute_tool.return_value = {"result": "ok"}
    return system, memory_mock, text_engine, tool_manager


def _assistant_rows(memory_mock):
    return [c.kwargs for c in memory_mock.log_message.call_args_list
            if c.kwargs.get("author_role") == "assistant"]


async def _drain(stream):
    return [ev async for ev in stream]


async def _stop(system):
    with pytest.raises(asyncio.CancelledError):
        await _drain(system.stream_response("test_persona", "user", "chan", "go"))


@pytest.mark.asyncio
async def test_stop_mid_loop_keeps_finished_steps_and_their_parks(chat_system_with_mocks):
    system, memory_mock, text_engine, _ = _setup(chat_system_with_mocks)
    _script(
        text_engine,
        [_say("Checking first."),
         *_calls(_call("r1", "search_tickets", query="q"),
                 _call("w1", "update_ticket", state="closed"))],
        ([_say("Next I'll look at the card.")], asyncio.CancelledError()),
    )

    await _stop(system)

    (row,) = _assistant_rows(memory_mock)
    # The reply is only what came after the finished step...
    assert row["content"] == "Next I'll look at the card."
    # ...and the step's own prose is on its entry, not repeated in the reply.
    ctx = json.loads(row["tool_context"])
    step = next(m for m in ctx if m.get("tool_calls"))
    assert step["content"] == "Checking first."
    answered = {m["tool_call_id"] for m in ctx if m.get("role") == "tool"}
    assert answered == {"r1", "w1"}
    # The write proposed in the finished step stays approvable.
    parks = system.confirmations.list_for("user", "test_persona")
    assert [p.write_call["name"] for p in parks] == ["update_ticket"]
    assert parks[0].parked_assistant_id == 11


@pytest.mark.asyncio
async def test_stop_mid_step_keeps_the_step_with_each_calls_real_outcome(
        chat_system_with_mocks):
    """The step in flight is kept, not dropped: dropping it left its prose as
    the reply -- the model's plan with no trace of the calls it made -- so the
    next turn could assume a search had run. A call that returned keeps its
    real result; one cut off mid-run is marked unknown, since some reads
    (`set_timer`) change state; a write never runs inside a turn, so an
    unanswered one is `not_executed`."""
    system, memory_mock, text_engine, tool_manager = _setup(chat_system_with_mocks)
    _script(
        text_engine,
        [_say("Step one."), *_calls(_call("r1", "search_tickets", query="a"))],
        [_say("Step two."), *_calls(_call("r2", "search_tickets", query="b"),
                                    _call("r3", "search_tickets", query="c"),
                                    _call("w1", "update_ticket", state="closed"))],
    )
    tool_manager.execute_tool.side_effect = [
        {"result": "first"}, {"result": "second"}, asyncio.CancelledError(),
    ]

    await _stop(system)

    (row,) = _assistant_rows(memory_mock)
    ctx = json.loads(row["tool_context"])
    steps = [m for m in ctx if m.get("tool_calls")]
    assert [m["content"] for m in steps] == ["Step one.", "Step two."]
    # Both steps' prose is on their entries, so none of it is the reply.
    assert row["content"] == ""
    results = {m["tool_call_id"]: json.loads(m["content"])
               for m in ctx if m.get("role") == "tool"}
    assert results["r2"] == {"result": "second"}
    assert results["r3"]["status"] == "interrupted"
    assert "unknown" in results["r3"]["error"]
    assert results["w1"]["status"] == "not_executed"
    # The write never parked, so nothing is approvable.
    assert system.confirmations.list_for("user", "test_persona") == []


@pytest.mark.asyncio
async def test_stop_during_the_answer_keeps_the_partial_answer(chat_system_with_mocks):
    system, memory_mock, text_engine, _ = _setup(chat_system_with_mocks)
    _script(
        text_engine,
        [_say("Looking it up."), *_calls(_call("r1", "search_tickets", query="q"))],
        ([_say("There are three"), _say(" open tickets")], asyncio.CancelledError()),
    )

    await _stop(system)

    (row,) = _assistant_rows(memory_mock)
    assert row["content"] == "There are three open tickets"
    assert "Looking it up." not in row["content"]
    assert json.loads(row["tool_context"])


@pytest.mark.asyncio
async def test_stop_after_tools_with_no_text_saves_a_tool_only_row(chat_system_with_mocks):
    system, memory_mock, text_engine, _ = _setup(chat_system_with_mocks)
    _script(
        text_engine,
        _calls(_call("r1", "search_tickets", query="q")),
        ([], asyncio.CancelledError()),
    )

    await _stop(system)

    (row,) = _assistant_rows(memory_mock)
    assert row["content"] == ""
    assert json.loads(row["tool_context"])


@pytest.mark.asyncio
async def test_stop_before_anything_saves_nothing(chat_system_with_mocks):
    system, memory_mock, text_engine, _ = _setup(chat_system_with_mocks)
    _script(text_engine, ([], asyncio.CancelledError()))

    await _stop(system)

    assert _assistant_rows(memory_mock) == []


@pytest.mark.parametrize("interrupt", [
    asyncio.CancelledError(), LLMCommunicationError("upstream 500"),
], ids=["stop", "error"])
@pytest.mark.asyncio
async def test_interrupted_turn_keeps_the_conversation_tainted(
        chat_system_with_mocks, interrupt):
    """The saved step replays its untrusted `search_tickets` result to the
    model next turn, so the conversation stays tainted, as after a finished
    turn. Only `_LoopFinishedEvent` used to set the sticky bit."""
    system, _, text_engine, _ = _setup(chat_system_with_mocks)
    _script(
        text_engine,
        [_say("Looking."), *_calls(_call("r1", "search_tickets", query="q"))],
        ([_say("Found")], interrupt),
    )

    if isinstance(interrupt, asyncio.CancelledError):
        await _stop(system)
    else:
        await _drain(system.stream_response("test_persona", "user", "chan", "go"))

    key = ("user", "test_persona", "chan", None)
    assert system.request_builder.conversation_taints.get(key) is True


@pytest.mark.asyncio
async def test_error_no_longer_repeats_step_prose_in_the_reply(chat_system_with_mocks):
    system, memory_mock, text_engine, _ = _setup(chat_system_with_mocks)
    _script(
        text_engine,
        [_say("Checking first."), *_calls(_call("r1", "search_tickets", query="q"))],
        ([_say("Half an ans")], LLMCommunicationError("upstream 500")),
    )

    events = await _drain(
        system.stream_response("test_persona", "user", "chan", "go")
    )

    assert isinstance(events[-1], ErrorEvent)
    (row,) = _assistant_rows(memory_mock)
    assert row["content"] == "Half an ans"
    step = next(m for m in json.loads(row["tool_context"]) if m.get("tool_calls"))
    assert step["content"] == "Checking first."
