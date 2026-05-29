# tests/integration/test_agent_loop_e2e.py
#
# End-to-end coverage of the agent tool-use loop driven through the real
# orchestration kernel (`ChatSystem._orchestrate` via `stream_response`) with
# a real MemoryManager + SQLite backend. Only the LLM provider is mocked.
#
# The three properties under test:
#   1. Context flow — at *each* iteration of the loop the model is handed a
#      message history that has accumulated the prior iterations' tool calls
#      and tool results (so the agent "gets proper context at each step").
#   2. Memory recording — the user turn and the final assistant turn are both
#      persisted (log_message rows) and pushed through the backend boundary
#      (`retain_turn`), and the assistant row carries the tool-call transcript
#      as tool_context.
#   3. Clean closure — the per-turn ContextVar is set during the turn and
#      reset to None afterward, and the stream terminates with exactly one
#      DoneEvent (carrying the real interaction ids) and nothing after it.

import copy
import json

import pytest

from src.chat_system import (
    DoneEvent, ErrorEvent, TokenEvent,
    ToolCallResultEvent, ToolCallStartEvent, ResponseType,
)
from src.engine import LLMCommunicationError
from src.tools.turn_context import get_turn_context

pytestmark = pytest.mark.integration


async def _drain(stream):
    return [ev async for ev in stream]


def _script_engine(chat_system, scripted, *, capture):
    """Point the mocked `generate_response` at a sequence of scripted
    provider results. Each call records the live turn context, the system
    prompt, and a *deep copy* of the message history it was handed, so the
    per-iteration context snapshots can't be mutated out from under the
    assertions by the in-place history append the loop performs.

    `scripted` is a list of `(result_dict, api_payload)` tuples.
    `capture` is a dict that accumulates `turn_ctx_seen` and `histories`.
    """
    it = iter(scripted)

    async def fake_generate_response(persona_config, history_object, *args, **kwargs):
        capture["turn_ctx_seen"].append(get_turn_context())
        capture["histories"].append({
            "persona_prompt": history_object.get("persona_prompt"),
            "message_history": copy.deepcopy(history_object.get("message_history")),
        })
        return next(it)

    chat_system.text_engine.generate_response.side_effect = fake_generate_response


@pytest.mark.asyncio
async def test_tool_loop_context_flow_memory_and_clean_close(mocked_chat_system):
    """Two sequential read-tool iterations then a final text answer, driven
    end-to-end. Asserts context accumulation per step, memory persistence,
    and clean turn closure."""
    chat_system, memory_manager = mocked_chat_system
    persona = chat_system.personas["test_persona"]
    persona.set_enabled_tools(["*"])

    user_id, channel = "user-e2e", "chan-e2e"

    # The model: call get_agent_status, then get_agent_history, then answer.
    # Both tools are read-only and produces_untrusted=False, so the loop runs
    # them inline and the turn stays untainted (clean close).
    scripted = [
        ({"type": "tool_calls",
          "calls": [{"id": "call_1", "name": "get_agent_status",
                     "arguments": {"agent_id": "zammad_bot"}}]}, {}),
        ({"type": "tool_calls",
          "calls": [{"id": "call_2", "name": "get_agent_history",
                     "arguments": {"agent_id": "zammad_bot"}}]}, {}),
        ({"type": "text", "content": "Both agents are healthy."}, {}),
    ]
    capture = {"turn_ctx_seen": [], "histories": []}
    _script_engine(chat_system, scripted, capture=capture)

    # Real tool_manager, but stub the actual tool implementations: the loop
    # dispatches by name regardless of advertisement, and we want
    # deterministic, network-free results.
    tool_results = {
        "get_agent_status": {"agent": "zammad_bot", "status": "running"},
        "get_agent_history": {"agent": "zammad_bot", "events": ["triaged #42"]},
    }

    async def fake_execute(name, **kwargs):
        return tool_results[name]

    chat_system.tool_manager.execute_tool = fake_execute  # type: ignore[assignment]

    # Spy on the backend boundary without breaking its (noop) behavior.
    from unittest.mock import AsyncMock
    retain_spy = AsyncMock()
    chat_system.memory_backend.retain_turn = retain_spy  # type: ignore[assignment]

    # Sanity: no turn context leaking in from a previous turn.
    assert get_turn_context() is None

    events = await _drain(chat_system.stream_response(
        "test_persona", user_id, channel, "check the agents",
    ))

    # --- (1) Context flow: the model was called once per loop iteration, and
    #         each call's history had accumulated the prior tool round-trips. ---
    assert chat_system.text_engine.generate_response.call_count == 3
    histories = capture["histories"]

    # System prompt is present on every iteration.
    for h in histories:
        assert h["persona_prompt"] == persona.get_prompt()

    # Iteration 0: only the user turn — no tool traffic yet.
    iter0 = histories[0]["message_history"]
    assert iter0[-1] == {"role": "user", "content": "check the agents"}
    assert all("tool_calls" not in m and m.get("role") != "tool" for m in iter0)

    # Iteration 1: user turn + assistant tool_calls(get_agent_status) + its result.
    iter1 = histories[1]["message_history"]
    assert iter1[0]["role"] == "user"
    assistant_tc_1 = next(m for m in iter1 if m.get("role") == "assistant" and "tool_calls" in m)
    assert assistant_tc_1["tool_calls"][0]["name"] == "get_agent_status"
    tool_msg_1 = next(m for m in iter1 if m.get("role") == "tool")
    assert tool_msg_1["tool_call_id"] == "call_1"
    assert "running" in tool_msg_1["content"]
    # The second tool's round-trip has NOT happened yet at iteration 1.
    assert not any(m.get("tool_call_id") == "call_2" for m in iter1)

    # Iteration 2: both tool round-trips are now visible, in order.
    iter2 = histories[2]["message_history"]
    tool_ids = [m.get("tool_call_id") for m in iter2 if m.get("role") == "tool"]
    assert tool_ids == ["call_1", "call_2"]
    status_names = [m["tool_calls"][0]["name"] for m in iter2
                    if m.get("role") == "assistant" and "tool_calls" in m]
    assert status_names == ["get_agent_status", "get_agent_history"]
    # History only grows — iteration 2 is a strict superset of iteration 1.
    assert len(iter2) > len(iter1) > len(iter0)

    # --- (2) Memory recording. ---
    # User + assistant rows persisted to the real DB.
    rows = memory_manager.get_channel_history(channel, "test_persona", None, 50)
    assert [r["author_role"] for r in rows] == ["user", "assistant"]
    assert rows[0]["content"] == "check the agents"
    assert rows[1]["content"] == "Both agents are healthy."

    # The assistant row carries the full tool transcript as tool_context.
    tool_ctx = json.loads(rows[1]["tool_context"])
    ctx_tool_ids = [m.get("tool_call_id") for m in tool_ctx if m.get("role") == "tool"]
    assert ctx_tool_ids == ["call_1", "call_2"]

    # Backend boundary received both turns with the right roles + content,
    # and the assistant turn is marked untrusted=False (no tainting tool ran).
    assert retain_spy.await_count == 2
    roles = [c.kwargs["role"] for c in retain_spy.await_args_list]
    assert roles == ["user", "assistant"]
    user_retain, assistant_retain = retain_spy.await_args_list
    assert user_retain.kwargs["content"] == "check the agents"
    assert user_retain.kwargs["bank_id"] == "test_persona"
    assert assistant_retain.kwargs["content"] == "Both agents are healthy."
    assert assistant_retain.kwargs["untrusted"] is False

    # --- (3) Clean closure. ---
    # Turn context was live during every model call...
    assert all(ctx is not None for ctx in capture["turn_ctx_seen"])
    seen = capture["turn_ctx_seen"][0]
    assert seen.persona_name == "test_persona"
    assert seen.user_identifier == user_id
    assert seen.channel == channel
    # ...and is reset once the turn ends.
    assert get_turn_context() is None

    # Exactly one DoneEvent, and it is the final event (nothing trails it).
    done_events = [e for e in events if isinstance(e, DoneEvent)]
    assert len(done_events) == 1
    assert isinstance(events[-1], DoneEvent)
    done = done_events[0]
    assert done.text == "Both agents are healthy."
    assert done.response_type == ResponseType.LLM_GENERATION
    assert done.assistant_id == rows[1]["interaction_id"]
    assert done.user_interaction_id == rows[0]["interaction_id"]
    assert not any(isinstance(e, ErrorEvent) for e in events)

    # Tool lifecycle events are well-formed and paired (start before result).
    starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
    results = [e for e in events if isinstance(e, ToolCallResultEvent)]
    assert [s.tool_name for s in starts] == ["get_agent_status", "get_agent_history"]
    assert {r.call_id for r in results} == {"call_1", "call_2"}
    types = [type(e).__name__ for e in events]
    assert types.index("ToolCallStartEvent") < types.index("ToolCallResultEvent")
    # Final TokenEvent (the answer) lands after the last tool result.
    assert types.index("TokenEvent") > max(
        i for i, t in enumerate(types) if t == "ToolCallResultEvent"
    )


@pytest.mark.asyncio
async def test_tool_loop_closes_cleanly_on_llm_error(mocked_chat_system):
    """If the provider fails mid-loop, the turn still closes neatly: an
    ErrorEvent (no DoneEvent) is emitted and the per-turn ContextVar is reset
    rather than leaking into the next request."""
    chat_system, memory_manager = mocked_chat_system
    persona = chat_system.personas["test_persona"]
    persona.set_enabled_tools(["*"])

    chat_system.text_engine.generate_response.side_effect = LLMCommunicationError("upstream 500")

    assert get_turn_context() is None
    events = await _drain(chat_system.stream_response(
        "test_persona", "user-err", "chan-err", "do something",
    ))

    assert any(isinstance(e, ErrorEvent) for e in events)
    assert not any(isinstance(e, DoneEvent) for e in events)
    # Context var must not leak past the failed turn.
    assert get_turn_context() is None

    # The user turn is still pinned even though the model errored mid-flight.
    rows = memory_manager.get_channel_history("chan-err", "test_persona", None, 50)
    assert any(r["author_role"] == "user" and r["content"] == "do something" for r in rows)
