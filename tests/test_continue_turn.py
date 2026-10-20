# tests/test_continue_turn.py
"""DP-447: continue the trailing assistant reply, and edit a turn as raw text.

Continue resumes a reply in the model's own words: the wire array ends on the
row's text as a prefix (KoboldCPP appends it verbatim after the generation
prompt), and what streams is appended to the same row. The raw edit is the way
in: one text holding `<think>` + reasoning + `</think>` + reply.
"""
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from src.generation_events import (
    ReasoningEvent, ThinkCloseSplitter, TokenEvent, render_continue_prefix,
    split_think_text,
)
from src.memory.memory_manager import MemoryManager
from src.request_builder import ContinueTurn
from tests.test_chat_system import _drain_events, chat_system_with_mocks  # noqa: F401

_TS = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


# --- think-text helpers ------------------------------------------------------

def test_split_think_text_closed_open_and_plain():
    assert split_think_text("<think>\nplan\n</think>\n\nReply.") == ("Reply.", "plan", False)
    assert split_think_text("<think>\nstill going") == ("", "still going", True)
    # Closed with an empty reply is still closed — the case the row can't keep.
    assert split_think_text("<think>plan</think>") == ("", "plan", False)
    assert split_think_text("Just a reply.") is None


def test_continue_prefix_per_row_shape():
    # Open thinking: resume inside it (the template already opened the block).
    assert render_continue_prefix("", "17*20 = 340", None) == ("17*20 = 340", True)
    # A reply exists: thinking is closed whatever the caller says.
    assert render_continue_prefix("The answer is", "391", True) == \
        ("391\n</think>\n\nThe answer is", False)
    # Closed thinking, no reply yet: only the caller can say so.
    assert render_continue_prefix("", "391", False) == ("391\n</think>\n\n", False)
    # No thinking: close the empty block the generation prompt opened, or the
    # model reads the reply as thinking (seen live: "yellow</think>Red, blue…").
    assert render_continue_prefix("Red, blue,", "", None) == ("\n</think>\n\nRed, blue,", False)
    # Thinking switched off: the prompt already closed the block; reply only.
    assert render_continue_prefix("Red, blue,", "plan", True, template_opens_think=False) == \
        ("Red, blue,", False)


def _route(deltas: List[str]) -> List[Any]:
    splitter = ThinkCloseSplitter()
    out: List[Any] = []
    for d in deltas:
        out.extend(splitter.feed(d))
    return out + splitter.flush()


def _joined(events: List[Any], kind: type) -> str:
    return "".join(e.delta for e in events if isinstance(e, kind))


def test_splitter_routes_thinking_until_close_tag():
    # The shape KoboldCPP streamed for an open-think continue on the live box.
    events = _route([" = 51, so 340 + 51 = 391.\n", "</think>", "\n\n", "391"])
    assert _joined(events, ReasoningEvent) == " = 51, so 340 + 51 = 391.\n"
    assert _joined(events, TokenEvent) == "391"


def test_splitter_holds_a_close_tag_split_across_deltas():
    events = _route(["thinking </th", "ink>\n\nThe answer"])
    assert _joined(events, ReasoningEvent) == "thinking "
    assert _joined(events, TokenEvent) == "The answer"
    # A partial that turns out not to be the tag is released as reasoning.
    assert _joined(_route(["a </t", "able>"]), ReasoningEvent) == "a </table>"
    assert _joined(_route(["ends on </thi"]), ReasoningEvent) == "ends on </thi"


# --- storage -------------------------------------------------------------------

@pytest.fixture
def mm():
    manager = MemoryManager(db_path=":memory:")
    manager.create_schema()
    yield manager
    manager.close()


def _log(mm, role, content, reasoning=None, tool_context=None):
    return mm.log_message(
        user_identifier="portal", persona_name="p", channel="web_ui",
        author_role=role, author_name="p", content=content, timestamp=_TS,
        tool_context=tool_context, reasoning_content=reasoning,
    )


def _row(mm, rid):
    return dict(mm._get_connection().execute(
        "SELECT content, reasoning_content FROM User_Interactions WHERE interaction_id = ?",
        (rid,)).fetchone())


def test_raw_edit_splits_the_turn_back_into_its_columns(mm):
    rid = _log(mm, "assistant", "", reasoning="looping looping")
    mm.update_assistant_raw_turn(rid, "<think>\nthe answer is 391\n</think>\n\nIt's 391.")
    assert _row(mm, rid) == {"content": "It's 391.", "reasoning_content": "the answer is 391"}
    # Dropping the block clears the reasoning; no PATCH can leave it stale.
    mm.update_assistant_raw_turn(rid, "Just the reply.")
    assert _row(mm, rid) == {"content": "Just the reply.", "reasoning_content": None}


def test_raw_edit_refuses_user_rows_and_tool_turns(mm):
    user = _log(mm, "user", "hi")
    tooled = _log(mm, "assistant", "done", reasoning="step", tool_context="[]x")
    with pytest.raises(ValueError):
        mm.update_assistant_raw_turn(user, "<think>x</think>y")
    with pytest.raises(ValueError):
        mm.update_assistant_raw_turn(tooled, "<think>x</think>y")


def test_latest_portal_assistant_is_the_trailing_row_only(mm):
    _log(mm, "user", "hi")
    rid = _log(mm, "assistant", "Red, blue,")
    row = mm.latest_portal_assistant("p", "portal", "web_ui")
    assert row is not None and row["interaction_id"] == rid
    _log(mm, "user", "and?")
    assert mm.latest_portal_assistant("p", "portal", "web_ui") is None


# --- kernel ------------------------------------------------------------------

def _scripted(system, deltas: List[Dict[str, Any]], seen: Dict[str, Any]):
    async def stream(_config, messages, _params, tools=None, **_kwargs):
        seen["messages"] = messages
        seen["tools"] = tools
        for d in deltas:
            yield d
        yield {"type": "done", "full_text": ""}
    system.text_engine.stream_messages = stream


def _continue_setup(chat_system_with_mocks, row):  # noqa: F811
    system, memory_mock, _, persona, _ = chat_system_with_mocks
    persona._model_name = "local"
    memory_mock.get_channel_history.return_value = [
        {"interaction_id": 1, "author_role": "user", "author_name": "u",
         "content": "Name three primary colors.", "timestamp": _TS},
        {"interaction_id": 2, "author_role": "assistant", "author_name": "test_persona",
         "content": row["content"], "reasoning_content": row["reasoning_content"],
         "timestamp": _TS},
    ]
    memory_mock.latest_portal_assistant.return_value = {
        "interaction_id": 2, "author_role": "assistant", "tool_context": None, **row,
    }
    memory_mock.update_interaction_content.return_value = True
    return system, memory_mock


@pytest.mark.asyncio
async def test_continue_resumes_the_reply_and_appends_in_place(chat_system_with_mocks):  # noqa: F811
    system, memory_mock = _continue_setup(
        chat_system_with_mocks, {"content": "Red, blue,", "reasoning_content": None})
    seen: Dict[str, Any] = {}
    _scripted(system, [{"type": "text_delta", "text": " yellow."}], seen)

    events = await _drain_events(system.stream_response(
        "test_persona", "portal", "web_ui", "", continue_turn=ContinueTurn()))

    # The wire array ends on the reply as a prefix — once, not duplicated —
    # with no new user turn and no tools.
    assert seen["messages"][-1] == {"role": "assistant", "content": "\n</think>\n\nRed, blue,"}
    assert seen["messages"][-2]["role"] == "user"
    assert not seen["tools"]
    # Same row, joined exactly (the leading space survives), nothing archived.
    memory_mock.update_interaction_content.assert_called_once()
    args, kwargs = memory_mock.update_interaction_content.call_args
    assert args[:2] == (2, "Red, blue, yellow.")
    memory_mock.handle_portal_retry.assert_not_called()
    memory_mock.log_message.assert_not_called()
    assert events[-1].assistant_id == 2


@pytest.mark.asyncio
async def test_continue_with_open_thinking_routes_text_back_to_reasoning(
        chat_system_with_mocks):  # noqa: F811
    system, memory_mock = _continue_setup(
        chat_system_with_mocks, {"content": "", "reasoning_content": "17*20 = 340, and 17*3"})
    seen: Dict[str, Any] = {}
    _scripted(system, [
        {"type": "text_delta", "text": " = 51, so 391.\n</th"},
        {"type": "text_delta", "text": "ink>\n\n391"},
    ], seen)

    events = await _drain_events(system.stream_response(
        "test_persona", "portal", "web_ui", "", continue_turn=ContinueTurn()))

    # A reasoning-only row is not in replayed history, so the prefix is added.
    assert seen["messages"][-1] == {"role": "assistant", "content": "17*20 = 340, and 17*3"}
    assert "".join(e.delta for e in events if isinstance(e, ReasoningEvent)) == " = 51, so 391.\n"
    assert "".join(e.delta for e in events if isinstance(e, TokenEvent)) == "391"
    _, kwargs = memory_mock.update_interaction_content.call_args
    assert memory_mock.update_interaction_content.call_args.args[1] == "391"
    assert kwargs["reasoning_content"] == "17*20 = 340, and 17*3 = 51, so 391."


@pytest.mark.asyncio
async def test_continue_closed_thinking_writes_the_reply(chat_system_with_mocks):  # noqa: F811
    """The thinking-loop rescue: thinking cut and closed in an edit, no reply yet."""
    system, memory_mock = _continue_setup(
        chat_system_with_mocks, {"content": "", "reasoning_content": "it's 391"})
    seen: Dict[str, Any] = {}
    _scripted(system, [{"type": "text_delta", "text": "391."}], seen)

    await _drain_events(system.stream_response(
        "test_persona", "portal", "web_ui", "",
        continue_turn=ContinueTurn(think_open=False)))

    assert seen["messages"][-1]["content"] == "it's 391\n</think>\n\n"
    args, kwargs = memory_mock.update_interaction_content.call_args
    assert args[1] == "391." and kwargs["reasoning_content"] == "it's 391"


@pytest.mark.asyncio
async def test_stopped_continue_keeps_the_row_queued(chat_system_with_mocks):  # noqa: F811
    """A stopped continue writes prefix + partial and, unlike a stopped retry,
    leaves the row's retain alone — it is still the reply the operator kept."""
    system, memory_mock = _continue_setup(
        chat_system_with_mocks, {"content": "Red, blue,", "reasoning_content": None})

    async def stream(*_a, **_k):
        yield {"type": "text_delta", "text": " yel"}
        raise asyncio.CancelledError
    system.text_engine.stream_messages = stream

    with pytest.raises(asyncio.CancelledError):
        await _drain_events(system.stream_response(
            "test_persona", "portal", "web_ui", "", continue_turn=ContinueTurn()))
    assert memory_mock.update_interaction_content.call_args.args[:2] == (2, "Red, blue, yel")
    memory_mock.discard_pending_retain.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("model, row, reason", [
    ("mock_model", {"content": "x", "reasoning_content": None}, "model local"),
    ("local", None, "Nothing to continue"),
    ("local", {"content": "x", "reasoning_content": None, "tool_context": "[]"}, "tool steps"),
])
async def test_continue_refusals_touch_nothing(chat_system_with_mocks, model, row, reason):  # noqa: F811
    system, memory_mock, _, persona, _ = chat_system_with_mocks
    persona._model_name = model
    memory_mock.latest_portal_assistant.return_value = row and {
        "interaction_id": 2, "author_role": "assistant", "tool_context": None, **row}

    events = await _drain_events(system.stream_response(
        "test_persona", "portal", "web_ui", "", continue_turn=ContinueTurn()))

    assert type(events[-1]).__name__ == "ErrorEvent" and reason in events[-1].message
    memory_mock.update_interaction_content.assert_not_called()
    memory_mock.log_message.assert_not_called()
