# tests/test_prompt_timestamps.py
"""DP-431: user turns carry their stored time in the prompt; assistant turns
and the system prompt carry none."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from config import global_config
from src.persona import Persona
from src.request_builder import RequestBuilder, RequestContext, _PROMPT_STAMP
from src.utils.timeutil import prompt_stamp

T1 = datetime(2026, 10, 14, 18, 5, tzinfo=timezone.utc)   # Wed 14:05 EDT
T2 = datetime(2026, 10, 15, 13, 30)                       # naive UTC, Thu 09:30 EDT


@pytest.fixture(autouse=True)
def eastern(monkeypatch):
    monkeypatch.setattr(global_config, "LOCAL_TZ", "America/New_York")


def _builder(rows=()):
    mm = MagicMock()
    mm.get_channel_history.return_value = [dict(r) for r in rows]
    tools = MagicMock()
    tools.get_tool_definitions.return_value = []
    return RequestBuilder(
        memory_manager=mm, memory_backend=MagicMock(),
        tool_manager_lookup=lambda: tools, persona_lookup=lambda _n: None,
    )


def _persona(inject=True):
    return Persona(persona_name="p", model_name="mock", prompt="sys",
                   inject_timestamp=inject, long_term_memory=False)


ROWS = [
    {"interaction_id": 1, "author_role": "user", "author_name": "Adam",
     "content": "hi", "timestamp": T1},
    {"interaction_id": 2, "author_role": "assistant", "author_name": "p",
     "content": "hello", "timestamp": T1},
    {"interaction_id": 3, "author_role": "assistant", "author_name": "other",
     "content": "me too", "timestamp": T2},
]


def test_user_turns_stamped_own_assistant_turns_not():
    out = _builder().format_raw_history_for_llm(ROWS, "channel", "p", None, stamp=True)
    assert out == [
        {"role": "user", "content": "[Wed 2026-10-14 14:05] hi"},
        {"role": "assistant", "content": "hello"},
        # Another persona's turn is presented as a user turn, so it is stamped.
        {"role": "user", "content": "[Thu 2026-10-15 09:30] other: me too"},
    ]


def test_group_chat_stamp_precedes_speaker():
    out = _builder().format_raw_history_for_llm(ROWS[:1], "channel", "p", "guild", stamp=True)
    assert out[0]["content"] == "[Wed 2026-10-14 14:05] Adam: hi"


def test_stamp_off_leaves_history_bare():
    out = _builder().format_raw_history_for_llm(ROWS, "channel", "p", None, stamp=False)
    assert [m["content"] for m in out] == ["hi", "hello", "other: me too"]


@pytest.mark.parametrize("ts", [None, "not a time"])
def test_unreadable_time_passes_through_unstamped(ts):
    rows = [{"author_role": "user", "author_name": "Adam", "content": "hi", "timestamp": ts}]
    out = _builder().format_raw_history_for_llm(rows, "channel", "p", None, stamp=True)
    assert out[0]["content"] == "hi"


async def test_new_turn_stamp_matches_its_future_history_stamp():
    """The cache invariant: the new turn's prompt text is exactly what the same
    row renders as once it is history, so the next request's prefix is
    unchanged."""
    persona = _persona()
    ctx = RequestContext(persona=persona, persona_name="p", user_identifier="u",
                         channel="c", message="what time is it", timestamp=T2)
    await _builder(ROWS[:2]).prepare_request(ctx)
    new_turn = ctx.conversation_history[-1]
    assert new_turn == {"role": "user", "content": "[Thu 2026-10-15 09:30] what time is it"}

    stored = {"author_role": "user", "author_name": "Adam",
              "content": "what time is it", "timestamp": T2}
    as_history = _builder().format_raw_history_for_llm([stored], "channel", "p", None, stamp=True)
    assert as_history[0] == new_turn


async def test_inject_off_sends_bare_turns():
    ctx = RequestContext(persona=_persona(inject=False), persona_name="p",
                         user_identifier="u", channel="c", message="yo", timestamp=T2)
    await _builder(ROWS[:2]).prepare_request(ctx)
    assert [m["content"] for m in ctx.conversation_history] == ["hi", "hello", "yo"]


def test_recall_query_strip_matches_the_formatter():
    """The retry recall query strips the stamp by regex; it must keep matching
    what `prompt_stamp` emits, or a date leaks into recall as a time filter."""
    assert _PROMPT_STAMP.sub("", f"{prompt_stamp(T1)} Adam: hi") == "Adam: hi"
