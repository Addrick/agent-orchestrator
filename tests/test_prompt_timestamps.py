# tests/test_prompt_timestamps.py
"""DP-431: user turns carry their stored time in the prompt; assistant turns
and the system prompt carry none."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

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


# --- LTM block placement (DP-431) ---

BLOCK = "<memory>\nfact\n</memory>"


def test_block_rides_on_newest_user_turn():
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "[Thu 2026-10-15 09:30] new"},
    ]
    RequestBuilder._attach_memory_block(history, BLOCK)
    assert history[0]["content"] == "old"
    assert history[-1]["content"] == f"{BLOCK}\n\n[Thu 2026-10-15 09:30] new"
    # Roles still alternate: no extra message.
    assert [m["role"] for m in history] == ["user", "assistant", "user"]


def test_block_on_multimodal_turn_is_a_leading_text_part():
    parts = [{"type": "image_url", "image_url": {"url": "x"}}]
    history = [{"role": "user", "content": list(parts)}]
    RequestBuilder._attach_memory_block(history, BLOCK)
    assert history[0]["content"] == [{"type": "text", "text": BLOCK}] + parts


def test_block_without_a_user_turn_is_dropped():
    history = [{"role": "assistant", "content": "a"}]
    RequestBuilder._attach_memory_block(history, BLOCK)
    assert history == [{"role": "assistant", "content": "a"}]


async def test_prepare_request_attaches_block_to_new_turn_not_head():
    builder = _builder(ROWS[:2])
    builder.retrieve_memory_block = AsyncMock(return_value=(BLOCK, False))
    ctx = RequestContext(persona=_persona(), persona_name="p", user_identifier="u",
                         channel="c", message="q", timestamp=T2)
    await builder.prepare_request(ctx)
    assert [m["content"] for m in ctx.conversation_history] == [
        "[Wed 2026-10-14 14:05] hi",
        "hello",
        f"{BLOCK}\n\n[Thu 2026-10-15 09:30] q",
    ]
    sources = RequestBuilder.derive_message_sources(
        [{"role": "system", "content": "sys"}] + ctx.conversation_history, is_retry=False)
    assert sources == ["persona.prompt", "history", "history", "composer+ltm"]


async def test_retry_attaches_block_to_the_replayed_user_turn():
    rows = ROWS[:2] + [{"interaction_id": 3, "author_role": "user", "author_name": "Adam",
                        "content": "again", "timestamp": T2},
                       {"interaction_id": 4, "author_role": "assistant", "author_name": "p",
                        "content": "discarded", "timestamp": T2}]
    builder = _builder(rows)
    builder.retrieve_memory_block = AsyncMock(return_value=(BLOCK, False))
    ctx = RequestContext(persona=_persona(), persona_name="p", user_identifier="u",
                         channel="c", message="again", timestamp=T2)
    await builder.prepare_request(ctx, is_retry=True)
    assert ctx.conversation_history[-1]["content"] == f"{BLOCK}\n\n[Thu 2026-10-15 09:30] again"
    # The recall query was taken before the block was attached.
    assert ctx.conversation_history[0]["content"] == "[Wed 2026-10-14 14:05] hi"


def test_block_labels_are_absolute_and_preamble_disowns_user_authorship():
    from src.memory.backend.base import MemoryHit
    block = RequestBuilder.format_memory_block([
        MemoryHit(id="7", content="likes tea", score=1.0,
                  tags=["channel:web_ui"], timestamp=T1),
    ])
    assert "[#web_ui, ID:7, Wed 2026-10-14 14:05]" in block
    assert "not part of what the user wrote" in block
    assert " ago" not in block
