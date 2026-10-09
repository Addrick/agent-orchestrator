# tests/integration/test_deferred_retain.py
#
# DP-423: a turn reaches the semantic backend once newer turns push it out
# of the sliding history window — not while the model still reads it
# verbatim, or recall hands the same conversation back twice — or once its
# conversation has sat idle for SESSION_GAP_SECONDS, so a short chat that
# simply stops is still remembered. The window keeps priority: it is never
# trimmed to make room for memory.
# Driven through the real kernel and a real MemoryManager; only the LLM and
# the backend's `retain_turn` are mocked.

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from src.memory.backend.base import SESSION_GAP_SECONDS
from src.persona import MemoryMode

pytestmark = pytest.mark.integration


async def _turn(chat_system, message, *, user="u1", channel="chan"):
    chat_system.text_engine.generate_response.return_value = (
        {"type": "text", "content": f"re: {message}"}, {},
    )
    async for _ in chat_system.stream_response("test_persona", user, channel, message):
        pass
    await chat_system.turn_persistence.drain_flushes()


def _retained(spy):
    return [c.kwargs["content"].split(": ", 1)[-1] for c in spy.await_args_list]


@pytest.fixture
def setup(mocked_chat_system):
    chat_system, memory_manager = mocked_chat_system
    persona = chat_system.personas["test_persona"]
    persona.set_history_messages(2)
    persona.set_inject_timestamp(False)
    persona.set_enabled_tools([])
    spy = AsyncMock()
    chat_system.memory_backend.retain_turn = spy  # type: ignore[assignment]
    return chat_system, memory_manager, persona, spy


@pytest.mark.asyncio
async def test_turns_are_retained_only_after_eviction(setup):
    chat_system, _, _, spy = setup

    await _turn(chat_system, "one")
    await _turn(chat_system, "two")
    # Window of 2: turn two was built from [one, re: one] — nothing evicted.
    assert spy.await_count == 0

    await _turn(chat_system, "three")
    # Turn three's window is [two, re: two]; turn one's pair has left it.
    assert _retained(spy) == ["one", "re: one"]
    roles = [c.kwargs["role"] for c in spy.await_args_list]
    assert roles == ["user", "assistant"]

    await _turn(chat_system, "four")
    assert _retained(spy) == ["one", "re: one", "two", "re: two"]


@pytest.mark.asyncio
async def test_edit_and_delete_inside_window_are_honoured(setup):
    chat_system, memory_manager, _, spy = setup

    await _turn(chat_system, "draft")
    rows = memory_manager.get_channel_history("chan", "test_persona", None, 10)
    user_id, assistant_id = rows[0]["interaction_id"], rows[1]["interaction_id"]
    memory_manager.update_interaction_content(user_id, "final")
    memory_manager.suppress_interaction(assistant_id)

    await _turn(chat_system, "two")
    await _turn(chat_system, "three")
    assert "draft" not in _retained(spy)
    assert "re: draft" not in _retained(spy)
    assert _retained(spy)[0] == "final"


@pytest.mark.asyncio
async def test_idle_conversation_is_flushed_by_the_sweep(setup):
    chat_system, _, _, spy = setup

    await _turn(chat_system, "abandoned", channel="quiet")
    assert spy.await_count == 0

    # Nothing will ever push these out of a window of 2; after a day idle
    # the sweep (run on any turn, anywhere) retains them anyway.
    later = datetime.now(timezone.utc) + timedelta(seconds=SESSION_GAP_SECONDS + 60)
    sent = await chat_system.turn_persistence.flush_idle_sessions(now=later)
    assert sent == 2
    assert _retained(spy) == ["abandoned", "re: abandoned"]
    assert {c.kwargs["scope_tags"][0] for c in spy.await_args_list} == {"channel:quiet"}
    assert await chat_system.turn_persistence.flush_idle_sessions(now=later) == 0


@pytest.mark.asyncio
async def test_window_keeps_priority_after_a_break(setup):
    """Coming back a day later is not a blank chat: the window still holds
    yesterday, and since the conversation resumed before any sweep cut it,
    those turns are not retained while they are still in the window."""
    chat_system, memory_manager, _, spy = setup

    await _turn(chat_system, "yesterday")
    old = datetime.now(timezone.utc) - timedelta(seconds=SESSION_GAP_SECONDS + 60)
    with memory_manager.transaction() as conn:
        conn.execute("UPDATE User_Interactions SET timestamp = ?", (old,))

    await _turn(chat_system, "today")
    history = chat_system.text_engine.generate_response.call_args.args[1]["message_history"]
    assert any("yesterday" in str(m.get("content")) for m in history)
    assert spy.await_count == 0

    await _turn(chat_system, "later today")
    # Now pushed out by newer turns: retained, with their original stamps.
    assert _retained(spy) == ["yesterday", "re: yesterday"]
    assert all(c.kwargs["timestamp"] < old + timedelta(seconds=1)
               for c in spy.await_args_list)


@pytest.mark.asyncio
async def test_refused_retain_stays_queued_and_is_retried(setup):
    """A turn leaves the queue only once the backend accepted it — a down
    Hindsight must not cost the memory of a turn already out of the window."""
    chat_system, memory_manager, _, spy = setup
    backend = chat_system.memory_backend
    accept = {"ok": False}

    async def confirmed(**kwargs):
        if accept["ok"]:
            await spy(**kwargs)
        return accept["ok"]

    backend.retain_turn_confirmed = confirmed  # type: ignore[assignment]

    await _turn(chat_system, "one")
    await _turn(chat_system, "two")
    await _turn(chat_system, "three")  # evicts one's pair — refused
    rows = memory_manager.get_channel_history("chan", "test_persona", None, 10)
    assert memory_manager.is_retain_pending(rows[0]["interaction_id"])
    assert spy.await_count == 0

    accept["ok"] = True
    await _turn(chat_system, "four")  # retries one's pair, evicts two's
    assert _retained(spy) == ["one", "re: one", "two", "re: two"]
    assert not memory_manager.is_retain_pending(rows[0]["interaction_id"])


@pytest.mark.asyncio
async def test_global_mode_flushes_turns_another_channel_evicted(setup):
    """A global window spans channels, so a turn in one channel pushes out
    another channel's turns — and must retain them, or they'd be stranded."""
    chat_system, _, persona, spy = setup
    persona.set_memory_mode(MemoryMode.GLOBAL)

    await _turn(chat_system, "over here", channel="a")
    await _turn(chat_system, "b1", channel="b")
    await _turn(chat_system, "b2", channel="b")
    assert _retained(spy) == ["over here", "re: over here"]
    assert {c.kwargs["scope_tags"][0] for c in spy.await_args_list} == {"channel:a"}


@pytest.mark.asyncio
async def test_personal_mode_flushes_only_the_speakers_turns(setup):
    chat_system, _, persona, spy = setup
    persona.set_memory_mode(MemoryMode.PERSONAL)

    await _turn(chat_system, "a1", user="a")
    await _turn(chat_system, "b1", user="b")
    await _turn(chat_system, "a2", user="a")
    await _turn(chat_system, "a3", user="a")
    # a3's window is [a2, re: a2]: a1's pair is out of a's window. b1 is in a
    # channel-wide id range below the cutoff but is still in b's own window.
    assert _retained(spy) == ["a1", "re: a1"]


@pytest.mark.asyncio
async def test_ticket_mode_empty_window_flushes_on_next_turn(setup):
    chat_system, _, persona, spy = setup
    persona.set_memory_mode(MemoryMode.TICKET_ISOLATED)

    await _turn(chat_system, "first")
    assert spy.await_count == 0
    await _turn(chat_system, "second")
    assert _retained(spy) == ["first", "re: first"]
