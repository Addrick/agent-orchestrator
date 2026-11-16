# tests/test_retry_row_guard.py
"""DP-523: a retry replaces the row on screen, or is refused.

A retry archives and overwrites (or answers) its *channel's* latest row, but a
merged-memory persona's screen ends on the newest row of *any* channel. Viewed
from another channel, the regenerated reply used to stream and then land on
that channel's old reply, off screen. The portal now names the row it means,
and the kernel refuses a mismatch before anything is generated or written.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from src.memory.memory_manager import MemoryManager
from tests.test_chat_system import _drain_events, chat_system_with_mocks  # noqa: F401

_TS = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def mm():
    manager = MemoryManager(db_path=":memory:")
    manager.create_schema()
    yield manager
    manager.close()


def _log(mm, channel, role, minutes):
    return mm.log_message(
        user_identifier="portal", persona_name="p", channel=channel,
        author_role=role, author_name="p", content=f"{channel} {role}",
        timestamp=_TS + timedelta(minutes=minutes),
    )


def test_latest_row_is_per_channel_not_the_merged_view(mm):
    """The GLOBAL two-channel shape behind the bug: the merged view ends on
    web_ui's reply, but the other channel's own latest is its older reply."""
    _log(mm, "web_ui_old", "user", 0)
    old_reply = _log(mm, "web_ui_old", "assistant", 1)
    _log(mm, "web_ui", "user", 2)
    new_reply = _log(mm, "web_ui", "assistant", 3)

    assert mm.latest_portal_row_id("p", "portal", "web_ui") == new_reply
    assert mm.latest_portal_row_id("p", "portal", "web_ui_old") == old_reply
    assert mm.latest_portal_row_id("p", "portal", "never_used") is None


def test_latest_row_counts_a_trailing_user_turn(mm):
    """A retry on a reply-less user turn answers that turn, so the id it
    checks against is the user row's."""
    _log(mm, "web_ui", "assistant", 0)
    user = _log(mm, "web_ui", "user", 1)
    assert mm.latest_portal_row_id("p", "portal", "web_ui") == user


def _scripted(system, seen: Dict[str, Any]):
    async def stream(_config, messages, _params, tools=None, **_kwargs):
        seen["called"] = True
        yield {"type": "token", "delta": "again"}
        yield {"type": "done", "full_text": "again"}
    system.text_engine.stream_messages = stream


def _history() -> List[Dict[str, Any]]:
    return [
        {"interaction_id": 3, "author_role": "user", "author_name": "u",
         "content": "hi", "timestamp": _TS},
        {"interaction_id": 4, "author_role": "assistant", "author_name": "test_persona",
         "content": "hello", "timestamp": _TS},
    ]


@pytest.mark.asyncio
async def test_retry_of_a_row_this_channel_does_not_end_on_is_refused(
        chat_system_with_mocks):  # noqa: F811
    system, memory_mock, _, _, _ = chat_system_with_mocks
    memory_mock.get_channel_history.return_value = _history()
    memory_mock.latest_portal_row_id.return_value = 2  # the channel's own old reply
    seen: Dict[str, Any] = {}
    _scripted(system, seen)

    events = await _drain_events(system.stream_response(
        "test_persona", "portal", "web_ui_old", " ", is_retry=True, retry_row_id=4))

    assert type(events[-1]).__name__ == "ErrorEvent" and "on screen" in events[-1].message
    assert "called" not in seen
    memory_mock.handle_portal_retry.assert_not_called()
    memory_mock.update_interaction_content.assert_not_called()
    memory_mock.log_message.assert_not_called()


@pytest.mark.asyncio
async def test_retry_of_the_row_on_screen_regenerates_it(chat_system_with_mocks):  # noqa: F811
    system, memory_mock, _, _, _ = chat_system_with_mocks
    memory_mock.get_channel_history.return_value = _history()
    memory_mock.latest_portal_row_id.return_value = 4
    memory_mock.handle_portal_retry.return_value = 4
    memory_mock.update_interaction_content.return_value = True
    seen: Dict[str, Any] = {}
    _scripted(system, seen)

    events = await _drain_events(system.stream_response(
        "test_persona", "portal", "web_ui", " ", is_retry=True, retry_row_id=4))

    assert type(events[-1]).__name__ == "DoneEvent"
    assert seen.get("called")
    memory_mock.handle_portal_retry.assert_called_once()
    assert memory_mock.update_interaction_content.call_args.args[:2] == (4, "again")
