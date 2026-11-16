# tests/test_retry_row_guard.py
"""DP-523: a retry replaces the row on screen, or is refused.

A retry archives and overwrites (or answers) its *channel's* latest row, but a
merged-memory persona's screen ends on the newest row of *any* channel. Viewed
from another channel, the regenerated reply used to stream and then land on
that channel's old reply, off screen. The portal now names the row it means
(`derpr_retry_id`), and `handle_portal_retry` refuses a mismatch inside the
same lock as the archive, before anything is generated or written.
"""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.memory.errors import RetryTargetMismatch
from src.memory.memory_manager import MemoryManager
from src.persona import MemoryMode
from src.request_builder import PORTAL_MERGED_MODES
from tests.interfaces.test_kobold_engine_adapter import (  # noqa: F401
    _bypass_control_plane_auth, _make_real_adapter,
)

_TS = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
_PERSONA = "test_persona"


def _log(mm, channel, role, minutes, content=None, tool_context=None):
    return mm.log_message(
        user_identifier="portal", persona_name=_PERSONA, channel=channel,
        author_role=role, author_name=_PERSONA,
        content=f"{channel} {role}" if content is None else content,
        timestamp=_TS + timedelta(minutes=minutes), tool_context=tool_context,
    )


def _content(mm, rid):
    return mm._get_connection().execute(
        "SELECT content FROM User_Interactions WHERE interaction_id = ?", (rid,),
    ).fetchone()[0]


def _archived(mm, rid):
    return mm._get_connection().execute(
        "SELECT COUNT(*) FROM Interaction_Edit_History WHERE interaction_id = ?", (rid,),
    ).fetchone()[0]


def _two_channels(mm):
    """The shape behind the bug: the merged view ends on web_ui's reply, but
    web_ui_old's own latest row is its older reply."""
    _log(mm, "web_ui_old", "user", 0)
    old_reply = _log(mm, "web_ui_old", "assistant", 1)
    _log(mm, "web_ui", "user", 2)
    new_reply = _log(mm, "web_ui", "assistant", 3)
    return old_reply, new_reply


# --- storage -------------------------------------------------------------------

@pytest.fixture
def mm():
    manager = MemoryManager(db_path=":memory:")
    manager.create_schema()
    yield manager
    manager.close()


def test_mismatch_raises_before_anything_is_archived(mm):
    old_reply, new_reply = _two_channels(mm)
    with pytest.raises(RetryTargetMismatch):
        mm.handle_portal_retry(_PERSONA, "portal", "web_ui_old", expected_row_id=new_reply)
    assert _archived(mm, old_reply) == 0


def test_matching_target_archives_as_before(mm):
    _, new_reply = _two_channels(mm)
    assert mm.handle_portal_retry(
        _PERSONA, "portal", "web_ui", expected_row_id=new_reply) == new_reply
    assert _archived(mm, new_reply) == 1


# --- over the wire -------------------------------------------------------------

def _retry(client, channel, retry_id, stream=True):
    return client.post("/v1/chat/completions", json={
        "stream": stream, "channel": channel, "derpr_user_text": " ",
        "derpr_retry": True, "derpr_retry_id": retry_id,
    })


@pytest.mark.parametrize("stream", [True, False])
def test_global_retry_from_another_channel_is_refused(stream):
    adapter, mm, persona, _ = _make_real_adapter(persona_name=_PERSONA, deltas=("never",))
    persona.set_memory_mode("GLOBAL")
    old_reply, new_reply = _two_channels(mm)

    with TestClient(adapter.app) as client:
        r = _retry(client, "web_ui_old", new_reply, stream=stream)

    assert "on screen" in r.text
    assert _content(mm, old_reply) == "web_ui_old assistant"  # not overwritten off screen
    assert _content(mm, new_reply) == "web_ui assistant"
    assert _archived(mm, old_reply) == 0
    mm.close()


def test_retry_of_the_row_on_screen_regenerates_it():
    adapter, mm, persona, _ = _make_real_adapter(persona_name=_PERSONA, deltas=("again",))
    persona.set_memory_mode("GLOBAL")
    _, new_reply = _two_channels(mm)

    with TestClient(adapter.app) as client:
        r = _retry(client, "web_ui", new_reply)

    assert r.status_code == 200
    assert _content(mm, new_reply) == "again"
    assert _archived(mm, new_reply) == 1
    mm.close()


def test_retry_after_a_tool_cards_only_reply_answers_the_user_turn():
    """DP-444 keeps a stopped turn's tool cards on screen with no text. The
    portal names the user turn before it (the last row with text), which is
    also the engine's lookup, so the retry adds a fresh reply — it must not
    be refused for ending on a row the lookup skips."""
    adapter, mm, _, _ = _make_real_adapter(persona_name=_PERSONA, deltas=("fresh",))
    user = _log(mm, "web_ui", "user", 0)
    tools_only = _log(mm, "web_ui", "assistant", 1, content="",
                      tool_context='[{"role": "assistant", "content": ""}]')

    with TestClient(adapter.app) as client:
        r = _retry(client, "web_ui", user)

    assert r.status_code == 200 and "on screen" not in r.text
    newest = mm._get_connection().execute(
        "SELECT interaction_id, content FROM User_Interactions"
        " ORDER BY interaction_id DESC LIMIT 1").fetchone()
    assert newest["interaction_id"] > tools_only and newest["content"] == "fresh"
    mm.close()


def test_retry_id_is_ignored_on_a_plain_send():
    """derpr_retry_id only means something with derpr_retry; a send carrying
    a stale one must not be refused."""
    adapter, mm, _, _ = _make_real_adapter(persona_name=_PERSONA, deltas=("ok",))
    _two_channels(mm)

    with TestClient(adapter.app) as client:
        r = client.post("/v1/chat/completions", json={
            "stream": True, "channel": "web_ui", "derpr_user_text": "hi",
            "derpr_retry_id": 1,
        })

    assert r.status_code == 200 and "on screen" not in r.text
    mm.close()


# --- the merged-mode list ------------------------------------------------------

@pytest.mark.parametrize("mode", list(MemoryMode))
def test_portal_merged_modes_match_the_history_dispatch(mode):
    """`channels_merge` (what the portal's rail obeys) must say exactly
    whether the engine's view of one portal channel includes another's rows."""
    adapter, mm, persona, chat_system = _make_real_adapter(persona_name=_PERSONA)
    persona.set_memory_mode(mode.name)
    _two_channels(mm)

    history, _ = chat_system.get_view_history(_PERSONA, "portal", "web_ui", None)
    merged = any(r.get("channel") == "web_ui_old"
                 or r.get("content", "").startswith("web_ui_old") for r in history)

    assert merged == (mode in PORTAL_MERGED_MODES)
    with TestClient(adapter.app) as client:
        body = client.get(f"/api/v1/persona/{_PERSONA}").json()
    assert body["channels_merge"] == (mode in PORTAL_MERGED_MODES)
    mm.close()
