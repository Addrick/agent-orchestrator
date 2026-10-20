# tests/interfaces/test_portal_continue.py
"""DP-447 over the wire: continue, the raw-think PATCH, and the stale-view guard.

Real ChatSystem + in-memory DB behind the adapter, LLM stubbed at
`text_engine.stream_messages` — the same harness as the retry route tests.
"""
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from fastapi.testclient import TestClient

from tests.interfaces.test_kobold_engine_adapter import (  # noqa: F401
    _bypass_control_plane_auth, _fetch_portal_rows, _make_real_adapter,
)

_BASE = datetime(2026, 10, 20, 12, 0, 0)


@pytest.fixture(autouse=True)
def _kcpp_backend(monkeypatch):
    """No network: the live server is KoboldCPP (probe tested in test_continue_turn)."""
    monkeypatch.setattr("src.chat_system.supports_assistant_prefill",
                        AsyncMock(return_value=True))


def _seed(mm, reply="Red, blue,", reasoning=None, tool_context=None):
    user_id = mm.log_message(
        user_identifier="portal", persona_name="test_persona", channel="web_ui",
        author_role="user", author_name=None, content="Name three primary colors.",
        timestamp=_BASE,
    )
    assistant_id = mm.log_message(
        user_identifier="portal", persona_name="test_persona", channel="web_ui",
        author_role="assistant", author_name="test_persona", content=reply,
        timestamp=_BASE + timedelta(seconds=1), reply_to_id=user_id,
        reasoning_content=reasoning, tool_context=tool_context,
    )
    return user_id, assistant_id


def _body(**extra):
    return {"stream": True, "channel": "web_ui", "derpr_user_text": "", **extra}


def _archived(mm, rid):
    return mm._get_connection().execute(
        "SELECT COUNT(*) FROM Interaction_Edit_History WHERE interaction_id = ?", (rid,),
    ).fetchone()[0]


def test_continue_appends_to_the_trailing_reply_in_place():
    adapter, mm, _, _ = _make_real_adapter(deltas=(" yellow.",))
    _, assistant_id = _seed(mm)

    with TestClient(adapter.app) as client:
        r = client.post("/v1/chat/completions", json=_body(
            derpr_continue=True, derpr_expected_last_id=assistant_id))
    assert r.status_code == 200

    rows = _fetch_portal_rows(mm, "test_persona")
    assert len(rows) == 2  # no user row, no new assistant row
    assert rows[1]["interaction_id"] == assistant_id
    assert rows[1]["content"] == "Red, blue, yellow."
    assert _archived(mm, assistant_id) == 0  # not a version
    mm.close()


def test_stale_view_is_refused_before_anything_is_written():
    """Computer A's screen ends on the reply; computer B has since sent a turn."""
    adapter, mm, _, _ = _make_real_adapter(deltas=("never",))
    _, seen_by_a = _seed(mm)
    newer = mm.log_message(
        user_identifier="portal", persona_name="test_persona", channel="web_ui",
        author_role="user", author_name=None, content="sent from computer B",
        timestamp=_BASE + timedelta(seconds=5),
    )

    with TestClient(adapter.app) as client:
        for extra in ({"derpr_user_text": "from A"}, {"derpr_retry": True},
                      {"derpr_continue": True}):
            r = client.post("/v1/chat/completions", json=_body(
                derpr_expected_last_id=seen_by_a, **extra))
            assert r.status_code == 409
            assert r.json()["error"]["code"] == "stale_view"
            assert r.json()["derpr_last_id"] == newer

    assert len(_fetch_portal_rows(mm, "test_persona")) == 3
    assert _archived(mm, seen_by_a) == 0
    mm.close()


def test_current_view_and_empty_conversation_pass_the_guard():
    adapter, mm, _, _ = _make_real_adapter(deltas=("hi there",))
    with TestClient(adapter.app) as client:
        # An empty conversation: the screen ends on nothing.
        r = client.post("/v1/chat/completions", json=_body(
            derpr_user_text="hello", derpr_expected_last_id=None))
        assert r.status_code == 200
        last = _fetch_portal_rows(mm, "test_persona")[-1]["interaction_id"]
        r = client.post("/v1/chat/completions", json=_body(
            derpr_user_text="again", derpr_expected_last_id=last))
        assert r.status_code == 200
    assert len(_fetch_portal_rows(mm, "test_persona")) == 4
    mm.close()


def test_patch_raw_think_rewrites_reasoning_and_reply():
    adapter, mm, _, _ = _make_real_adapter()
    _, assistant_id = _seed(mm, reply="", reasoning="loop loop loop")
    _, tooled = _seed(mm, reply="done", reasoning="step", tool_context='[{"x": 1}]')

    with TestClient(adapter.app) as client:
        r = client.patch(f"/api/v1/interaction/{assistant_id}", json={
            "content": "<think>\nit's 391\n</think>\n\nIt's 391.", "raw_think": True})
        assert r.status_code == 200
        refused = client.patch(f"/api/v1/interaction/{tooled}", json={
            "content": "<think>x</think>y", "raw_think": True})
        assert refused.status_code == 400

    row = mm._get_connection().execute(
        "SELECT content, reasoning_content FROM User_Interactions WHERE interaction_id = ?",
        (assistant_id,)).fetchone()
    assert (row["content"], row["reasoning_content"]) == ("It's 391.", "it's 391")
    mm.close()
