# tests/test_dispatch_events.py
"""Unit tests for the DP-227 common event schema + Claude stream adapter."""

import json

from src.self_edit.events import (
    DONE,
    ERROR,
    PROGRESS,
    QUESTION,
    STARTED,
    WAKE_TYPES,
    AgentEvent,
    ClaudeStreamAdapter,
    get_adapter,
)


def _line(obj) -> str:
    return json.dumps(obj)


def test_agent_event_jsonl_roundtrip():
    ev = AgentEvent("a1", 3, QUESTION, {"text": "which?", "session_id": "s"})
    back = AgentEvent.from_jsonl(ev.to_jsonl())
    assert back.agent_id == "a1"
    assert back.seq == 3
    assert back.type == QUESTION
    assert back.payload["text"] == "which?"
    assert back.is_wake is True


def test_wake_and_terminal_classification():
    assert WAKE_TYPES == {QUESTION, DONE, ERROR}
    assert AgentEvent("a", 0, PROGRESS).is_wake is False
    assert AgentEvent("a", 0, STARTED).is_terminal is False
    assert AgentEvent("a", 0, DONE).is_terminal is True
    assert AgentEvent("a", 0, ERROR).is_terminal is True


def test_adapter_init_captures_session_and_emits_started():
    a = ClaudeStreamAdapter("a1")
    evs = a.parse_line(_line({"type": "system", "subtype": "init", "session_id": "sess-123"}))
    assert len(evs) == 1
    assert evs[0].type == STARTED
    assert a.session_id == "sess-123"
    assert evs[0].payload["session_id"] == "sess-123"


def test_adapter_assistant_text_is_progress():
    a = ClaudeStreamAdapter("a1")
    evs = a.parse_line(_line({
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": "looking at auth.py"}]},
    }))
    assert len(evs) == 1
    assert evs[0].type == PROGRESS
    assert "auth.py" in evs[0].payload["text"]


def test_adapter_question_sentinel():
    a = ClaudeStreamAdapter("a1")
    a.parse_line(_line({"type": "system", "subtype": "init", "session_id": "s9"}))
    evs = a.parse_line(_line({
        "type": "result", "subtype": "success", "is_error": False,
        "result": "FIXR_QUESTION: should I bump the timeout or the retry count?",
        "session_id": "s9",
    }))
    assert len(evs) == 1
    assert evs[0].type == QUESTION
    assert evs[0].payload["session_id"] == "s9"
    assert "timeout" in evs[0].payload["text"]


def test_adapter_done_sentinel_extracts_pr_url():
    a = ClaudeStreamAdapter("a1")
    evs = a.parse_line(_line({
        "type": "result", "subtype": "success", "is_error": False,
        "result": "FIXR_DONE: https://github.com/Addrick/agent-orchestrator/pull/42 fixed it",
    }))
    assert evs[0].type == DONE
    assert evs[0].payload["pr_url"] == "https://github.com/Addrick/agent-orchestrator/pull/42"


def test_adapter_error_from_sentinel_and_from_is_error():
    a = ClaudeStreamAdapter("a1")
    e1 = a.parse_line(_line({"type": "result", "subtype": "success",
                             "result": "FIXR_ERROR: cannot reproduce"}))
    assert e1[0].type == ERROR
    assert "cannot reproduce" in e1[0].payload["text"]

    b = ClaudeStreamAdapter("a2")
    e2 = b.parse_line(_line({"type": "result", "subtype": "error_max_turns", "is_error": True}))
    assert e2[0].type == ERROR
    assert e2[0].payload["detail"] == "error_max_turns"


def test_adapter_clean_finish_without_sentinel_is_done():
    a = ClaudeStreamAdapter("a1")
    evs = a.parse_line(_line({"type": "result", "subtype": "success",
                              "is_error": False, "result": "all done"}))
    assert evs[0].type == DONE
    assert evs[0].payload["pr_url"] is None


def test_adapter_ignores_garbage_and_unknown():
    a = ClaudeStreamAdapter("a1")
    assert a.parse_line("not json at all") == []
    assert a.parse_line("") == []
    assert a.parse_line(_line({"type": "user", "message": {}})) == []
    assert a.parse_line(_line(["a", "list"])) == []


def test_seq_increments_across_events():
    a = ClaudeStreamAdapter("a1")
    e0 = a.parse_line(_line({"type": "system", "subtype": "init"}))
    e1 = a.parse_line(_line({"type": "assistant", "message": {"content": "hi"}}))
    assert e0[0].seq == 0
    assert e1[0].seq == 1


def test_get_adapter_defaults_to_claude():
    assert isinstance(get_adapter("claude", "x"), ClaudeStreamAdapter)
    assert isinstance(get_adapter("unknown-platform", "x"), ClaudeStreamAdapter)
