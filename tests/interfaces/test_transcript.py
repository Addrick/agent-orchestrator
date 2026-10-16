# tests/interfaces/test_transcript.py

"""DP-130 transcript projection (`build_transcript`) and the tool-context
parser it shares with the parked-confirmation path."""

import json

from src.interfaces.transcript import (
    build_transcript,
    _parse_tool_context,
)


def _rows(n):
    """n alternating user/assistant rows with sequential interaction_ids."""
    out = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        out.append({
            "author_role": role,
            "content": f"msg {i}",
            "interaction_id": 100 + i,
        })
    return out


# -------- DP-130 transcript projection (build_transcript) --------

def test_transcript_c1_id_xor_ephemeral():
    rows = _rows(4)
    transcript = build_transcript(rows)
    chunks = transcript["chunks"]
    assert len(chunks) == 4
    for c in chunks:
        assert (c["interaction_id"] is not None) != (c["ephemeral"] is True)
        assert c["ephemeral"] is False

def test_transcript_includes_role_content_and_versions_flag():
    rows = _rows(2)
    transcript = build_transcript(rows, ids_with_versions={101})
    chunks = transcript["chunks"]
    assert chunks[0]["role"] == "user"
    assert chunks[0]["has_versions"] is False
    assert chunks[1]["role"] == "assistant"
    assert chunks[1]["has_versions"] is True  # id 101 has versions

def test_transcript_keeps_reasoning_apart_from_content():
    rows = [
        {"author_role": "assistant", "content": "answer",
         "reasoning_content": "thinking", "interaction_id": 5},
    ]
    chunks = build_transcript(rows)["chunks"]
    assert chunks[0]["content"] == "answer"
    assert chunks[0]["reasoning"] == "thinking"

def test_transcript_parses_tool_context_json():
    rows = [
        {"author_role": "assistant", "content": "done", "interaction_id": 5,
         "tool_context": json.dumps([
             {"role": "assistant", "tool_calls": [
                 {"id": "c1", "name": "lookup", "arguments": {"q": "x"}}]},
             {"role": "tool", "tool_call_id": "c1", "content": "x"},
         ])},
    ]
    chunks = build_transcript(rows)["chunks"]
    assert chunks[0]["tool_context"] == [{
        "call_id": "c1", "group_id": None, "tool_name": "lookup",
        "arguments": {"q": "x"}, "result": "x", "error": None,
    }]

def test_transcript_skips_non_renderable_rows():
    rows = [
        {"author_role": "system", "content": "boot"},
        {"author_role": "user", "content": "hi", "interaction_id": 1},
        {"author_role": "assistant", "content": "", "interaction_id": 2},  # tool-only
    ]
    chunks = build_transcript(rows)["chunks"]
    assert len(chunks) == 1
    assert chunks[0]["interaction_id"] == 1

def test_transcript_appends_pending_ephemeral_chunk():
    rows = _rows(2)
    pending = [{"ephemeral_chunk_id": "tok123", "content": "awaiting approval",
                "tool_context": None}]
    chunks = build_transcript(rows, pending=pending)["chunks"]
    assert len(chunks) == 3
    last = chunks[-1]
    assert last["ephemeral"] is True
    assert last["interaction_id"] is None
    assert last["ephemeral_chunk_id"] == "tok123"
    assert last["content"] == "awaiting approval"


def test_transcript_appends_one_chunk_per_pending_action():
    """DP-297 made `pending` a list — every live proposal gets its own chunk.

    Before, the projection took a single dict, so a reload showed one
    proposal and silently dropped the rest.
    """
    rows = _rows(2)
    pending = [
        {"ephemeral_chunk_id": "tok1", "content": "one", "tool_context": None},
        {"ephemeral_chunk_id": "tok2", "content": "two", "tool_context": None},
    ]
    chunks = build_transcript(rows, pending=pending)["chunks"]
    assert len(chunks) == 4
    assert [c["ephemeral_chunk_id"] for c in chunks[-2:]] == ["tok1", "tok2"]
    assert all(c["ephemeral"] is True for c in chunks[-2:])


def test_transcript_with_no_pending_actions():
    """An empty list and None both mean 'nothing awaiting approval'."""
    rows = _rows(2)
    assert len(build_transcript(rows, pending=[])["chunks"]) == 2
    assert len(build_transcript(rows, pending=None)["chunks"]) == 2


# -------- _parse_tool_context: raw-OpenAI-message -> frontend ToolContext --------
# The parked-CONFIRM path slices the still-in-flight conversation_history (raw
# OpenAI messages) into _parse_tool_context so the pending chunk can render the
# tool call that is awaiting approval. Persisted rows already store the structured
# shape and must pass through untouched.

def test_parse_tool_context_transforms_openai_tool_call_and_result():
    raw = [
        {"role": "assistant", "tool_calls": [
            {"id": "call_1", "name": "search_tickets", "group_id": "g1",
             "arguments": {"query": "vpn"}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"result": []}'},
    ]
    out = _parse_tool_context(raw)
    assert out == [{
        "call_id": "call_1",
        "group_id": "g1",
        "tool_name": "search_tickets",
        "arguments": {"query": "vpn"},
        "result": '{"result": []}',
        "error": None,
    }]


def test_parse_tool_context_function_shape_and_stringified_args():
    # OpenAI's nested `function` envelope with arguments as a JSON string.
    raw = [
        {"role": "assistant", "tool_calls": [
            {"id": "call_9", "function": {"name": "create_ticket"},
             "arguments": '{"title": "x"}'},
        ]},
    ]
    out = _parse_tool_context(raw)
    assert len(out) == 1
    assert out[0]["tool_name"] == "create_ticket"
    assert out[0]["arguments"] == {"title": "x"}
    assert out[0]["result"] is None


def test_parse_tool_context_surfaces_tool_error():
    raw = [
        {"role": "assistant", "tool_calls": [{"id": "c", "name": "t"}]},
        {"role": "tool", "tool_call_id": "c",
         "content": '{"error": "boom"}'},
    ]
    out = _parse_tool_context(raw)
    assert out[0]["result"] == '{"error": "boom"}'
    assert out[0]["error"] == "boom"


def test_parse_tool_context_accepts_json_string():
    raw = json.dumps([
        {"role": "assistant", "tool_calls": [{"id": "c", "name": "t"}]},
    ])
    out = _parse_tool_context(raw)
    assert out == [{
        "call_id": "c", "group_id": None, "tool_name": "t",
        "arguments": {}, "result": None, "error": None,
    }]


def test_parse_tool_context_passthrough_already_structured():
    # Persisted rows store the structured shape (no `role` keys) — must be
    # returned unchanged so existing transcript rendering keeps working.
    structured = [{"call_id": "c", "tool_name": "t", "arguments": {},
                   "result": "ok", "error": None}]
    assert _parse_tool_context(structured) == structured


def test_parse_tool_context_none_and_unparseable():
    assert _parse_tool_context(None) is None
    assert _parse_tool_context("") is None
    assert _parse_tool_context("not json{") is None


def test_parse_tool_context_non_list_returned_as_is():
    assert _parse_tool_context({"a": 1}) == {"a": 1}


def test_parse_tool_context_malformed_elements_do_not_raise():
    # A corrupted tool_context (non-dict list element, or a non-dict tool_call)
    # must not raise AttributeError — that would escape the (TypeError, ValueError)
    # guard and 500 the entire /transcript endpoint rather than one row.
    assert _parse_tool_context(["x"]) == ["x"]
    # role-bearing message that resolves nothing → None (DP-143), but the point
    # here is that the non-dict tool_call doesn't raise.
    assert _parse_tool_context(
        [{"role": "assistant", "tool_calls": ["bad"]}]
    ) is None


def test_parse_tool_context_orphaned_call_id_returns_none():
    # Truncated/orphaned history: a tool row references a call_id with no
    # matching assistant tool_calls in the slice → nothing resolves. Must NOT
    # leak the raw {role, content} OpenAI messages (they lack call_id/tool_name/
    # arguments → garbled ToolCard). Return None so no tool panel renders. DP-143.
    raw = [
        {"role": "assistant", "content": "let me check"},
        {"role": "tool", "tool_call_id": "orphan", "content": "{}"},
    ]
    assert _parse_tool_context(raw) is None


def test_parse_tool_context_structured_without_role_still_passes_through():
    # Already-structured ToolContext[] carries no `role` key → passthrough
    # unchanged even though nothing "resolves". DP-143 must not break this.
    structured = [{"call_id": "c", "tool_name": "t", "arguments": {},
                   "result": None, "error": None}]
    assert _parse_tool_context(structured) == structured
