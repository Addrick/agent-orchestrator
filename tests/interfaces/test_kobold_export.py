# tests/interfaces/test_kobold_export.py

"""Phase 2.1: DERPR DB → kobold-lite savefile exporter.

Covers:
  - Unit: build_kobold_savefile produces a valid v1 'oldui' savefile shape
    with user turns wrapped in {{[INPUT]}}/{{[OUTPUT]}} placeholders.
  - Integration: round-trip through MemoryManager.get_global_history.
  - Regression: skipped row counter accounts for system / empty / tool rows.
"""

import json
from datetime import datetime, timedelta

import pytest

from memory.memory_manager import MemoryManager
from src.interfaces.kobold_export import (
    build_kobold_savefile,
    build_transcript,
    _parse_tool_context,
)


REQUIRED_SAVEFILE_KEYS = {
    "gamestarted", "prompt", "memory", "authorsnote", "anotetemplate",
    "actions", "actions_metadata", "worldinfo", "wifolders_d", "wifolders_l",
}


# -------- Unit tests --------

def test_empty_history_returns_valid_skeleton():
    savefile, skipped = build_kobold_savefile([])
    assert REQUIRED_SAVEFILE_KEYS.issubset(savefile.keys())
    assert savefile["gamestarted"] is True
    assert savefile["prompt"] == ""
    assert savefile["actions"] == []
    # memory stays empty — persona system prompt lives in kobold's
    # instruct_sysprompt setting, not the savefile memory block.
    assert savefile["memory"] == ""
    assert skipped == 0


def test_user_turn_is_wrapped_with_instruct_placeholders():
    rows = [{"author_role": "user", "content": "hello"}]
    savefile, _ = build_kobold_savefile(rows)
    # First entry becomes `prompt`, not `actions[0]`.
    assert "{{[INPUT]}}" in savefile["prompt"]
    assert "{{[OUTPUT]}}" in savefile["prompt"]
    assert "hello" in savefile["prompt"]
    assert savefile["actions"] == []


def test_assistant_turn_is_emitted_raw():
    rows = [{"author_role": "assistant", "content": "raw response"}]
    savefile, _ = build_kobold_savefile(rows)
    assert savefile["prompt"] == "raw response"
    assert "{{[INPUT]}}" not in savefile["prompt"]


def test_user_assistant_alternation_preserves_order():
    rows = [
        {"author_role": "user", "content": "u1"},
        {"author_role": "assistant", "content": "a1"},
        {"author_role": "user", "content": "u2"},
        {"author_role": "assistant", "content": "a2"},
    ]
    savefile, _ = build_kobold_savefile(rows)
    rendered = [savefile["prompt"]] + savefile["actions"]
    assert len(rendered) == 4
    assert "u1" in rendered[0] and "{{[INPUT]}}" in rendered[0]
    assert rendered[1] == "a1"
    assert "u2" in rendered[2] and "{{[INPUT]}}" in rendered[2]
    assert rendered[3] == "a2"


def test_system_rows_skipped_and_counted():
    rows = [
        {"author_role": "system", "content": "boot"},
        {"author_role": "user", "content": "hi"},
    ]
    savefile, skipped = build_kobold_savefile(rows)
    assert skipped == 1
    assert "hi" in savefile["prompt"]
    assert "boot" not in savefile["prompt"]
    assert savefile["actions"] == []


def test_empty_content_rows_skipped_and_counted():
    rows = [
        {"author_role": "user", "content": "   "},
        {"author_role": "user", "content": "real"},
    ]
    savefile, skipped = build_kobold_savefile(rows)
    assert skipped == 1
    assert "real" in savefile["prompt"]


def test_assistant_with_tool_context_and_content_is_emitted():
    # Assistant rows can carry both a `tool_context` (tool-call metadata) and
    # visible `content`. Phase 2.1 has no tool-turn rendering — we just emit
    # the textual content as a normal assistant turn. Tool-call expansion is
    # backlog (see web_ui_roadmap Phase 2 tool-use note).
    rows = [
        {
            "author_role": "assistant",
            "content": "final answer",
            "tool_context": json.dumps([{"role": "tool", "content": "..."}]),
        }
    ]
    savefile, skipped = build_kobold_savefile(rows)
    assert savefile["prompt"] == "final answer"
    assert skipped == 0


def test_tool_only_row_with_empty_content_skipped():
    # A tool-call-only assistant row (no visible content) has nothing to
    # render into kobold's text stream — skip it, count once.
    rows = [
        {
            "author_role": "assistant",
            "content": "",
            "tool_context": json.dumps([{"role": "tool", "content": "..."}]),
        },
        {"author_role": "user", "content": "hi"},
    ]
    savefile, skipped = build_kobold_savefile(rows)
    assert skipped == 1
    assert "hi" in savefile["prompt"]


# -------- Integration: round-trip through MemoryManager --------

@pytest.mark.integration
def test_export_from_seeded_memory_manager():
    mm = MemoryManager(db_path=":memory:")
    mm.create_schema()

    base = datetime(2026, 4, 1, 12, 0, 0)
    seed = [
        ("user", "user_a", "first user msg"),
        ("assistant", "test_persona", "first reply"),
        ("user", "user_a", "second user msg"),
        ("assistant", "test_persona", "second reply"),
    ]
    for i, (role, name, content) in enumerate(seed):
        mm.log_message(
            user_identifier="user_a",
            persona_name="test_persona",
            channel="chan",
            author_role=role,
            author_name=name,
            content=content,
            timestamp=base + timedelta(seconds=i),
        )

    raw = mm.get_global_history("test_persona", limit=10)
    savefile, skipped = build_kobold_savefile(raw)

    assert skipped == 0
    rendered = [savefile["prompt"]] + savefile["actions"]
    assert len(rendered) == 4
    assert "first user msg" in rendered[0]
    assert rendered[1] == "first reply"
    assert "second user msg" in rendered[2]
    assert rendered[3] == "second reply"
    # Exporter no longer stuffs persona prompt into memory — the UI writes it
    # to kobold's instruct_sysprompt field instead.
    assert savefile["memory"] == ""

    mm.close()

def test_build_kobold_savefile_wraps_reasoning_in_think_tags():
    from src.interfaces.kobold_export import build_kobold_savefile
    history = [
        {"author_role": "user", "content": "hello", "interaction_id": 1},
        {
            "author_role": "assistant", 
            "content": "final answer", 
            "reasoning_content": "my thoughts",
            "interaction_id": 2
        }
    ]
    savefile, skipped = build_kobold_savefile(history)
    actions = [savefile["prompt"]] + savefile["actions"]
    assert "<think>\nmy thoughts\n</think>\nfinal answer" in actions[1]


# -------- DP-130 contract invariant C2: gametext alignment (ids == actions + 1) --------

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


def test_c2_ids_are_gametext_aligned():
    # Gametext alignment: one id per VISIBLE chunk including the prompt, so
    # len(interaction_ids) == len(actions) + 1 == len([prompt, *actions]).
    # This matches how the portal keys derpr_interaction_ids[gametext_index]
    # (index 0 == prompt). Anything else drifts the array on restore.
    savefile, _ = build_kobold_savefile(_rows(4))
    gametext_len = len([savefile["prompt"]] + savefile["actions"])
    assert len(savefile["interaction_ids"]) == gametext_len == 4
    assert len(savefile["interaction_ids"]) == len(savefile["actions"]) + 1

def test_c2_ids_align_positionally_with_gametext():
    # interaction_ids[i] addresses gametext_arr[i]; ids[0] is the prompt's id.
    savefile, _ = build_kobold_savefile(_rows(4))
    # Rows have ids 100,101,102,103 → gametext [prompt(100),101,102,103].
    assert savefile["interaction_ids"] == [100, 101, 102, 103]

def test_c2_holds_for_13_rows_regression():
    # The reported actions=12 ids=13 was CORRECT gametext alignment (13 chunks:
    # prompt + 12 actions). Lock that in: ids == chunks, actions == chunks - 1.
    savefile, _ = build_kobold_savefile(_rows(13))
    assert len(savefile["actions"]) == 12
    assert len(savefile["interaction_ids"]) == 13

def test_c2_single_chunk_has_one_id_zero_actions():
    savefile, _ = build_kobold_savefile(_rows(1))
    assert savefile["actions"] == []
    assert savefile["interaction_ids"] == [100]

def test_c2_holds_when_a_row_lacks_an_int_id():
    # A renderable row with a missing/None id still gets a slot (None) so the
    # array stays 1:1 with the story — the portal's `if (interactionId)` guard
    # no-ops the None slot instead of mis-targeting a neighbour.
    rows = [
        {"author_role": "user", "content": "u0", "interaction_id": 1},
        {"author_role": "assistant", "content": "a0"},  # no interaction_id
        {"author_role": "user", "content": "u1", "interaction_id": 3},
    ]
    savefile, _ = build_kobold_savefile(rows)
    gametext_len = len([savefile["prompt"]] + savefile["actions"])
    assert len(savefile["interaction_ids"]) == gametext_len == 3
    assert savefile["interaction_ids"] == [1, None, 3]

def test_c2_holds_with_skipped_system_and_empty_rows():
    rows = [
        {"author_role": "system", "content": "boot"},
        {"author_role": "user", "content": "u0", "interaction_id": 1},
        {"author_role": "assistant", "content": "  ", "interaction_id": 2},  # empty
        {"author_role": "assistant", "content": "a0", "interaction_id": 3},
    ]
    savefile, skipped = build_kobold_savefile(rows)
    assert skipped == 2
    gametext_len = len([savefile["prompt"]] + savefile["actions"])
    assert len(savefile["interaction_ids"]) == gametext_len


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

def test_transcript_folds_reasoning_into_think_block():
    rows = [
        {"author_role": "assistant", "content": "answer",
         "reasoning_content": "thinking", "interaction_id": 5},
    ]
    chunks = build_transcript(rows)["chunks"]
    assert chunks[0]["content"] == "<think>\nthinking\n</think>\nanswer"
    assert chunks[0]["reasoning"] == "thinking"

def test_transcript_parses_tool_context_json():
    rows = [
        {"author_role": "assistant", "content": "done", "interaction_id": 5,
         "tool_context": json.dumps([{"role": "tool", "content": "x"}])},
    ]
    chunks = build_transcript(rows)["chunks"]
    assert chunks[0]["tool_context"] == [{"role": "tool", "content": "x"}]

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
    pending = {"ephemeral_chunk_id": "tok123", "content": "awaiting approval",
               "tool_context": None}
    chunks = build_transcript(rows, pending=pending)["chunks"]
    assert len(chunks) == 3
    last = chunks[-1]
    assert last["ephemeral"] is True
    assert last["interaction_id"] is None
    assert last["ephemeral_chunk_id"] == "tok123"
    assert last["content"] == "awaiting approval"


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
