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
from src.interfaces.kobold_export import build_kobold_savefile, build_transcript


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


# -------- DP-130 contract invariant C2: len(actions) == len(interaction_ids) --------

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


def test_c2_actions_and_ids_equal_length_simple():
    # The live bug was actions=12 ids=13 — popping the prompt out of `actions`
    # only. Both arrays must end equal length, and the prompt id is preserved.
    savefile, _ = build_kobold_savefile(_rows(4))
    assert len(savefile["actions"]) == len(savefile["interaction_ids"])
    assert len(savefile["actions"]) == 3  # 4 rows - 1 prompt
    assert savefile["prompt_interaction_id"] == 100  # first row's id

def test_c2_holds_for_13_rows_regression():
    # Direct regression on the reported actions=12 ids=13 divergence.
    savefile, _ = build_kobold_savefile(_rows(13))
    assert len(savefile["actions"]) == 12
    assert len(savefile["interaction_ids"]) == 12

def test_c2_ids_align_positionally_with_actions():
    # interaction_ids[i] must address actions[i] (the post-prompt turn).
    savefile, _ = build_kobold_savefile(_rows(4))
    # Rows: ids 100(prompt),101,102,103 ; actions are rows 101,102,103.
    assert savefile["interaction_ids"] == [101, 102, 103]

def test_c2_holds_when_a_row_lacks_an_int_id():
    # A renderable row with a missing/None id still gets a slot (None) so the
    # arrays stay aligned — the portal's `if (interactionId)` guard no-ops it.
    rows = [
        {"author_role": "user", "content": "u0", "interaction_id": 1},
        {"author_role": "assistant", "content": "a0"},  # no interaction_id
        {"author_role": "user", "content": "u1", "interaction_id": 3},
    ]
    savefile, _ = build_kobold_savefile(rows)
    assert len(savefile["actions"]) == len(savefile["interaction_ids"]) == 2
    assert savefile["interaction_ids"] == [None, 3]

def test_c2_holds_with_skipped_system_and_empty_rows():
    rows = [
        {"author_role": "system", "content": "boot"},
        {"author_role": "user", "content": "u0", "interaction_id": 1},
        {"author_role": "assistant", "content": "  ", "interaction_id": 2},  # empty
        {"author_role": "assistant", "content": "a0", "interaction_id": 3},
    ]
    savefile, skipped = build_kobold_savefile(rows)
    assert skipped == 2
    assert len(savefile["actions"]) == len(savefile["interaction_ids"])


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
