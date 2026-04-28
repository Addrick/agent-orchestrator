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
from src.interfaces.kobold_export import build_kobold_savefile


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
