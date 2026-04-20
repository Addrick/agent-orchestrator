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

from src.database.memory_manager import MemoryManager
from src.interfaces.kobold_export import build_kobold_savefile


REQUIRED_SAVEFILE_KEYS = {
    "gamestarted", "prompt", "memory", "authorsnote", "anotetemplate",
    "actions", "actions_metadata", "worldinfo", "wifolders_d", "wifolders_l",
}


# -------- Unit tests --------

def test_empty_history_returns_valid_skeleton():
    savefile, skipped = build_kobold_savefile([], system_prompt="sys")
    assert REQUIRED_SAVEFILE_KEYS.issubset(savefile.keys())
    assert savefile["gamestarted"] is True
    assert savefile["prompt"] == ""
    assert savefile["actions"] == []
    assert savefile["memory"] == "sys"
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


def test_tool_context_rows_counted_but_content_kept():
    rows = [
        {
            "author_role": "assistant",
            "content": "final answer",
            "tool_context": json.dumps([{"role": "tool", "content": "..."}]),
        }
    ]
    savefile, skipped = build_kobold_savefile(rows)
    # tool_context expansion is dropped (kobold has no slot for tool turns)
    # but the assistant's textual content survives.
    assert savefile["prompt"] == "final answer"
    assert skipped == 1


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
    savefile, skipped = build_kobold_savefile(raw, system_prompt="you are test")

    assert skipped == 0
    rendered = [savefile["prompt"]] + savefile["actions"]
    assert len(rendered) == 4
    assert "first user msg" in rendered[0]
    assert rendered[1] == "first reply"
    assert "second user msg" in rendered[2]
    assert rendered[3] == "second reply"
    assert savefile["memory"] == "you are test"

    mm.close()
