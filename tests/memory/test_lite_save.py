"""DP-252: KoboldCpp Lite save → transcript parser."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.memory.date_extraction import extract_regex_dates
from src.memory.lite_save import (
    NotALiteSave,
    export_time_from_filename,
    looks_like_lite_save,
    parse_lite_save,
)

NY = ZoneInfo("America/New_York")
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lite_save_instruct.json"

_SETTINGS = {"opmode": "4", "start_thinking_tag": "<think>", "stop_thinking_tag": "</think>"}


def _save(prompt: str, actions=(), **settings) -> dict:
    return {"prompt": prompt, "actions": list(actions),
            "savedsettings": {**_SETTINGS, **settings}}


def _roles(t):
    return [(m.role, m.content) for m in t.messages]


# ---------- the real export (sanitized) ----------

def test_real_export_joins_actions_and_splits_on_placeholders():
    t = parse_lite_save(json.loads(FIXTURE.read_text(encoding="utf-8")), tz=NY)
    assert [m.role for m in t.messages] == ["user", "assistant", "user", "assistant"]
    assert t.messages[0].content.startswith("I'm just here to test speed.")
    assert t.messages[3].content.startswith("The case was simple.")


def test_real_export_strips_think_that_spans_two_actions():
    t = parse_lite_save(json.loads(FIXTURE.read_text(encoding="utf-8")), tz=NY)
    rendered = t.render()
    assert "<think>" not in rendered and "</think>" not in rendered
    assert "let me recount" not in rendered.lower()
    assert "let me recount" in t.messages[3].reasoning_content.lower()
    assert t.reasoning_stripped


def test_real_export_never_renders_setup_or_settings():
    t = parse_lite_save(json.loads(FIXTURE.read_text(encoding="utf-8")), tz=NY)
    rendered = t.render()
    assert "SENTINEL" not in rendered
    assert "{{[" not in rendered


# ---------- turn splitting ----------

def test_user_and_assistant_turns_render_as_transcript():
    t = parse_lite_save(_save("\n{{[INPUT]}}\nhi\n{{[OUTPUT]}}\n", ["hello there"]), tz=NY)
    assert t.render() == "User: hi\n\nAssistant: hello there"
    assert t.turn_count() == 2


def test_raw_instruct_tags_split_on_literal_tags():
    save = _save("<|user|>hi<|assistant|>", ["yo"], raw_instruct_tags=True,
                 instruct_starttag="<|user|>", instruct_endtag="<|assistant|>")
    assert _roles(parse_lite_save(save, tz=NY)) == [("user", "hi"), ("assistant", "yo")]


def test_separate_end_tags_close_turns():
    save = _save("{{[INPUT]}}hi{{[INPUT_END]}}{{[OUTPUT]}}yo{{[OUTPUT_END]}}")
    assert _roles(parse_lite_save(save, tz=NY)) == [("user", "hi"), ("assistant", "yo")]


def test_system_turns_and_leading_setup_are_not_rendered():
    save = _save("be terse\n{{[SYSTEM]}}\nrules\n{{[INPUT]}}\nhi\n{{[OUTPUT]}}\nyo")
    t = parse_lite_save(save, tz=NY)
    assert [m.role for m in t.messages] == ["system", "system", "user", "assistant"]
    assert t.render() == "User: hi\n\nAssistant: yo"


def test_reasoning_only_turn_disappears():
    t = parse_lite_save(_save("{{[INPUT]}}hi{{[OUTPUT]}}<think>hmm</think>  "), tz=NY)
    assert _roles(t) == [("user", "hi")]


# ---------- reasoning edge cases ----------

def test_close_tag_without_opener_strips_from_turn_start():
    t = parse_lite_save(_save("{{[INPUT]}}hi{{[OUTPUT]}}injected thinking</think>answer"), tz=NY)
    assert t.messages[1].content == "answer"
    assert t.messages[1].reasoning_content == "injected thinking"


def test_opener_without_close_strips_to_turn_end():
    t = parse_lite_save(_save("{{[INPUT]}}hi{{[OUTPUT]}}answer<think>cut off"), tz=NY)
    assert t.messages[1].content == "answer"
    assert t.messages[1].reasoning_content == "cut off"


def test_custom_thinking_tags_from_settings():
    save = _save("{{[INPUT]}}hi{{[OUTPUT]}}[r]x[/r]answer",
                 start_thinking_tag="[r]", stop_thinking_tag="[/r]")
    assert parse_lite_save(save, tz=NY).messages[1].content == "answer"


# ---------- timestamps ----------

def test_lite_stamp_rewritten_to_iso_and_anchorable():
    t = parse_lite_save(_save("{{[INPUT]}}[9/18/2026, 01:44 AM] hi{{[OUTPUT]}}yo"), tz=NY)
    assert t.messages[0].content == "[2026-09-18 01:44] hi"
    assert t.stamps == [datetime(2026, 9, 18, 1, 44, tzinfo=NY)]
    assert [d.date().isoformat() for d in extract_regex_dates(t.render())] == ["2026-09-18"]


def test_pm_stamp_with_narrow_nbsp_converts_to_24h():
    t = parse_lite_save(_save("{{[INPUT]}}[9/18/2026, 11:05 PM] hi"), tz=NY)
    assert t.messages[0].content == "[2026-09-18 23:05] hi"


def test_unrecognised_stamp_left_verbatim():
    # A non-en-US locale (day-first, 24h) is not guessed at.
    t = parse_lite_save(_save("{{[INPUT]}}[18.9.2026, 13:44] hi"), tz=NY)
    assert t.messages[0].content == "[18.9.2026, 13:44] hi"
    assert t.stamps == []


def test_impossible_stamp_left_verbatim():
    t = parse_lite_save(_save("{{[INPUT]}}[13/40/2026, 01:44 AM] hi"), tz=NY)
    assert t.messages[0].content.startswith("[13/40/2026")


def test_export_time_from_filename():
    got = export_time_from_filename("Untitled_Instruct_6_30_2026__7_32_26_PM.json", tz=NY)
    assert got == datetime(2026, 6, 30, 19, 32, 26, tzinfo=NY)
    assert export_time_from_filename("notes.json", tz=NY) is None


# ---------- rejection ----------

@pytest.mark.parametrize("data", [[], {"prompt": "x"}, {"actions": [], "savedsettings": {}}])
def test_non_saves_rejected(data):
    assert not looks_like_lite_save(data)
    with pytest.raises(NotALiteSave, match="not a KoboldCpp Lite save"):
        parse_lite_save(data, tz=NY)


def test_chat_mode_rejected():
    with pytest.raises(NotALiteSave, match="instruct-mode"):
        parse_lite_save(_save("User: hi\nBot: yo", opmode="3"), tz=NY)


def test_empty_conversation_rejected():
    with pytest.raises(NotALiteSave, match="no conversation turns"):
        parse_lite_save(_save("just setup, no turns"), tz=NY)
