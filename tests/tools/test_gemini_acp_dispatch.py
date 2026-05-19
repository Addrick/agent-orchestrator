"""Phase 1 tests for tools/gemini_acp_dispatch.py.

Covers result-block parsing, summary building (with 1KB cap), and argparse.
JSON-RPC client + process lifecycle live in later phases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.gemini_acp_dispatch import (
    ResultBlockParseError,
    build_summary,
    main,
    parse_args,
    parse_result_block,
    write_outputs,
)


# ---------- parse_result_block ----------


def test_parse_result_block_happy_inline_lists() -> None:
    text = (
        "Some preceding text from gemini.\n\n"
        "## Result\n"
        "stop_reason: ok\n"
        "files_modified: src/foo.py, tests/test_foo.py\n"
        "key_changes: added Foo, wired Bar\n"
        "acceptance_self_check: ran -- pytest passes\n"
        "blockers: none\n"
    )
    parsed = parse_result_block(text)
    assert parsed is not None
    assert parsed["stop_reason"] == "ok"
    assert parsed["files_modified"] == ["src/foo.py", "tests/test_foo.py"]
    assert parsed["key_changes"] == ["added Foo", "wired Bar"]
    assert parsed["acceptance_self_check"] == "ran -- pytest passes"
    assert parsed["blockers"] == []


def test_parse_result_block_happy_bullet_lists() -> None:
    text = (
        "## Result\n"
        "stop_reason: ok\n"
        "files_modified:\n"
        "- src/foo.py\n"
        "- tests/test_foo.py\n"
        "key_changes:\n"
        "- added Foo with bar\n"
        "- wired into Baz\n"
        "acceptance_self_check: ran -- pytest passes\n"
        "blockers:\n"
        "- none\n"
    )
    parsed = parse_result_block(text)
    assert parsed is not None
    assert parsed["files_modified"] == ["src/foo.py", "tests/test_foo.py"]
    assert parsed["key_changes"] == ["added Foo with bar", "wired into Baz"]
    assert parsed["blockers"] == []


def test_parse_result_block_missing_returns_none() -> None:
    text = "Gemini wandered off and never wrote a result block."
    assert parse_result_block(text) is None


def test_parse_result_block_malformed_missing_required_key_raises() -> None:
    text = (
        "## Result\n"
        "stop_reason: ok\n"
        "files_modified: src/foo.py\n"
        # key_changes missing
        "acceptance_self_check: ran -- ok\n"
        "blockers: none\n"
    )
    with pytest.raises(ResultBlockParseError):
        parse_result_block(text)


def test_parse_result_block_malformed_unknown_stop_reason_raises() -> None:
    text = (
        "## Result\n"
        "stop_reason: vibes\n"
        "files_modified: none\n"
        "key_changes: nothing\n"
        "acceptance_self_check: skipped\n"
        "blockers: none\n"
    )
    with pytest.raises(ResultBlockParseError):
        parse_result_block(text)


def test_parse_result_block_picks_last_result_heading() -> None:
    text = (
        "## Result\n"
        "stop_reason: failed\n"
        "files_modified: none\n"
        "key_changes: tried, didn't work\n"
        "acceptance_self_check: skipped\n"
        "blockers: none\n"
        "\nActually wait, let me retry.\n\n"
        "## Result\n"
        "stop_reason: ok\n"
        "files_modified: src/foo.py\n"
        "key_changes: fixed it\n"
        "acceptance_self_check: ran -- pass\n"
        "blockers: none\n"
    )
    parsed = parse_result_block(text)
    assert parsed is not None
    assert parsed["stop_reason"] == "ok"


# ---------- build_summary ----------


def _parsed_ok() -> dict[str, object]:
    return {
        "stop_reason": "ok",
        "files_modified": ["src/foo.py"],
        "key_changes": ["did the thing"],
        "acceptance_self_check": "ran -- pass",
        "blockers": [],
    }


def test_build_summary_happy() -> None:
    summary = build_summary(
        session_id="sp-1-attempt-1",
        wall_clock_seconds=42.0,
        outer_stop_reason="ok",
        parsed=_parsed_ok(),
        raw_final_message=None,
    )
    assert summary["session_id"] == "sp-1-attempt-1"
    assert summary["stop_reason"] == "ok"
    assert summary["wall_clock_seconds"] == 42.0
    assert summary["files_modified"] == ["src/foo.py"]
    assert summary["key_changes"] == ["did the thing"]
    assert summary["parse_failed"] is False
    assert summary["raw_final_message"] is None
    # 1KB cap on happy summary
    assert len(json.dumps(summary).encode("utf-8")) <= 1024


def test_build_summary_truncation_drops_key_changes_then_blockers_then_files() -> None:
    bloated = {
        "stop_reason": "ok",
        "files_modified": [f"src/file_{i}.py" for i in range(20)],
        "key_changes": [f"change number {i} with words" for i in range(20)],
        "acceptance_self_check": "ran -- pass",
        "blockers": [f"blocker number {i}" for i in range(20)],
    }
    summary = build_summary(
        session_id="sp-1-attempt-1",
        wall_clock_seconds=10.0,
        outer_stop_reason="ok",
        parsed=bloated,
        raw_final_message=None,
    )
    serialized_len = len(json.dumps(summary).encode("utf-8"))
    assert serialized_len <= 1024
    # key_changes should be truncated first (possibly to empty),
    # files_modified should be the last thing trimmed
    assert len(summary["key_changes"]) < 20
    if len(summary["files_modified"]) < 20:
        # if files trimmed, key_changes and blockers must already be empty
        assert summary["key_changes"] == []
        assert summary["blockers"] == []


def test_build_summary_parse_failed_carries_raw_message_truncated_to_2kb() -> None:
    huge = "x" * 5000
    summary = build_summary(
        session_id="sp-1-attempt-1",
        wall_clock_seconds=10.0,
        outer_stop_reason="parse-failed",
        parsed=None,
        raw_final_message=huge,
    )
    assert summary["parse_failed"] is True
    assert summary["stop_reason"] == "parse-failed"
    assert summary["raw_final_message"] is not None
    assert len(summary["raw_final_message"].encode("utf-8")) <= 2048
    # When parse_failed, the structured fields are empty / defaults
    assert summary["files_modified"] == []
    assert summary["key_changes"] == []
    assert summary["blockers"] == []


def test_build_summary_timeout_marks_outer_stop_reason() -> None:
    summary = build_summary(
        session_id="sp-1-attempt-1",
        wall_clock_seconds=600.0,
        outer_stop_reason="timeout",
        parsed=None,
        raw_final_message=None,
    )
    assert summary["stop_reason"] == "timeout"
    assert summary["parse_failed"] is True
    assert summary["raw_final_message"] is None


# ---------- write_outputs ----------


def test_write_outputs_creates_summary_and_events(tmp_path: Path) -> None:
    out_dir = tmp_path / "sp-1-attempt-1"
    events: list[dict[str, object]] = [
        {"type": "session/update", "n": 1},
        {"type": "session/update", "n": 2},
    ]
    summary = {"session_id": "sp-1-attempt-1", "stop_reason": "ok"}
    write_outputs(out_dir, summary, events)
    summary_path = out_dir / "summary.json"
    events_path = out_dir / "events.jsonl"
    assert summary_path.exists()
    assert events_path.exists()
    assert json.loads(summary_path.read_text()) == summary
    lines = events_path.read_text().strip().split("\n")
    assert [json.loads(line) for line in lines] == events


def test_write_outputs_empty_events_writes_empty_file(tmp_path: Path) -> None:
    out_dir = tmp_path / "sp-2"
    write_outputs(out_dir, {"a": 1}, [])
    assert (out_dir / "events.jsonl").read_text() == ""


# ---------- argparse ----------


def test_parse_args_requires_packet_and_outdir(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["--packet", str(tmp_path / "p.md")])


def test_parse_args_happy(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--packet", str(tmp_path / "p.md"),
            "--worktree", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--session-id", "sp-1-attempt-1",
            "--wall-clock", "5",
        ]
    )
    assert args.session_id == "sp-1-attempt-1"
    assert args.wall_clock == 5
    assert args.packet == tmp_path / "p.md"


def test_parse_args_defaults_wall_clock_to_10(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--packet", str(tmp_path / "p.md"),
            "--worktree", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--session-id", "sp-1-attempt-1",
        ]
    )
    assert args.wall_clock == 10


# ---------- main entry (Phase 1 just argparses, no ACP yet) ----------


def test_main_phase1_writes_skeleton_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 1 main() should be invokable end-to-end but is a stub that writes
    a 'not-implemented' summary so the integration path is exercisable.
    Phase 2 replaces the stub with the actual ACP roundtrip."""
    packet = tmp_path / "packet.md"
    packet.write_text("# SP-1: stub\n\nDoes nothing yet.\n")
    out_dir = tmp_path / "out"
    rc = main(
        [
            "--packet", str(packet),
            "--worktree", str(tmp_path),
            "--out-dir", str(out_dir),
            "--session-id", "sp-1-attempt-1",
            "--wall-clock", "1",
        ]
    )
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["session_id"] == "sp-1-attempt-1"
    assert summary["stop_reason"] == "not-implemented"
