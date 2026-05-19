"""Tests for scripts/gemini_acp_dispatch.py.

Phase 1: parser, summary builder, argparse, write_outputs.
Phase 2: ACP JSON-RPC roundtrip against fake-gemini fixture.
Phase 3: wall-clock timeout, SIGTERM cancel, drain cap.
"""

from __future__ import annotations

import json
import signal
import threading
import time
from pathlib import Path

import pytest

from scripts.gemini_acp_dispatch import (
    ResultBlockParseError,
    build_summary,
    main,
    parse_args,
    parse_result_block,
    run_dispatch,
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


# ---------- main entry ----------


def test_main_swaps_in_spawn_via_monkeypatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_gemini_spawner,
) -> None:
    """main() drives the full dispatcher; in tests we monkeypatch the
    module-level spawn function to swap real gemini for the fake."""
    packet = tmp_path / "packet.md"
    packet.write_text("# SP-1\n\nDo nothing\n")
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        "scripts.gemini_acp_dispatch.spawn_gemini",
        fake_gemini_spawner(),
    )
    rc = main(
        [
            "--packet", str(packet),
            "--worktree", str(tmp_path),
            "--out-dir", str(out_dir),
            "--session-id", "sp-1-attempt-1",
            # 0.5 minutes = 30s; default fake has no delay so completes instantly
            "--wall-clock", "0.5",
        ]
    )
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["session_id"] == "sp-1-attempt-1"
    assert summary["stop_reason"] == "ok"
    assert summary["files_modified"] == ["src/foo.py"]


# ---------- Phase 2: ACP roundtrip against fake-gemini ----------


def _run(
    *,
    out_dir: Path,
    spawn_fn,
    wall_clock_seconds: float = 30.0,
    packet_text: str = "# SP-1\nDo work\n",
    session_id: str = "sp-1-attempt-1",
) -> int:
    return run_dispatch(
        packet_text=packet_text,
        worktree=out_dir.parent,
        out_dir=out_dir,
        session_id=session_id,
        wall_clock_seconds=wall_clock_seconds,
        spawn_fn=spawn_fn,
    )


def test_acp_handshake_and_summary(tmp_path: Path, fake_gemini_spawner) -> None:
    out_dir = tmp_path / "out"
    rc = _run(out_dir=out_dir, spawn_fn=fake_gemini_spawner())
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "ok"
    assert summary["parse_failed"] is False
    assert summary["files_modified"] == ["src/foo.py"]
    assert summary["key_changes"] == ["implemented foo"]
    # events.jsonl populated with at least handshake responses + stream
    events = [
        json.loads(line)
        for line in (out_dir / "events.jsonl").read_text().splitlines()
        if line
    ]
    methods = [e.get("method") for e in events]
    # Initialize and session/new responses don't carry "method"; updates do
    assert "session/update" in methods
    # At least one response (no method, has id) present
    assert any("method" not in e and "id" in e for e in events)


def test_streamed_chunks_concatenate_into_result_block(
    tmp_path: Path, fake_gemini_spawner
) -> None:
    """The dispatcher must accumulate all agent_message_chunk text and parse
    the result block out of the concatenated whole, not individual chunks."""
    out_dir = tmp_path / "out"
    custom = (
        "preface line\n\n"
        "## Result\n"
        "stop_reason: failed\n"
        "files_modified: none\n"
        "key_changes: tried, gave up\n"
        "acceptance_self_check: skipped -- could not run\n"
        "blockers: missing fixture\n"
    )
    rc = _run(out_dir=out_dir, spawn_fn=fake_gemini_spawner(final_message=custom))
    # outer_stop is "failed" → not in _EXIT_CODES → exit 0 (Claude reads summary)
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "failed"
    assert summary["parse_failed"] is False
    assert summary["blockers"] == ["missing fixture"]


def test_missing_result_block_marks_parse_failed(
    tmp_path: Path, fake_gemini_spawner
) -> None:
    out_dir = tmp_path / "out"
    rc = _run(
        out_dir=out_dir,
        spawn_fn=fake_gemini_spawner(final_message="just chatter, no result block"),
    )
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "parse-failed"
    assert summary["parse_failed"] is True
    assert summary["raw_final_message"] is not None
    assert "just chatter" in summary["raw_final_message"]


def test_malformed_result_block_marks_parse_failed(
    tmp_path: Path, fake_gemini_spawner
) -> None:
    out_dir = tmp_path / "out"
    bad = "## Result\nstop_reason: vibes\nfiles_modified: none\nkey_changes: x\nacceptance_self_check: y\nblockers: none\n"
    rc = _run(out_dir=out_dir, spawn_fn=fake_gemini_spawner(final_message=bad))
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "parse-failed"
    assert summary["parse_failed"] is True
    assert "vibes" in (summary["raw_final_message"] or "")


def test_auto_approves_permission_requests(
    tmp_path: Path, fake_gemini_spawner
) -> None:
    """request_permission events from the server must be auto-responded with
    'allow_always' so the prompt can complete. The fake fires N requests
    before the prompt response; the dispatcher must not hang and must
    record both the inbound requests and its outbound responses."""
    out_dir = tmp_path / "out"
    rc = _run(out_dir=out_dir, spawn_fn=fake_gemini_spawner(permission_requests=3))
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "ok"
    events = [
        json.loads(line)
        for line in (out_dir / "events.jsonl").read_text().splitlines()
        if line
    ]
    perm_requests = [e for e in events if e.get("method") == "session/request_permission"]
    assert len(perm_requests) == 3


def test_handshake_failure_yields_error_summary(
    tmp_path: Path, fake_gemini_spawner
) -> None:
    out_dir = tmp_path / "out"
    rc = _run(out_dir=out_dir, spawn_fn=fake_gemini_spawner(fail_init=True))
    assert rc == 1
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "error"
    assert summary["parse_failed"] is True
    assert "handshake failed" in (summary["raw_final_message"] or "")


def test_spawn_failure_yields_error_summary(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"

    def boom(_worktree: Path):
        raise RuntimeError("no gemini for you")

    rc = run_dispatch(
        packet_text="x",
        worktree=tmp_path,
        out_dir=out_dir,
        session_id="sp-1-attempt-1",
        wall_clock_seconds=5.0,
        spawn_fn=boom,
    )
    assert rc == 1
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "error"
    assert "no gemini for you" in (summary["raw_final_message"] or "")


# ---------- Phase 3: lifecycle (timeout, cancel) ----------


def test_wall_clock_timeout_writes_timeout_summary_and_exits_124(
    tmp_path: Path, fake_gemini_spawner
) -> None:
    """Fake sleeps 5s before responding; dispatcher caps at 0.5s and should
    cancel, write a timeout summary, and exit 124."""
    out_dir = tmp_path / "out"
    rc = _run(
        out_dir=out_dir,
        spawn_fn=fake_gemini_spawner(prompt_delay=5.0),
        wall_clock_seconds=0.5,
    )
    assert rc == 124
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "timeout"
    assert summary["parse_failed"] is True
    # Wall-clock recorded should be roughly the cap, not the fake's 5s sleep
    assert summary["wall_clock_seconds"] < 4.0


def test_cancel_via_flag_writes_cancelled_summary_and_exits_130(
    tmp_path: Path, fake_gemini_spawner
) -> None:
    """We can't reliably send SIGTERM to ourselves on Windows. Instead we
    drive cancellation by firing SIGINT from a background thread once
    run_dispatch is well into the prompt — same code path the SIGTERM
    handler would take on Unix."""
    out_dir = tmp_path / "out"

    def _interrupter() -> None:
        time.sleep(0.4)
        # signal.raise_signal targets the current process; SIGINT is supported
        # on both POSIX and Windows.
        try:
            signal.raise_signal(signal.SIGINT)
        except Exception:
            pass

    t = threading.Thread(target=_interrupter, daemon=True)
    t.start()
    rc = _run(
        out_dir=out_dir,
        spawn_fn=fake_gemini_spawner(prompt_delay=5.0),
        wall_clock_seconds=30.0,
    )
    t.join(timeout=2.0)
    assert rc == 130
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "cancelled"


def test_drain_does_not_hang_beyond_5s_after_cancel(
    tmp_path: Path, fake_gemini_spawner
) -> None:
    """If gemini ignores our cancel, the dispatcher must still bound its
    total wait by the drain cap (5s) plus a little wiggle room."""
    out_dir = tmp_path / "out"
    start = time.monotonic()
    rc = _run(
        out_dir=out_dir,
        spawn_fn=fake_gemini_spawner(prompt_delay=20.0),
        wall_clock_seconds=0.3,
    )
    elapsed = time.monotonic() - start
    assert rc == 124
    # Cap (0.3s) + drain (5s) + close (≤2s for stdin-close wait + 2s terminate) → < 12s
    assert elapsed < 12.0
