"""Tests for scripts/agy_dispatch.py.

The parser/summary/argparse-helper core is imported verbatim from the gemini
dispatcher and is exercised in test_gemini_acp_dispatch.py; these tests cover
the agy-specific transport: subprocess capture of stdout→result-block /
stderr→events, plus the lifecycle (timeout, cancel, spawn failure) on the new
``communicate``-based driver.
"""

from __future__ import annotations

import json
import signal
import threading
import time
from pathlib import Path

import pytest

from scripts.agy_config import DispatchConfig
from scripts.agy_dispatch import (
    _clean_agy_stdout,
    main,
    parse_args,
    run_dispatch,
)


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


# ---------- happy path: stdout → result block, stderr → events ----------


def test_happy_run_parses_stdout_and_logs_stderr(
    tmp_path: Path, fake_agy_spawner
) -> None:
    out_dir = tmp_path / "out"
    rc = _run(
        out_dir=out_dir,
        spawn_fn=fake_agy_spawner(stderr="listing files...\nwrote src/foo.py\n"),
    )
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "ok"
    assert summary["parse_failed"] is False
    assert summary["files_modified"] == ["src/foo.py"]
    assert summary["key_changes"] == ["implemented foo"]
    # stderr lines land in events.jsonl as opaque _stderr events
    events = [
        json.loads(line)
        for line in (out_dir / "events.jsonl").read_text().splitlines()
        if line
    ]
    stderr_events = [e for e in events if "_stderr" in e]
    assert any("wrote src/foo.py" in e["_stderr"] for e in stderr_events)


def test_failed_result_block_exits_zero(tmp_path: Path, fake_agy_spawner) -> None:
    """agy's own 'failed' is not a dispatcher error → exit 0, Claude reads it."""
    out_dir = tmp_path / "out"
    custom = (
        "tried hard\n\n"
        "## Result\n"
        "stop_reason: failed\n"
        "files_modified: none\n"
        "key_changes: tried, gave up\n"
        "acceptance_self_check: skipped -- could not run\n"
        "blockers: missing fixture\n"
    )
    rc = _run(out_dir=out_dir, spawn_fn=fake_agy_spawner(final_message=custom))
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "failed"
    assert summary["parse_failed"] is False
    assert summary["blockers"] == ["missing fixture"]


def test_missing_result_block_marks_parse_failed(
    tmp_path: Path, fake_agy_spawner
) -> None:
    out_dir = tmp_path / "out"
    rc = _run(
        out_dir=out_dir,
        spawn_fn=fake_agy_spawner(final_message="just chatter, no result block"),
    )
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "parse-failed"
    assert summary["parse_failed"] is True
    assert "just chatter" in (summary["raw_final_message"] or "")


def test_malformed_result_block_marks_parse_failed(
    tmp_path: Path, fake_agy_spawner
) -> None:
    out_dir = tmp_path / "out"
    bad = (
        "## Result\nstop_reason: vibes\nfiles_modified: none\n"
        "key_changes: x\nacceptance_self_check: y\nblockers: none\n"
    )
    rc = _run(out_dir=out_dir, spawn_fn=fake_agy_spawner(final_message=bad))
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "parse-failed"
    assert summary["parse_failed"] is True
    assert "vibes" in (summary["raw_final_message"] or "")


def test_nonzero_exit_still_parses_result_block(
    tmp_path: Path, fake_agy_spawner
) -> None:
    """A well-formed block is authoritative even if agy exits nonzero; the
    block's stop_reason drives the summary, not the process exit code."""
    out_dir = tmp_path / "out"
    rc = _run(out_dir=out_dir, spawn_fn=fake_agy_spawner(exit_code=3))
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "ok"


# ---------- failure / lifecycle ----------


def test_spawn_failure_yields_error_summary(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"

    def boom(_worktree: Path):
        raise RuntimeError("no agy for you")

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
    assert "no agy for you" in (summary["raw_final_message"] or "")


def test_wall_clock_timeout_writes_timeout_summary_and_exits_124(
    tmp_path: Path, fake_agy_spawner
) -> None:
    out_dir = tmp_path / "out"
    rc = _run(
        out_dir=out_dir,
        spawn_fn=fake_agy_spawner(prompt_delay=5.0),
        wall_clock_seconds=0.5,
    )
    assert rc == 124
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "timeout"
    assert summary["parse_failed"] is True
    assert summary["wall_clock_seconds"] < 4.0


def test_cancel_via_signal_writes_cancelled_summary_and_exits_130(
    tmp_path: Path, fake_agy_spawner
) -> None:
    out_dir = tmp_path / "out"

    def _interrupter() -> None:
        time.sleep(0.4)
        try:
            signal.raise_signal(signal.SIGINT)
        except Exception:
            pass

    t = threading.Thread(target=_interrupter, daemon=True)
    t.start()
    rc = _run(
        out_dir=out_dir,
        spawn_fn=fake_agy_spawner(prompt_delay=5.0),
        wall_clock_seconds=30.0,
    )
    t.join(timeout=2.0)
    assert rc == 130
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "cancelled"


def test_timeout_is_bounded_in_wall_clock_time(
    tmp_path: Path, fake_agy_spawner
) -> None:
    """A child that runs far longer than the cap must be terminated promptly,
    not waited out — total time stays near the cap + terminate grace."""
    out_dir = tmp_path / "out"
    start = time.monotonic()
    rc = _run(
        out_dir=out_dir,
        spawn_fn=fake_agy_spawner(prompt_delay=20.0),
        wall_clock_seconds=0.3,
    )
    elapsed = time.monotonic() - start
    assert rc == 124
    assert elapsed < 10.0


# ---------- argparse + main ----------


def test_parse_args_requires_packet_and_outdir(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["--packet", str(tmp_path / "p.md")])


def test_parse_args_wall_clock_unset_by_default(tmp_path: Path) -> None:
    # Both unit flags default to None; main() resolves the unset case to 10 min.
    args = parse_args(
        [
            "--packet", str(tmp_path / "p.md"),
            "--worktree", str(tmp_path),
            "--out-dir", str(tmp_path / "out"),
            "--session-id", "sp-1-attempt-1",
        ]
    )
    assert args.wall_clock_minutes is None
    assert args.wall_clock_seconds is None


# ---------- stdout cleaning (background-task noise) ----------


def test_clean_agy_stdout_strips_system_message_envelopes() -> None:
    raw = (
        "working on it\n"
        "<SYSTEM_MESSAGE>\n[Message] task-47 finished with huge log...\n"
        "lots of noise\n</SYSTEM_MESSAGE>\n"
        "## Result\nstop_reason: ok\nfiles_modified: none\n"
        "key_changes: did it\nacceptance_self_check: ok\nblockers: none\n"
    )
    cleaned = _clean_agy_stdout(raw)
    assert "SYSTEM_MESSAGE" not in cleaned
    assert "huge log" not in cleaned
    assert "## Result" in cleaned


def test_clean_agy_stdout_handles_multiple_and_preserves_result() -> None:
    raw = (
        "<SYSTEM_MESSAGE>a</SYSTEM_MESSAGE>"
        "mid text"
        "<SYSTEM_MESSAGE>b</SYSTEM_MESSAGE>"
        "## Result\nstop_reason: ok\n"
    )
    cleaned = _clean_agy_stdout(raw)
    assert cleaned.count("SYSTEM_MESSAGE") == 0
    assert "mid text" in cleaned
    assert "## Result" in cleaned


def test_clean_agy_stdout_noop_when_no_envelopes() -> None:
    raw = "just a clean result\n## Result\nstop_reason: ok\n"
    assert _clean_agy_stdout(raw) == raw


def test_tool_noise_run_still_parses_result_block(
    tmp_path: Path, fake_agy_spawner
) -> None:
    """End-to-end: a tool-using run whose stdout interleaves SYSTEM_MESSAGE
    envelopes around the result block must still parse to stop_reason ok."""
    out_dir = tmp_path / "out"
    noisy = (
        "<SYSTEM_MESSAGE>\nbackground task task-47 finished\n</SYSTEM_MESSAGE>\n"
        "Did the work.\n\n"
        "## Result\nstop_reason: ok\nfiles_modified: src/foo.py\n"
        "key_changes: used a tool\nacceptance_self_check: ran -- pass\n"
        "blockers: none\n"
    )
    rc = _run(out_dir=out_dir, spawn_fn=fake_agy_spawner(final_message=noisy))
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "ok"
    assert summary["parse_failed"] is False
    assert summary["files_modified"] == ["src/foo.py"]


def test_config_sets_cli_model_during_run_and_restores_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """config.model is written to the agy CLI settings file *while agy runs* and
    the prior value is restored after."""
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"model": "Gemini 3.5 Flash (Low)"}), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.agy_config.DEFAULT_AGY_CLI_SETTINGS_PATH", settings
    )
    seen: dict[str, object] = {}

    def spawn(wt: Path):
        seen["model_during_run"] = json.loads(settings.read_text())["model"]
        import subprocess
        import sys
        return subprocess.Popen(
            [sys.executable, "-c", "print('## Result\\nstop_reason: ok\\n"
             "files_modified: none\\nkey_changes: x\\n"
             "acceptance_self_check: y\\nblockers: none')"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    rc = run_dispatch(
        packet_text="x",
        worktree=tmp_path,
        out_dir=tmp_path / "out",
        session_id="sp-1",
        wall_clock_seconds=30.0,
        spawn_fn=spawn,
        config=DispatchConfig(model="Gemini 3.5 Flash (High)"),
    )
    assert rc == 0
    assert seen["model_during_run"] == "Gemini 3.5 Flash (High)"
    # Prior value restored after the run.
    assert json.loads(settings.read_text())["model"] == "Gemini 3.5 Flash (Low)"


# ---------- config integration (AGENTS.md / MCP / model) ----------


def test_config_stages_agents_md_during_run_and_restores_after(
    tmp_path: Path,
) -> None:
    """AGENTS.md must exist in the worktree *while agy runs* and be gone after."""
    worktree = tmp_path / "wt"
    worktree.mkdir()
    out_dir = worktree / "out"
    seen: dict[str, bool] = {}

    def spawn(wt: Path):
        seen["agents_md_present"] = (wt / "AGENTS.md").exists()
        import subprocess
        import sys
        return subprocess.Popen(
            [sys.executable, "-c", "print('## Result\\nstop_reason: ok\\n"
             "files_modified: none\\nkey_changes: x\\n"
             "acceptance_self_check: y\\nblockers: none')"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    rc = run_dispatch(
        packet_text="do work",
        worktree=worktree,
        out_dir=out_dir,
        session_id="sp-1",
        wall_clock_seconds=30.0,
        spawn_fn=spawn,
        config=DispatchConfig(agents_md="you are a subagent"),
    )
    assert rc == 0
    assert seen["agents_md_present"] is True
    # Restored afterwards — we created it, so it should be gone.
    assert not (worktree / "AGENTS.md").exists()


def test_config_ensures_mcp_servers_and_logs_event(
    tmp_path: Path, fake_agy_spawner, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp_path = tmp_path / "mcp_config.json"
    monkeypatch.setattr("scripts.agy_config.DEFAULT_MCP_CONFIG_PATH", mcp_path)
    out_dir = tmp_path / "out"
    rc = run_dispatch(
        packet_text="x",
        worktree=tmp_path,
        out_dir=out_dir,
        session_id="sp-1",
        wall_clock_seconds=30.0,
        spawn_fn=fake_agy_spawner(),
        config=DispatchConfig(mcp_servers={"t": {"command": "python3"}}),
    )
    assert rc == 0
    data = json.loads(mcp_path.read_text())
    assert data["mcpServers"]["t"] == {"command": "python3"}
    events = [
        json.loads(line)
        for line in (out_dir / "events.jsonl").read_text().splitlines()
        if line
    ]
    assert any(e.get("_config") == "ensured mcp_servers" for e in events)


def test_main_swaps_in_spawn_via_monkeypatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_agy_spawner,
) -> None:
    packet = tmp_path / "packet.md"
    packet.write_text("# SP-1\n\nDo nothing\n")
    out_dir = tmp_path / "out"

    # spawn_agy takes (worktree, packet_text, wall_clock_seconds); the fake
    # spawner only needs the worktree, so adapt its signature.
    fake = fake_agy_spawner()
    monkeypatch.setattr(
        "scripts.agy_dispatch.spawn_agy",
        lambda worktree, packet_text, wall_clock_seconds: fake(worktree),
    )
    rc = main(
        [
            "--packet", str(packet),
            "--worktree", str(tmp_path),
            "--out-dir", str(out_dir),
            "--session-id", "sp-1-attempt-1",
            "--wall-clock", "0.5",
        ]
    )
    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["session_id"] == "sp-1-attempt-1"
    assert summary["stop_reason"] == "ok"
    assert summary["files_modified"] == ["src/foo.py"]


# ---------- raw stdout persistence ----------


def test_persists_full_raw_stdout_on_happy_path(
    tmp_path: Path, fake_agy_spawner
) -> None:
    out_dir = tmp_path / "out"
    narration = (
        "Reading files...\n"
        "<SYSTEM_MESSAGE>tool noise</SYSTEM_MESSAGE>\n"
        "## Result\n"
        "stop_reason: ok\n"
        "files_modified: src/foo.py\n"
        "key_changes: did it\n"
        "acceptance_self_check: ran -- pass\n"
        "blockers: none\n"
    )
    _run(out_dir=out_dir, spawn_fn=fake_agy_spawner(final_message=narration))
    raw = (out_dir / "stdout.txt").read_text()
    # Full, UNCLEANED stdout — the SYSTEM_MESSAGE envelope is kept for debugging
    # even though it is stripped before result-block parsing.
    assert raw == narration
    assert "<SYSTEM_MESSAGE>" in raw


def test_persists_raw_stdout_even_when_parse_fails(
    tmp_path: Path, fake_agy_spawner
) -> None:
    out_dir = tmp_path / "out"
    chatter = "did the work but my print turn ended before the result block"
    _run(out_dir=out_dir, spawn_fn=fake_agy_spawner(final_message=chatter))
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["parse_failed"] is True
    # The full narration is recoverable from stdout.txt regardless of parse state.
    assert (out_dir / "stdout.txt").read_text() == chatter


# ---------- wall-clock minutes/seconds CLI ----------


def test_wall_clock_minutes_and_seconds_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args(_min_argv(tmp_path) + ["--wall-clock", "5", "--wall-clock-seconds", "30"])


def test_wall_clock_seconds_flag_passed_through(
    tmp_path: Path, fake_agy_spawner, monkeypatch
) -> None:
    captured: dict = {}
    packet = tmp_path / "packet.md"
    packet.write_text("# SP-1\nx\n")
    out_dir = tmp_path / "out"
    fake = fake_agy_spawner()

    def fake_spawn(worktree, packet_text, wall_clock_seconds):
        captured["wall_clock_seconds"] = wall_clock_seconds
        return fake(worktree)

    monkeypatch.setattr("scripts.agy_dispatch.spawn_agy", fake_spawn)
    rc = main([
        "--packet", str(packet), "--worktree", str(tmp_path),
        "--out-dir", str(out_dir), "--session-id", "s",
        "--wall-clock-seconds", "42",
    ])
    assert rc == 0
    assert captured["wall_clock_seconds"] == 42.0  # seconds, NOT x60


def test_wall_clock_minutes_converts_to_seconds(
    tmp_path: Path, fake_agy_spawner, monkeypatch
) -> None:
    captured: dict = {}
    packet = tmp_path / "packet.md"
    packet.write_text("# SP-1\nx\n")
    out_dir = tmp_path / "out"
    fake = fake_agy_spawner()

    def fake_spawn(worktree, packet_text, wall_clock_seconds):
        captured["wall_clock_seconds"] = wall_clock_seconds
        return fake(worktree)

    monkeypatch.setattr("scripts.agy_dispatch.spawn_agy", fake_spawn)
    main([
        "--packet", str(packet), "--worktree", str(tmp_path),
        "--out-dir", str(out_dir), "--session-id", "s",
        "--wall-clock", "2",
    ])
    assert captured["wall_clock_seconds"] == 120.0  # 2 minutes


def _min_argv(tmp_path: Path) -> list[str]:
    return [
        "--packet", str(tmp_path / "p.md"),
        "--worktree", str(tmp_path),
        "--out-dir", str(tmp_path / "out"),
        "--session-id", "s",
    ]
