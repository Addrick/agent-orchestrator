"""agy (Antigravity) subprocess dispatcher (DP-127).

A strict simplification of the DP-119 gemini ACP dispatcher
(``scripts/gemini_acp_dispatch.py``). One process per sprint attempt: spawns
``agy -p`` as a child in the worktree, hands it the packet as the prompt,
captures stdout (agy's narration + final ``## Result`` block) and stderr
(→ ``events.jsonl``), and writes ``summary.json`` (≤1KB) as the only thing
Claude reads.

Why a subprocess and not the SDK: the Antigravity SDK is API-key-only and can
only observe (never engine-execute) tool calls, whereas the interactive CLI
``agy -p`` runs on the cached OAuth tier and does real autonomous Model-A tool
execution in its own worktree. agy has no ``--acp``/JSON-RPC mode, so there is
no transport to drive — we just capture stdout. See the decision record
``memory/project/decisions/2026-05-29-agy-sdk-oauth-finding.md``.

The parsing/summary/event/signal/argparse core is **imported verbatim** from
the gemini dispatcher; only the ~250-line ACP transport (spawn + AcpClient +
_drive_prompt) is replaced by the ``subprocess`` capture below.

Exit codes (identical to the gemini dispatcher):
- 0   — ran end-to-end (agy's own pass/fail is in ``summary.json``)
- 1   — dispatcher-internal error (spawn / IO)
- 124 — wall-clock timeout
- 130 — SIGTERM / SIGINT (cancelled by parent)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from scripts.agy_config import (
    DispatchConfig,
    DispatchConfigError,
    ensure_mcp_servers,
    load_dispatch_config,
    stage_agents_md,
)
from scripts.gemini_acp_dispatch import (  # reused verbatim — DP-119 core
    _CANCEL_DRAIN_SECONDS,
    _derive_outer_stop_and_message,
    _finalize_and_exit,
    _install_cancel_handlers,
    _restore_handlers,
    StreamingEventLog,
)

# A little headroom over our own wall-clock so OUR cap is the authority, not
# agy's internal --print-timeout (which would otherwise fire first and emit a
# half-message). Go-duration seconds string.
_PRINT_TIMEOUT_HEADROOM_SECONDS = 30
# Grace between terminate() and kill() when bounding a runaway child.
_TERMINATE_GRACE_SECONDS = 2.0

# agy interleaves tool/background-task results into stdout wrapped in
# <SYSTEM_MESSAGE>...</SYSTEM_MESSAGE> envelopes. On tool-using runs these can
# bury the final ``## Result`` block (observed 2026-05-30). We strip the
# clearly-delimited envelopes before parsing; everything else is preserved.
_SYSTEM_MESSAGE_RE = re.compile(r"<SYSTEM_MESSAGE>.*?</SYSTEM_MESSAGE>", re.DOTALL)


def _clean_agy_stdout(text: str) -> str:
    """Remove agy's ``<SYSTEM_MESSAGE>`` tool-result envelopes from stdout.

    Only strips balanced ``<SYSTEM_MESSAGE>...</SYSTEM_MESSAGE>`` blocks so a
    final ``## Result`` block isn't lost in background-task noise. Non-envelope
    text (including the result block) is left untouched.
    """
    return _SYSTEM_MESSAGE_RE.sub("", text)


# --------------------------------------------------------------------------
# Process spawn (transport — the only part that differs from DP-119)
# --------------------------------------------------------------------------


def spawn_agy(
    worktree: Path,
    packet_text: str,
    wall_clock_seconds: float,
    model: str | None = None,
) -> subprocess.Popen[str]:
    """Spawn ``agy -p`` in the worktree with the packet as its prompt.

    ``--dangerously-skip-permissions`` auto-approves agy's tool calls (the
    equivalent of the ACP auto-approve path), and ``--add-dir`` scopes the
    workspace to the worktree. ``--print-timeout`` is set above our wall-clock
    so the outer cap wins. ``agy`` has no model-selection flag, so ``model`` is
    accepted only for interface parity with the gemini dispatcher and ignored
    here.
    """
    binary = shutil.which("agy")
    if not binary:
        raise RuntimeError("agy CLI not found on PATH")
    wt = str(worktree.resolve())
    print_timeout = f"{int(wall_clock_seconds) + _PRINT_TIMEOUT_HEADROOM_SECONDS}s"
    argv = [
        binary,
        "--dangerously-skip-permissions",
        "--add-dir", wt,
        "--print-timeout", print_timeout,
        "-p", packet_text,
    ]
    return subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=wt,
    )


def _terminate(proc: subprocess.Popen[str]) -> None:
    """Terminate ``proc``, escalating to kill if it ignores SIGTERM.

    The child's stdout/stderr are being drained by ``communicate()`` on the
    worker thread, so we only poll() here (never wait()) to avoid two threads
    reaping the same process.
    """
    try:
        proc.terminate()
    except Exception:
        pass
    deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)
    try:
        proc.kill()
    except Exception:
        pass


def _log_stderr(event_log: StreamingEventLog, stderr_text: str) -> None:
    """Record agy's stderr lines as opaque events (one per non-empty line)."""
    for line in stderr_text.splitlines():
        if line.strip():
            event_log.write({"_stderr": line})


def _drive_subprocess(
    proc: subprocess.Popen[str],
    event_log: StreamingEventLog,
    wall_clock_seconds: float,
    cancel_flag: threading.Event,
    start: float,
) -> tuple[str, dict[str, Any] | None, str | None, float | None]:
    """Run one ``agy -p`` child with cancel + wall-clock monitoring.

    Mirrors the DP-119 ``_drive_prompt`` contract: returns
    ``(outer_stop, parsed, raw_message, elapsed_at_decision)`` where
    ``elapsed_at_decision`` is None on the natural-completion path.
    """
    result_holder: dict[str, Any] = {}

    def _worker() -> None:
        try:
            stdout, stderr = proc.communicate()
            result_holder["stdout"] = stdout
            result_holder["stderr"] = stderr
        except Exception as exc:  # pragma: no cover - defensive
            result_holder["error"] = exc

    worker = threading.Thread(target=_worker, daemon=True, name="agy-comm")
    worker.start()

    outer_stop = "ok"
    elapsed_at_decision: float | None = None
    while worker.is_alive():
        if cancel_flag.is_set():
            outer_stop = "cancelled"
            elapsed_at_decision = time.monotonic() - start
            _terminate(proc)
            break
        if time.monotonic() - start > wall_clock_seconds:
            outer_stop = "timeout"
            elapsed_at_decision = time.monotonic() - start
            _terminate(proc)
            break
        time.sleep(0.05)

    worker.join(timeout=_CANCEL_DRAIN_SECONDS)
    _log_stderr(event_log, result_holder.get("stderr") or "")

    if outer_stop != "ok":
        return outer_stop, None, (result_holder.get("stdout") or None), elapsed_at_decision
    if "error" in result_holder:
        return "error", None, f"agy subprocess failed: {result_holder['error']}", None
    stdout_text = _clean_agy_stdout(result_holder.get("stdout") or "")
    outer_stop, parsed, raw_message = _derive_outer_stop_and_message(stdout_text)
    return outer_stop, parsed, raw_message, None


# --------------------------------------------------------------------------
# Main run loop (reuses DP-119 finalize/exit machinery)
# --------------------------------------------------------------------------


def _apply_config(
    config: DispatchConfig,
    worktree: Path,
    event_log: StreamingEventLog,
) -> Callable[[], None]:
    """Apply config side effects before spawn; return a restore callable.

    Stages ``AGENTS.md`` into the worktree (restored on teardown) and merges any
    declared MCP servers into agy's global config. What was applied is recorded
    to the event log. MCP merges are global and not torn down (tool capabilities
    are stable across dispatches); only the worktree ``AGENTS.md`` is restored.
    """
    restore = stage_agents_md(worktree, config)
    if config.agents_md is not None or config.agents_md_path is not None:
        event_log.write({"_config": "staged AGENTS.md into worktree"})
    ensured = ensure_mcp_servers(config.mcp_servers)
    if ensured:
        event_log.write({"_config": "ensured mcp_servers", "servers": ensured})
    return restore


def _warn_if_orchestrator_root(worktree: Path, event_log: StreamingEventLog) -> None:
    """Warn when dispatching into the orchestrator's own repo root.

    A code agent loose in the dir that holds this dispatcher can see — and run —
    the orchestrator's source and tests, which causes the agent to nerd-snipe
    itself into exploring its own plumbing instead of doing the task (observed
    2026-05-30: the agent ran our pytest suite in a loop until timeout). Dispatch
    into an isolated worktree, never the repo root.
    """
    repo_root = Path(__file__).resolve().parent.parent
    if worktree.resolve() == repo_root:
        msg = (
            f"worktree is the orchestrator repo root ({repo_root}); the agent "
            f"can see/run the dispatcher itself — use an isolated worktree"
        )
        event_log.write({"_warning": msg})
        sys.stderr.write(f"WARNING: {msg}\n")


def run_dispatch(
    *,
    packet_text: str,
    worktree: Path,
    out_dir: Path,
    session_id: str,
    wall_clock_seconds: float,
    spawn_fn: Callable[[Path], subprocess.Popen[str]] | None = None,
    model: str | None = None,
    config: DispatchConfig | None = None,
) -> int:
    """Execute one sprint attempt via ``agy -p``. Returns the process exit code.

    Test seam: ``spawn_fn`` lets the suite swap in a fake-agy child. Wall-clock
    is in seconds (``main()`` converts from minutes). ``config`` carries the
    optional dispatcher config (model / AGENTS.md / MCP servers); an explicit
    ``model`` arg overrides ``config.model``.
    """
    if config is None:
        config = DispatchConfig()
    effective_model = model if model is not None else config.model

    if spawn_fn is None:
        def spawn_fn(wt: Path) -> subprocess.Popen[str]:
            return spawn_agy(wt, packet_text, wall_clock_seconds, model=effective_model)

    out_dir.mkdir(parents=True, exist_ok=True)
    event_log = StreamingEventLog(out_dir / "events.jsonl")
    cancel_flag = threading.Event()
    prev_handlers = _install_cancel_handlers(cancel_flag)

    proc: subprocess.Popen[str] | None = None
    restore_context: Callable[[], None] = lambda: None
    start = time.monotonic()

    try:
        _warn_if_orchestrator_root(worktree, event_log)
        if effective_model is not None:
            event_log.write({
                "_warning": "model override set; --model can trigger print-mode "
                            "loops/timeouts on the OAuth tier",
                "model": effective_model,
            })
        try:
            restore_context = _apply_config(config, worktree, event_log)
        except DispatchConfigError as e:
            return _finalize_and_exit(
                out_dir, event_log, session_id, start,
                "error", None, f"config failed: {e}",
            )

        try:
            proc = spawn_fn(worktree)
        except Exception as e:
            return _finalize_and_exit(
                out_dir, event_log, session_id, start,
                "error", None, f"spawn failed: {e}",
            )

        outer_stop, parsed, raw_message, elapsed_at_decision = _drive_subprocess(
            proc, event_log, wall_clock_seconds, cancel_flag, start,
        )
        return _finalize_and_exit(
            out_dir, event_log, session_id, start,
            outer_stop, parsed, raw_message,
            elapsed_override=elapsed_at_decision,
        )
    finally:
        _restore_handlers(prev_handlers)
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        restore_context()
        event_log.close()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agy_dispatch",
        description="Dispatch one sprint attempt to the agy CLI via 'agy -p'.",
    )
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--session-id", type=str, required=True)
    # Accept float so tests can use sub-minute caps without a hidden flag.
    parser.add_argument("--wall-clock", type=float, default=10)
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to a dispatcher config JSON "
                             "(model / agents_md / mcp_servers). See "
                             "scripts/agy_config.py for the schema.")
    parser.add_argument("--model", type=str, default=None,
                        help="Passed to agy as --model (e.g. "
                             "MODEL_GOOGLE_GEMINI_2_5_PRO); overrides "
                             "config.model. WARNING: on the OAuth print tier this "
                             "is unreliable and can trigger diagnostic loops / "
                             "timeouts; leave unset to use agy's default.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        packet_text = args.packet.read_text(encoding="utf-8")
    except OSError as e:
        sys.stderr.write(f"failed reading packet {args.packet}: {e}\n")
        return 1
    config: DispatchConfig | None = None
    if args.config is not None:
        try:
            config = load_dispatch_config(args.config)
        except DispatchConfigError as e:
            sys.stderr.write(f"{e}\n")
            return 1
    return run_dispatch(
        packet_text=packet_text,
        worktree=args.worktree,
        out_dir=args.out_dir,
        session_id=args.session_id,
        wall_clock_seconds=float(args.wall_clock) * 60.0,
        model=args.model,
        config=config,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
