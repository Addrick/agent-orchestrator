"""Gemini ACP dispatcher (DP-119).

One process per sprint attempt: spawns ``gemini --acp`` as a child, drives a
single ACP session over JSON-RPC on stdio, streams every event to
``events.jsonl``, parses the final ``## Result`` block out of gemini's last
message, and writes ``summary.json`` (≤1KB) as the only thing Claude reads.

Spec: ``memory/project/specs/2026-05-19-gemini-acp-transport-design.md``
Plan: ``memory/project/plans/DP-119-acp-dispatcher.md``

Wire-format details (initialize shape, ``mcpServers:[]`` on session/new,
array-wrapped prompt content) are sourced from the working
``GeminiACPClient`` in ``eval_harnesses/suites/memory_recall/lme_judge.py``,
which has been roundtripping gemini-cli 0.42.0 since 2026-05-19. We don't
authenticate; OAuth-cached creds make the step a no-op in practice.

Exit codes:
- 0   — ran end-to-end (gemini's own pass/fail is in ``summary.json``)
- 1   — dispatcher-internal error (spawn / handshake / IO)
- 124 — wall-clock timeout
- 130 — SIGTERM / SIGINT (cancelled by parent)
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

_ALLOWED_PARSED_STOP_REASONS = frozenset({"ok", "failed", "need-scope-expansion"})
_REQUIRED_RESULT_KEYS = (
    "stop_reason",
    "files_modified",
    "key_changes",
    "acceptance_self_check",
    "blockers",
)
_LIST_KEYS = frozenset({"files_modified", "key_changes", "blockers"})
_SUMMARY_BYTE_CAP = 1024
_RAW_MESSAGE_BYTE_CAP = 2048
_CANCEL_DRAIN_SECONDS = 5.0
_HANDSHAKE_TIMEOUT = 30.0


class ResultBlockParseError(ValueError):
    """Raised when a ``## Result`` block is present but malformed."""


# --------------------------------------------------------------------------
# Result-block parsing (Phase 1)
# --------------------------------------------------------------------------


def _extract_body_lines(text: str) -> list[str] | None:
    """Return the lines under the last ``## Result`` heading, or None."""
    lines = text.splitlines()
    last_heading_index = -1
    for i, line in enumerate(lines):
        if line.strip() == "## Result":
            last_heading_index = i
    if last_heading_index == -1:
        return None
    body: list[str] = []
    for line in lines[last_heading_index + 1:]:
        if line.startswith("## ") or line.startswith("# "):
            break
        body.append(line)
    return body


def _handle_bullet_line(
    stripped: str, parsed: dict[str, Any], current_list_key: str | None
) -> None:
    if current_list_key is None:
        raise ResultBlockParseError(
            f"Bullet '{stripped}' appeared before any list-valued key"
        )
    item = stripped[2:].strip()
    existing = parsed.setdefault(current_list_key, [])
    if not isinstance(existing, list):
        raise ResultBlockParseError(f"Internal: key '{current_list_key}' is not a list")
    if item.lower() != "none":
        existing.append(item)


def _handle_key_value_line(line: str, parsed: dict[str, Any]) -> str | None:
    if ":" not in line:
        raise ResultBlockParseError(f"Unparseable line in Result block: {line!r}")
    key, _, value = line.partition(":")
    key = key.strip()
    value = value.strip()
    if not key:
        raise ResultBlockParseError(f"Empty key in Result block: {line!r}")
    if key in _LIST_KEYS:
        if value == "":
            parsed[key] = []
            return key
        if value.lower() == "none":
            parsed[key] = []
            return None
        parsed[key] = [item.strip() for item in value.split(",") if item.strip()]
        return None
    parsed[key] = value
    return None


def _validate_parsed(parsed: dict[str, Any]) -> None:
    for required in _REQUIRED_RESULT_KEYS:
        if required not in parsed:
            raise ResultBlockParseError(f"Missing required key: {required}")
    stop_reason = parsed["stop_reason"]
    if not isinstance(stop_reason, str) or stop_reason not in _ALLOWED_PARSED_STOP_REASONS:
        raise ResultBlockParseError(
            f"Unknown stop_reason: {stop_reason!r}; "
            f"allowed: {sorted(_ALLOWED_PARSED_STOP_REASONS)}"
        )


def parse_result_block(text: str) -> dict[str, Any] | None:
    """Parse the final ``## Result`` block from a gemini turn.

    Returns the parsed dict, or ``None`` if no ``## Result`` heading is
    present. Raises :class:`ResultBlockParseError` if a heading is present
    but the block is malformed. Last ``## Result`` wins when multiple appear.
    """
    body_lines = _extract_body_lines(text)
    if body_lines is None:
        return None
    parsed: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in body_lines:
        line = raw_line.rstrip()
        if not line.strip():
            current_list_key = None
            continue
        stripped = line.lstrip()
        if stripped.startswith("- "):
            _handle_bullet_line(stripped, parsed, current_list_key)
            continue
        current_list_key = _handle_key_value_line(line, parsed)
    _validate_parsed(parsed)
    return parsed


# --------------------------------------------------------------------------
# Summary building (Phase 1)
# --------------------------------------------------------------------------


def _truncate_str_bytes(s: str, cap: int) -> str:
    encoded = s.encode("utf-8")
    if len(encoded) <= cap:
        return s
    return encoded[:cap].decode("utf-8", errors="ignore")


def build_summary(
    *,
    session_id: str,
    wall_clock_seconds: float,
    outer_stop_reason: str,
    parsed: dict[str, Any] | None,
    raw_final_message: str | None,
) -> dict[str, Any]:
    """Build the final summary dict (≤1KB on the happy path)."""
    summary: dict[str, Any] = {
        "session_id": session_id,
        "stop_reason": outer_stop_reason,
        "wall_clock_seconds": wall_clock_seconds,
        "files_modified": [],
        "key_changes": [],
        "acceptance_self_check": "",
        "blockers": [],
        "parse_failed": parsed is None,
        "raw_final_message": (
            _truncate_str_bytes(raw_final_message, _RAW_MESSAGE_BYTE_CAP)
            if raw_final_message is not None
            else None
        ),
    }

    if parsed is None:
        return summary

    files = parsed.get("files_modified", [])
    changes = parsed.get("key_changes", [])
    blockers = parsed.get("blockers", [])
    summary["files_modified"] = list(files) if isinstance(files, list) else []
    summary["key_changes"] = list(changes) if isinstance(changes, list) else []
    summary["blockers"] = list(blockers) if isinstance(blockers, list) else []
    acc = parsed.get("acceptance_self_check", "")
    summary["acceptance_self_check"] = acc if isinstance(acc, str) else ""

    for trim_key in ("key_changes", "blockers", "files_modified"):
        while (
            len(json.dumps(summary).encode("utf-8")) > _SUMMARY_BYTE_CAP
            and summary[trim_key]
        ):
            summary[trim_key].pop()

    return summary


def write_outputs(
    out_dir: Path,
    summary: dict[str, Any],
    events: list[dict[str, Any]],
) -> None:
    """Write ``summary.json`` and ``events.jsonl`` into ``out_dir``.

    Test/helper path. The live dispatcher streams events directly via
    ``StreamingEventLog`` to avoid buffering an unbounded list in RAM.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    events_text = "".join(json.dumps(e) + "\n" for e in events)
    (out_dir / "events.jsonl").write_text(events_text)


# --------------------------------------------------------------------------
# Event log (Phase 2)
# --------------------------------------------------------------------------


class StreamingEventLog:
    """Thread-safe newline-delimited JSON sink for ACP events."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = path.open("w", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False) + "\n"
        with self._lock:
            self._fp.write(line)
            self._fp.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._fp.close()
            except Exception:
                pass


# --------------------------------------------------------------------------
# ACP client (Phase 2)
# --------------------------------------------------------------------------


EventSink = Callable[[dict[str, Any]], None]


class AcpClient:
    """Synchronous JSON-RPC client for one gemini ACP child process.

    Wire format learned from the lme_judge ``GeminiACPClient`` (working
    against gemini-cli 0.42.0, 2026-05-19):

    - line-delimited JSON-RPC 2.0, one object per line
    - ``initialize`` params: ``{"protocolVersion": 1, "capabilities": {}}``
    - ``session/new`` params: ``{"cwd": <abs>, "mcpServers": []}``
    - ``session/prompt`` params: ``{"sessionId": ..., "prompt":
      [{"type": "text", "text": ...}]}``; the response is the terminal
      event, carrying ``result.stopReason``
    - assistant text arrives via ``session/update`` notifications with
      ``params.update.sessionUpdate == "agent_message_chunk"``

    Reader thread writes EVERY inbound message to ``event_sink`` so the
    raw transcript ends up in ``events.jsonl``.
    """

    def __init__(self, proc: subprocess.Popen[str], event_sink: EventSink) -> None:
        self._proc = proc
        self._event_sink = event_sink
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._id_counter = 0
        self._lock = threading.Lock()
        self._current_message: list[str] = []
        self.session_id: Optional[str] = None
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="acp-reader"
        )
        self._reader_thread.start()

    # ---- inbound side -----------------------------------------------------

    def _reader_loop(self) -> None:
        if self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Non-JSON noise (banners, warnings) — log as opaque event
                # and keep going.
                try:
                    self._event_sink({"_raw_non_json": line})
                except Exception:
                    pass
                continue
            try:
                self._event_sink(data)
            except Exception:
                pass
            self._dispatch_inbound(data)

    def _dispatch_inbound(self, data: dict[str, Any]) -> None:
        method = data.get("method")
        msg_id = data.get("id")

        # Notification (streaming update): no id, has method.
        if method and msg_id is None:
            if method == "session/update":
                upd = (data.get("params") or {}).get("update") or {}
                if upd.get("sessionUpdate") == "agent_message_chunk":
                    text = (upd.get("content") or {}).get("text") or ""
                    if text:
                        self._current_message.append(text)
            return

        # Server-to-client request: has method AND id. The only one we
        # expect is session/request_permission; auto-approve everything.
        if method and msg_id is not None:
            if method == "session/request_permission":
                self._send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"outcome": {"outcome": "selected", "optionId": "allow_always"}},
                })
            else:
                # Unknown server request — return empty result, don't crash.
                self._send({"jsonrpc": "2.0", "id": msg_id, "result": {}})
            return

        # Response to one of our requests: has id, no method.
        if msg_id is not None:
            self._responses.put(data)

    # ---- outbound side ----------------------------------------------------

    def _send(self, msg: dict[str, Any]) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("gemini stdin is closed")
        line = json.dumps(msg) + "\n"
        with self._lock:
            try:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError) as e:
                raise RuntimeError(f"failed writing to gemini stdin: {e}") from e

    def _call(
        self, method: str, params: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        with self._lock:
            self._id_counter += 1
            msg_id = self._id_counter
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = self._responses.get(timeout=0.25)
            except queue.Empty:
                if self._proc.poll() is not None:
                    raise RuntimeError(
                        f"gemini exited (code {self._proc.returncode}) "
                        f"before responding to {method}"
                    )
                continue
            if resp.get("id") == msg_id:
                if "error" in resp:
                    raise RuntimeError(f"ACP error in {method}: {resp['error']}")
                return resp
            # Mismatched id — non-sequential call would cause this; we don't.
        raise TimeoutError(f"ACP call {method} timed out after {timeout}s")

    # ---- public surface ---------------------------------------------------

    def initialize(self) -> dict[str, Any]:
        return self._call(
            "initialize",
            {"protocolVersion": 1, "capabilities": {}},
            timeout=_HANDSHAKE_TIMEOUT,
        )

    def session_new(self, cwd: str) -> str:
        resp = self._call(
            "session/new",
            {"cwd": cwd, "mcpServers": []},
            timeout=_HANDSHAKE_TIMEOUT,
        )
        sid = ((resp.get("result") or {}).get("sessionId"))
        if not isinstance(sid, str):
            raise RuntimeError(f"session/new returned no sessionId: {resp}")
        self.session_id = sid
        self._current_message = []
        return sid

    def session_prompt(self, text: str, timeout: float) -> dict[str, Any]:
        if self.session_id is None:
            raise RuntimeError("session_prompt called before session_new")
        self._current_message = []
        return self._call(
            "session/prompt",
            {
                "sessionId": self.session_id,
                "prompt": [{"type": "text", "text": text}],
            },
            timeout=timeout,
        )

    def session_cancel(self) -> None:
        """Fire-and-forget cancel. Doesn't wait for a response.

        Per the 2026-05-19 open question, we don't yet know whether cancel
        resolves the in-flight prompt with a stopReason or an error. Either
        way, the caller is already in shutdown mode by the time this runs.
        """
        if self.session_id is None:
            return
        with self._lock:
            self._id_counter += 1
            msg_id = self._id_counter
        try:
            self._send({
                "jsonrpc": "2.0",
                "id": msg_id,
                "method": "session/cancel",
                "params": {"sessionId": self.session_id},
            })
        except Exception:
            pass

    def final_message(self) -> str:
        return "".join(self._current_message)

    def close(self, drain_seconds: float = _CANCEL_DRAIN_SECONDS) -> None:
        # Close stdin to let gemini wind down cleanly.
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=drain_seconds)
        except subprocess.TimeoutExpired:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        # Reader exits naturally when stdout closes.
        self._reader_thread.join(timeout=2.0)


# --------------------------------------------------------------------------
# Process spawn (Phase 3)
# --------------------------------------------------------------------------


def spawn_gemini(worktree: Path) -> subprocess.Popen[str]:
    """Spawn ``gemini --acp`` in the worktree.

    Windows: ``shutil.which`` resolves the ``.CMD`` shim correctly. See
    ``memory/gemini_cli_subprocess_recipe.md``.
    """
    binary = shutil.which("gemini")
    if not binary:
        raise RuntimeError("gemini CLI not found on PATH")
    env = os.environ.copy()
    return subprocess.Popen(
        [binary, "--acp", "--skip-trust", "--yolo"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(worktree.resolve()),
        env=env,
    )


# --------------------------------------------------------------------------
# Main run loop (Phases 2 + 3)
# --------------------------------------------------------------------------


def _derive_outer_stop_and_message(
    final_text: str,
) -> tuple[str, dict[str, Any] | None, str | None]:
    """From gemini's final assistant text, derive (outer_stop, parsed, raw).

    Maps the parsed block's ``stop_reason`` into ``outer_stop`` directly
    (ok / failed / need-scope-expansion). Missing or malformed blocks
    yield ``parse-failed`` with ``raw`` set to the truncated final text.
    """
    try:
        parsed = parse_result_block(final_text)
    except ResultBlockParseError:
        return "parse-failed", None, final_text
    if parsed is None:
        return "parse-failed", None, final_text
    return parsed["stop_reason"], parsed, None


def _install_cancel_handlers(
    flag: threading.Event,
) -> dict[int, Any]:
    """Register SIGTERM/SIGINT handlers that set ``flag``. Returns the
    previous handler map so the caller can restore them on exit."""
    prev: dict[int, Any] = {}

    def handler(_signum: int, _frame: Any) -> None:
        flag.set()

    for sig_name in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            prev[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):
            pass
    return prev


def _restore_handlers(prev: dict[int, Any]) -> None:
    for sig, h in prev.items():
        try:
            signal.signal(sig, h)
        except (ValueError, OSError):
            pass


def _drive_prompt(
    client: AcpClient,
    packet_text: str,
    wall_clock_seconds: float,
    cancel_flag: threading.Event,
    start: float,
) -> tuple[str, dict[str, Any] | None, str | None, float | None]:
    """Run one session/prompt with cancel + wall-clock monitoring.

    Returns ``(outer_stop, parsed, raw_message, elapsed_at_decision)``.
    ``elapsed_at_decision`` is None for the natural-completion path.
    """
    result_holder: dict[str, Any] = {}

    def _worker() -> None:
        try:
            result_holder["resp"] = client.session_prompt(
                packet_text, timeout=wall_clock_seconds + 30.0,
            )
        except Exception as exc:
            result_holder["error"] = exc

    worker = threading.Thread(target=_worker, daemon=True, name="acp-prompt")
    worker.start()

    outer_stop = "ok"
    elapsed_at_decision: float | None = None
    while worker.is_alive():
        if cancel_flag.is_set():
            outer_stop = "cancelled"
            elapsed_at_decision = time.monotonic() - start
            client.session_cancel()
            break
        if time.monotonic() - start > wall_clock_seconds:
            outer_stop = "timeout"
            elapsed_at_decision = time.monotonic() - start
            client.session_cancel()
            break
        time.sleep(0.05)

    worker.join(timeout=_CANCEL_DRAIN_SECONDS)

    if outer_stop != "ok":
        return outer_stop, None, client.final_message() or None, elapsed_at_decision
    if "error" in result_holder:
        return "error", None, f"session/prompt failed: {result_holder['error']}", None
    outer_stop, parsed, raw_message = _derive_outer_stop_and_message(
        client.final_message()
    )
    return outer_stop, parsed, raw_message, None


def run_dispatch(
    *,
    packet_text: str,
    worktree: Path,
    out_dir: Path,
    session_id: str,
    wall_clock_seconds: float,
    spawn_fn: Callable[[Path], subprocess.Popen[str]] | None = None,
) -> int:
    """Execute one sprint attempt. Returns the process exit code.

    Test seam: ``spawn_fn`` lets the test suite swap in a fake-gemini child.
    Wall-clock is in seconds (not minutes); ``main()`` converts from minutes.
    """
    if spawn_fn is None:
        # Look up at call time (not def time) so monkeypatch on the module
        # attribute reaches us. Default arg binding would freeze the original.
        spawn_fn = spawn_gemini

    out_dir.mkdir(parents=True, exist_ok=True)
    event_log = StreamingEventLog(out_dir / "events.jsonl")
    cancel_flag = threading.Event()
    prev_handlers = _install_cancel_handlers(cancel_flag)

    proc: subprocess.Popen[str] | None = None
    client: AcpClient | None = None
    start = time.monotonic()

    try:
        try:
            proc = spawn_fn(worktree)
        except Exception as e:
            return _finalize_and_exit(
                out_dir, event_log, session_id, start,
                "error", None, f"spawn failed: {e}",
            )

        client = AcpClient(proc, event_log.write)

        try:
            client.initialize()
            client.session_new(str(worktree.resolve()))
        except Exception as e:
            return _finalize_and_exit(
                out_dir, event_log, session_id, start,
                "error", None, f"handshake failed: {e}",
                client=client,
            )

        outer_stop, parsed, raw_message, elapsed_at_decision = _drive_prompt(
            client, packet_text, wall_clock_seconds, cancel_flag, start,
        )
        return _finalize_and_exit(
            out_dir, event_log, session_id, start,
            outer_stop, parsed, raw_message,
            client=client,
            elapsed_override=elapsed_at_decision,
        )
    finally:
        _restore_handlers(prev_handlers)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        elif proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        event_log.close()


_EXIT_CODES = {
    "timeout": 124,
    "cancelled": 130,
    "error": 1,
}


def _finalize_and_exit(
    out_dir: Path,
    event_log: StreamingEventLog,
    session_id: str,
    start: float,
    outer_stop: str,
    parsed: dict[str, Any] | None,
    raw_message: str | None,
    *,
    client: AcpClient | None = None,
    elapsed_override: float | None = None,
) -> int:
    if elapsed_override is not None:
        elapsed = round(elapsed_override, 2)
    else:
        elapsed = round(time.monotonic() - start, 2)
    summary = build_summary(
        session_id=session_id,
        wall_clock_seconds=elapsed,
        outer_stop_reason=outer_stop,
        parsed=parsed,
        raw_final_message=raw_message,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if client is not None:
        try:
            client.close()
        except Exception:
            pass
    event_log.close()
    return _EXIT_CODES.get(outer_stop, 0)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gemini_acp_dispatch",
        description="Dispatch one sprint attempt to gemini-cli over ACP.",
    )
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--session-id", type=str, required=True)
    # Accept float so tests can use sub-minute caps without a hidden flag.
    parser.add_argument("--wall-clock", type=float, default=10)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        packet_text = args.packet.read_text(encoding="utf-8")
    except OSError as e:
        sys.stderr.write(f"failed reading packet {args.packet}: {e}\n")
        return 1
    return run_dispatch(
        packet_text=packet_text,
        worktree=args.worktree,
        out_dir=args.out_dir,
        session_id=args.session_id,
        wall_clock_seconds=float(args.wall_clock) * 60.0,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
