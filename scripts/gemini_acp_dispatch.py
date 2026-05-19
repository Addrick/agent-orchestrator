"""Gemini ACP dispatcher (DP-119).

Phase 1: argparse, result-block parser, summary builder, output writer.
The JSON-RPC client and process lifecycle land in Phase 2 and Phase 3
respectively. Phase 1 ``main()`` writes a stub summary so the integration
shape is exercisable end-to-end.

See ``memory/project/specs/2026-05-19-gemini-acp-transport-design.md`` and
``memory/project/plans/DP-119-acp-dispatcher.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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


class ResultBlockParseError(ValueError):
    """Raised when a ``## Result`` block is present but malformed."""


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
    """Parse a ``key: value`` line into ``parsed``; return the active list-key
    if the line opens a bullet block (e.g. ``files_modified:`` with no value).
    """
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
    but the block is malformed (missing required key, unknown
    ``stop_reason``, etc.). When multiple ``## Result`` headings appear,
    the last one wins.
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
    """Build the final summary dict.

    ``outer_stop_reason`` is the dispatcher's view: one of ``ok``,
    ``failed``, ``timeout``, ``cancelled``, ``parse-failed``, ``error``,
    ``need-scope-expansion``. When ``parsed`` is None, the structured
    fields default to empty and ``parse_failed`` is set to True.

    Happy summaries are capped at 1KB by progressively dropping items
    from ``key_changes`` → ``blockers`` → ``files_modified``. Parse-failed
    summaries skip the cap because ``raw_final_message`` (≤2KB) is the
    payload of interest.
    """
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

    # Truncate happy summaries to 1KB. Order: key_changes, blockers, files_modified.
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
    """Write ``summary.json`` and ``events.jsonl`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    events_text = "".join(json.dumps(e) + "\n" for e in events)
    (out_dir / "events.jsonl").write_text(events_text)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gemini_acp_dispatch",
        description="Dispatch one sprint attempt to gemini-cli over ACP.",
    )
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--session-id", type=str, required=True)
    parser.add_argument("--wall-clock", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """Phase 1 stub.

    Parses args, writes a ``not-implemented`` summary so callers can
    exercise the integration shape. Phase 2 replaces the body with the
    real ACP roundtrip.
    """
    args = parse_args(argv)
    summary = build_summary(
        session_id=args.session_id,
        wall_clock_seconds=0.0,
        outer_stop_reason="not-implemented",
        parsed=None,
        raw_final_message=None,
    )
    write_outputs(args.out_dir, summary, [])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
