"""Gemini SDK-based dispatcher v2.

Runs a subagent task inside a clean worktree using the google-antigravity SDK.
It runs autonomously without requiring interaction/permission prompting from the host,
and outputs a simple greppable final response.

Exit codes:
- 0   — ran successfully
- 1   — dispatcher-internal error (imports / config / execution failure)
- 124 — wall-clock timeout
- 130 — SIGTERM / SIGINT (cancelled by parent)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import the SDK components
try:
    from google.antigravity import Agent, LocalAgentConfig, CapabilitiesConfig
    from google.antigravity.hooks.policy import allow
except ImportError as e:
    sys.stderr.write(f"Failed to import google-antigravity SDK: {e}\n")
    # We will exit with 1 to indicate dependency/setup failure
    sys.exit(1)

_EXIT_CODES = {
    "timeout": 124,
    "cancelled": 130,
    "error": 1,
}

_SUMMARY_BYTE_CAP = 1024
_RAW_MESSAGE_BYTE_CAP = 2048


def _truncate_str_bytes(s: str, cap: int) -> str:
    encoded = s.encode("utf-8")
    if len(encoded) <= cap:
        return s
    return encoded[:cap].decode("utf-8", errors="ignore")


def parse_output(text: str) -> dict[str, Any]:
    """Parse the subagent's final output for success status.
    
    Supports the new simple 'RESULT: SUCCESS' / 'RESULT: FAILED' pattern,
    and falls back to looking for a '## Result' block if present.
    """
    # Initialize default summary fields
    result = {
        "stop_reason": "failed",
        "files_modified": [],
        "key_changes": [],
        "acceptance_self_check": "skipped",
        "blockers": [],
        "parse_failed": False,
    }
    
    # 1. Look for the simple greppable signature
    if "RESULT: SUCCESS" in text:
        result["stop_reason"] = "ok"
        result["acceptance_self_check"] = "ran — reported success"
        return result
    elif "RESULT: FAILED" in text:
        result["stop_reason"] = "failed"
        result["acceptance_self_check"] = "ran — reported failure"
        return result

    # 2. Fallback: Parse the ## Result block if it exists
    lines = text.splitlines()
    last_heading_index = -1
    for i, line in enumerate(lines):
        if line.strip() == "## Result":
            last_heading_index = i
            
    if last_heading_index == -1:
        # No signature and no block -> we mark as parse_failed
        result["parse_failed"] = True
        return result

    # Simple block parsing logic
    current_key = None
    for line in lines[last_heading_index + 1:]:
        line = line.strip()
        if line.startswith("#"):
            break  # Reached another section
        if not line:
            continue
            
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower()
            val = val.strip()
            if key in ("stop_reason", "stopreason"):
                result["stop_reason"] = "ok" if val.lower() in ("ok", "success") else val
            elif key in ("files_modified", "filesmodified"):
                if val.lower() != "none":
                    result["files_modified"] = [f.strip() for f in val.split(",") if f.strip()]
            elif key in ("acceptance_self_check", "acceptanceselfcheck"):
                result["acceptance_self_check"] = val
            elif key in ("key_changes", "keychanges", "blockers"):
                current_key = key
                if val and val.lower() != "none":
                    result[key] = [val]
            else:
                current_key = None
        elif line.startswith("- ") and current_key in ("key_changes", "blockers"):
            bullet = line[2:].strip()
            if bullet.lower() != "none":
                result[current_key].append(bullet)
                
    return result


def build_summary(
    *,
    session_id: str,
    wall_clock_seconds: float,
    outer_stop_reason: str,
    parsed: dict[str, Any],
    raw_final_message: str | None,
) -> dict[str, Any]:
    """Build the final summary dict (capped at 1KB on happy path)."""
    summary = {
        "session_id": session_id,
        "stop_reason": outer_stop_reason,
        "wall_clock_seconds": wall_clock_seconds,
        "files_modified": parsed.get("files_modified", []),
        "key_changes": parsed.get("key_changes", []),
        "acceptance_self_check": parsed.get("acceptance_self_check", "skipped"),
        "blockers": parsed.get("blockers", []),
        "parse_failed": parsed.get("parse_failed", False),
        "raw_final_message": (
            _truncate_str_bytes(raw_final_message, _RAW_MESSAGE_BYTE_CAP)
            if raw_final_message is not None
            else None
        ),
    }

    # Trim to respect 1KB cap
    for trim_key in ("key_changes", "blockers", "files_modified"):
        while (
            len(json.dumps(summary).encode("utf-8")) > _SUMMARY_BYTE_CAP
            and summary[trim_key]
        ):
            summary[trim_key].pop()

    return summary


async def run_dispatch(
    *,
    packet_text: str,
    worktree: Path,
    out_dir: Path,
    session_id: str,
    wall_clock_seconds: float,
    model: str | None = None,
) -> int:
    """Execute the subagent task via the google-antigravity SDK."""
    out_dir.mkdir(parents=True, exist_ok=True)
    events_log_file = out_dir / "events.jsonl"

    # Configure the Agent with write capability and permissive policies (emulating --yolo)
    # Note: If the SDK supports setting the model on LocalAgentConfig or via environment variables,
    # it will be respected. We set up the Config dynamically.
    config_args = {
        "capabilities": CapabilitiesConfig(write=True),
        "policies": [allow("*")],
    }
    
    # Optional system instructions to guide the subagent on output format
    config_args["system_instructions"] = (
        "You are an autonomous subagent working in a clean development worktree.\n"
        "Your task is self-contained. Execute the requested tasks, run tests to verify your progress, "
        "and edit files as required.\n"
        "Do not ask for permission before running tools or commands.\n"
        "At the very end of your final turn, you MUST write a single line summarizing the status:\n"
        "RESULT: SUCCESS (if tasks are complete and tests pass)\n"
        "RESULT: FAILED (if tasks could not be completed or tests failed)"
    )

    config = LocalAgentConfig(**config_args)
    
    # Enter target worktree CWD
    original_cwd = os.getcwd()
    os.chdir(worktree.resolve())

    start_time = asyncio.get_event_loop().time()
    outer_stop = "ok"
    final_text = ""
    raw_message = None
    parsed_block = {}

    # Signal handlers logic
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()
    
    def cancel_handler():
        nonlocal outer_stop
        outer_stop = "cancelled"
        main_task.cancel()

    for sig in ("SIGTERM", "SIGINT"):
        if hasattr(signal, sig):
            try:
                loop.add_signal_handler(getattr(signal, sig), cancel_handler)
            except (ValueError, OSError):
                pass

    try:
        async with Agent(config) as agent:
            # Replicates session creation and prompt
            response = await agent.chat(packet_text)
            
            # Start background task to log reasoning and tool actions to events.jsonl
            async def log_agent_events():
                try:
                    with open(events_log_file, "a", encoding="utf-8") as f:
                        async for thought in response.thoughts:
                            f.write(json.dumps({"type": "thought", "content": thought}, ensure_ascii=False) + "\n")
                            f.flush()
                        async for call in response.tool_calls:
                            f.write(json.dumps({"type": "tool_call", "name": call.name, "args": call.args}, ensure_ascii=False) + "\n")
                            f.flush()
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    sys.stderr.write(f"Logging task error: {e}\n")

            logger_task = asyncio.create_task(log_agent_events())

            # Read token stream with timeout check
            try:
                async def stream_output():
                    nonlocal final_text
                    async for token in response:
                        sys.stdout.write(token)
                        sys.stdout.flush()
                        final_text += token
                        # Write the token to events.jsonl for visibility
                        with open(events_log_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"type": "chunk", "text": token}, ensure_ascii=False) + "\n")
                            f.flush()

                await asyncio.wait_for(stream_output(), timeout=wall_clock_seconds)

            except asyncio.TimeoutError:
                outer_stop = "timeout"
            except asyncio.CancelledError:
                if outer_stop != "cancelled":
                    outer_stop = "cancelled"
                raise
            finally:
                logger_task.cancel()
                try:
                    await logger_task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        if outer_stop == "ok":
            outer_stop = "error"
        raw_message = f"Agent failed: {e}"
        sys.stderr.write(f"Agent failed with error: {e}\n")
    finally:
        # Restore directory and remove signal handlers
        os.chdir(original_cwd)
        for sig in ("SIGTERM", "SIGINT"):
            if hasattr(signal, sig):
                try:
                    loop.remove_signal_handler(getattr(signal, sig))
                except (ValueError, OSError):
                    pass

    elapsed = round(asyncio.get_event_loop().time() - start_time, 2)

    # 5. Process outcome and summary writing
    if outer_stop == "ok":
        parsed_block = parse_output(final_text)
        if parsed_block.get("parse_failed"):
            outer_stop = "parse-failed"
            raw_message = final_text
        else:
            outer_stop = parsed_block["stop_reason"]
    elif outer_stop in ("timeout", "cancelled"):
        # Still attempt to parse partial text in case we got a result
        parsed_block = parse_output(final_text)
        raw_message = final_text or raw_message

    summary = build_summary(
        session_id=session_id,
        wall_clock_seconds=elapsed,
        outer_stop_reason=outer_stop,
        parsed=parsed_block,
        raw_final_message=raw_message,
    )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return _EXIT_CODES.get(outer_stop, 0)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gemini_acp_dispatch_v2",
        description="Dispatch one sprint attempt to google-antigravity SDK agent.",
    )
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--session-id", type=str, required=True)
    parser.add_argument("--wall-clock", type=float, default=10.0)
    parser.add_argument("--model", type=str, default=None,
                        help="Gemini model id (e.g. gemini-3-flash-preview).")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        packet_text = args.packet.read_text(encoding="utf-8")
    except OSError as e:
        sys.stderr.write(f"failed reading packet {args.packet}: {e}\n")
        return 1

    # wall_clock is passed in minutes, convert to seconds
    wall_clock_seconds = float(args.wall_clock) * 60.0

    return asyncio.run(
        run_dispatch(
            packet_text=packet_text,
            worktree=args.worktree,
            out_dir=args.out_dir,
            session_id=args.session_id,
            wall_clock_seconds=wall_clock_seconds,
            model=args.model,
        )
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
