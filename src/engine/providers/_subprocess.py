# src/engine/providers/_subprocess.py
"""Subprocess machinery shared by the agy and cc (Claude Code) providers (DP-244).

Both providers are POSIX-only one-shot CLI routes: they render the full message
history into a single transcript, spawn a CLI in a workspace dir, and adapt the
process's exit-time output into the unified event shape. The pieces that are
literally shared — the subprocess runner, the transcript renderer, and the
workspace-name sanitiser — live here so agy.py and cc.py don't duplicate them.

NOTE: this module carries a whole-file ``ignore_errors`` in mypy.ini — the
process-group kill path (``os.killpg`` / ``os.getpgid`` / ``signal.SIGKILL`` /
``start_new_session``) is POSIX-only and trips the Windows-typed os/signal stubs,
the same legacy noise that kept ``[mypy-src.engine.driver]`` ignored before the
extraction.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from src.llm_errors import LLMCommunicationError

logger = logging.getLogger(__name__)


def render_transcript(history: List[Dict[str, Any]]) -> str:
    """Flatten a message history into a single role-tagged transcript for the
    agy / cc routes.

    The subprocess CLIs accept only one prompt turn and offer no API to seed
    prior assistant turns, while the engine is stateless and rebuilds the full
    context on every call. We therefore render the entire ``history`` — which
    already ends with the current user turn (see ``_extract_system_prompt``) —
    into one deterministic, auditable transcript so the CLI contributes nothing
    of its own. This is also what lets the engine's multi-turn tool loop work: a
    ``tool``-role result from a prior iteration is just another rendered line.

    The system prompt is delivered separately (agy: ``CustomSystemInstructions``;
    cc: ``--system-prompt``) and is intentionally not included here.
    ``current_message["text"]`` is a duplicate of the final user turn already
    present in ``history``, so it is not appended (doing so would duplicate the
    last message).
    """
    lines: List[str] = []
    for item in history:
        role = item.get("role")
        if role == "tool":
            lines.append(f"Tool({item.get('name', 'unknown')}): {item.get('content', '')}")
        elif role == "assistant":
            if item.get("content"):
                lines.append(f"Assistant: {item['content']}")
            for call in item.get("tool_calls", []) or []:
                args = json.dumps(call.get("arguments", {}), ensure_ascii=False)
                lines.append(f"Assistant (tool call {call.get('name', 'unknown')}): {args}")
        else:  # user (and any unlabeled turn) renders as the user
            lines.append(f"User: {item.get('content', '')}")
    return "\n\n".join(lines)


def sanitize_workspace_name(persona_name: Optional[str]) -> Optional[str]:
    """Persona names come from config and may contain path separators or other
    filesystem-hostile characters; reduce to a safe slug. Returns None when
    nothing usable remains (caller falls back to global)."""
    if not persona_name:
        return None
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", persona_name).strip("._")
    return slug or None


async def exec_cli(binary: str, args: List[str], workspace_dir: str, timeout: float,
                   label: str = "agy", env: Optional[Dict[str, str]] = None) -> str:
    # `label` names the provider in error messages — this CLI runner is shared
    # by the agy and cc (Claude Code) routes, so a failure must point at the
    # route the caller actually invoked. `env` overrides the child environment
    # (cc passes a subscription-scrubbed env; agy passes None = inherit unchanged).
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace_dir,
            env=env,
            start_new_session=True
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError as e:
            raise LLMCommunicationError(f"{label} CLI timed out after {timeout} seconds.") from e

        if proc.returncode != 0:
            stderr_excerpt = stderr.decode("utf-8", errors="replace").strip()
            excerpt = stderr_excerpt[-200:] if len(stderr_excerpt) > 200 else stderr_excerpt
            raise LLMCommunicationError(
                f"{label} CLI failed with exit code {proc.returncode}. Stderr: {excerpt}"
            )

        return stdout.decode("utf-8", errors="replace")
    finally:
        if proc is not None:
            try:
                import signal
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
