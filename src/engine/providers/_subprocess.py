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
from typing import Any, Dict, List, Optional, Tuple

from src.llm_errors import LLMCommunicationError

logger = logging.getLogger(__name__)

#: Linux caps a *single* argv/env string at ``MAX_ARG_STRLEN`` (32 pages = 128
#: KiB); ``execve()`` refuses anything larger with ``E2BIG`` — surfacing as
#: ``OSError: [Errno 7] Argument list too long`` from ``create_subprocess_exec``.
#: The rendered prompt is the only unbounded argv entry these routes pass (the
#: engine bounds history by *tokens* — ~131k tokens is ~0.5 MB of text, four
#: times the cap), so it must be bounded in bytes before the spawn.
MAX_ARG_STRLEN = 128 * 1024
#: Budget for the rendered prompt. Headroom under the hard cap covers the other
#: argv entries and the child env, which share the overall ``ARG_MAX`` budget.
MAX_CLI_PROMPT_BYTES = 96 * 1024
#: Marker left in place of the history elided to fit the budget.
ELISION_NOTICE = "[...older conversation elided to fit the CLI prompt size limit...]"

_BLOCK_SEP = "\n\n"


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _keep_head_bytes(text: str, max_bytes: int) -> str:
    """First ``max_bytes`` UTF-8 bytes of ``text`` (never splits a codepoint)."""
    if max_bytes <= 0:
        return ""
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def _keep_tail_bytes(text: str, max_bytes: int) -> str:
    """Last ``max_bytes`` UTF-8 bytes of ``text`` (never splits a codepoint)."""
    if max_bytes <= 0:
        return ""
    return text.encode("utf-8")[-max_bytes:].decode("utf-8", errors="ignore")


def fit_cli_prompt(
    preamble: str, transcript: str, max_bytes: int = MAX_CLI_PROMPT_BYTES
) -> Tuple[str, int]:
    """Join ``preamble`` + ``transcript`` into one prompt that fits ``max_bytes``.

    The agy / cc routes hand the whole prompt to the CLI as a single argv entry,
    so an oversized prompt is not a slow call — it is an ``E2BIG`` spawn failure
    (``OSError`` errno 7) before the CLI ever runs. The CLIs expose no stdin or
    prompt-file transport (``agy --print`` / ``claude -p`` take the prompt as the
    flag's value), so the payload has to fit the OS contract.

    ``preamble`` (system prompt + tool protocol) is required for a correct answer
    and is kept whole; the ``transcript`` is trimmed OLDEST-first, block by block
    (``render_transcript`` joins turns with a blank line), mirroring how the
    engine's token-budget pruning drops the oldest turns. Returns
    ``(prompt, dropped_blocks)``.
    """
    parts = [p for p in (preamble, transcript) if p]
    prompt = _BLOCK_SEP.join(parts)
    if _utf8_len(prompt) <= max_bytes:
        return prompt, 0

    blocks = transcript.split(_BLOCK_SEP) if transcript else []
    # Room left for history once the preamble and the elision marker are paid for.
    fixed = _utf8_len(ELISION_NOTICE) + len(_BLOCK_SEP)
    if preamble:
        fixed += _utf8_len(preamble) + len(_BLOCK_SEP)
    budget = max_bytes - fixed
    if budget <= 0:
        # Pathological: the preamble alone blows the limit. Nothing else fits.
        logger.warning(
            "CLI prompt preamble alone exceeds %d bytes — truncating it.", max_bytes
        )
        return _keep_head_bytes(prompt, max_bytes), len(blocks)

    kept: List[str] = []
    used = 0
    for block in reversed(blocks):  # newest turns first
        cost = _utf8_len(block) + (len(_BLOCK_SEP) if kept else 0)
        if used + cost > budget:
            break
        kept.insert(0, block)
        used += cost
    dropped = len(blocks) - len(kept)
    if not kept and blocks:
        # Even the newest turn alone overflows: keep its tail (the most recent
        # text) rather than dropping the current turn entirely.
        kept = [_keep_tail_bytes(blocks[-1], budget)]
        dropped = len(blocks) - 1
        logger.warning(
            "CLI prompt: newest history block alone exceeds %d bytes — kept its tail.",
            budget,
        )

    prompt = _BLOCK_SEP.join(
        [p for p in (preamble, ELISION_NOTICE, _BLOCK_SEP.join(kept)) if p]
    )
    logger.warning(
        "CLI prompt exceeded %d bytes — elided %d oldest history block(s).",
        max_bytes, dropped,
    )
    return prompt, dropped


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
        except OSError as e:
            # execve refuses an oversized argv/env with E2BIG ("Argument list too
            # long"). Callers clamp the prompt (`fit_cli_prompt`), so reaching
            # here means some other argv entry or the child env blew the budget —
            # surface it as a provider error instead of an unhandled OSError that
            # takes the whole turn down.
            argv_bytes = sum(len(a.encode("utf-8")) for a in args)
            raise LLMCommunicationError(
                f"{label} CLI could not be spawned ({e.strerror or e}); "
                f"argv payload was {argv_bytes} bytes."
            ) from e
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
