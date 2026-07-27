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
import errno
import json
import logging
import os
import re
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from src.llm_errors import LLMCommunicationError

logger = logging.getLogger(__name__)

#: Linux caps a *single* argv/env string at ``MAX_ARG_STRLEN`` (32 pages = 128
#: KiB); ``execve()`` refuses anything larger with ``E2BIG`` — surfacing as
#: ``OSError: [Errno 7] Argument list too long`` from ``create_subprocess_exec``.
#: The rendered prompt is the largest argv entry these routes pass (the engine
#: bounds history by *tokens* — ~131k tokens is ~0.5 MB of text, four times the
#: cap), so it must be bounded in bytes before the spawn.
MAX_ARG_STRLEN = 128 * 1024
#: Budget for a single rendered argv entry. The headroom under the hard cap is
#: per-entry slack, not a shared pool: each argv string gets its own
#: ``MAX_ARG_STRLEN`` ceiling, and the *total* (``ARG_MAX``, ~2 MB) is far away
#: even with every clamped entry at budget.
MAX_CLI_PROMPT_BYTES = 96 * 1024
#: Marker left in place of the history elided to fit the budget.
ELISION_NOTICE = "[...older conversation elided to fit the CLI prompt size limit...]"
#: Marker appended to a single block that had to be cut mid-content.
TRUNCATION_NOTICE = "[...truncated to fit the CLI prompt size limit...]"

_BLOCK_SEP = "\n\n"


class TranscriptBlock(NamedTuple):
    """One rendered *message*, kept whole.

    ``role`` is the source message's role, retained so the elision policy can
    tell a user question from a tool result. ``text`` is that message's rendered
    form — possibly several lines (an assistant turn carrying both content and
    tool calls renders as one line each), joined internally with the same
    separator used between blocks so that
    ``_BLOCK_SEP.join(b.text for b in blocks)`` reproduces the flat transcript
    byte for byte.
    """

    role: str
    text: str


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _keep_head_bytes(text: str, max_bytes: int) -> str:
    """First ``max_bytes`` UTF-8 bytes of ``text`` (never splits a codepoint)."""
    if max_bytes <= 0:
        return ""
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def _truncate_block(text: str, max_bytes: int) -> str:
    """Cut one block down to ``max_bytes``, keeping its **head**.

    Head, not tail, because every block starts with its role label
    (``User:`` / ``Assistant:`` / ``Tool(name):``). Cutting from the front
    strips that label and hands the model an unattributed fragment of, most
    likely, raw tool output — content it would then read as conversation. The
    label is the one part that must survive.
    """
    marker = _BLOCK_SEP + TRUNCATION_NOTICE
    room = max_bytes - _utf8_len(marker)
    if room <= 0:
        return _keep_head_bytes(text, max_bytes)
    return _keep_head_bytes(text, room) + marker


def clamp_cli_arg(text: str, max_bytes: int = MAX_CLI_PROMPT_BYTES,
                  label: str = "argument") -> str:
    """Bound a standalone argv entry that has no block structure to elide.

    Used for the cc route's ``--system-prompt`` value, which travels as its own
    argv entry and is therefore subject to the same ``MAX_ARG_STRLEN`` ceiling
    as the prompt. Clamping it is strictly better than the alternative (an
    ``E2BIG`` spawn failure that loses the turn outright), but a truncated
    system prompt is a degraded prompt — hence the warning.
    """
    if _utf8_len(text) <= max_bytes:
        return text
    logger.warning(
        "CLI %s exceeded %d bytes (%d) — truncating; the model will see a "
        "shortened instruction block.", label, max_bytes, _utf8_len(text),
    )
    return _truncate_block(text, max_bytes)


def fit_cli_prompt(
    preamble: str,
    blocks: Sequence[TranscriptBlock],
    max_bytes: int = MAX_CLI_PROMPT_BYTES,
) -> Tuple[str, int]:
    """Join ``preamble`` + rendered ``blocks`` into one prompt fitting ``max_bytes``.

    The agy / cc routes hand the whole prompt to the CLI as a single argv entry,
    so an oversized prompt is not a slow call — it is an ``E2BIG`` spawn failure
    (``OSError`` errno 7) before the CLI ever runs. The CLIs expose no stdin or
    prompt-file transport (``agy --print`` / ``claude -p`` take the prompt as the
    flag's value), so the payload has to fit the OS contract.

    ``preamble`` (system prompt + tool protocol) is required for a correct answer
    and is kept whole. History is dropped OLDEST-first, one **message** at a
    time, mirroring how the engine's token-budget pruning drops the oldest turns.

    Elision is message-granular on purpose: the blocks arrive pre-split by
    ``render_transcript_blocks`` rather than being recovered from the flat
    transcript by splitting on blank lines. A message's own content can contain
    blank lines — multi-paragraph user text, pretty-printed JSON tool results —
    so re-splitting the rendered string yields *fragments*, not turns, and the
    cut lands mid-message.

    When not even the newest message fits, the newest **user** block is
    preferred over a newer tool result: in the tool loop the last block is the
    tool output, and keeping it while dropping the question it answers leaves
    the model with a blob and no task. Returns ``(prompt, dropped_blocks)``.
    """
    texts = [b.text for b in blocks]
    parts = [p for p in (preamble, _BLOCK_SEP.join(texts) if texts else "") if p]
    prompt = _BLOCK_SEP.join(parts)
    if _utf8_len(prompt) <= max_bytes:
        return prompt, 0

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
        cost = _utf8_len(block.text) + (len(_BLOCK_SEP) if kept else 0)
        if used + cost > budget:
            break
        kept.insert(0, block.text)
        used += cost
    dropped = len(blocks) - len(kept)

    if not kept and blocks:
        # Even the newest message alone overflows. Keep the newest *user* turn
        # if there is one — in the tool loop the final block is the tool result,
        # and answering is impossible without the question. Fall back to the
        # newest block of any role.
        anchor = next((b for b in reversed(blocks) if b.role == "user"), blocks[-1])
        kept = [_truncate_block(anchor.text, budget)]
        dropped = len(blocks) - 1
        logger.warning(
            "CLI prompt: no history block fits in %d bytes — kept a truncated "
            "%r block.", budget, anchor.role,
        )

    prompt = _BLOCK_SEP.join(
        [p for p in (preamble, ELISION_NOTICE, _BLOCK_SEP.join(kept)) if p]
    )
    logger.warning(
        "CLI prompt exceeded %d bytes — elided %d oldest history message(s).",
        max_bytes, dropped,
    )
    return prompt, dropped


def render_transcript_blocks(history: List[Dict[str, Any]]) -> List[TranscriptBlock]:
    """Render ``history`` to one :class:`TranscriptBlock` per message.

    This is the structured form of :func:`render_transcript` — same rendering,
    but the message boundaries are preserved instead of being flattened away, so
    the argv-budget elision can drop whole turns. Messages that render to
    nothing (an assistant turn with neither content nor tool calls) contribute
    no block, matching the flat renderer's line output exactly.
    """
    blocks: List[TranscriptBlock] = []
    for item in history:
        role = item.get("role")
        lines: List[str] = []
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
        if lines:
            blocks.append(TranscriptBlock(role=role or "user", text=_BLOCK_SEP.join(lines)))
    return blocks


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

    Delegates to :func:`render_transcript_blocks` and flattens: one renderer,
    so the flat transcript and the block list can never disagree about what a
    message looks like.
    """
    return _BLOCK_SEP.join(b.text for b in render_transcript_blocks(history))


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
            # Surface any spawn failure as a provider error rather than an
            # unhandled OSError that takes the whole turn down — but say which
            # failure it was. E2BIG (oversized argv/env) is the one the byte
            # budget exists for, so only that message quotes the payload size;
            # FileNotFoundError and PermissionError are OSError subclasses too,
            # and pointing those at the size limit sends the investigation the
            # wrong way.
            if e.errno == errno.E2BIG:
                argv_bytes = sum(len(a.encode("utf-8")) for a in args)
                raise LLMCommunicationError(
                    f"{label} CLI could not be spawned: argv/env too large for "
                    f"execve ({e.strerror or e}); argv payload was {argv_bytes} bytes."
                ) from e
            raise LLMCommunicationError(
                f"{label} CLI could not be spawned: {e.strerror or e}."
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
