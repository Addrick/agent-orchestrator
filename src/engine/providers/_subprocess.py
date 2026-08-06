# src/engine/providers/_subprocess.py
"""Subprocess machinery shared by the agy and cc (Claude Code) providers (DP-244).

Both providers are one-shot CLI routes: they render the full message history
into a single transcript, spawn a CLI in a workspace dir, and adapt the
process's exit-time output into the unified event shape. The pieces that are
literally shared — the subprocess runner, the transcript renderer, and the
workspace-name sanitiser — live here so agy.py and cc.py don't duplicate them.

The runner itself is platform-agnostic (DP-324): process isolation and the
cleanup kill are the only OS-specific parts, and each has a POSIX and a Windows
implementation (``_spawn_isolation_kwargs`` / ``_kill_process_tree``).

NOTE: this module carries a whole-file ``ignore_errors`` in mypy.ini — the POSIX
kill path (``os.killpg`` / ``os.getpgid`` / ``signal.SIGKILL``) does not exist in
the Windows-typed os/signal stubs (and vice versa for the Windows creation
flags), the same legacy noise that kept ``[mypy-src.engine.driver]`` ignored
before the extraction.
"""

import asyncio
import errno
import json
import logging
import os
import re
import subprocess
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
#: Windows has no per-argument cap; ``CreateProcess`` caps the *whole* command
#: line at 32767 characters (``ERROR_FILENAME_EXCED_RANGE``, WinError 206, which
#: Python raises as ``FileNotFoundError``). That is a shared pool across every
#: argv entry and an order of magnitude tighter than the POSIX per-entry ceiling,
#: so the Windows budgets below must leave room for *all* the other args a route
#: passes (cc: ``--system-prompt``, ``--settings`` JSON, flags).
MAX_COMMAND_LINE_CHARS = 32767
#: Budget for the rendered prompt argv entry on POSIX. The headroom under the
#: hard cap is per-entry slack, not a shared pool: each argv string gets its own
#: ``MAX_ARG_STRLEN`` ceiling, and the *total* (``ARG_MAX``, ~2 MB) is far away
#: even with every clamped entry at budget.
MAX_CLI_PROMPT_BYTES_POSIX = 96 * 1024
#: Same, on Windows, where the total command line IS the ceiling — hence the far
#: smaller number: the prompt, the cc route's ``--system-prompt``, the
#: ``--settings`` JSON and the flags all come out of one 32767-char pool.
MAX_CLI_PROMPT_BYTES_WINDOWS = 20 * 1024
#: Budget for a secondary argv entry with no block structure to elide (the cc
#: route's ``--system-prompt``) on Windows. On POSIX it gets the same per-entry
#: ceiling as the prompt.
MAX_CLI_ARG_BYTES_WINDOWS = 6 * 1024


def cli_prompt_budget() -> int:
    """Byte budget for the rendered prompt on THIS host.

    Resolved per call rather than frozen at import: the two platforms differ by
    a factor of five, so a constant baked in at import time would be untestable
    for the other platform (and every test would silently assert the budget of
    whatever host the suite happens to run on).
    """
    return MAX_CLI_PROMPT_BYTES_POSIX if os.name == "posix" else MAX_CLI_PROMPT_BYTES_WINDOWS


def cli_arg_budget() -> int:
    """Byte budget for a secondary argv entry on THIS host. See
    :func:`cli_prompt_budget`."""
    return MAX_CLI_PROMPT_BYTES_POSIX if os.name == "posix" else MAX_CLI_ARG_BYTES_WINDOWS


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


def clamp_cli_arg(text: str, max_bytes: Optional[int] = None,
                  label: str = "argument") -> str:
    """Bound a standalone argv entry that has no block structure to elide.

    Used for the cc route's ``--system-prompt`` value, which travels as its own
    argv entry and is therefore subject to the same OS ceiling as the prompt
    (``MAX_ARG_STRLEN`` per entry on POSIX; the shared
    ``MAX_COMMAND_LINE_CHARS`` pool on Windows). Clamping it is strictly better
    than the alternative (a spawn failure that loses the turn outright), but a
    truncated system prompt is a degraded prompt — hence the warning.
    """
    if max_bytes is None:
        max_bytes = cli_arg_budget()
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
    max_bytes: Optional[int] = None,
) -> Tuple[str, int]:
    """Join ``preamble`` + rendered ``blocks`` into one prompt fitting ``max_bytes``.

    The agy / cc routes hand the whole prompt to the CLI as a single argv entry,
    so an oversized prompt is not a slow call — it is a spawn failure before the
    CLI ever runs (POSIX: ``E2BIG``, ``OSError`` errno 7; Windows: WinError 206,
    the 32767-char command-line cap). The CLIs expose no stdin or
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

    ``max_bytes`` defaults to this host's budget (:func:`cli_prompt_budget`).
    """
    if max_bytes is None:
        max_bytes = cli_prompt_budget()
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


def _spawn_isolation_kwargs() -> Dict[str, Any]:
    """Spawn kwargs that put the CLI in its own process group / session.

    The point is the same on both platforms — the CLI spawns helpers of its own,
    and killing only the direct child would strand them — but the mechanism is
    not: POSIX gets ``setsid()`` (so ``killpg`` reaches the whole session),
    Windows gets ``CREATE_NEW_PROCESS_GROUP`` (so the tree is walkable by
    ``taskkill /T`` and a Ctrl-C in derpr's console is not forwarded to it).
    ``start_new_session`` is silently ignored by the Windows implementation, so
    passing it there would look like isolation while providing none.
    """
    if os.name == "posix":
        return {"start_new_session": True}
    return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}


def _kill_process_tree(proc: Any) -> None:
    """Best-effort teardown of the spawned CLI and anything it spawned.

    POSIX: kill the whole session unconditionally, including after a clean exit —
    the CLI is known to leave background helpers behind, and the session id is
    still valid for them once the leader is gone.

    Windows: there is no session-wide signal. ``taskkill /T`` walks the tree from
    a *live* parent pid, so it is only meaningful while the process is running;
    after exit the parent link is gone and the call would only spawn a doomed
    ``taskkill`` per turn. Skip it in that case.
    """
    if os.name == "posix":
        import signal
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        return
    if proc.returncode is not None:
        return
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False, timeout=10,
        )
    except Exception:
        proc.kill()


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
                **_spawn_isolation_kwargs()
            )
        except OSError as e:
            # Surface any spawn failure as a provider error rather than an
            # unhandled OSError that takes the whole turn down — but say which
            # failure it was. The oversized-payload failure is the one the byte
            # budget exists for, so only that message quotes the payload size;
            # FileNotFoundError and PermissionError are OSError subclasses too,
            # and pointing those at the size limit sends the investigation the
            # wrong way. POSIX signals it as E2BIG; Windows signals a too-long
            # command line as WinError 206 — which Python raises as a
            # FileNotFoundError, so it must be matched on winerror, not errno,
            # or an oversized prompt reads as "binary not found".
            if e.errno == errno.E2BIG or getattr(e, "winerror", None) == 206:
                argv_bytes = sum(len(a.encode("utf-8")) for a in args)
                raise LLMCommunicationError(
                    f"{label} CLI could not be spawned: command line too large "
                    f"for the OS ({e.strerror or e}); argv payload was "
                    f"{argv_bytes} bytes."
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
                _kill_process_tree(proc)
            except Exception:
                pass
