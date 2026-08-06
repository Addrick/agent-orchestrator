# src/engine/providers/_subprocess.py
"""Subprocess machinery shared by the agy and cc (Claude Code) providers (DP-244).

Both providers are one-shot CLI routes: they render the full message history
into a single transcript, spawn a CLI in a workspace dir, and adapt the
process's exit-time output into the unified event shape. The pieces that are
literally shared — the subprocess runner, the transcript renderer, and the
workspace-name sanitiser — live here so agy.py and cc.py don't duplicate them.

The runner itself is platform-agnostic (DP-324). Three things differ per OS, and
each has a POSIX and a Windows implementation:

* what an argv entry *costs* against the OS ceiling (``_cmdline_cost``),
* how the child is isolated at spawn (``_spawn_isolation_kwargs``),
* how the child *and its descendants* are torn down (``_adopt_process_tree`` /
  ``_kill_process_tree``).

NOTE: this module carries a whole-file ``ignore_errors`` in mypy.ini — the POSIX
kill path (``os.killpg`` / ``os.getpgid`` / ``signal.SIGKILL``) does not exist in
the Windows-typed os/signal stubs (and vice versa for the Windows creation flags
and the ``ctypes.wintypes`` job-object calls), the same legacy noise that kept
``[mypy-src.engine.driver]`` ignored before the extraction.
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
#: Budget for the rendered prompt argv entry on POSIX, in UTF-8 bytes. The
#: headroom under the hard cap is per-entry slack, not a shared pool: each argv
#: string gets its own ``MAX_ARG_STRLEN`` ceiling, and the *total* (``ARG_MAX``,
#: ~2 MB) is far away even with every clamped entry at budget.
MAX_CLI_PROMPT_BYTES_POSIX = 96 * 1024
#: Same, on Windows, where the total command line IS the ceiling — hence the far
#: smaller number: the prompt, the cc route's ``--system-prompt``, the
#: ``--settings`` JSON and the flags all come out of one 32767-char pool.
#: The unit here is **escaped command-line characters, not bytes** — see
#: :func:`_cmdline_cost`.
MAX_CLI_PROMPT_CHARS_WINDOWS = 20 * 1024
#: Budget for a secondary argv entry with no block structure to elide (the cc
#: route's ``--system-prompt``) on Windows. On POSIX it gets the same per-entry
#: ceiling as the prompt.
MAX_CLI_ARG_CHARS_WINDOWS = 6 * 1024


def cli_prompt_budget() -> int:
    """Budget for the rendered prompt on THIS host, in this host's argv unit
    (UTF-8 bytes on POSIX, escaped command-line chars on Windows —
    :func:`_cmdline_cost` is what converts a string to that unit).

    Resolved per call rather than frozen at import: the two platforms differ by
    a factor of five, so a constant baked in at import time would be untestable
    for the other platform (and every test would silently assert the budget of
    whatever host the suite happens to run on).
    """
    return MAX_CLI_PROMPT_BYTES_POSIX if os.name == "posix" else MAX_CLI_PROMPT_CHARS_WINDOWS


def cli_arg_budget() -> int:
    """Budget for a secondary argv entry on THIS host. See
    :func:`cli_prompt_budget`."""
    return MAX_CLI_PROMPT_BYTES_POSIX if os.name == "posix" else MAX_CLI_ARG_CHARS_WINDOWS


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


def _cmdline_cost(text: str) -> int:
    """What one argv entry costs against THIS host's command-line ceiling.

    POSIX passes argv as a NUL-separated byte vector, so an entry costs exactly
    its UTF-8 length and ``MAX_ARG_STRLEN`` is a byte cap.

    Windows has no argv vector at all: ``CreateProcess`` takes ONE string, and
    Python builds it with ``subprocess.list2cmdline`` — which wraps an entry
    containing whitespace in quotes and escapes every embedded ``"`` as ``\\"``.
    So the OS measures the *escaped* form, not the raw text, and the two diverge
    by a lot on exactly the content these routes carry (measured: x1.00 for plain
    ASCII, x1.14 for a JSON tool-result transcript, x2.00 for quote-dense text).
    Budgeting raw bytes against a cap the OS applies to the escaped string is how
    a prompt the trimmer believed fit still died at the spawn with WinError 206 —
    the exact failure ``fit_cli_prompt`` exists to prevent. Cost the real thing.
    """
    if os.name == "posix":
        return _utf8_len(text)
    return len(subprocess.list2cmdline([text]))


def _keep_head_within(text: str, max_cost: int) -> str:
    """Longest head of ``text`` costing at most ``max_cost``, never splitting a
    codepoint.

    Binary search over the UTF-8 prefix length rather than a straight slice: on
    Windows the cost of a prefix is not its length (see :func:`_cmdline_cost`),
    so there is no arithmetic that maps a budget to a cut point. Cost is
    non-decreasing in prefix length, so the search converges; tracking the best
    known-good candidate keeps the result valid even if it were not.
    """
    if max_cost <= 0:
        return ""
    if _cmdline_cost(text) <= max_cost:
        return text
    raw = text.encode("utf-8")
    lo, hi, best = 0, len(raw), ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = raw[:mid].decode("utf-8", errors="ignore")
        if _cmdline_cost(candidate) <= max_cost:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _truncate_block(text: str, max_cost: int) -> str:
    """Cut one block down to ``max_cost``, keeping its **head**.

    Head, not tail, because every block starts with its role label
    (``User:`` / ``Assistant:`` / ``Tool(name):``). Cutting from the front
    strips that label and hands the model an unattributed fragment of, most
    likely, raw tool output — content it would then read as conversation. The
    label is the one part that must survive.
    """
    marker = _BLOCK_SEP + TRUNCATION_NOTICE
    room = max_cost - _cmdline_cost(marker)
    if room <= 0:
        return _keep_head_within(text, max_cost)
    return _keep_head_within(text, room) + marker


def clamp_cli_arg(text: str, max_bytes: Optional[int] = None,
                  label: str = "argument") -> str:
    """Bound a standalone argv entry that has no block structure to elide.

    Used for the cc route's ``--system-prompt`` value, which travels as its own
    argv entry and is therefore subject to the same OS ceiling as the prompt
    (``MAX_ARG_STRLEN`` per entry on POSIX; the shared
    ``MAX_COMMAND_LINE_CHARS`` pool on Windows). Clamping it is strictly better
    than the alternative (a spawn failure that loses the turn outright), but a
    truncated system prompt is a degraded prompt — hence the warning.

    ``max_bytes`` is in this host's argv unit (:func:`_cmdline_cost`), which is
    UTF-8 bytes only on POSIX.
    """
    if max_bytes is None:
        max_bytes = cli_arg_budget()
    cost = _cmdline_cost(text)
    if cost <= max_bytes:
        return text
    logger.warning(
        "CLI %s exceeded the %d-unit argv budget (%d) — truncating; the model "
        "will see a shortened instruction block.", label, max_bytes, cost,
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

    ``max_bytes`` defaults to this host's budget (:func:`cli_prompt_budget`) and
    is in this host's argv unit — UTF-8 bytes on POSIX, escaped command-line
    characters on Windows (:func:`_cmdline_cost`). Per-part costs are summed
    rather than measured on the joined string, which over-counts the quoting
    Windows applies once to the whole entry; erring high is the safe direction.
    """
    if max_bytes is None:
        max_bytes = cli_prompt_budget()
    texts = [b.text for b in blocks]
    parts = [p for p in (preamble, _BLOCK_SEP.join(texts) if texts else "") if p]
    prompt = _BLOCK_SEP.join(parts)
    if _cmdline_cost(prompt) <= max_bytes:
        return prompt, 0

    # Room left for history once the preamble and the elision marker are paid for.
    fixed = _cmdline_cost(ELISION_NOTICE) + len(_BLOCK_SEP)
    if preamble:
        fixed += _cmdline_cost(preamble) + len(_BLOCK_SEP)
    budget = max_bytes - fixed
    if budget <= 0:
        # Pathological: the preamble alone blows the limit. Nothing else fits.
        logger.warning(
            "CLI prompt preamble alone exceeds the %d-unit argv budget — "
            "truncating it.", max_bytes
        )
        return _keep_head_within(prompt, max_bytes), len(blocks)

    kept: List[str] = []
    used = 0
    for block in reversed(blocks):  # newest turns first
        cost = _cmdline_cost(block.text) + (len(_BLOCK_SEP) if kept else 0)
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
            "CLI prompt: no history block fits in %d argv units — kept a "
            "truncated %r block.", budget, anchor.role,
        )

    prompt = _BLOCK_SEP.join(
        [p for p in (preamble, ELISION_NOTICE, _BLOCK_SEP.join(kept)) if p]
    )
    logger.warning(
        "CLI prompt exceeded the %d-unit argv budget — elided %d oldest history "
        "message(s).", max_bytes, dropped,
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
    """Spawn kwargs that detach the CLI from derpr's own console/session.

    POSIX gets ``setsid()``: it is what makes the whole descendant set reachable
    later through one ``killpg``, so here isolation and teardown are the same
    mechanism.

    Windows gets ``CREATE_NEW_PROCESS_GROUP``, which does strictly less than it
    looks like. It only stops a Ctrl-C/Ctrl-Break in derpr's console from being
    delivered to the child — that is the entire benefit, and it is worth having.
    It is emphatically NOT what lets the tree be torn down: ``taskkill /T`` walks
    parent-pid links from a process snapshot and reaches the same descendants
    with or without the flag (verified). Teardown comes from the job object in
    :func:`_adopt_process_tree`.

    ``start_new_session`` is silently ignored by the Windows implementation, so
    passing it there would look like isolation while providing none.
    """
    if os.name == "posix":
        return {"start_new_session": True}
    return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}


#: ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``: when the last handle to the job closes
#: — including because derpr died and the kernel closed it for us — every process
#: still assigned to that job is terminated.
_JOB_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_EXTENDED_LIMIT_INFORMATION = 9
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100


def _adopt_process_tree(proc: Any) -> Any:
    """Make ``proc``'s whole descendant set killable as a unit. Returns an opaque
    handle for :func:`_kill_process_tree`, or ``None`` when the platform needs
    none (POSIX: ``setsid`` at spawn already did this).

    Windows has no session to signal, and ``taskkill /T`` is only half an answer
    because it needs a *live* parent pid to walk from — once the CLI exits
    cleanly, its helpers are orphaned and unreachable by pid (verified: the
    grandchild outlives the parent and ``taskkill`` then reports "process not
    found"). POSIX kills the session unconditionally *because* these CLIs are
    known to leave helpers behind, so Windows needs an equivalent that survives
    the parent — a job object with ``KILL_ON_JOB_CLOSE``. Closing its handle
    kills whatever is still inside, live parent or not.

    Best-effort by design: every failure path returns ``None``, which degrades to
    the ``taskkill`` fallback rather than failing the turn. There is a small race
    between ``CreateProcess`` and the assignment here, so a helper spawned in
    that window escapes the job; the fallback still covers the live-parent case.
    """
    if os.name == "posix":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class _BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimits),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Declaring these matters: HANDLE is pointer-sized and the default
        # restype (c_int) truncates it on 64-bit, so every later call would be
        # handed a bogus handle and silently fail.
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        k32.CloseHandle.argtypes = [wintypes.HANDLE]

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = _JOB_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(
                job, _JOB_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(limits), ctypes.sizeof(limits)):
            k32.CloseHandle(job)
            return None
        handle = k32.OpenProcess(
            _PROCESS_TERMINATE | _PROCESS_SET_QUOTA, False, proc.pid)
        if not handle:
            k32.CloseHandle(job)
            return None
        try:
            if not k32.AssignProcessToJobObject(job, handle):
                k32.CloseHandle(job)
                return None
        finally:
            k32.CloseHandle(handle)
        return job
    except Exception:
        logger.debug("Could not put the CLI in a job object; "
                     "falling back to taskkill.", exc_info=True)
        return None


def _win_close_job(job: Any) -> None:
    """Close a job handle from :func:`_adopt_process_tree` — which, because of
    ``KILL_ON_JOB_CLOSE``, is what actually kills the tree."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Same reason as in _adopt_process_tree: without argtypes the pointer-sized
    # handle is marshalled as a 32-bit int, and the close — and with it the
    # kill — silently does nothing.
    k32.CloseHandle.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle(job)


def _kill_process_tree(proc: Any, job: Any = None) -> None:
    """Best-effort teardown of the spawned CLI and anything it spawned.

    POSIX: kill the whole session unconditionally, including after a clean exit —
    the CLI is known to leave background helpers behind, and the session id is
    still valid for them once the leader is gone.

    Windows: close the job handle from :func:`_adopt_process_tree`. That is the
    equivalent of the POSIX branch — it reaches the helpers whether or not the
    parent is still alive, which is why it is preferred over ``taskkill``.

    Only when there is no job (the assignment failed) does it fall back to
    ``taskkill /T``, which needs a live parent pid and is therefore skipped once
    the process has exited. That fallback is spawned and NOT waited on: this runs
    in the ``finally`` of an async call, and blocking there stalls the one event
    loop that also serves Discord, the portal and every other provider.
    """
    if os.name == "posix":
        import signal
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        return
    if job is not None:
        _win_close_job(job)
        return
    if proc.returncode is not None:
        return
    try:
        subprocess.Popen(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
    job = None
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
        # Claim the descendant set before the CLI has had time to build one.
        job = _adopt_process_tree(proc)
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
                _kill_process_tree(proc, job)
            except Exception:
                pass
