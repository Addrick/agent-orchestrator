# src/self_edit/dispatcher.py
"""Dispatch + supervise detached Claude Code coding agents (DP-227).

One ``dispatch`` = one bug = one ``git worktree`` (via clone_manager) + one
detached ``claude`` subprocess whose ``--output-format stream-json`` stdout is
redirected to a per-agent raw log file. A per-agent *bridge* task tails that
file, runs each line through a platform adapter (events.py), appends the
resulting common-schema ``AgentEvent``s to an audit log, and on a wake event
({question, done, error}) calls the injected ``on_wake`` coroutine — which the
integration wires to ``chat_system.generate_response("fixr", …)``.

The agent is detached: the bridge tails the *file*, never the live pipe, so a
slow fixr turn never backpressures the agent and a malformed line can't wedge
anything. Approval lives at two boundaries above this module: the dispatch tool
parks for confirmation before ``dispatch`` is ever called, and a human merges
the PR the agent opens. This module never merges or pushes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from config import global_config
from src.self_edit import clone_manager
from src.self_edit.events import (
    DONE,
    ERROR,
    QUESTION,
    STARTED,
    AgentEvent,
    DispatchAdapter,
    get_adapter,
)
from src.self_edit.prompts import DISPATCH_AGENT_PROMPT
from src.self_edit.registry import (
    AgentRecord,
    AgentRegistry,
)
from src.self_edit import registry as reg
from src.utils.claude_cli_env import build_claude_cli_env

logger = logging.getLogger(__name__)

# Seconds the bridge sleeps when it reaches EOF on the raw log before re-reading.
_TAIL_POLL_SECONDS = 1.0
# Hard ceiling on a single agent's wall-clock life (safety net for a hung agent).
_AGENT_MAX_LIFE_SECONDS = 60 * 60

#: on_wake(record, event) -> Awaitable — called for each {question,done,error}.
WakeCallback = Callable[[AgentRecord, AgentEvent], Awaitable[None]]
#: on_event(record, event) -> Awaitable — called for EVERY event (DP-230), so the
#: transcript sink (per-agent Discord thread) sees progress, not just wakes.
EventCallback = Callable[[AgentRecord, AgentEvent], Awaitable[None]]


class DispatcherError(RuntimeError):
    """Raised when a dispatch cannot start (bad worktree, missing claude, …)."""


class Dispatcher:
    def __init__(
        self,
        registry: AgentRegistry,
        *,
        on_wake: WakeCallback,
        on_event: Optional[EventCallback] = None,
        platform: str = "claude",
        model_arg: Optional[str] = None,
        clone_dir: Optional[str] = None,
    ) -> None:
        self._registry = registry
        self._on_wake = on_wake
        self._on_event = on_event
        self._platform = platform
        self._model_arg = model_arg or "sonnet"
        self._clone_dir = clone_dir
        self._procs: Dict[str, asyncio.subprocess.Process] = {}
        self._bridges: Dict[str, "asyncio.Task[None]"] = {}

    # -- public API ----------------------------------------------------------

    async def dispatch(self, bug_id: str, description: str) -> AgentRecord:
        """Create an isolated worktree and spawn a detached coding agent in it.

        Caller (the dispatch WRITE tool) has already been confirmed by the
        ConfirmationManager. Returns the registered ``AgentRecord``."""
        if await self._registry.has_active_for_bug(bug_id):
            raise DispatcherError(
                f"An agent for {bug_id} is already in flight. Inspect or kill it "
                "before dispatching again."
            )

        worktree = await asyncio.to_thread(
            clone_manager.create_worktree, bug_id, clone_dir=self._clone_dir
        )
        branch = f"bugfix/{bug_id}-fix"
        fixr_dir = os.path.join(worktree, ".fixr")
        os.makedirs(fixr_dir, exist_ok=True)
        raw_log = os.path.join(fixr_dir, "raw.jsonl")
        events_log = os.path.join(fixr_dir, "events.jsonl")

        agent_id = f"{bug_id}-{int(time.time())}"
        prompt = f"Bug {bug_id}: {description}"
        proc = await self._spawn(
            prompt=prompt,
            system_prompt=DISPATCH_AGENT_PROMPT,
            cwd=worktree,
            raw_log=raw_log,
        )

        record = AgentRecord(
            agent_id=agent_id,
            bug_id=bug_id,
            description=description,
            worktree=worktree,
            branch=branch,
            raw_log=raw_log,
            events_log=events_log,
            pid=proc.pid,
            status=reg.RUNNING,
        )
        await self._registry.add(record)
        self._procs[agent_id] = proc
        self._start_bridge(record)
        logger.info("Dispatched agent %s (pid %s) for %s", agent_id, proc.pid, bug_id)
        return record

    async def answer_agent(self, agent_id: str, message: str) -> AgentRecord:
        """Resume a WAITING agent headlessly with ``claude --resume <sid> -p``.

        Only an agent in ``WAITING`` (it asked a question and parked) is
        resumable. Resuming a RUNNING agent would spawn a *second* ``claude``
        appending to the same raw log and orphan the first process; resuming a
        terminal agent (DONE/ERROR/KILLED/ORPHANED) would resurrect it. We claim
        the agent with an atomic WAITING→RUNNING compare-and-set, so concurrent
        replies can't both resume it, then spawn. On spawn failure the status is
        reverted to WAITING so the answer can be retried.

        The resumed run appends to the SAME raw log; the bridge (restarted if it
        had stopped) keeps converting from where it left off."""
        record = await self._registry.get(agent_id)
        if record is None:
            raise DispatcherError(f"Unknown agent_id: {agent_id}")
        if not record.session_id:
            raise DispatcherError(
                f"Agent {agent_id} has no session_id yet — cannot resume."
            )
        # Atomic claim: only one caller wins WAITING→RUNNING; a non-WAITING
        # agent (still running, finished, killed) is rejected.
        if not await self._registry.compare_and_set_status(
            agent_id, reg.WAITING, reg.RUNNING
        ):
            raise DispatcherError(
                f"Agent {agent_id} is not waiting (status={record.status}); "
                "nothing to answer."
            )

        proc = None
        try:
            proc = await self._spawn(
                prompt=message,
                system_prompt=None,            # session carries the system prompt
                cwd=record.worktree,
                raw_log=record.raw_log,
                resume_session=record.session_id,
            )
            await self._registry.update(agent_id, pid=proc.pid)
            self._procs[agent_id] = proc
            # Re-arm the bridge if the previous one finished on the question event.
            if agent_id not in self._bridges or self._bridges[agent_id].done():
                self._start_bridge(record, resume_tail=True)
        except Exception:
            # Anything between the claim and a fully wired resume fails the
            # answer: kill any process we did spawn so it can't run un-bridged,
            # drop our handle to it, and roll the claim back so a later answer
            # can retry this agent.
            if proc is not None and proc.returncode is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
            self._procs.pop(agent_id, None)
            await self._registry.update(agent_id, status=reg.WAITING, pid=None)
            raise
        logger.info("Resumed agent %s (pid %s)", agent_id, proc.pid)
        return record

    async def kill(self, agent_id: str, *, remove_worktree: bool = False) -> bool:
        """Terminate an agent's process + bridge. Optionally drop its worktree."""
        record = await self._registry.get(agent_id)
        if record is None:
            return False
        proc = self._procs.pop(agent_id, None)
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        task = self._bridges.pop(agent_id, None)
        if task is not None and not task.done():
            task.cancel()
        await self._registry.update(agent_id, status=reg.KILLED)
        if remove_worktree:
            await asyncio.to_thread(
                clone_manager.remove_worktree,
                record.bug_id, clone_dir=self._clone_dir, force=True,
            )
        logger.info("Killed agent %s", agent_id)
        return True

    async def shutdown(self) -> None:
        """Cancel all bridge tasks (process cleanup is best-effort on exit)."""
        for task in list(self._bridges.values()):
            if not task.done():
                task.cancel()
        if self._bridges:
            await asyncio.gather(*self._bridges.values(), return_exceptions=True)
        self._bridges.clear()

    # -- internals -----------------------------------------------------------

    async def _spawn(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str],
        cwd: str,
        raw_log: str,
        resume_session: Optional[str] = None,
    ) -> asyncio.subprocess.Process:
        binary = os.environ.get("CLAUDE_CLI_PATH") or shutil.which("claude")
        if not binary:
            raise DispatcherError("Claude Code 'claude' binary not found on PATH.")
        argv = self._build_argv(prompt, system_prompt, resume_session)
        # Force the dispatched agent onto the Claude subscription (strip the
        # inherited ANTHROPIC_API_KEY so `-p` mode doesn't silently bill the API).
        env = build_claude_cli_env()
        # Append-mode so a resume continues the same audit trail.
        log_fh = open(raw_log, "a", encoding="utf-8")
        try:
            proc = await asyncio.create_subprocess_exec(
                binary, *argv,
                cwd=cwd,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log_fh,
                stderr=asyncio.subprocess.STDOUT,
            )
        finally:
            # The child inherited the fd; the parent's copy can close.
            log_fh.close()
        return proc

    def _build_argv(
        self,
        prompt: str,
        system_prompt: Optional[str],
        resume_session: Optional[str],
    ) -> List[str]:
        """Assemble the detached ``claude`` argv. Mirrors engine._build_cc_args
        for the sandbox/yolo treatment so dispatched agents get the same OS
        confinement as the cc-* engine route."""
        argv: List[str] = ["-p", prompt, "--output-format", "stream-json",
                           "--verbose", "--model", self._model_arg]
        if resume_session:
            argv += ["--resume", resume_session]
        elif system_prompt:
            argv += ["--system-prompt", system_prompt]
        if global_config.CC_SANDBOX:
            argv += ["--dangerously-skip-permissions"]
        elif global_config.CC_ALLOWED_TOOLS:
            argv += ["--allowedTools", *global_config.CC_ALLOWED_TOOLS]
        if global_config.CC_MAX_TURNS > 0:
            argv += ["--max-turns", str(global_config.CC_MAX_TURNS)]
        sandbox = self._sandbox_settings()
        if sandbox is not None:
            argv += ["--settings", json.dumps(sandbox)]
        return argv

    @staticmethod
    def _sandbox_settings() -> Optional[Dict[str, Any]]:
        if not global_config.CC_SANDBOX:
            return None
        sandbox: Dict[str, Any] = {"enabled": True, "autoAllowBashIfSandboxed": True}
        if global_config.CC_SANDBOX_WEAKER_NESTED:
            sandbox["enableWeakerNestedSandbox"] = True
        if global_config.CC_SANDBOX_ALLOWED_DOMAINS:
            sandbox["network"] = {"allowedDomains": list(global_config.CC_SANDBOX_ALLOWED_DOMAINS)}
        return {"sandbox": sandbox}

    def _start_bridge(self, record: AgentRecord, *, resume_tail: bool = False) -> None:
        adapter = get_adapter(self._platform, record.agent_id)
        task = asyncio.create_task(
            self._bridge(record, adapter, resume_tail=resume_tail),
            name=f"fixr-bridge-{record.agent_id}",
        )
        self._bridges[record.agent_id] = task

    async def _bridge(
        self, record: AgentRecord, adapter: DispatchAdapter, *, resume_tail: bool
    ) -> None:
        """Tail record.raw_log, convert lines, append common events, fire wakes."""
        agent_id = record.agent_id
        deadline = time.time() + _AGENT_MAX_LIFE_SECONDS
        # On a resume we keep reading from where the file already is; on a fresh
        # dispatch we start from the top (offset 0).
        offset = os.path.getsize(record.raw_log) if (resume_tail and os.path.exists(record.raw_log)) else 0
        try:
            while True:
                if time.time() > deadline:
                    await self._emit(record, AgentEvent(
                        agent_id, -1, ERROR,
                        {"text": "agent exceeded max life", "detail": "timeout"}))
                    await self.kill(agent_id)
                    return
                new_lines, offset = await asyncio.to_thread(_read_from, record.raw_log, offset)
                if await self._process_lines(record, adapter, new_lines):
                    return
                if not new_lines and await self._synthesize_exit(record):
                    return
                await asyncio.sleep(_TAIL_POLL_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — bridge must never crash silently
            logger.exception("Bridge for agent %s crashed", agent_id)
            await self._registry.update(agent_id, status=reg.ERROR)

    async def _process_lines(
        self, record: AgentRecord, adapter: DispatchAdapter, lines: List[str]
    ) -> bool:
        """Emit events for each parsed line. Returns True when the bridge should
        stop (a question parked the agent, or a terminal done/error arrived)."""
        for raw in lines:
            for ev in adapter.parse_line(raw):
                if adapter.session_id and adapter.session_id != record.session_id:
                    await self._registry.update(record.agent_id, session_id=adapter.session_id)
                    record.session_id = adapter.session_id
                await self._emit(record, ev)
                if ev.type == QUESTION:
                    # answer_agent re-arms the bridge when it resumes the agent.
                    await self._registry.update(record.agent_id, status=reg.WAITING)
                    return True
                if ev.is_terminal:
                    return True
        return False

    async def _synthesize_exit(self, record: AgentRecord) -> bool:
        """If the process exited and the log is drained without a terminal event,
        synthesize one so fixr always gets a final wake. Returns True if it did."""
        proc = self._procs.get(record.agent_id)
        if proc is None or proc.returncode is None:
            return False
        if proc.returncode != 0:
            await self._emit(record, AgentEvent(
                record.agent_id, -1, ERROR,
                {"text": f"agent exited code {proc.returncode}", "detail": "nonzero_exit"}))
        else:
            await self._emit(record, AgentEvent(
                record.agent_id, -1, DONE,
                {"summary": "agent exited cleanly (no FIXR_DONE sentinel)"}))
        return True

    async def _emit(self, record: AgentRecord, ev: AgentEvent) -> None:
        """Append the event to the audit log, reflect status, fire wake if due."""
        await asyncio.to_thread(_append_jsonl, record.events_log, ev.to_jsonl())
        fields: Dict[str, Any] = {"last_event": ev.type}
        if ev.type == DONE:
            fields["status"] = reg.DONE
            if ev.payload.get("pr_url"):
                fields["pr_url"] = ev.payload["pr_url"]
        elif ev.type == ERROR:
            fields["status"] = reg.ERROR
        elif ev.type == STARTED and ev.payload.get("session_id"):
            fields["session_id"] = ev.payload["session_id"]
        await self._registry.update(record.agent_id, **fields)
        # Transcript sink (DP-230) sees EVERY event; the fixr wake only fires for
        # wake types. A failing sink must never kill the bridge.
        if self._on_event is not None:
            try:
                await self._on_event(record, ev)
            except Exception:  # noqa: BLE001
                logger.exception("on_event failed for agent %s", record.agent_id)
        if ev.is_wake:
            try:
                await self._on_wake(record, ev)
            except Exception:  # noqa: BLE001 — a failed wake must not kill the bridge
                logger.exception("on_wake failed for agent %s", record.agent_id)


def _read_from(path: str, offset: int) -> tuple[List[str], int]:
    """Read complete lines appended after byte ``offset``; return (lines,
    new_offset). Binary mode so offsets are exact bytes (text-mode seek cookies
    aren't addable). Only advances past the last newline, so a partially-written
    line is re-read whole on the next poll."""
    if not os.path.exists(path):
        return [], offset
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    if not data:
        return [], offset
    last_nl = data.rfind(b"\n")
    if last_nl == -1:
        return [], offset  # no complete line yet
    complete = data[: last_nl + 1]
    text = complete.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines, offset + len(complete)


def _append_jsonl(path: str, line: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
