# src/agents/codr_agent.py
"""codr agent (DP-449): lets the opencode persona `codr` work on its own.

Each tick looks at codr's one opencode session (the same one chat uses,
`providers/opencode`) and does at most one of:

  - relay: every completed `notify_adam` tool call not yet in the action log
    is logged, then posted to the configured channel (rate-limited) and
    stored as codr's reply in its latest conversation;
  - watch: a turn this agent started is still running, so check it for loops,
    a stall or an overrun, and nudge or abort it;
  - finish: that turn has ended, so run the task's check, record the attempt
    and post the result;
  - start: codr is idle and a task in the queue is still open, so start the
    next attempt with `prompt_async` (fire-and-forget).

All turn state lives on the opencode server, in the metadata of the user part
that started the turn (`AUTO_FIELD`: task, attempt, nudges, start time,
result), so a derpr restart picks up where it left off. Attempt outcomes go to
the agent action log, which is what decides whether a task is still open.

The task's check never enters the workspace: derpr keeps it and runs it
through opencode's shell endpoint after the turn, so codr cannot edit it.
"""

import asyncio
import io
import json
import logging
import re
import shlex
import tarfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import aiohttp

from src.agents.base import Agent
from src.chat_system import ChatSystem
from src.clients.notification import NotificationRouter
from src.generation_events import ResponseType
from src.engine.providers.opencode import (
    AUTO_FIELD, INJECTED_FIELD, OpencodeClient, auth_headers, current_turn_start,
    ensure_opencode_supported, message_meta, open_auto, opencode_model, part_meta, session_title,
    turn_end, turn_finished, workspace_directory, workspace_lock,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTIFY_TOOL = "notify_adam"
RELAY_ACTION = "notify_relay"
ATTEMPT_ACTION = "task_attempt"
REPEAT_LIMIT = 3            # identical tool calls / reasoning steps in a row
# Stored in the action log, whose payloads are capped at 4,000 chars.
CHECK_TAIL_CHARS = 2500
_EXIT_MARK = "__DERPR_EXIT="
_CHANGED_MARK = "__DERPR_CHANGED="
_SEEDED_MARK = "__DERPR_SEEDED"

DEFAULTS: Dict[str, Any] = {
    "persona": "codr",
    "tasks_dir": "config/codr_tasks",
    "max_attempts": 3,
    "max_turn_seconds": 7200,
    "stall_seconds": 900,
    "check_timeout_seconds": 600,
    "notify_per_hour": 20,
    "start_tasks": True,
    "pool_url": "http://10.0.0.70:5002/v1",
    "pool_context": 65536,
}


@dataclass(frozen=True)
class CodrTask:
    id: str
    text: str
    check: str
    max_attempts: int
    # Starting files (path relative to the task folder, content), from `seed/`.
    seed: Tuple[Tuple[str, bytes], ...] = ()


def load_tasks(tasks_dir: Path, default_attempts: int) -> List[CodrTask]:
    """Every `<tasks_dir>/<id>/` holding both TASK.md and check.sh, oldest
    (by name) first. An optional `max_attempts: N` first line in TASK.md
    overrides the default and is not shown to codr. Files under `seed/` are
    the task's starting files."""
    tasks = []
    for folder in sorted(p for p in tasks_dir.iterdir() if p.is_dir()) if tasks_dir.is_dir() else []:
        task_md, check = folder / "TASK.md", folder / "check.sh"
        if not (task_md.is_file() and check.is_file()):
            continue
        text = task_md.read_text(encoding="utf-8")
        attempts = default_attempts
        first, _, rest = text.partition("\n")
        match = re.fullmatch(r"\s*max_attempts:\s*(\d+)\s*", first)
        if match:
            attempts, text = int(match.group(1)), rest
        seed_dir = folder / "seed"
        seed = tuple(
            (f.relative_to(seed_dir).as_posix(), f.read_bytes())
            for f in sorted(seed_dir.rglob("*")) if f.is_file() and "__pycache__" not in f.parts
        ) if seed_dir.is_dir() else ()
        tasks.append(CodrTask(folder.name, text.strip(), check.read_text(encoding="utf-8"), attempts, seed))
    return tasks


# --- loop guard ---------------------------------------------------------------


def _role(message: Dict[str, Any]) -> Any:
    return (message.get("info") or {}).get("role")


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def detect_loop(turn: List[Dict[str, Any]]) -> Optional[str]:
    """Why the turn looks stuck, or None. `turn` is the turn's messages after
    the user message that started it. opencode's own doom_loop guard only sees
    repeats inside one step; these look across steps."""
    calls = [
        (p.get("tool"), json.dumps((p.get("state") or {}).get("input"), sort_keys=True))
        for m in turn if _role(m) == "assistant"
        for p in m.get("parts", [])
        if p.get("type") == "tool" and (p.get("state") or {}).get("status") != "pending"
    ]
    last = calls[-REPEAT_LIMIT:]
    if len(last) == REPEAT_LIMIT and len(set(last)) == 1:
        return f"the same `{last[0][0]}` call with the same arguments {REPEAT_LIMIT} times in a row"

    reasoning = [
        _norm(" ".join(p.get("text", "") for p in m.get("parts", []) if p.get("type") == "reasoning"))
        for m in turn if _role(m) == "assistant"
    ]
    reasoning = [r for r in reasoning if r]
    last_r = reasoning[-REPEAT_LIMIT:]
    if len(last_r) == REPEAT_LIMIT and len(set(last_r)) == 1 and len(last_r[0]) >= 40:
        return f"the same reasoning in {REPEAT_LIMIT} steps in a row"
    return None


def progress_signature(turn: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """Changes whenever the turn makes visible progress (a new step, part,
    streamed text or tool state)."""
    last = turn[-1] if turn else {}
    parts = last.get("parts", [])
    size = sum(len(p.get("text") or "") + len(json.dumps(p.get("state") or {})) for p in parts)
    return len(turn), len(parts), size


# --- workspace setup ----------------------------------------------------------

_NOTIFY_JS = """// Written by DERPR (DP-449). Overwritten on its next start; edits are lost.
// DERPR reads each call of this tool from the session and posts it to Adam.
export default {
  description:
    "Send Adam, your operator, a short message in your Discord channel now, without ending your turn. " +
    "Use it for progress worth knowing or a problem he can help with, not for routine status. " +
    "At most %(per_hour)d an hour are delivered. His reply, if any, arrives later as a new message.",
  args: { message: { type: "string", description: "The message, under 1500 characters." } },
  async execute(args) {
    return "Queued for Adam. Keep working.";
  },
}
"""

_POOL_AGENT_MD = """---
description: Worker on the shared inference pool (other model servers). Give it one self-contained subtask with everything it needs; run it in the background to keep working meanwhile. Pool servers may run a different, weaker model, so check what it returns.
mode: subagent
model: kcpp-pool/local
---
You are a worker subagent. Do the one subtask you are given, in the files and folder it names, and report the result briefly: what you did, what you verified, what is left.
"""


def validate_mcp(mcp: Any) -> None:
    """Refuse a malformed `mcp` setting (DP-462) before it reaches codr's
    opencode.json, where opencode would reject the whole file at instance
    start. The shape is opencode's own, passed through as is."""
    if mcp is None:
        return
    if not isinstance(mcp, dict):
        raise ValueError("codr agent: `mcp` must be an object mapping server names to their config")
    for name, entry in mcp.items():
        where = f"codr agent: mcp.{name}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} must be an object")
        kind = entry.get("type")
        if kind == "local":
            command = entry.get("command")
            if not (isinstance(command, list) and command and all(isinstance(c, str) and c for c in command)):
                raise ValueError(f"{where}: a local server needs `command`, a list of strings, program first")
        elif kind == "remote":
            if not (isinstance(entry.get("url"), str) and re.match(r"^https?://\S+$", entry["url"])):
                raise ValueError(f"{where}: a remote server needs `url`, an http(s) URL")
        elif not (set(entry) == {"enabled"} and isinstance(entry["enabled"], bool)):
            raise ValueError(f"{where}: `type` must be \"local\" or \"remote\" (or the entry only {{\"enabled\": bool}})")


def workspace_files(cfg: Dict[str, Any]) -> Dict[str, str]:
    """The `.opencode/` files DERPR keeps in codr's workspace. The empty
    `node_modules/` plus package files stop opencode's npm install of its plugin
    package, which hangs ~60 s per instance start with the sandbox offline."""
    pkg = {"dependencies": {"@opencode-ai/plugin": "*"}}
    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {"kcpp-pool": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Inference pool (any free local server)",
            "options": {"baseURL": cfg["pool_url"]},
            "models": {"local": {"name": "whatever the pool server holds",
                                 "limit": {"context": int(cfg["pool_context"]), "output": 16384}}},
        }},
    }
    if cfg.get("mcp"):
        config["mcp"] = cfg["mcp"]
    return {
        ".opencode/tool/notify_adam.js": _NOTIFY_JS % {"per_hour": int(cfg["notify_per_hour"])},
        ".opencode/agent/pool.md": _POOL_AGENT_MD,
        ".opencode/opencode.json": json.dumps(config, indent=2) + "\n",
        ".opencode/package.json": json.dumps(pkg) + "\n",
        ".opencode/package-lock.json": json.dumps({"packages": {"": pkg}}) + "\n",
    }


def sync_script(files: Dict[str, str]) -> str:
    """Bash that writes each file only when its content differs, then prints
    whether anything changed."""
    lines = ["set -e", "mkdir -p .opencode/tool .opencode/agent .opencode/node_modules", "changed=0"]
    for i, (path, content) in enumerate(files.items()):
        lines += [f"cat > '{path}.new' <<'DERPR_EOF_{i}'", content.rstrip("\n"), f"DERPR_EOF_{i}",
                  f"if cmp -s '{path}.new' '{path}'; then rm -f '{path}.new'; "
                  f"else mv '{path}.new' '{path}'; changed=1; fi"]
    lines.append(f'echo "{_CHANGED_MARK}$changed"')
    return "\n".join(lines) + "\n"


def seed_archive(task: CodrTask) -> Optional[bytes]:
    """The task's seed files as one tar.gz, or None if it has none."""
    if not task.seed:
        return None
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel, data in task.seed:
            info = tarfile.TarInfo(rel)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


async def seed_task(client: OpencodeClient, task: CodrTask) -> None:
    """Create `tasks/<id>/` and unpack each seed file that is not there yet,
    so a later attempt keeps what codr changed. The archive goes in through
    `write_file`, so a seed of any size fits (DP-455)."""
    base = f"tasks/{task.id}"
    archive = seed_archive(task)
    if archive is None:
        await client.run_shell(f"mkdir -p -- '{base}'")
        return
    staged = f".derpr-seed-{task.id}.tar.gz"
    await client.write_file(staged, archive)
    out = await client.run_shell(
        f"mkdir -p -- '{base}' && tar -xzf '{staged}' -C '{base}' --skip-old-files "
        f"&& echo {_SEEDED_MARK}; rm -f -- '{staged}'")
    if _SEEDED_MARK not in out:
        raise RuntimeError(f"codr: seeding {base} failed: {out[-500:]}")


def check_script(task: CodrTask, timeout_s: int) -> str:
    # The check travels as an argument, not on stdin: fed as a heredoc, any
    # command in it that reads stdin swallows the lines after it, and the check
    # exits 0 without having run them (DP-463).
    return (
        f"cd -- 'tasks/{task.id}' 2>/dev/null || {{ echo 'no tasks/{task.id} folder in the workspace'; exit 1; }}\n"
        f"timeout {int(timeout_s)} bash -c {shlex.quote(task.check.rstrip())} </dev/null\n"
        f'echo "{_EXIT_MARK}$?"\n'
    )


def parse_check(output: str) -> Tuple[bool, str]:
    """(passed, output without the exit marker). No marker = failed."""
    match = re.search(rf"{_EXIT_MARK}(\d+)\s*$", output)
    body = output[:match.start()] if match else output
    return bool(match and match.group(1) == "0"), body


# --- the agent ----------------------------------------------------------------


class CodrAgent(Agent):
    agent_name: str = "codr"
    schedule: Dict[str, Any] = {"interval": 30}

    def __init__(
        self,
        chat_system: ChatSystem,
        notification_router: NotificationRouter,
        agent_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(chat_system, inject_personas=False)
        self.notification_router = notification_router
        self.agent_config = agent_config or {}
        self.cfg = {**DEFAULTS, **{k: v for k, v in self.agent_config.items() if not k.startswith("_")}}
        validate_mcp(self.cfg.get("mcp"))
        self.persona_name: str = self.cfg["persona"]
        self.root, self.slug, self.directory = workspace_directory(self.persona_name)
        self._workspace_synced = False
        self._dispose_pending = False
        self._sent_at: Deque[float] = deque()
        self._dropped = 0
        self._undelivered = 0
        self._progress: Tuple[Any, float] = (None, 0.0)

    # -- plumbing

    def _tasks(self) -> List[CodrTask]:
        tasks_dir = Path(self.cfg["tasks_dir"])
        return load_tasks(tasks_dir if tasks_dir.is_absolute() else REPO_ROOT / tasks_dir,
                          int(self.cfg["max_attempts"]))

    def _attempts(self, task_id: str) -> List[Dict[str, Any]]:
        """Recorded attempts of `task_id`, oldest first."""
        rows = self.memory_manager.get_agent_actions(self.agent_name, limit=10000, action_type=ATTEMPT_ACTION)
        return [r for r in reversed(rows) if r.get("trigger_context") == f"task:{task_id}"]

    async def _post(self, body: str) -> bool:
        target = self.cfg.get("notification_target") or {}
        if not target.get("recipient"):
            logger.warning("codr agent: no notification_target configured; not posting: %s", body[:200])
            return False
        try:
            return await self.notification_router.send(
                channel=target.get("channel", "discord_channel"), recipient=str(target["recipient"]),
                subject="", body=body[:1900])
        except Exception as e:  # noqa: BLE001 — a failed post must not stop the agent
            logger.warning("codr agent: post failed: %s", e)
            return False

    # -- the tick

    async def deploy(self) -> None:
        ensure_opencode_supported()
        async with aiohttp.ClientSession(headers=auth_headers()) as http:
            client = OpencodeClient(http, self.directory)
            sid = await client.ensure_session(self.root, self.slug, session_title(self.persona_name))
            if not self._workspace_synced:
                await self._sync_workspace(client)
            msgs = await client.turn_messages(sid)
            await self._relay(msgs)

            start = self._open_turn(msgs)
            if start is not None:
                auto = open_auto(msgs[start]) or {}
                # A later turn (a chat message sent between ticks) means this one ended.
                end = turn_end(msgs, start)
                if end < len(msgs) or turn_finished(msgs, start):
                    await self._finish(client, msgs[:end], start, auto)
                else:
                    await self._watch(client, sid, msgs, start, auto)
                return

            busy = await client.is_busy(sid)
            if self._dispose_pending and not busy:
                await self._dispose(client)
            if not busy and self.cfg["start_tasks"]:
                await self._start_next(client, sid)

    async def _sync_workspace(self, client: OpencodeClient) -> None:
        out = await client.run_shell(sync_script(workspace_files(self.cfg)))
        if f"{_CHANGED_MARK}1" in out:
            # opencode reads `.opencode/` when it starts the directory's
            # instance; disposing it (only while idle) makes it read again.
            self._dispose_pending = True
        elif f"{_CHANGED_MARK}0" not in out:
            raise RuntimeError(f"codr workspace setup failed: {out[-500:]}")
        self._workspace_synced = True

    @staticmethod
    def _open_turn(msgs: List[Dict[str, Any]]) -> Optional[int]:
        """Index of the newest turn this agent started and has not closed. It
        is not always the latest turn: a chat message sent after the task turn
        ended but before this tick ran starts a turn of its own after it."""
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if _role(m) == "user" and not message_meta(m, INJECTED_FIELD) and open_auto(m) is not None:
                return i
        return None

    async def _dispose(self, client: OpencodeClient) -> None:
        lock = workspace_lock(self.directory)
        if lock.locked():
            return                  # a chat turn is starting; next tick
        async with lock:
            await client.request("POST", "/instance/dispose")
        self._dispose_pending = False
        logger.info("codr agent: workspace config changed; opencode instance reloaded.")

    # -- relay

    def _rate_ok(self) -> bool:
        now = time.monotonic()
        while self._sent_at and now - self._sent_at[0] > 3600:
            self._sent_at.popleft()
        if len(self._sent_at) >= int(self.cfg["notify_per_hour"]):
            return False
        self._sent_at.append(now)
        return True

    def _relayed(self) -> Set[str]:
        rows = self.memory_manager.get_agent_actions(self.agent_name, limit=2000, action_type=RELAY_ACTION)
        return {str(r.get("trigger_context")) for r in rows}

    async def _relay(self, msgs: List[Dict[str, Any]]) -> None:
        """Post each `notify_adam` call not relayed yet. Which ones were is
        kept in the action log, not on the part: opencode sends an assistant
        part's metadata to the model as provider metadata, and a foreign key
        there breaks every later request in the session (verified 2026-10-21:
        "messages do not match the ModelMessage[] schema")."""
        done: Optional[Set[str]] = None
        for m in msgs:
            for p in m.get("parts", []):
                state = p.get("state") or {}
                if p.get("tool") != NOTIFY_TOOL or state.get("status") != "completed":
                    continue
                done = self._relayed() if done is None else done
                key = f"part:{p.get('id')}"
                if key in done:
                    continue
                done.add(key)
                text = str((state.get("input") or {}).get("message") or "").strip()
                allowed = self._rate_ok()
                # Record first: a crash after the post must not post it twice.
                row = self._log_task_root(RELAY_ACTION, trigger_context=key,
                                          action_payload={"message": text[:1500]},
                                          outcome="sent" if allowed else "dropped")
                if not allowed:
                    self._dropped += 1
                    logger.info("codr agent: notify_adam rate limit hit; dropped: %s", text[:200])
                    continue
                if await self._post(f"codr: {text[:1500]}"):
                    await self._commit_reply(text[:1500])
                else:
                    self.memory_manager.update_agent_action_outcome(row, "failed")
                    self._undelivered += 1

    async def _commit_reply(self, text: str) -> None:
        """Store a relayed `notify_adam` message as codr's reply in the
        conversation it was posted to, so derpr's history has codr's side of
        the chat."""
        target = self.cfg.get("notification_target") or {}
        key = await self.notification_router.conversation_key(
            target.get("channel", "discord_channel"), str(target.get("recipient")))
        if key is None:
            logger.info("codr agent: no conversation for the notification target; relay not stored in history")
            return
        channel, server_id = key
        persistence = self.chat_system.turn_persistence
        row_id = persistence.commit_or_update_assistant(
            persona_name=self.persona_name, user_identifier=f"agent:{self.agent_name}",
            channel=channel, server_id=server_id, final_text=text,
            response_type=ResponseType.LLM_GENERATION, user_interaction_id=None,
            retry_assistant_id=None, tool_context_json=None,
        )
        if row_id is not None:
            persistence.queue_retain_safe(interaction_id=row_id, persona_name=self.persona_name,
                                          channel=channel, untrusted=False)

    # -- start

    async def _start_next(self, client: OpencodeClient, sid: str) -> None:
        for task in self._tasks():
            attempts = self._attempts(task.id)
            if any(a.get("outcome") == "pass" for a in attempts) or len(attempts) >= task.max_attempts:
                continue
            await self._start(client, sid, task, len(attempts) + 1, attempts[-1] if attempts else None)
            return

    def _task_prompt(self, task: CodrTask, attempt: int, previous: Optional[Dict[str, Any]]) -> str:
        lines = [
            f"[Task `{task.id}`, attempt {attempt} of {task.max_attempts}, started by DERPR's task queue]",
            "",
            task.text,
            "",
            f"Work in `tasks/{task.id}/` in your workspace. Keep an attempts log at "
            f"`tasks/{task.id}/attempts.md`: what you tried and what happened. Read it first if it exists.",
            f"`{NOTIFY_TOOL}` tells Adam something now without ending your turn. Your final reply is not "
            f"posted, so when you are done, tell Adam with `{NOTIFY_TOOL}` what you did and anything he "
            f"should know, then end your turn. DERPR then runs its own check on `tasks/{task.id}/`; you "
            "will see its output if it fails.",
            "For an independent subtask you can use the `pool` subagent (task tool, background if you "
            "want to keep working).",
        ]
        if previous:
            try:
                detail = json.loads(previous.get("outcome_payload") or "{}")
            except ValueError:
                detail = {}
            lines += ["", f"The previous attempt ended `{previous.get('outcome')}`."]
            if detail.get("reason"):
                lines.append(f"Reason: {detail['reason']}")
            if detail.get("check_tail"):
                lines += ["Check output (tail):", "```", detail["check_tail"], "```"]
        return "\n".join(lines)

    async def _start(self, client: OpencodeClient, sid: str, task: CodrTask, attempt: int,
                     previous: Optional[Dict[str, Any]]) -> None:
        persona = self.chat_system.personas.get(self.persona_name)
        if persona is None:
            logger.error("codr agent: persona %r not found; not starting tasks.", self.persona_name)
            return
        lock = workspace_lock(self.directory)
        if lock.locked():
            return                  # a chat turn is starting or running; next tick
        async with lock:
            msgs = await client.turn_messages(sid)
            start = current_turn_start(msgs)
            if await client.is_busy(sid) or (start is not None and not turn_finished(msgs, start)
                                             and open_auto(msgs[start]) is not None):
                return
            await seed_task(client, task)
            meta = {"task": task.id, "attempt": attempt, "nudges": 0, "started": time.time()}
            await self._prompt(client, sid, persona, self._task_prompt(task, attempt, previous), meta)
        logger.info("codr agent: started task %s attempt %d", task.id, attempt)

    async def _prompt(self, client: OpencodeClient, sid: str, persona: Any, text: str,
                      meta: Dict[str, Any]) -> None:
        body: Dict[str, Any] = {
            "parts": [{"type": "text", "text": text, "metadata": {AUTO_FIELD: meta}}],
            "system": persona.get_prompt(),
        }
        model = opencode_model(persona.get_model_name())
        if model:
            body["model"] = model
        await client.prompt_async(sid, body)
        self._progress = (None, time.monotonic())
        await self._wait_posted(client, sid, meta)

    async def _wait_posted(self, client: OpencodeClient, sid: str, meta: Dict[str, Any],
                           limit: float = 15.0) -> None:
        """Hold the caller's workspace lock until the prompt is in the session.
        `prompt_async` returns before the message is written, and a chat call
        that looks in that gap sees no running task and posts a blocking turn
        that folds into this one."""
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            msgs = await client.turn_messages(sid)
            start = current_turn_start(msgs)
            if start is not None and message_meta(msgs[start], AUTO_FIELD) == meta:
                return
            await asyncio.sleep(0.5)
        logger.warning("codr agent: prompt not visible in the session after %.0f s", limit)

    # -- watch

    def _stuck_reason(self, turn: List[Dict[str, Any]], auto: Dict[str, Any]) -> Tuple[Optional[str], bool]:
        """(reason, nudgeable). An overrun is never nudged: the attempt is out of time."""
        if time.time() - float(auto.get("started") or time.time()) > float(self.cfg["max_turn_seconds"]):
            return f"the attempt ran past {int(self.cfg['max_turn_seconds'])} s", False
        reason = detect_loop(turn)
        if reason:
            return reason, True
        sig, since = self._progress
        now = time.monotonic()
        current = progress_signature(turn)
        if current != sig:
            self._progress = (current, now)
        elif now - since > float(self.cfg["stall_seconds"]):
            return f"nothing new for {int(self.cfg['stall_seconds'])} s", True
        return None, False

    async def _watch(self, client: OpencodeClient, sid: str, msgs: List[Dict[str, Any]],
                     start: int, auto: Dict[str, Any]) -> None:
        reason, nudgeable = self._stuck_reason(msgs[start + 1:], auto)
        if not reason:
            return
        # Held from the abort to the nudge being in the session: with the turn
        # stopped, a chat message would otherwise start a turn in between and
        # the nudge would land inside it.
        async with workspace_lock(self.directory):
            await self._stop(client, sid, msgs, start, auto, reason, nudgeable)

    async def _stop(self, client: OpencodeClient, sid: str, msgs: List[Dict[str, Any]], start: int,
                    auto: Dict[str, Any], reason: str, nudgeable: bool) -> None:
        await client.abort(sid)
        await self._wait_idle(client, sid)
        start_part = self._auto_part(msgs[start])
        task = next((t for t in self._tasks() if t.id == auto.get("task")), None)
        persona = self.chat_system.personas.get(self.persona_name)
        if nudgeable and int(auto.get("nudges", 0)) == 0 and task and persona:
            await client.patch_part(start_part, **{AUTO_FIELD: {**auto, "result": "nudged", "reason": reason}})
            nudge = (f"[DERPR loop guard] I stopped your turn: {reason}. Change approach rather than "
                     f"repeating it, then carry on with task `{task.id}`.")
            await self._prompt(client, sid, persona, nudge, {**auto, "nudges": 1})
            logger.info("codr agent: nudged task %s: %s", task.id, reason)
            return
        await client.patch_part(start_part, **{AUTO_FIELD: {**auto, "result": "aborted", "reason": reason}})
        await self._record(auto, "aborted", {"reason": reason}, task)

    async def _wait_idle(self, client: OpencodeClient, sid: str, limit: float = 20.0) -> None:
        deadline = time.monotonic() + limit
        while await client.is_busy(sid) and time.monotonic() < deadline:
            await asyncio.sleep(1.0)

    @staticmethod
    def _auto_part(message: Dict[str, Any]) -> Dict[str, Any]:
        return next(p for p in message.get("parts", []) if AUTO_FIELD in part_meta(p))

    # -- finish

    async def _finish(self, client: OpencodeClient, msgs: List[Dict[str, Any]],
                      start: int, auto: Dict[str, Any]) -> None:
        task = next((t for t in self._tasks() if t.id == auto.get("task")), None)
        final = next((m for m in reversed(msgs[start + 1:]) if _role(m) == "assistant"), {})
        reply = "\n".join(p.get("text", "") for p in final.get("parts", [])
                          if p.get("type") == "text" and not p.get("synthetic")).strip()
        error = ((final.get("info") or {}).get("error") or {})
        detail: Dict[str, Any] = {"reply": reply[:600]}
        if task is None:
            outcome, detail["reason"] = "aborted", f"task {auto.get('task')!r} is no longer in the queue"
        else:
            output = await client.run_shell(
                check_script(task, int(self.cfg["check_timeout_seconds"])),
                timeout=aiohttp.ClientTimeout(total=int(self.cfg["check_timeout_seconds"]) + 60))
            passed, body = parse_check(output)
            outcome = "pass" if passed else "fail"
            detail["check_tail"] = body[-CHECK_TAIL_CHARS:]
        if error:
            detail["turn_error"] = f"{error.get('name', 'error')} {(error.get('data') or {}).get('message', '')}".strip()
        # Claim before recording and posting, so a crash cannot post twice.
        await client.patch_part(self._auto_part(msgs[start]), **{AUTO_FIELD: {**auto, "result": outcome}})
        await self._record(auto, outcome, detail, task)

    def _fit_payload(self, detail: Dict[str, Any]) -> Dict[str, Any]:
        """Shrink `detail` until its stored (escaped) form fits the action
        log's cap. Cut as stored, the JSON would not parse and the next
        attempt's prompt would lose the reason and the check output. The reply
        goes first (the next prompt does not use it), then the start of the
        check output (its end says what failed)."""
        detail = dict(detail)
        for field, keep_end in (("reply", False), ("check_tail", True)):
            while detail.get(field):
                over = len(json.dumps(detail, default=str, ensure_ascii=True)) - self.MAX_PAYLOAD_CHARS
                if over <= 0:
                    return detail
                value = detail[field]
                cut = max(over, 50)
                detail[field] = value[cut:] if keep_end else value[:-cut]
        return detail

    async def _record(self, auto: Dict[str, Any], outcome: str, detail: Dict[str, Any],
                      task: Optional[CodrTask]) -> None:
        task_id = str(auto.get("task"))
        seconds = int(time.time() - float(auto.get("started") or time.time()))
        detail = {**detail, "seconds": seconds, "nudges": int(auto.get("nudges", 0))}
        action_id = self._log_task_root(
            ATTEMPT_ACTION, trigger_context=f"task:{task_id}",
            action_payload={"task": task_id, "attempt": auto.get("attempt")},
            contexts=[("task", task_id)])
        self._finalize_action(action_id, outcome, self._fit_payload(detail))
        total = task.max_attempts if task else "?"
        line = (f"codr task `{task_id}` attempt {auto.get('attempt')}/{total}: **{outcome.upper()}** "
                f"({seconds // 60} min)")
        if detail.get("reason"):
            line += f": {detail['reason']}"
        if self._dropped:
            line += f" ({self._dropped} notify_adam message(s) dropped by the rate limit)"
            self._dropped = 0
        if self._undelivered:
            line += f" ({self._undelivered} notify_adam message(s) failed to post)"
            self._undelivered = 0
        await self._post(line)
        logger.info("codr agent: task %s attempt %s -> %s", task_id, auto.get("attempt"), outcome)
