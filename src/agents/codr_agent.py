# src/agents/codr_agent.py
"""codr agent (DP-449): lets the opencode persona `codr` work on its own.

Each tick looks at codr's one opencode session (the same one chat uses,
`providers/opencode`) and does at most one of:

  - relay: every completed `notify_adam` tool call not yet in the action log
    is logged, then posted to the configured channel (rate-limited) and
    stored as codr's reply in its latest conversation;
  - watch: a turn this agent started is still running, so check it for loops,
    a stall or an overrun, and nudge or abort it;
  - finish: that turn has ended, so run the task's check in the ticket's
    worktree, merge the ticket branch into its project on a pass, record the
    attempt and post the result;
  - start: codr is idle and a task in the queue is open (`queue_states`:
    attempts left, every dependency passed), so get its project repo and
    worktree ready and start the next attempt with `prompt_async`
    (fire-and-forget).

The workspace layout (DP-488): `AGENTS.md` (rules, synced with `.opencode/`),
`memory/` (codr's notes, a git repo), `projects/<project>/` (a git repo per
project) and `worktrees/<ticket>/` (branch `ticket/<ticket>`).

All turn state lives on the opencode server, in the metadata of the user part
that started the turn (`AUTO_FIELD`: task, attempt, nudges, start time,
result), so a derpr restart picks up where it left off. Attempt outcomes go to
the agent action log, which is what decides whether a task is still open.

The task's check never enters the workspace: derpr keeps it and runs it
through opencode's shell endpoint after the turn, so codr cannot edit it.
"""

import asyncio
import hashlib
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

from config import global_config
from src.agents.base import Agent
from src.chat_system import ChatSystem
from src.clients.notification import NotificationRouter
from src.generation_events import ResponseType
# The ticket format's one home, shared with its writer, stagr's
# `queue_codr_task` (DP-489); `src.stagr` sits below `src.agents`.
from src.stagr.codr_ticket import ATTEMPT_ACTION, HEADER_KEYS, PROJECT_RE, attempt_context
from src.engine.providers.opencode import (
    AUTO_FIELD, INJECTED_FIELD, OpencodeClient, auth_headers, current_turn_start,
    ensure_opencode_supported, message_meta, open_auto, opencode_model, part_meta, session_title,
    turn_end, turn_finished, workspace_directory, workspace_lock,
)

logger = logging.getLogger(__name__)

NOTIFY_TOOL = "notify_adam"
RELAY_ACTION = "notify_relay"
NOTICE_ACTION = "ticket_notice"
REPEAT_LIMIT = 3            # identical tool calls / reasoning steps in a row
# Stored in the action log, whose payloads are capped at 4,000 chars.
CHECK_TAIL_CHARS = 2500
_EXIT_MARK = "__DERPR_EXIT="
_CHANGED_MARK = "__DERPR_CHANGED="
_READY_MARK = "__DERPR_TICKET_READY"
_MEMORY_MARK = "__DERPR_MEMORY_READY"
_MERGED_MARK = "__DERPR_MERGED="
_CONFLICT_MARK = "__DERPR_CONFLICT"
_MAIN_MARK = "__DERPR_MAIN="
_NOTED_MARK = "__DERPR_NOTED"
# Commits DERPR makes in codr's repos carry its own name, so `git log` tells
# them apart from codr's.
_GIT_AS_DERPR = "git -c user.name=DERPR -c user.email=derpr@codr.local"
AGENTS_MD = global_config.CONFIG_DIR / "codr_workspace" / "AGENTS.md"

DEFAULTS: Dict[str, Any] = {
    "persona": "codr",
    "max_attempts": 3,
    "max_turn_seconds": 604800,
    "stall_seconds": 900,
    "check_timeout_seconds": 600,
    "notify_per_hour": 20,
    "start_tasks": True,
    "pool_url": "http://10.0.0.70:5001/v1",
    "pool_context": 262144,
}


@dataclass(frozen=True)
class CodrTask:
    id: str
    text: str
    check: str
    max_attempts: int
    # Starting files (path relative to the task folder, content), from `seed/`.
    seed: Tuple[Tuple[str, bytes], ...] = ()
    title: str = ""
    # Empty = the ticket is its own project (`default_project`).
    project: str = ""
    depends_on: Tuple[str, ...] = ()

    @property
    def project_name(self) -> str:
        return self.project or default_project(self.id)

    @property
    def worktree(self) -> str:
        return f"worktrees/{self.id}"

    @property
    def branch(self) -> str:
        return f"ticket/{self.id}"

    @property
    def project_dir(self) -> str:
        return f"projects/{self.project_name}"


# Any folder name is a ticket ID (CD-NNN is stagr's, `codr_ticket.TICKET_ID_RE`),
# but it names a worktree and the branch `ticket/<id>`, so it must be a valid
# git ref component too.
BRANCH_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_HEADER_LINE = re.compile(r"([a-z][a-z_]*):[ \t]*(.*?)\s*")


def default_project(ticket_id: str) -> str:
    """A ticket without `project:` is its own project, named after its folder."""
    return ticket_id.lower()


def _ticket_id_error(ticket_id: str) -> Optional[str]:
    if (not BRANCH_SAFE_ID_RE.fullmatch(ticket_id) or ".." in ticket_id
            or ticket_id.endswith((".", ".lock"))):
        return (f"the folder name {ticket_id!r} cannot be a git branch name: use letters, digits, "
                "'.', '_' and '-', starting with a letter or digit")
    return None


def parse_header(text: str) -> Tuple[Dict[str, str], str]:
    """(header, ticket text). The header is the run of `key: value` lines at
    the top, ended by the first blank line or the first line that is not
    `key: value` (the pre-DP-488 form was one `max_attempts: N` line with the
    text right under it). No such first line = no header."""
    header: Dict[str, str] = {}
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        match = _HEADER_LINE.fullmatch(lines[i])
        if not match:
            break
        key, value = match.groups()
        if key in header:
            raise ValueError(f"`{key}:` appears twice in the header")
        header[key] = value
        i += 1
    if i < len(lines) and header and not lines[i].strip():
        i += 1
    return header, "\n".join(lines[i:])


def _ticket_fields(ticket_id: str, header: Dict[str, str], default_attempts: int) -> Dict[str, Any]:
    """The header's values, checked. Raises ValueError naming what is wrong."""
    unknown = sorted(set(header) - set(HEADER_KEYS))
    if unknown:
        raise ValueError(f"unknown header key(s): {', '.join(unknown)} (known: {', '.join(HEADER_KEYS)})")
    fields: Dict[str, Any] = {"max_attempts": default_attempts, "title": header.get("title", "")}
    if "max_attempts" in header:
        if not re.fullmatch(r"\d+", header["max_attempts"]) or int(header["max_attempts"]) < 1:
            raise ValueError(f"`max_attempts:` must be a whole number of at least 1, not {header['max_attempts']!r}")
        fields["max_attempts"] = int(header["max_attempts"])
    project = header.get("project", "") or default_project(ticket_id)
    if not PROJECT_RE.fullmatch(project):
        raise ValueError(f"project name {project!r} must be lowercase letters, digits and '-'")
    fields["project"] = header.get("project", "")
    deps = tuple(d.strip() for d in header.get("depends_on", "").split(",") if d.strip())
    if ticket_id in deps:
        raise ValueError("a ticket cannot depend on itself")
    fields["depends_on"] = deps
    return fields


def _name_order(name: str) -> List[Any]:
    """Sort key that compares the numbers in a name as numbers, so CD-1000
    comes after CD-999 (a plain string sort puts it before CD-101). Split on
    a captured group, strings and ints alternate, so they never meet."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", name)]


def load_tickets(tasks_dir: Path, default_attempts: int) -> Tuple[List[CodrTask], Dict[str, str]]:
    """(tickets, {folder name: why it was skipped}). A ticket is every
    `<tasks_dir>/<id>/` holding both TASK.md and check.sh, oldest (by name)
    first. TASK.md may open with a header (`parse_header`, keys
    `HEADER_KEYS`), which codr never sees. Files under `seed/` are the
    ticket's starting files. Dot-folders are skipped: `queue_codr_task`
    stages a task in one before it renames it into place."""
    tasks, errors = [], {}
    for folder in sorted((p for p in tasks_dir.iterdir() if p.is_dir() and not p.name.startswith(".")),
                         key=lambda p: _name_order(p.name)) if tasks_dir.is_dir() else []:
        task_md, check = folder / "TASK.md", folder / "check.sh"
        if not (task_md.is_file() and check.is_file()):
            continue
        try:
            id_error = _ticket_id_error(folder.name)
            if id_error:
                raise ValueError(id_error)
            header, text = parse_header(task_md.read_text(encoding="utf-8"))
            fields = _ticket_fields(folder.name, header, default_attempts)
        except ValueError as e:
            errors[folder.name] = str(e)
            continue
        seed_dir = folder / "seed"
        seed = tuple(
            (f.relative_to(seed_dir).as_posix(), f.read_bytes())
            for f in sorted(seed_dir.rglob("*")) if f.is_file() and "__pycache__" not in f.parts
        ) if seed_dir.is_dir() else ()
        tasks.append(CodrTask(folder.name, text.strip(), check.read_text(encoding="utf-8"),
                              seed=seed, **fields))
    return tasks, errors


def load_tasks(tasks_dir: Path, default_attempts: int) -> List[CodrTask]:
    """The tickets `load_tickets` accepts, without the skipped ones."""
    return load_tickets(tasks_dir, default_attempts)[0]


# --- queue order --------------------------------------------------------------

OPEN, PASSED, FAILED, WAITING, HELD, BLOCKED = "open", "passed", "failed", "waiting", "held", "blocked"
# Posted once each: a ticket that will never start, or waits on a ticket nobody queued.
NOTICE_STATES = (BLOCKED, HELD)


def queue_states(tasks: List[CodrTask], outcomes: Dict[str, List[str]]) -> Dict[str, Tuple[str, str]]:
    """{ticket id: (state, why)} from each ticket's recorded attempt
    outcomes. `passed`; `failed` (no pass and no attempts left); `blocked`
    (a dependency failed for good, is blocked itself, or is in a cycle: it
    will never start); `held` (a dependency is not in the queue and never
    passed); `waiting` (a dependency has not passed yet); else `open`, ready
    to start. `outcomes` may hold tickets no longer in the queue: one that
    passed and whose folder was removed still satisfies its dependents, as
    stagr's `queue_codr_task` accepts it as a dependency."""
    by_id = {t.id: t for t in tasks}
    states: Dict[str, Tuple[str, str]] = {}

    def dep_state(dep: str, path: Tuple[str, ...]) -> Tuple[str, str]:
        """What `dep` makes of the ticket at the end of `path`."""
        if dep in path:
            return BLOCKED, f"its dependencies form a cycle ({' -> '.join(path + (dep,))})"
        if dep not in by_id:
            if "pass" in outcomes.get(dep, []):
                return OPEN, ""
            return HELD, f"it depends on `{dep}`, which is not in the queue"
        found = state(dep, path + (dep,))[0]
        if found in (FAILED, BLOCKED):
            return BLOCKED, f"it depends on `{dep}`, which {'failed for good' if found == FAILED else 'is blocked'}"
        return (OPEN, "") if found == PASSED else (WAITING, f"it waits for `{dep}`")

    def state(tid: str, path: Tuple[str, ...]) -> Tuple[str, str]:
        if tid not in states:
            done = outcomes.get(tid, [])
            if "pass" in done:
                states[tid] = (PASSED, "")
            elif len(done) >= by_id[tid].max_attempts:
                states[tid] = (FAILED, "")
            else:
                found = [dep_state(dep, path) for dep in by_id[tid].depends_on]
                # The worst one counts: blocked, then held, then waiting.
                states[tid] = next((f for rank in (BLOCKED, HELD, WAITING) for f in found if f[0] == rank),
                                   (OPEN, ""))
        return states[tid]

    for task in tasks:
        state(task.id, (task.id,))
    return states


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
  args: { message: { type: "string", description: "The message. Keep it short; long ones are split into several Discord posts." } },
  async execute(args) {
    return "Queued for Adam. Keep working.";
  },
}
"""

_POOL_AGENT_MD = """---
description: Worker running the same model as you, on whichever local model server is free. Give it one self-contained subtask with everything it needs; run it in the background to keep working meanwhile. Check what it returns.
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
        _validate_mcp_options(where, entry)


def _validate_mcp_options(where: str, entry: Dict[str, Any]) -> None:
    """The optional fields opencode types; a wrong one fails the whole file too."""
    if "enabled" in entry and not isinstance(entry["enabled"], bool):
        raise ValueError(f"{where}: `enabled` must be true or false")
    timeout = entry.get("timeout")
    if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0):
        raise ValueError(f"{where}: `timeout` must be a positive whole number of milliseconds")
    for field in ("environment", "headers"):
        value = entry.get(field)
        if value is not None and not (isinstance(value, dict) and all(isinstance(v, str) for v in value.values())):
            raise ValueError(f"{where}: `{field}` must map names to strings")


def workspace_files(cfg: Dict[str, Any]) -> Dict[str, str]:
    """The files DERPR keeps in codr's workspace: `AGENTS.md` (codr's rules,
    which opencode loads into every session, DP-488) and `.opencode/`. The
    empty `node_modules/` plus package files stop opencode's npm install of its
    plugin package, which hangs ~60 s per instance start with the sandbox
    offline."""
    pkg = {"dependencies": {"@opencode-ai/plugin": "*"}}
    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {"kcpp-pool": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Inference router (any free local server)",
            "options": {"baseURL": cfg["pool_url"]},
            "models": {"local": {"name": "the model the router is serving",
                                 "limit": {"context": int(cfg["pool_context"]), "output": 16384}}},
        }},
    }
    if cfg.get("mcp"):
        config["mcp"] = cfg["mcp"]
    return {
        "AGENTS.md": AGENTS_MD.read_text(encoding="utf-8"),
        ".opencode/tool/notify_adam.js": _NOTIFY_JS % {"per_hour": int(cfg["notify_per_hour"])},
        ".opencode/agent/pool.md": _POOL_AGENT_MD,
        ".opencode/opencode.json": json.dumps(config, indent=2) + "\n",
        ".opencode/package.json": json.dumps(pkg) + "\n",
        ".opencode/package-lock.json": json.dumps({"packages": {"": pkg}}) + "\n",
    }


def sync_script(files: Dict[str, str]) -> str:
    """Bash that writes each file only when its content differs, then prints
    whether anything changed."""
    dirs = sorted({p.rsplit("/", 1)[0] for p in files if "/" in p} | {".opencode/node_modules"})
    lines = ["set -e", f"mkdir -p {' '.join(shlex.quote(d) for d in dirs)}", "changed=0"]
    for i, (path, content) in enumerate(files.items()):
        lines += [f"cat > '{path}.new' <<'DERPR_EOF_{i}'", content.rstrip("\n"), f"DERPR_EOF_{i}",
                  f"if cmp -s '{path}.new' '{path}'; then rm -f '{path}.new'; "
                  f"else mv '{path}.new' '{path}'; changed=1; fi"]
    lines.append(f'echo "{_CHANGED_MARK}$changed"')
    return "\n".join(lines) + "\n"


# --- tickets in codr's workspace (DP-488) -------------------------------------
#
# Every script below runs from the workspace root through `run_shell`. Paths
# and names are quoted with `shlex.quote`; ticket IDs and project names are
# also limited by `BRANCH_SAFE_ID_RE` / `PROJECT_RE`.

_MEMORY_FILES = {
    "MEMORY.md": (
        "# Memory index (L0)\n\n"
        "An index only: one line per directory. Read a directory's `_overview.md` (L1) before\n"
        "any detail file (L2) in it. Write bottom-up: the detail file, then its `_overview.md`,\n"
        "then this index if a directory's summary changed, and commit in the same turn.\n\n"
        "- `tickets/`: one file per ticket: the ticket, your notes on it, DERPR's attempt lines.\n"
        "- `projects/`: one file per project: what it is, its layout, how to test it.\n"),
    "tickets/_overview.md": (
        "# Tickets (L1)\n\n"
        "One line per ticket: ID, title, state, what it left behind. Detail in `<ID>.md`.\n"),
    "projects/_overview.md": (
        "# Projects (L1)\n\n"
        "One line per project under `projects/`: what it is. Detail in `<project>.md`.\n"),
}


def _git_commit_lines(repo: str, message: str, paths: List[str]) -> List[str]:
    """Bash that commits `paths` (relative to `repo`) as DERPR, and only them,
    if any of them changed: whatever codr has staged stays out of it."""
    r, quoted = shlex.quote(repo), " ".join(shlex.quote(p) for p in paths)
    return [
        f"git -C {r} add -- {quoted}",
        f"git -C {r} diff --cached --quiet -- {quoted} || "
        f"{_GIT_AS_DERPR} -C {r} commit -q --only -m {shlex.quote(message)} -- {quoted}",
    ]


def _git_identity_lines(repo: str) -> List[str]:
    """codr commits in its repos too; give it an author if the box has none."""
    r = shlex.quote(repo)
    return [f"git -C {r} config user.name >/dev/null || git -C {r} config user.name codr",
            f"git -C {r} config user.email >/dev/null || git -C {r} config user.email codr@codr.local"]


def memory_skeleton_lines() -> List[str]:
    """Bash that makes `memory/` a git repo and writes (and commits) the
    skeleton files that are missing. A file that exists is never touched: the
    notes are codr's."""
    lines = ["mkdir -p memory/tickets memory/projects",
             "[ -e memory/.git ] || git -C memory init -q -b main", *_git_identity_lines("memory")]
    for i, (path, content) in enumerate(_MEMORY_FILES.items()):
        target = shlex.quote(f"memory/{path}")
        lines += [f"if [ ! -e {target} ]; then cat > {target} <<'DERPR_MEM_{i}'", content.rstrip("\n"),
                  f"DERPR_MEM_{i}", *_git_commit_lines("memory", f"DERPR: add {path}", [path]), "fi"]
    return lines


def memory_skeleton_script() -> str:
    return "\n".join(["set -e", *memory_skeleton_lines(), f"echo {_MEMORY_MARK}"]) + "\n"


def ticket_record(task: CodrTask) -> str:
    """The first version of `memory/tickets/<id>.md`. From then on it is
    codr's, except for the attempt lines DERPR appends."""
    title = f"{task.id}: {task.title}" if task.title else task.id
    deps = ", ".join(f"`{d}`" for d in task.depends_on) or "none"
    return (f"# {title}\n\nProject: `{task.project_name}`. Depends on: {deps}. "
            f"Worktree: `{task.worktree}/`, branch `{task.branch}`.\n\n"
            f"## Ticket\n\n{task.text}\n\n"
            "## Notes\n\nWhat you tried, what happened, what you decided.\n\n"
            "## Attempts (one line each, added by DERPR)\n\n")


def ticket_archive(task: CodrTask) -> bytes:
    """One tar.gz holding the ticket record (`ticket.md`) and the seed files
    (`seed/<path>`). It goes in through `write_file`, so a seed of any size
    fits (DP-455)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel, data in (("ticket.md", ticket_record(task).encode()),
                          *((f"seed/{rel}", data) for rel, data in task.seed)):
            info = tarfile.TarInfo(rel)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _copy_missing(src: str, dest: str) -> str:
    """Bash copying the tree under `src` into `dest`, skipping files `dest`
    already has (what `tar --skip-old-files` does; `cp -n`'s exit status
    differs between coreutils versions)."""
    return f"tar -C {src} -cf - . | tar -xf - -C {dest} --skip-old-files"


def ticket_script(task: CodrTask, staged: str) -> str:
    """Bash that gets a ticket ready for an attempt, from `staged`
    (`ticket_archive`):

    - the `memory/` skeleton, and `memory/tickets/<id>.md` from the ticket if
      it is not there yet (committed);
    - `projects/<project>/` as a git repo with an empty first commit on
      `main`, unless a repo is already there (its checked-out branch is its
      main);
    - `worktrees/<id>/` on branch `ticket/<id>`, cut from main, if it is
      missing (a retry reuses it). A new one first gets the files of a
      pre-DP-488 `tasks/<id>/` folder, so an in-flight task keeps its work;
    - the seed files the worktree does not have yet, so a retry keeps what
      codr changed."""
    q = shlex.quote
    proj, wt, br, rec = q(task.project_dir), q(task.worktree), q(task.branch), f"tickets/{task.id}.md"
    legacy = q(f"tasks/{task.id}")
    return "\n".join([
        "set -e",
        f"tmp=$(mktemp -d .derpr-ticket-XXXXXX); trap 'rm -rf -- \"$tmp\" {q(staged)}' EXIT",
        f'tar -xzf {q(staged)} -C "$tmp"',
        *memory_skeleton_lines(),
        f"if [ ! -e {q('memory/' + rec)} ]; then cp -- \"$tmp/ticket.md\" {q('memory/' + rec)}",
        *_git_commit_lines("memory", f"{task.id}: ticket record", [rec]), "fi",
        f"if [ ! -e {proj}/.git ]; then",
        f"  if [ -d {proj} ] && [ -n \"$(ls -A {proj})\" ]; then",
        f"    echo {q(task.project_dir + ' exists but is not a git repo; not touching it')}; exit 1; fi",
        f"  mkdir -p {proj} && git -C {proj} init -q -b main",
        "fi",
        *_git_identity_lines(task.project_dir),
        f"git -C {proj} rev-parse -q --verify HEAD >/dev/null || "
        f"{_GIT_AS_DERPR} -C {proj} commit -q --allow-empty -m {q('DERPR: start ' + task.project_name)}",
        f"main=$(git -C {proj} symbolic-ref -q --short HEAD) || "
        f"{{ echo {q(task.project_dir + ' has no branch checked out')}; exit 1; }}",
        f"if [ ! -d {wt} ]; then",
        f"  git -C {proj} worktree prune",
        f"  if git -C {proj} show-ref -q --verify {q('refs/heads/' + task.branch)}; then",
        f"    git -C {proj} worktree add -q \"$PWD\"/{wt} {br}",
        f"  else git -C {proj} worktree add -q -b {br} \"$PWD\"/{wt} \"$main\"; fi",
        f"  if [ -d {legacy} ]; then {_copy_missing(legacy, wt)}; fi",
        "fi",
        f'if [ -d "$tmp/seed" ]; then {_copy_missing(chr(34) + "$tmp/seed" + chr(34), wt)}; fi',
        f"echo {_READY_MARK}",
    ]) + "\n"


async def prepare_ticket(client: OpencodeClient, task: CodrTask) -> None:
    """Run `ticket_script` on codr's box. Raises if it did not finish."""
    staged = f".derpr-ticket-{task.id}.tar.gz"
    await client.write_file(staged, ticket_archive(task))
    out = await client.run_shell(ticket_script(task, staged))
    if _READY_MARK not in out:
        raise RuntimeError(f"codr: preparing ticket {task.id} failed: {out[-500:]}")


def check_script(task: CodrTask, timeout_s: int) -> str:
    # The check travels as an argument, not on stdin: fed as a heredoc, any
    # command in it that reads stdin swallows the lines after it, and the check
    # exits 0 without having run them (DP-463).
    return (
        f"cd -- {shlex.quote(task.worktree)} 2>/dev/null || "
        f"{{ echo {shlex.quote(f'no {task.worktree} folder in the workspace')}; exit 1; }}\n"
        f"timeout {int(timeout_s)} bash -c {shlex.quote(task.check.rstrip())} </dev/null\n"
        f'echo "{_EXIT_MARK}$?"\n'
    )


def merge_script(task: CodrTask) -> str:
    """Bash run after a passing check: commit what codr left uncommitted,
    merge the ticket branch into the project's main with a merge commit, then
    remove the worktree (the branch stays). A conflicted merge is aborted and
    its files listed after `_CONFLICT_MARK`; the worktree stays."""
    q = shlex.quote
    proj, wt, br = q(task.project_dir), q(task.worktree), q(task.branch)
    return "\n".join([
        "set -u",
        f"main=$(git -C {proj} symbolic-ref -q --short HEAD) || "
        f"{{ echo {q('no branch checked out in ' + task.project_dir)}; exit 1; }}",
        f'echo "{_MAIN_MARK}$main"',
        f"git -C {wt} add -A || exit 1",
        f"git -C {wt} diff --cached --quiet || "
        f"{_GIT_AS_DERPR} -C {wt} commit -q -m {q(task.id + ': uncommitted at check pass')} || exit 1",
        f"if out=$({_GIT_AS_DERPR} -C {proj} merge --no-ff -m {q(task.id + ': merge ' + task.branch)} {br} 2>&1); then",
        f'  echo "{_MERGED_MARK}$(git -C {proj} rev-parse --short HEAD)"',
        f"  git -C {proj} worktree remove --force \"$PWD\"/{wt} || echo 'worktree not removed'",
        f"  git -C {proj} worktree prune",
        "else",
        f"  files=$(git -C {proj} diff --name-only --diff-filter=U)",
        f"  git -C {proj} merge --abort 2>/dev/null",
        f'  if [ -n "$files" ]; then echo "{_CONFLICT_MARK}"; echo "$files"; else echo "$out"; fi',
        "fi",
    ]) + "\n"


def parse_merge(output: str) -> Tuple[Optional[str], Optional[str], str]:
    """(merge commit, main branch, problem). A conflict's problem lists its
    files; no merge commit and no conflict = the merge failed some other way."""
    main = re.search(rf"{_MAIN_MARK}(\S+)", output)
    merged = re.search(rf"{_MERGED_MARK}(\S+)", output)
    branch = main.group(1) if main else None
    if merged:
        return merged.group(1), branch, ""
    if _CONFLICT_MARK in output:
        files = output.split(_CONFLICT_MARK, 1)[1].strip()
        return None, branch, f"merging into `{branch}` conflicted in:\n{files}"
    return None, branch, f"the merge into the project's main failed:\n{output.strip()[-1500:]}"


def ticket_note_script(task_id: str, line: str) -> str:
    """Bash that appends one line to `memory/tickets/<id>.md` and commits it."""
    rec = f"tickets/{task_id}.md"
    return "\n".join([
        "set -e",
        f"printf '%s\\n' {shlex.quote(line)} >> {shlex.quote('memory/' + rec)}",
        *_git_commit_lines("memory", f"{task_id}: {line[:60]}", [rec]),
        f"echo {_NOTED_MARK}",
    ]) + "\n"


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
        return load_tasks(global_config.CODR_TASKS_DIR, int(self.cfg["max_attempts"]))

    def _all_attempts(self) -> Dict[str, List[Dict[str, Any]]]:
        """Recorded attempts by ticket id, oldest first."""
        rows = self.memory_manager.get_agent_actions(self.agent_name, limit=10000, action_type=ATTEMPT_ACTION)
        by_task: Dict[str, List[Dict[str, Any]]] = {}
        for r in reversed(rows):
            context = str(r.get("trigger_context") or "")
            if context.startswith(attempt_context("")):
                by_task.setdefault(context[len(attempt_context("")):], []).append(r)
        return by_task

    def _attempts(self, task_id: str) -> List[Dict[str, Any]]:
        """Recorded attempts of `task_id`, oldest first."""
        return self._all_attempts().get(task_id, [])

    async def _notice_once(self, task_id: str, message: str) -> None:
        """Log and post `message` about a ticket, once: the action log keeps
        which notices went out, so a restart does not repeat them either. A
        changed message (a different problem) is a new notice."""
        key = f"notice:{task_id}:{hashlib.sha1(message.encode()).hexdigest()[:12]}"
        rows = self.memory_manager.get_agent_actions(self.agent_name, limit=2000, action_type=NOTICE_ACTION)
        if any(r.get("trigger_context") == key for r in rows):
            return
        logger.warning("codr agent: ticket %s: %s", task_id, message)
        self._log_task_root(NOTICE_ACTION, trigger_context=key, action_payload={"message": message},
                            outcome="posted")
        await self._post(f"codr ticket `{task_id}`: {message}")

    async def _post(self, body: str) -> bool:
        target = self.cfg.get("notification_target") or {}
        if not target.get("recipient"):
            logger.warning("codr agent: no notification_target configured; not posting: %s", body[:200])
            return False
        try:
            return await self.notification_router.send(
                channel=target.get("channel", "discord_channel"), recipient=str(target["recipient"]),
                subject="", body=body)
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
            # opencode reads `.opencode/` and `AGENTS.md` when it starts the
            # directory's instance; disposing it (only while idle) makes it
            # read them again.
            self._dispose_pending = True
        elif f"{_CHANGED_MARK}0" not in out:
            raise RuntimeError(f"codr workspace setup failed: {out[-500:]}")
        out = await client.run_shell(memory_skeleton_script())
        if _MEMORY_MARK not in out:
            raise RuntimeError(f"codr workspace setup failed (memory/): {out[-500:]}")
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
                # Whole, here and in `_post`: the Discord client splits long
                # posts (DP-485). Only the action-log copy is capped, to fit its row.
                if await self._post(f"codr: {text}"):
                    await self._commit_reply(text)
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
        """Start the oldest open ticket (`queue_states`), after posting, once
        each, the tickets that were skipped or will not start."""
        tasks, errors = load_tickets(global_config.CODR_TASKS_DIR, int(self.cfg["max_attempts"]))
        for task_id, error in errors.items():
            await self._notice_once(task_id, f"skipped: {error}")
        attempts = self._all_attempts()
        # Every attempted ticket, queued or not: a passed one whose folder is gone still counts.
        states = queue_states(tasks, {tid: [str(a.get("outcome")) for a in rows]
                                      for tid, rows in attempts.items()})
        for task in tasks:
            state, why = states[task.id]
            if state in NOTICE_STATES:
                await self._notice_once(task.id, f"{state}: {why}")
        for task in tasks:
            if states[task.id][0] == OPEN:
                done = attempts.get(task.id, [])
                if not await self._start(client, sid, task, len(done) + 1, done[-1] if done else None):
                    return
                # Its project could not be prepared: try the next open ticket
                # rather than stall the whole queue behind it.

    def _task_prompt(self, task: CodrTask, attempt: int, previous: Optional[Dict[str, Any]]) -> str:
        title = f" {task.title}" if task.title else ""
        lines = [
            f"[Ticket `{task.id}`{title}, attempt {attempt} of {task.max_attempts}, "
            f"project `{task.project_name}`, started by DERPR's task queue]",
            "",
            task.text,
            "",
            f"Work only in your worktree `{task.worktree}/` (branch `{task.branch}`, cut from the main "
            f"branch of `{task.project_dir}/`), and commit there. Follow `AGENTS.md`.",
            f"Your notes on this ticket go in `memory/tickets/{task.id}.md`: what you tried, what "
            "happened, what you decided. Read it, and `memory/MEMORY.md`, before you start.",
            f"`{NOTIFY_TOOL}` tells Adam something now without ending your turn. Your final reply is not "
            f"posted, so when you are done, tell Adam with `{NOTIFY_TOOL}` what you did and anything he "
            f"should know, then end your turn. DERPR then runs its own check in `{task.worktree}/`; you "
            "will see its output if it fails. If it passes, DERPR merges your branch into main.",
            "For an independent subtask you can use the `pool` subagent (task tool, background if you "
            "want to keep working).",
        ]
        if previous:
            lines += self._previous_lines(task, previous)
        return "\n".join(lines)

    @staticmethod
    def _previous_lines(task: CodrTask, previous: Dict[str, Any]) -> List[str]:
        try:
            detail = json.loads(previous.get("outcome_payload") or "{}")
        except ValueError:
            detail = {}
        lines = ["", f"The previous attempt ended `{previous.get('outcome')}`."]
        if detail.get("reason"):
            lines.append(f"Reason: {detail['reason']}")
        if detail.get("merge_conflict"):
            main = detail.get("main_branch") or "main"
            lines.append(f"The check passed, but merging `{task.branch}` into `{main}` conflicted, so "
                         f"nothing was merged. Merge `{main}` into your branch in `{task.worktree}/`, "
                         "resolve the conflicts, run the tests and commit.")
        if detail.get("check_tail"):
            lines += ["Check output (tail):", "```", detail["check_tail"], "```"]
        return lines

    async def _start(self, client: OpencodeClient, sid: str, task: CodrTask, attempt: int,
                     previous: Optional[Dict[str, Any]]) -> bool:
        """Start `task`. True only when its project could not be prepared, so
        the caller may try the next open ticket; False once it started, or
        when nothing can start this tick (busy, locked, no persona)."""
        persona = self.chat_system.personas.get(self.persona_name)
        if persona is None:
            logger.error("codr agent: persona %r not found; not starting tasks.", self.persona_name)
            return False
        lock = workspace_lock(self.directory)
        if lock.locked():
            return False            # a chat turn is starting or running; next tick
        async with lock:
            msgs = await client.turn_messages(sid)
            start = current_turn_start(msgs)
            if await client.is_busy(sid) or (start is not None and not turn_finished(msgs, start)
                                             and open_auto(msgs[start]) is not None):
                return False
            try:
                await prepare_ticket(client, task)
            except RuntimeError as e:
                # Retried every tick, but posted once: e.g. a non-repo folder
                # in the project's place needs a human.
                await self._notice_once(task.id, f"could not start: {e}")
                return True
            meta = {"task": task.id, "attempt": attempt, "nudges": 0, "started": time.time()}
            await self._prompt(client, sid, persona, self._task_prompt(task, attempt, previous), meta)
        logger.info("codr agent: started task %s attempt %d", task.id, attempt)
        return False

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
        if task is not None:
            await self._note_attempt(client, task, auto, "aborted", {"reason": reason})

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
            outcome = await self._check(client, task, detail)
        if error:
            detail["turn_error"] = f"{error.get('name', 'error')} {(error.get('data') or {}).get('message', '')}".strip()
        # Claim before recording and posting, so a crash cannot post twice.
        await client.patch_part(self._auto_part(msgs[start]), **{AUTO_FIELD: {**auto, "result": outcome}})
        await self._record(auto, outcome, detail, task)
        if task is not None:
            await self._note_attempt(client, task, auto, outcome, detail)

    async def _check(self, client: OpencodeClient, task: CodrTask, detail: Dict[str, Any]) -> str:
        """Run the check in the ticket's worktree and, if it passes, merge the
        branch (`merge_script`). Fills `detail`; returns the outcome. A merge
        that does not go through makes the attempt a `fail`."""
        output = await client.run_shell(
            check_script(task, int(self.cfg["check_timeout_seconds"])),
            timeout=aiohttp.ClientTimeout(total=int(self.cfg["check_timeout_seconds"]) + 60))
        passed, body = parse_check(output)
        detail["check_tail"] = body[-CHECK_TAIL_CHARS:]
        if not passed:
            return "fail"
        merged, main, problem = parse_merge(await client.run_shell(merge_script(task)))
        detail["main_branch"] = main
        if merged:
            detail["merged"] = merged
            return "pass"
        detail["reason"] = problem.split("\n", 1)[0]
        detail["merge_conflict"] = "conflicted" in detail["reason"]
        detail["check_tail"] = problem[-CHECK_TAIL_CHARS:]
        return "fail"

    async def _note_attempt(self, client: OpencodeClient, task: CodrTask, auto: Dict[str, Any],
                            outcome: str, detail: Dict[str, Any]) -> None:
        """One line in `memory/tickets/<id>.md` per recorded attempt. Best
        effort: the attempt is already recorded and posted."""
        line = f"- attempt {auto.get('attempt')}: {outcome}"
        if detail.get("merged"):
            line += f", merged into `{detail.get('main_branch')}` as {detail['merged']}"
        elif detail.get("reason"):
            line += f": {detail['reason']}"
        try:
            out = await client.run_shell(ticket_note_script(task.id, line))
        except Exception as e:  # noqa: BLE001 — the note must not undo the recorded attempt
            out = str(e)
        if _NOTED_MARK not in out:
            logger.warning("codr agent: could not note %s attempt %s in memory/: %s",
                           task.id, auto.get("attempt"), out[-300:])

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
            ATTEMPT_ACTION, trigger_context=attempt_context(task_id),
            action_payload={"task": task_id, "attempt": auto.get("attempt")},
            contexts=[("task", task_id)])
        self._finalize_action(action_id, outcome, self._fit_payload(detail))
        total = task.max_attempts if task else "?"
        line = (f"codr task `{task_id}` attempt {auto.get('attempt')}/{total}: **{outcome.upper()}** "
                f"({seconds // 60} min)")
        if detail.get("merged") and task:
            line += f", merged into `{task.project_name}` {detail.get('main_branch')}"
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
