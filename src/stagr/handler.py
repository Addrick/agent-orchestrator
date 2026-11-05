"""The stagr tools (DP-456): a setup agent's hands on the setup VM.

stagr is a second `opencode serve`, used here only for its shell endpoint
(`OpencodeClient.run_shell`), exactly as the codr agent uses codr's. The VM's
LAN and WAN are switches, off by default (DP-465, ``src/stagr/net.py``), and
its WAN goes only through a proxy allowlist, so `stagr_shell` runs unapproved:
the network is gated, not the local work. What crosses a boundary is gated
(``is_write``) instead:

- `request_network` turns stagr's LAN or WAN on, and `request_web_access`
  approves one more host while its WAN is on; both through the node verbs over
  the shared node gate (`run_node_command`). The operator's portal panel flips
  any switch of either box directly (`set_sandbox_net`);
- `copy_to_codr` / `copy_from_codr` copy a file or folder between stagr and
  codr's offline sandbox, to a path that must not exist yet. Any `.git` is left
  behind unless asked for, so upstream history (a later fix, say) never reaches
  codr by accident;
- `queue_codr_task` writes a codr ticket (DP-489) into the codr task queue
  (`global_config.CODR_TASKS_DIR`, DP-481): TASK.md (the header of
  `src/stagr/codr_ticket.py`, then stagr's Spec, Plan and Packages) and the
  hidden check.sh. DERPR picks the ID, `CD-NNN`, one past the highest it has
  ever seen in the queue folder or the codr agent's attempt log, so an ID is
  never reused. The ticket is staged in a dot-folder named after its ID (the
  agent skips it, the next allocation counts it) and renamed into place, so a
  tick never sees half a ticket. stagr does the planning before it: packages
  are resolved against codr's installed state and installed there (DP-479,
  the persona prompt).

`codr_shell` runs unapproved too, in codr's workspace (DP-462): codr-vm is
offline and is restored from a backup if a command wrecks it. It holds codr's
workspace lock and runs only while every session there is idle, so the
environment changes between tasks, never under one. It refuses a command that
runs `opencode` (DP-481): that starts a codr session the codr agent never
watches (no loop guard, time limit or check), held open only by this call.

Files move with `OpencodeClient.write_file` / `read_file` (DP-455), as one
tar.gz each way; derpr is the only thing that talks to both boxes.
"""

from __future__ import annotations

import asyncio
import logging
import posixpath
import re
import shlex
import shutil
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import aiohttp

from config import global_config
from src.engine.providers.opencode import (
    OpencodeClient, auth_headers, open_auto, persona_sessions, workspace_directory, workspace_held, workspace_lock,
)
from src.deferral_kinds import DEFERRAL_KIND_CODR_TICKET, declare_deferral
from src.llm_errors import LLMCommunicationError
from src.proxmox.ssh import SSHRunner
from src.stagr.codr_ticket import (
    ATTEMPT_ACTION, CODR_AGENT_NAME, PROJECT_RE, attempt_context, render_task_md, ticket_id, ticket_number,
)
from src.stagr.net import BOXES, ZONES, SandboxNet

if TYPE_CHECKING:
    from src.memory.memory_manager import MemoryManager
    from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)

CODR_PERSONA = "codr"
STAGR_BOX = "stagr"
PATH_RE = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")
MAX_SHELL_SECONDS = 3600
DEFAULT_SHELL_SECONDS = 600
# What one copy may carry, compressed. write_file moves ~0.5 MB/s.
MAX_TRANSFER_BYTES = 200 * 1024 * 1024
_EXIT_MARK = "__STAGR_EXIT="
_OK_MARK = "__STAGR_OK"
_LONG = aiohttp.ClientTimeout(total=600)
# `opencode` as a command (`opencode run`, `~/.opencode/bin/opencode`), not
# `.opencode/` or `opencode.json`.
_OPENCODE_CMD = re.compile(r"(?:^|[^\w.-])opencode(?=\s|$|[;&|)])")
TASK_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# The highest ticket number handed out, kept in the queue folder so a ticket
# deleted before its first attempt still keeps its number.
HIGH_WATER_FILE = ".last-ticket-number"
_QUEUE_LOCK = threading.Lock()


class _Refused(Exception):
    """A transfer step that stopped for a reason the caller should be told."""


def _error(message: str) -> Dict[str, Any]:
    return {"status": "error", "message": message}


def _clip(text: str, limit: int) -> str:
    """Keep the start and (mostly) the end of long output: errors land last."""
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head
    return f"{text[:head]}\n[... {len(text) - limit} characters cut ...]\n{text[-tail:]}"


def _rel_path(path: str) -> Optional[str]:
    """`path` as a safe path relative to a workdir, or None."""
    path = (path or "").strip()
    if path.startswith("/"):
        return None
    path = path.rstrip("/")
    if path in ("", "."):
        return "."
    if not PATH_RE.match(path) or ".." in path.split("/"):
        return None
    return path


def _onoff(value: Optional[bool]) -> str:
    return "unknown" if value is None else ("on" if value else "off")


def _flag(value: Any) -> bool:
    """A boolean tool argument; models sometimes send it as a string."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def _shell_script(command: str, timeout_seconds: Optional[int]) -> Union[Tuple[str, int], Dict[str, Any]]:
    """The bash that runs `command` under a timeout and reports its exit code,
    plus that timeout; or the error to return."""
    if not (command or "").strip():
        return _error("command is empty.")
    try:
        seconds = DEFAULT_SHELL_SECONDS if timeout_seconds is None else int(timeout_seconds)
    except (TypeError, ValueError):
        return _error("timeout_seconds must be a whole number of seconds.")
    seconds = min(max(seconds, 1), MAX_SHELL_SECONDS)
    # The command runs in its own bash, so an `exit` in it still reports. It
    # travels as an argument, not on stdin: a heredoc would be read by any
    # command in it that reads stdin, swallowing the lines after it.
    return (f"timeout {seconds} bash -c {shlex.quote(command.rstrip())} </dev/null\n"
            f'echo "{_EXIT_MARK}$?"\n'), seconds


def _shell_result(out: str, seconds: int) -> Dict[str, Any]:
    # The last marker, wherever it is: stderr may arrive after stdout.
    marks = list(re.finditer(rf"{_EXIT_MARK}(\d+)\n?", out))
    body = out[:marks[-1].start()] + out[marks[-1].end():] if marks else out
    exit_code = int(marks[-1].group(1)) if marks else None
    result: Dict[str, Any] = {"status": "ok", "exit_code": exit_code,
                              "output": _clip(body, global_config.STAGR_OUTPUT_CHARS)}
    if exit_code == 124:
        result["note"] = f"timed out after {seconds} s"
    return result


class StagrToolHandler:
    def __init__(self, runner: SSHRunner | None = None, net_runner: SSHRunner | None = None,
                 memory_manager: "MemoryManager | None" = None) -> None:
        self._net = SandboxNet(runner, net_runner)
        # The codr agent's attempt log: where a ticket ID was used before.
        self._memory = memory_manager

    def register(self, manager: "ToolManager") -> None:
        manager.register("stagr_shell", self._stagr_shell)
        manager.register("list_web_access", self._list_web_access)
        manager.register("request_web_access", self._request_web_access)
        manager.register("request_network", self._request_network, enricher=self._request_network_warning)
        manager.register("sandbox_net_status", self._sandbox_net_status)
        manager.register("set_sandbox_net", self._set_sandbox_net)
        manager.register("codr_shell", self._codr_shell)
        manager.register("copy_to_codr", self._copy_to_codr)
        manager.register("copy_from_codr", self._copy_from_codr)
        manager.register("queue_codr_task", self._queue_codr_task, enricher=self._queue_preview)

    # -- clients ---------------------------------------------------------------

    @staticmethod
    def _configured() -> Optional[Dict[str, Any]]:
        if not global_config.STAGR_SERVER_URL:
            return _error("stagr is not configured (set STAGR_SERVER_URL and STAGR_SERVER_PASSWORD).")
        return None

    @staticmethod
    def _stagr(http: aiohttp.ClientSession) -> OpencodeClient:
        return OpencodeClient(http, global_config.STAGR_WORKDIR, base_url=global_config.STAGR_SERVER_URL)

    @staticmethod
    def _stagr_session() -> aiohttp.ClientSession:
        return aiohttp.ClientSession(headers=auth_headers(global_config.STAGR_SERVER_PASSWORD))

    # -- shells ------------------------------------------------------------------

    async def _stagr_shell(self, command: str, timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
        if (err := self._configured()):
            return err
        prepared = _shell_script(command, timeout_seconds)
        if isinstance(prepared, dict):
            return prepared
        script, seconds = prepared
        try:
            async with self._stagr_session() as http:
                out = await self._stagr(http).run_shell(
                    script, timeout=aiohttp.ClientTimeout(total=seconds + 60))
        except LLMCommunicationError as e:
            return _error(f"stagr shell failed: {e}")
        except TimeoutError:
            return _error(f"stagr shell gave no answer within {seconds + 60} s.")
        return _shell_result(out, seconds)

    async def _codr_shell(self, command: str, timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
        if _OPENCODE_CMD.search(command or ""):
            return _error("codr_shell does not run opencode: a codr started that way is not watched "
                          "by DERPR (no loop guard, time limit or check) and dies when DERPR restarts. "
                          "Queue the task with queue_codr_task instead.")
        prepared = _shell_script(command, timeout_seconds)
        if isinstance(prepared, dict):
            return prepared
        script, seconds = prepared
        _, _, codr_dir = workspace_directory(CODR_PERSONA)
        lock = workspace_lock(codr_dir)
        if lock.locked():
            return _error("codr is starting or running a turn; try again when it is idle.")
        try:
            # Held for the whole command, so codr starts nothing while it runs.
            async with lock, aiohttp.ClientSession(headers=auth_headers()) as http:
                with workspace_held(codr_dir, "stagr's codr_shell", seconds + 60):
                    codr = OpencodeClient(http, codr_dir)
                    if not await self._codr_idle(codr):
                        return _error("codr is working, or its last task turn is not checked yet; "
                                      "try again when it is idle.")
                    out = await codr.run_shell(script, timeout=aiohttp.ClientTimeout(total=seconds + 60))
        except LLMCommunicationError as e:
            return _error(f"codr shell failed: {e}")
        except TimeoutError:
            return _error(f"codr shell gave no answer within {seconds + 60} s.")
        return _shell_result(out, seconds)

    # -- network (DP-465) ----------------------------------------------------------

    async def _list_web_access(self) -> Dict[str, Any]:
        """stagr's own view: its two switches and the hosts its WAN lets through."""
        st = await self._net.status(STAGR_BOX, detail=False)
        if st.get("status") == "error":
            return st
        return {"status": "ok", "lan": _onoff(st["lan"]), "wan": _onoff(st["wan"]),
                "allowlist": st["allowlist"], **({"errors": st["errors"]} if "errors" in st else {})}

    async def _request_web_access(self, host: str = "", reason: str = "", domain: str = "",
                                  minutes: Any = None) -> Dict[str, Any]:
        # `domain` and `minutes` are the DP-456 shape: a call parked before
        # DP-465 deployed still resolves (minutes is ignored; there is no timer).
        result = await self._net.allow(STAGR_BOX, host or domain)
        if result.get("status") == "ok":
            logger.info("stagr: web access to %s approved (%s)", host or domain, reason)
        return result

    async def _request_network_warning(self, zone: str = "", **_: Any) -> Optional[str]:
        """The approval prompt's enrichment: the LAN+WAN warning when the other
        switch is already on, so Adam sees it before he approves."""
        zone = (zone or "").strip().lower()
        return await self._net.both_on_warning(STAGR_BOX, zone) if zone in ZONES else None

    async def _request_network(self, zone: str, reason: str = "") -> Dict[str, Any]:
        """Runs only once the operator approved it, so the approval is the
        confirmation: the prompt carried the LAN+WAN warning (the enricher
        above), and it does not stop the call a second time."""
        zone = (zone or "").strip().lower()
        if zone not in ZONES:
            return _error("zone must be lan or wan.")
        result = await self._net.set_switch(STAGR_BOX, zone, True, confirm=True)
        if result.get("status") == "ok":
            logger.info("stagr: %s turned on (%s)", zone, reason)
        return result

    async def _sandbox_net_status(self) -> Dict[str, Any]:
        """Every box, in full: the portal panel's read."""
        boxes = await asyncio.gather(*(self._net.status(b) for b in BOXES))
        return {"status": "ok", "boxes": list(boxes)}

    async def _set_sandbox_net(self, box: str, zone: str, state: str, confirm: Any = False) -> Dict[str, Any]:
        state = (state or "").strip().lower()
        if state not in ("on", "off"):
            return _error("state must be on or off.")
        return await self._net.set_switch((box or "").strip().lower(), (zone or "").strip().lower(),
                                          state == "on", confirm=_flag(confirm))

    # -- transfers -------------------------------------------------------------------

    @staticmethod
    async def _codr_idle(client: OpencodeClient) -> bool:
        """Every session in codr's workspace idle, and no task turn of the codr
        agent still open: a turn that has ended but whose check has not run
        yet (the next tick runs it) is still under its task."""
        status = await client.request("GET", "/session/status") or {}
        if not all((v or {}).get("type", "idle") == "idle" for v in status.values()):
            return False
        # codr's own session and each ticket's (DP-484).
        for s in persona_sessions(await client.request("GET", "/session") or [], CODR_PERSONA):
            if any(open_auto(m) is not None for m in await client.turn_messages(str(s["id"]))):
                return False
        return True

    @staticmethod
    async def _pack(client: OpencodeClient, path: str, *, keep_git: bool) -> bytes:
        """`path` (a file or folder) on that box as a tar.gz holding it under
        its own name."""
        parent, name = posixpath.split(path)
        if name == ".git" and not keep_git:
            # --exclude=.git would drop the one thing asked for.
            raise _Refused(f"{path} is a .git folder; set keep_git to copy it.")
        staged = f".stagr-out-{uuid.uuid4().hex[:8]}.tar.gz"
        exclude = "" if keep_git else "--exclude=.git "
        out = await client.run_shell(
            f"[ -e {shlex.quote(path)} ] && tar -czf {staged} {exclude}-C {shlex.quote(parent or '.')} "
            f'-- {shlex.quote(name)} && echo "{_OK_MARK} $(stat -c %s {staged})"', timeout=_LONG)
        try:
            packed = re.search(rf"{_OK_MARK} (\d+)", out)
            if not packed:
                raise _Refused(f"could not pack {path}: {out[-500:] or 'no such file or folder'}")
            # Checked on the box, before the archive is pulled into derpr's memory.
            size = int(packed.group(1))
            if size > MAX_TRANSFER_BYTES:
                raise _Refused(f"{path} is {size} bytes compressed; the limit is {MAX_TRANSFER_BYTES}.")
            return await client.read_file(staged)
        finally:
            await client.run_shell(f"rm -f -- {staged}")

    @staticmethod
    async def _refuse_existing(client: OpencodeClient, dest: str, box: str) -> None:
        """Checked before anything is packed, so a doomed copy moves no bytes."""
        exists = await client.run_shell(f"[ -e {shlex.quote(dest)} ] && echo EXISTS || true")
        if "EXISTS" in exists:
            raise _Refused(f"{dest} already exists on {box}, and a copy never overwrites: "
                           "pick a new path or remove it first.")

    @staticmethod
    async def _unpack(client: OpencodeClient, archive: bytes, name: str, dest: str) -> None:
        """Unpack `archive` (which holds `name`) and move it to `dest`, which
        must not exist. It is unpacked beside `dest` first and moved in one
        rename, so a failure leaves no half copy. The staged archive (and a
        partial upload of it) is removed whatever happens."""
        tag = uuid.uuid4().hex[:8]
        parent = posixpath.dirname(dest)
        staged = f".stagr-in-{tag}.tar.gz"
        tmp = shlex.quote(posixpath.join(parent, f".stagr-in-{tag}"))
        mkparent = f"mkdir -p -- {shlex.quote(parent)} && " if parent else ""
        try:
            await client.write_file(staged, archive)
            out = await client.run_shell(
                f"{mkparent}mkdir -- {tmp} && tar -xzf {staged} -C {tmp} && "
                f"[ ! -e {shlex.quote(dest)} ] && mv -T -- {tmp}/{shlex.quote(name)} {shlex.quote(dest)} "
                f"&& echo {_OK_MARK}; rm -rf -- {tmp}", timeout=_LONG)
        finally:
            await client.run_shell(f"rm -f -- {staged} {staged}.derpr-part")
        if _OK_MARK not in out:
            raise _Refused(f"unpacking to {dest} failed: {out[-500:] or 'it exists now, or the copy was empty'}")

    @staticmethod
    def _copy_paths(source: str, dest: str) -> Tuple[Optional[str], Optional[str]]:
        """Both as safe relative paths naming something (not the workdir
        itself), or None."""
        src, dst = _rel_path(source), _rel_path(dest)
        return (None if src == "." else src), (None if dst == "." else dst)

    async def _copy_to_codr(self, source: str, dest: str, keep_git: Any = False) -> Dict[str, Any]:
        if (err := self._configured()):
            return err
        src, dst = self._copy_paths(source, dest)
        if src is None:
            return _error(f"source must name a file or folder relative to {global_config.STAGR_WORKDIR}, "
                          f"with no '..': {source!r}")
        if dst is None:
            return _error(f"dest must be a new path relative to codr's workspace, with no '..': {dest!r}")
        keep = _flag(keep_git)
        _, _, codr_dir = workspace_directory(CODR_PERSONA)
        lock = workspace_lock(codr_dir)
        if lock.locked():
            return _error("codr is starting or running a turn; try again when it is idle.")
        try:
            async with aiohttp.ClientSession(headers=auth_headers()) as http:
                await self._refuse_existing(OpencodeClient(http, codr_dir), dst, "codr")
            async with self._stagr_session() as http:
                archive = await self._pack(self._stagr(http), src, keep_git=keep)
            async with lock, aiohttp.ClientSession(headers=auth_headers()) as http:
                with workspace_held(codr_dir, "stagr's copy_to_codr"):
                    codr = OpencodeClient(http, codr_dir)
                    if not await self._codr_idle(codr):
                        raise _Refused("codr is working, or its last task turn is not checked yet; "
                                       "copy when it is idle.")
                    await self._unpack(codr, archive, posixpath.basename(src), dst)
        except _Refused as e:
            return _error(str(e))
        except LLMCommunicationError as e:
            return _error(f"copy to codr failed: {e}")
        except TimeoutError:
            return _error("copy to codr failed: a call to stagr or codr's sandbox timed out.")
        return {"status": "ok", "copied": f"stagr {src} -> codr {dst}", "bytes": len(archive),
                "git": "kept" if keep else "any .git was left behind"}

    async def _copy_from_codr(self, source: str, dest: str, keep_git: Any = False) -> Dict[str, Any]:
        if (err := self._configured()):
            return err
        src, dst = self._copy_paths(source, dest)
        if src is None:
            return _error(f"source must name a file or folder relative to codr's workspace, "
                          f"with no '..': {source!r}")
        if dst is None:
            return _error(f"dest must be a new path relative to {global_config.STAGR_WORKDIR}, "
                          f"with no '..': {dest!r}")
        keep = _flag(keep_git)
        _, _, codr_dir = workspace_directory(CODR_PERSONA)
        try:
            async with self._stagr_session() as http:
                await self._refuse_existing(self._stagr(http), dst, "stagr")
            async with aiohttp.ClientSession(headers=auth_headers()) as http:
                archive = await self._pack(OpencodeClient(http, codr_dir), src, keep_git=keep)
            async with self._stagr_session() as http:
                await self._unpack(self._stagr(http), archive, posixpath.basename(src), dst)
        except _Refused as e:
            return _error(str(e))
        except LLMCommunicationError as e:
            return _error(f"copy from codr failed: {e}")
        except TimeoutError:
            return _error("copy from codr failed: a call to codr's sandbox or stagr timed out.")
        return {"status": "ok", "copied": f"codr {src} -> stagr {dst}", "bytes": len(archive),
                "git": "kept" if keep else "any .git was left behind"}

    # -- the codr task queue (DP-481; tickets, DP-489) ---------------------------------

    def _seen_tickets(self, queue: Path) -> Union[Tuple[int, Set[str]], Dict[str, Any]]:
        """(the highest CD-NNN ever handed out, every ticket a depends_on may
        name); or the error. Read from the queue folder (dot-folders being
        staged and the high-water mark included) and from the codr agent's
        attempt rows, so a ticket whose folder was deleted keeps its number."""
        if self._memory is None:
            return _error("the codr action log is not wired in, so a new ticket ID could repeat an old one.")
        highest, names = 0, set()
        mark = queue / HIGH_WATER_FILE
        if mark.is_file():
            try:
                highest = int(mark.read_text(encoding="utf-8").strip() or 0)
            except ValueError:
                return _error(f"{mark} does not hold a number; fix or remove it.")
        for entry in queue.iterdir() if queue.is_dir() else []:
            # Only a folder codr's agent would load (`load_tickets`): an empty
            # one would be accepted here and hold its dependents forever there.
            if (entry.is_dir() and not entry.name.startswith(".")
                    and (entry / "TASK.md").is_file() and (entry / "check.sh").is_file()):
                names.add(entry.name)
            # `.CD-007.queue-1a2b3c4d` is CD-007 being staged.
            highest = max(highest, ticket_number(entry.name.lstrip(".").split(".", 1)[0]) or 0)
        prefix = attempt_context("")
        for row in self._memory.get_agent_actions(CODR_AGENT_NAME, limit=100000, action_type=ATTEMPT_ACTION):
            context = str(row.get("trigger_context") or "")
            if context.startswith(prefix):
                names.add(context[len(prefix):])
                highest = max(highest, ticket_number(context[len(prefix):]) or 0)
        return highest, names

    @staticmethod
    def _depends_on(value: Any) -> List[str]:
        """The listed ticket IDs, in order, once each. Models send a list or a
        comma-separated string."""
        items = value if isinstance(value, (list, tuple)) else str(value or "").split(",")
        return list(dict.fromkeys(str(i).strip() for i in items if str(i).strip()))

    def _ticket(self, title: Any, project: Any, depends_on: Any, spec: Any, plan: Any, packages: Any,
                check_sh: Any, max_attempts: Any) -> Union[Dict[str, Any], Tuple[str, List[str]]]:
        """(TASK.md, depends_on) once every field checks out, or the error.
        Whether depends_on names real tickets is checked when it is written."""
        line = str(title or "").strip()
        if not line or "\n" in line or "\r" in line:
            return _error("title must be one non-empty line.")
        name = str(project or "").strip()
        if not PROJECT_RE.match(name):
            return _error(f"project must be lowercase letters, digits and '-': {name!r}")
        deps = self._depends_on(depends_on)
        if bad := [d for d in deps if not TASK_NAME_RE.match(d)]:
            return _error(f"depends_on must be ticket IDs such as CD-004: {bad}")
        for field, value in (("spec", spec), ("plan", plan), ("packages", packages), ("check_sh", check_sh)):
            if not str(value or "").strip():
                return _error(f"{field} must not be empty (packages may say none).")
        attempts: Optional[int] = None
        if max_attempts is not None:
            try:
                attempts = int(max_attempts)
            except (TypeError, ValueError):
                return _error("max_attempts must be a whole number.")
            if attempts < 1:
                return _error("max_attempts must be at least 1.")
        return render_task_md(title=line, project=name, depends_on=deps, max_attempts=attempts,
                              spec=str(spec), plan=str(plan), packages=str(packages)), deps

    async def _queue_preview(self, title: Any = None, project: Any = None, depends_on: Any = None,
                             spec: Any = None, plan: Any = None, packages: Any = None, check_sh: Any = None,
                             max_attempts: Any = None, **_: Any) -> Optional[str]:
        """The approval prompt's enrichment: the ID the ticket would get now,
        or why it would be refused. The prompt itself shows every field."""
        ticket = self._ticket(title, project, depends_on, spec, plan, packages, check_sh, max_attempts)
        if isinstance(ticket, dict):
            return f"⚠️ will be refused: {ticket['message']}"
        seen = self._seen_tickets(global_config.CODR_TASKS_DIR)
        if isinstance(seen, dict):
            return f"⚠️ will be refused: {seen['message']}"
        if missing := [d for d in ticket[1] if d not in seen[1]]:
            return f"⚠️ will be refused: depends_on names no known ticket: {', '.join(missing)}"
        return f"new ticket {ticket_id(seen[0] + 1)}, project {str(project).strip()}"

    async def _queue_codr_task(self, title: Any = None, project: Any = None, spec: Any = None, plan: Any = None,
                               packages: Any = None, check_sh: Any = None, depends_on: Any = None,
                               max_attempts: Any = None) -> Dict[str, Any]:
        """Runs only once the operator approved it: the prompt shows every
        field. DERPR picks the ID, so two tickets never share one."""
        ticket = self._ticket(title, project, depends_on, spec, plan, packages, check_sh, max_attempts)
        if isinstance(ticket, dict):
            return ticket
        text, deps = ticket
        project = str(project).strip()
        queue = global_config.CODR_TASKS_DIR
        # Nothing below awaits, and the lock covers threads: one allocation at
        # a time. The ID is in the staging folder's name, so it counts as taken
        # until the rename, and a second mkdir of that name fails.
        with _QUEUE_LOCK:
            try:
                queue.mkdir(parents=True, exist_ok=True)
                seen = self._seen_tickets(queue)
                if isinstance(seen, dict):
                    return seen
                highest, names = seen
                if missing := [d for d in deps if d not in names]:
                    return _error(f"depends_on names no queued or attempted ticket: {', '.join(missing)}")
                name = ticket_id(highest + 1)
                staged = queue / f".{name}.queue-{uuid.uuid4().hex[:8]}"
                staged.mkdir()
                try:
                    (staged / "TASK.md").write_text(text, encoding="utf-8")
                    (staged / "check.sh").write_text(str(check_sh).rstrip() + "\n", encoding="utf-8")
                    if (queue / name).exists():
                        return _error(f"{name} appeared while it was being queued; nothing was written. Try again.")
                    # The mark goes first: a failed rename leaves a gap, never a reuse.
                    tmp = queue / f"{HIGH_WATER_FILE}.tmp"
                    tmp.write_text(f"{highest + 1}\n", encoding="utf-8")
                    tmp.replace(queue / HIGH_WATER_FILE)
                    staged.rename(queue / name)
                finally:
                    shutil.rmtree(staged, ignore_errors=True)
            except OSError as e:
                return _error(f"could not queue the ticket: {e}")
        logger.info("stagr: queued codr ticket %s (project %s)", name, project)
        # DP-493: the ticket ID is the deferral's token. When codr's agent
        # records the ticket's final outcome it settles it, and this
        # conversation gets one turn with the result.
        return declare_deferral(
            {"status": "ok", "queued": name,
             "note": f"Ticket {name}. DERPR starts it when codr is idle and every ticket in depends_on "
                     f"has passed, in its own worktree of projects/{project}/. You will be re-invoked here "
                     f"with its result once it passes, or fails or is blocked for good."},
            DEFERRAL_KIND_CODR_TICKET, name)
