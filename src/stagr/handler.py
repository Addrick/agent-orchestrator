"""The stagr tools (DP-456): a setup agent's hands on the setup VM.

stagr is a second `opencode serve`, used here only for its shell endpoint
(`OpencodeClient.run_shell`), exactly as the codr agent uses codr's. The VM has
no LAN and reaches the internet only through the egress proxy's domain
allowlist, so `stagr_shell` runs unapproved: nothing it can reach is ours.
What crosses a boundary is gated (``is_write``) instead:

- `request_web_access` adds a domain to the allowlist for a while, through the
  node verb `derpr-egress` over the shared node gate (`run_node_command`), the
  same SSH key and wrapper as the proxmox tools;
- `copy_to_codr` / `copy_from_codr` copy a file or folder between stagr and
  codr's offline sandbox, to a path that must not exist yet. Any `.git` is left
  behind unless asked for, so upstream history (a later fix, say) never reaches
  codr by accident.

`codr_shell` runs unapproved too, in codr's workspace (DP-462): codr-vm is
offline and is restored from a backup if a command wrecks it. It holds codr's
workspace lock and runs only while every session there is idle, so the
environment changes between tasks, never under one.

Files move with `OpencodeClient.write_file` / `read_file` (DP-455), as one
tar.gz each way; derpr is the only thing that talks to both boxes.
"""

from __future__ import annotations

import logging
import posixpath
import re
import shlex
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import aiohttp

from config import global_config
from src.engine.providers.opencode import (
    OpencodeClient, auth_headers, open_auto, session_title, workspace_directory, workspace_lock,
)
from src.llm_errors import LLMCommunicationError
from src.proxmox.ssh import SSHRunner, run_node_command

if TYPE_CHECKING:
    from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)

EGRESS_VERB = "/usr/local/sbin/derpr-egress"
CODR_PERSONA = "codr"
# The same gate as the node verb and egress-ctl. The last label starts with a
# letter, so no IP literal is accepted.
DOMAIN_RE = re.compile(r"^([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]([a-z0-9-]{0,61}[a-z0-9])?$")
PATH_RE = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")
MAX_GRANT_MINUTES = 1440
MAX_SHELL_SECONDS = 3600
DEFAULT_SHELL_SECONDS = 600
# What one copy may carry, compressed. write_file moves ~0.5 MB/s.
MAX_TRANSFER_BYTES = 200 * 1024 * 1024
_EXIT_MARK = "__STAGR_EXIT="
_OK_MARK = "__STAGR_OK"
_LONG = aiohttp.ClientTimeout(total=600)


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
    def __init__(self, runner: SSHRunner | None = None) -> None:
        self._ssh = runner or SSHRunner()

    def register(self, manager: "ToolManager") -> None:
        manager.register("stagr_shell", self._stagr_shell)
        manager.register("list_web_access", self._list_web_access)
        manager.register("request_web_access", self._request_web_access)
        manager.register("codr_shell", self._codr_shell)
        manager.register("copy_to_codr", self._copy_to_codr)
        manager.register("copy_from_codr", self._copy_from_codr)

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

    # -- web access ----------------------------------------------------------------

    async def _list_web_access(self) -> Dict[str, Any]:
        result = await run_node_command(self._ssh, [EGRESS_VERB, "list"], exit_label="derpr-egress")
        if result.get("status") == "error":
            return result
        return {"status": "ok", "allowlist": result.get("stdout", "").strip()}

    async def _request_web_access(self, domain: str, minutes: int, reason: str = "") -> Dict[str, Any]:
        domain = (domain or "").strip().lower().rstrip(".")
        if len(domain) > 253 or not DOMAIN_RE.match(domain):
            return _error(f"not a domain name: {domain!r} (no IP addresses, no scheme or path).")
        try:
            minutes = int(minutes)
        except (TypeError, ValueError):
            return _error("minutes must be a whole number.")
        if not 1 <= minutes <= MAX_GRANT_MINUTES:
            return _error(f"minutes must be 1-{MAX_GRANT_MINUTES}.")
        result = await run_node_command(self._ssh, [EGRESS_VERB, "allow", domain, str(minutes)],
                                        exit_label="derpr-egress")
        if result.get("status") == "error":
            return result
        logger.info("stagr: web access to %s for %d min granted (%s)", domain, minutes, reason)
        return {"status": "ok", "granted": result.get("stdout", "").strip()}

    # -- transfers -------------------------------------------------------------------

    @staticmethod
    async def _codr_idle(client: OpencodeClient) -> bool:
        """Every session in codr's workspace idle, and no task turn of the codr
        agent still open: a turn that has ended but whose check has not run
        yet (the next tick runs it) is still under its task."""
        status = await client.request("GET", "/session/status") or {}
        if not all((v or {}).get("type", "idle") == "idle" for v in status.values()):
            return False
        title = session_title(CODR_PERSONA)
        for s in await client.request("GET", "/session") or []:
            if s.get("title") == title and not s.get("parentID"):
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
