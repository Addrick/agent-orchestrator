"""The stagr tools (DP-456): a setup agent's hands on the setup VM.

stagr is a second `opencode serve`, used here only for its shell endpoint
(`OpencodeClient.run_shell`), exactly as the codr agent uses codr's. The VM has
no LAN and reaches the internet only through the egress proxy's domain
allowlist, so `stagr_shell` runs unapproved: nothing it can reach is ours.
What crosses a boundary is gated (``is_write``) instead:

- `request_web_access` adds a domain to the allowlist for a while, through the
  node verb `derpr-egress` over the shared node gate (`run_node_command`), the
  same SSH key and wrapper as the proxmox tools;
- `deliver_to_codr` copies a folder from stagr into a task folder in codr's
  offline sandbox. Any `.git` is left behind and the folder arrives as a fresh
  one-commit repo, so upstream history (a later fix, say) never reaches codr;
- `retrieve_from_codr` copies a codr task folder back to stagr.

Files move with `OpencodeClient.write_file` / `read_file` (DP-455), as one
tar.gz each way; derpr is the only thing that talks to both boxes.
"""

from __future__ import annotations

import logging
import re
import shlex
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

import aiohttp

from config import global_config
from src.engine.providers.opencode import (
    OpencodeClient, auth_headers, workspace_directory, workspace_lock,
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
TASK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MAX_GRANT_MINUTES = 1440
MAX_SHELL_SECONDS = 3600
DEFAULT_SHELL_SECONDS = 600
# What one delivery may carry, compressed. write_file moves ~0.5 MB/s.
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


class StagrToolHandler:
    def __init__(self, runner: SSHRunner | None = None) -> None:
        self._ssh = runner or SSHRunner()

    def register(self, manager: "ToolManager") -> None:
        manager.register("stagr_shell", self._stagr_shell)
        manager.register("list_web_access", self._list_web_access)
        manager.register("request_web_access", self._request_web_access)
        manager.register("deliver_to_codr", self._deliver_to_codr)
        manager.register("retrieve_from_codr", self._retrieve_from_codr)

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

    # -- stagr_shell -------------------------------------------------------------

    async def _stagr_shell(self, command: str, timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
        if (err := self._configured()):
            return err
        if not (command or "").strip():
            return _error("command is empty.")
        seconds = min(max(int(timeout_seconds or DEFAULT_SHELL_SECONDS), 1), MAX_SHELL_SECONDS)
        # The command runs in its own bash, so an `exit` in it still reports.
        script = (f"timeout {seconds} bash <<'STAGR_CMD'\n{command.rstrip()}\nSTAGR_CMD\n"
                  f'echo "{_EXIT_MARK}$?"\n')
        try:
            async with self._stagr_session() as http:
                out = await self._stagr(http).run_shell(
                    script, timeout=aiohttp.ClientTimeout(total=seconds + 60))
        except LLMCommunicationError as e:
            return _error(f"stagr shell failed: {e}")
        except TimeoutError:
            return _error(f"stagr shell gave no answer within {seconds + 60} s.")
        # The last marker, wherever it is: stderr may arrive after stdout.
        marks = list(re.finditer(rf"{_EXIT_MARK}(\d+)\n?", out))
        body = out[:marks[-1].start()] + out[marks[-1].end():] if marks else out
        exit_code = int(marks[-1].group(1)) if marks else None
        result: Dict[str, Any] = {"status": "ok", "exit_code": exit_code,
                                  "output": _clip(body, global_config.STAGR_OUTPUT_CHARS)}
        if exit_code == 124:
            result["note"] = f"timed out after {seconds} s"
        return result

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
        status = await client.request("GET", "/session/status") or {}
        return all((v or {}).get("type", "idle") == "idle" for v in status.values())

    @staticmethod
    async def _pack(client: OpencodeClient, folder: str, *, exclude_git: bool) -> bytes:
        """`folder` on that box as a tar.gz of its contents."""
        staged = f".stagr-out-{uuid.uuid4().hex[:8]}.tar.gz"
        exclude = "--exclude=.git " if exclude_git else ""
        out = await client.run_shell(
            f"[ -d {shlex.quote(folder)} ] && tar -czf {staged} {exclude}-C {shlex.quote(folder)} . "
            f"&& echo {_OK_MARK}", timeout=_LONG)
        try:
            if _OK_MARK not in out:
                raise _Refused(f"could not pack {folder}: {out[-500:] or 'no such folder'}")
            archive = await client.read_file(staged)
        finally:
            await client.run_shell(f"rm -f -- {staged}")
        if len(archive) > MAX_TRANSFER_BYTES:
            raise _Refused(f"{folder} is {len(archive)} bytes compressed; the limit is {MAX_TRANSFER_BYTES}.")
        return archive

    async def _deliver_to_codr(self, path: str, task: str) -> Dict[str, Any]:
        if (err := self._configured()):
            return err
        src = _rel_path(path)
        if src is None:
            return _error(f"path must be relative to {global_config.STAGR_WORKDIR}, with no '..': {path!r}")
        if not TASK_RE.match(task or ""):
            return _error(f"not a task folder name: {task!r}")
        _, _, codr_dir = workspace_directory(CODR_PERSONA)
        lock = workspace_lock(codr_dir)
        if lock.locked():
            return _error("codr is starting or running a turn; try again when it is idle.")
        try:
            async with self._stagr_session() as http:
                archive = await self._pack(self._stagr(http), src, exclude_git=True)
            async with lock, aiohttp.ClientSession(headers=auth_headers()) as http:
                await self._unpack_into_codr(OpencodeClient(http, codr_dir), archive, f"tasks/{task}")
        except _Refused as e:
            return _error(str(e))
        except LLMCommunicationError as e:
            return _error(f"delivery failed: {e}")
        return {"status": "ok", "delivered": f"{src} -> codr tasks/{task}", "bytes": len(archive),
                "history": "fresh repo, one commit (any .git on stagr was not copied)"}

    async def _unpack_into_codr(self, codr: OpencodeClient, archive: bytes, target: str) -> None:
        """Unpack into a new folder in codr's workspace as a one-commit repo.
        Caller holds codr's workspace lock."""
        if not await self._codr_idle(codr):
            raise _Refused("codr is working; deliver when it is idle.")
        exists = await codr.run_shell(f"[ -e {target} ] && echo EXISTS || true")
        if "EXISTS" in exists:
            raise _Refused(f"{target} already exists in codr's workspace; pick a new task name.")
        await codr.write_file(".stagr-in.tar.gz", archive)
        out = await codr.run_shell(
            f"mkdir -p {target} && tar -xzf .stagr-in.tar.gz -C {target} && rm -f .stagr-in.tar.gz "
            f"&& cd {target} && git init -q && git add -A "
            "&& git -c user.name=stagr -c user.email=stagr@derpr commit -qm 'task start (delivered by stagr)' "
            "&& git rev-list --all --count", timeout=_LONG)
        if out.strip().splitlines()[-1:] != ["1"]:
            raise _Refused(f"unpacking in codr's sandbox failed: {out[-500:]}")

    async def _retrieve_from_codr(self, task: str) -> Dict[str, Any]:
        if (err := self._configured()):
            return err
        if not TASK_RE.match(task or ""):
            return _error(f"not a task folder name: {task!r}")
        _, _, codr_dir = workspace_directory(CODR_PERSONA)
        dest = f"from-codr/{task}"
        try:
            async with aiohttp.ClientSession(headers=auth_headers()) as http:
                archive = await self._pack(OpencodeClient(http, codr_dir), f"tasks/{task}", exclude_git=False)
            async with self._stagr_session() as http:
                stagr = self._stagr(http)
                await stagr.write_file(".stagr-in.tar.gz", archive)
                out = await stagr.run_shell(
                    f"rm -rf {dest} && mkdir -p {dest} && tar -xzf .stagr-in.tar.gz -C {dest} "
                    f"&& rm -f .stagr-in.tar.gz && echo {_OK_MARK}", timeout=_LONG)
        except _Refused as e:
            return _error(str(e))
        except LLMCommunicationError as e:
            return _error(f"retrieving tasks/{task} failed: {e}")
        if _OK_MARK not in out:
            return _error(f"unpacking on stagr failed: {out[-300:]}")
        return {"status": "ok", "retrieved": f"codr tasks/{task} -> stagr {dest}", "bytes": len(archive)}
