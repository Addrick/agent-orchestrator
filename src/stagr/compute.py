"""`run_python` (DP-522): a short Python script for any persona, unapproved.

It runs on stagr's setup VM over the same transport as `stagr_shell`
(`OpencodeClient.run_shell`, `_shell_script` / `_shell_result`), never as
stagr: the command is only `sudo -n derpr-compute <seconds>` with the script
on stdin, and that root-owned runner (``services/compute-sandbox/``) is what
makes it safe to run without approval -- a throwaway systemd unit as user
`compute` with no network at all, no /home, a read-only system, a private
/tmp, and memory/CPU/task/time caps. Nothing the script reads can leave the
box and nothing it writes outlives the call, so the tool is a read
(``exfil_capable`` False). Change the runner and this file together.
"""

from __future__ import annotations

import asyncio
import base64
from typing import TYPE_CHECKING, Any, Dict

import aiohttp

from config import global_config
from src.engine.providers.opencode import OpencodeClient, auth_headers
from src.llm_errors import LLMCommunicationError
from src.stagr.handler import _error, _shell_result, _shell_script

if TYPE_CHECKING:
    from src.tools.tool_manager import ToolManager

# The runner on stagr-vm; its own argument check is the authority on SECONDS.
RUNNER = "sudo -n /usr/local/bin/derpr-compute"
DEFAULT_SECONDS = 30
MAX_SECONDS = 120
# One shell command must stay under Linux's 128 KiB argument cap, and the
# script travels base64-encoded inside run_shell's own base64.
MAX_CODE_BYTES = 32 * 1024
# Each call may hold 512 MB of stagr-vm's 3 GiB; stagr needs the rest.
MAX_CONCURRENT = 2
# The outer `timeout` only backstops the runner, which stops the script itself.
_SLACK_SECONDS = 15


class ComputeToolHandler:
    def __init__(self) -> None:
        self._slots = asyncio.Semaphore(MAX_CONCURRENT)

    def register(self, manager: "ToolManager") -> None:
        manager.register("run_python", self.run_python)

    async def run_python(self, code: str = "", timeout_seconds: Any = None) -> Dict[str, Any]:
        if not global_config.STAGR_SERVER_URL:
            return _error("run_python is not configured (it runs on stagr's box: set STAGR_SERVER_URL "
                          "and STAGR_SERVER_PASSWORD).")
        code = code or ""
        if not code.strip():
            return _error("code is empty.")
        if len(code.encode("utf-8")) > MAX_CODE_BYTES:
            return _error(f"code is over {MAX_CODE_BYTES} bytes; send a shorter script.")
        try:
            seconds = DEFAULT_SECONDS if timeout_seconds in (None, "") else int(timeout_seconds)
        except (TypeError, ValueError):
            return _error("timeout_seconds must be a whole number of seconds.")
        seconds = min(max(seconds, 1), MAX_SECONDS)
        encoded = base64.b64encode(code.encode("utf-8")).decode()
        prepared = _shell_script(f"echo {encoded} | base64 -d | {RUNNER} {seconds}", seconds + _SLACK_SECONDS)
        if isinstance(prepared, dict):
            return prepared
        script, _ = prepared
        async with self._slots:
            try:
                async with aiohttp.ClientSession(headers=auth_headers(global_config.STAGR_SERVER_PASSWORD)) as http:
                    client = OpencodeClient(http, global_config.STAGR_WORKDIR, base_url=global_config.STAGR_SERVER_URL)
                    out = await client.run_shell(
                        script, timeout=aiohttp.ClientTimeout(total=seconds + _SLACK_SECONDS + 60))
            except LLMCommunicationError as e:
                return _error(f"run_python failed to reach its box: {e}")
            except TimeoutError:
                return _error(f"run_python gave no answer within {seconds + _SLACK_SECONDS + 60} s.")
        return _shell_result(out, seconds)
