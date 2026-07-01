"""Async SSH runner for the Proxmox tools (DP-262).

A thin wrapper over ``ssh -i <key> <user>@<host> <argv...>``. It runs the remote
command as an *argv list* (not a shell string) so callers never build shell
strings from model-supplied values — every argument is passed positionally to
``ssh``, which forwards them to the remote command without a second local shell.

The remote side still runs under the login shell, so we additionally reject any
argument containing shell metacharacters as defense in depth: the only values
that ever reach here are numeric vmids and config-pinned unit names, so a
metacharacter means misuse, not a legitimate call.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
from dataclasses import dataclass
from typing import Sequence

from config import global_config

logger = logging.getLogger(__name__)

# Characters that must never appear in a remote argument. vmids are digits and
# unit names are config-pinned [\w.-]; anything here signals misuse/injection.
_FORBIDDEN = set(";&|`$<>(){}[]!*?~\n\r\"'\\ ")


class SSHError(RuntimeError):
    """Raised when an SSH op cannot be attempted or the remote command fails."""


@dataclass
class SSHResult:
    returncode: int
    stdout: str
    stderr: str


def _reject_bad_args(argv: Sequence[str]) -> None:
    for arg in argv:
        bad = _FORBIDDEN.intersection(arg)
        if bad:
            raise SSHError(
                f"refusing SSH arg with forbidden characters {sorted(bad)!r}: {arg!r}"
            )


class SSHRunner:
    """Runs remote commands on the Proxmox node over key-based SSH.

    Config-driven (host/user/key/timeout from global_config) so tests inject a
    fake runner and production never hardcodes the target.
    """

    def __init__(
        self,
        *,
        host: str | None = None,
        user: str | None = None,
        key_path: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._host = host or global_config.PVE_SSH_HOST
        self._user = user or global_config.PVE_SSH_USER
        self._key = key_path or global_config.PVE_SSH_KEY
        self._timeout = timeout if timeout is not None else global_config.PVE_SSH_TIMEOUT

    async def run(self, argv: Sequence[str]) -> SSHResult:
        """Run ``argv`` on the node. Raises SSHError on transport failure/timeout.

        A non-zero remote exit is returned in the result (not raised) so callers
        can surface the node's stderr to the model; only the SSH transport
        itself failing (or timing out) raises.
        """
        argv = list(argv)
        _reject_bad_args(argv)
        ssh_cmd = [
            "ssh",
            "-i", self._key,
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"ConnectTimeout={int(self._timeout)}",
            f"{self._user}@{self._host}",
            *argv,
        ]
        logger.info("proxmox ssh: %s", shlex.join(argv))
        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:  # no ssh binary in the container
            raise SSHError(f"ssh binary not found: {e}") from e
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except asyncio.TimeoutError as e:
            proc.kill()
            raise SSHError(f"ssh timed out after {self._timeout:.0f}s") from e
        return SSHResult(
            returncode=proc.returncode if proc.returncode is not None else -1,
            stdout=out.decode("utf-8", "replace").strip(),
            stderr=err.decode("utf-8", "replace").strip(),
        )
