"""Async SSH runner for the Proxmox tools (DP-262).

A thin wrapper over ``ssh -i <key> <user>@<host> <argv...>``. It runs the remote
command as an *argv list* (not a shell string) so callers never build shell
strings from model-supplied values — every argument is passed positionally to
``ssh``, which forwards them to the remote command without a second local shell.

The remote side still runs under the login shell, so we additionally reject any
argument containing shell metacharacters as defense in depth: the only values
that ever reach here are numeric vmids and config-pinned unit names, so a
metacharacter means misuse, not a legitimate call.

``run_node_command`` is the **single gate** in front of the node, and every tool
that reaches the box goes through it — not just the proxmox ones. It owns the two
things that are properties of *the key and the box's sshd* rather than of any one
handler: the ``PVE_TOOLS_ENABLED`` transport switch and the in-flight cap.
DP-348: they used to live inside ``ProxmoxToolHandler._run``, so when the
HuggingFace tools grew their own copy of that method they inherited neither.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import weakref
from dataclasses import dataclass
from typing import Any, Dict, Sequence

from config import global_config

logger = logging.getLogger(__name__)

# Characters that must never appear in a remote argument. vmids are digits and
# unit names are config-pinned [\w.-]; anything here signals misuse/injection.
_FORBIDDEN = set(";&|`$<>(){}[]!*?~\n\r\"'\\ ")

#: Cap on SSH calls in flight at once. Every ``run`` is a fresh ``ssh`` process
#: doing a full auth handshake — there is no ControlMaster here — and sshd's
#: default MaxStartups (10:30:100) begins randomly dropping connections past ten
#: unauthenticated ones. The per-unit probes fan out over however many units the
#: box holds, a number DP-332 deliberately stopped bounding in config, so the
#: cap belongs at the transport rather than at each call site.
MAX_INFLIGHT_SSH = 4

#: The cap is a property of *a node's sshd*, not of one handler, so it is held
#: here and shared by every handler that talks to that box (proxmox and
#: huggingface both do). It used to be a per-instance ``asyncio.Semaphore`` on
#: ``ProxmoxToolHandler``, so even two proxmox handlers would not have shared it.
#: One cap per host (DP-465 added a second, host ``derpr``): each sshd has its
#: own MaxStartups, so calls to one must not spend the other's slots.
#: Keyed by event loop rather than being a single module global:
#: ``asyncio.Semaphore`` binds to the first loop that awaits it and raises on a
#: second one, which would make every test after the first fail.
_INFLIGHT: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Dict[str, asyncio.Semaphore]]" = (
    weakref.WeakKeyDictionary()
)


def inflight_semaphore(host: str = "") -> asyncio.Semaphore:
    """The running loop's SSH concurrency gate for ``host``, created on first use."""
    per_host = _INFLIGHT.setdefault(asyncio.get_running_loop(), {})
    sem = per_host.get(host)
    if sem is None:
        sem = asyncio.Semaphore(MAX_INFLIGHT_SSH)
        per_host[host] = sem
    return sem


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

    @property
    def host(self) -> str:
        return self._host

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


async def run_node_command(
    runner: SSHRunner,
    argv: Sequence[str],
    *,
    exit_label: str = "remote command",
) -> Dict[str, Any]:
    """Run one argv on the node, mapping every failure to a result dict.

    The single gate in front of the node for *all* tools, not just the proxmox
    ones. ``PVE_TOOLS_ENABLED`` and the concurrency cap are properties of the
    key and of the box's sshd, so a second handler that reached the node on its
    own terms would be exempt from both — which is exactly what happened when
    the HuggingFace tools grew their own copy of this function (DP-348): they
    shelled out to the node with the proxmox tools switched off, and their
    connections were invisible to the cap that exists to stay under sshd's
    MaxStartups.

    A per-tool *feature* switch (``HF_TOOLS_ENABLED``) is a different question
    and stays with its handler; this one is the *transport*.

    A non-zero remote exit comes back as an error dict carrying the node's own
    stderr; only the transport failing is turned into a message of ours.
    ``exit_label`` names what exited, so a caller running one specific node verb
    can say so instead of reporting a generic remote command.
    """
    if not global_config.PVE_TOOLS_ENABLED:
        return {
            "status": "error",
            "message": (
                "Proxmox tools are disabled (set PVE_TOOLS_ENABLED=true and mount "
                "the pve SSH key to enable)."
            ),
        }
    try:
        async with inflight_semaphore(getattr(runner, "host", "")):
            res = await runner.run(argv)
    except SSHError as e:
        return {"status": "error", "message": f"ssh failed: {e}"}
    if res.returncode != 0:
        return {
            "status": "error",
            "message": f"{exit_label} exited {res.returncode}",
            "stderr": res.stderr,
            "stdout": res.stdout,
        }
    return {"status": "ok", "stdout": res.stdout, "stderr": res.stderr}
