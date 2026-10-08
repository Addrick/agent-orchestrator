"""Node-artifact drift check (DP-418).

``services/pve/`` is the node-side half of every Proxmox and HuggingFace tool,
and it reaches the node by a hand ``scp`` that no CI step can see. From
2026-08-21 to 2026-10-08 the node ran copies five tickets old: DP-364 changed
the install argv from 9 to 10, the old wrapper refused it, and ``install_model``
failed on every call for four weeks with every gate green.

This compares the copies the image shipped with against the node's, once per
start, and says so when they differ. It reports; it never redeploys. Redeploying
would need write access to the wrapper that gates this key, which is the one
thing the key must never have.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

from src.proxmox.ssh import SSHRunner, run_node_command

logger = logging.getLogger(__name__)

#: Repo file under ``services/pve/`` -> where the deploy runbook puts it on the
#: node. ⚠️ The wrapper admits the hash request only with exactly these paths in
#: exactly this order (``services/pve/derpr-pve-wrapper``); change both or neither.
NODE_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("derpr-pve-wrapper", "/usr/local/bin/derpr-pve-wrapper"),
    ("derpr-model-install", "/usr/local/sbin/derpr-model-install"),
    ("derpr-model-tier", "/usr/local/sbin/derpr-model-tier"),
    ("koboldcpp-model.service.in", "/usr/local/share/derpr/koboldcpp-model.service.in"),
    ("gguf_header.py", "/usr/local/share/derpr/gguf_header.py"),
)

#: ``COPY . .`` puts the repo at the image root, so the shipped copies sit
#: beside ``src/``.
SERVICES_DIR = Path(__file__).resolve().parents[2] / "services" / "pve"


@dataclass
class DriftReport:
    #: Node paths whose sha256 differs from the shipped copy.
    stale: List[str] = field(default_factory=list)
    #: Node paths the node did not hash (absent, or unreadable).
    missing: List[str] = field(default_factory=list)
    #: Set when the comparison could not run at all, with the reason.
    unverifiable: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not (self.stale or self.missing or self.unverifiable)


def local_digests(services_dir: Path = SERVICES_DIR) -> Dict[str, str]:
    """sha256 of each shipped artifact, keyed by its node path.

    CRLF is folded to LF first. The node copies are deployed from ``git show``
    and carry LF; a Windows checkout with ``autocrlf`` would otherwise report
    every file stale on a dev box. A CRLF copy on the *node* still mismatches,
    correctly -- it would not run there (``bad interpreter: ^M``).
    """
    digests: Dict[str, str] = {}
    for repo_name, node_path in NODE_ARTIFACTS:
        data = (services_dir / repo_name).read_bytes().replace(b"\r\n", b"\n")
        digests[node_path] = hashlib.sha256(data).hexdigest()
    return digests


def parse_sha256sum(stdout: str) -> Dict[str, str]:
    """``<hex>  <path>`` lines -> ``{path: hex}``. Anything else is skipped."""
    out: Dict[str, str] = {}
    for line in stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and len(parts[0]) == 64:
            out[parts[1].lstrip("*")] = parts[0].lower()
    return out


async def check_node_artifacts(
    runner: SSHRunner, services_dir: Path = SERVICES_DIR
) -> DriftReport:
    try:
        expected = local_digests(services_dir)
    except OSError as e:
        return DriftReport(unverifiable=f"image is missing a shipped artifact: {e}")
    argv = ["sha256sum", *(node_path for _, node_path in NODE_ARTIFACTS)]
    res = await run_node_command(runner, argv, exit_label="sha256sum")
    # sha256sum exits 1 when any file is absent but still hashes the rest, so
    # the stdout is read whatever the exit; only an empty one means no answer.
    remote = parse_sha256sum(str(res.get("stdout") or ""))
    if not remote:
        reason = res.get("stderr") or res.get("message") or "no output"
        return DriftReport(unverifiable=str(reason))
    report = DriftReport()
    for node_path, digest in expected.items():
        if node_path not in remote:
            report.missing.append(node_path)
        elif remote[node_path] != digest:
            report.stale.append(node_path)
    return report


def format_report(report: DriftReport) -> str:
    lines: List[str] = []
    if report.unverifiable:
        lines.append(
            f"Could not compare the node's copies: {report.unverifiable}. A node "
            "wrapper older than DP-418 refuses the hash request, so this usually "
            "means the node is stale too."
        )
    lines += [f"- stale: `{p}`" for p in report.stale]
    lines += [f"- missing: `{p}`" for p in report.missing]
    lines.append(
        "Proxmox/HuggingFace tools may fail until `services/pve/` is redeployed "
        "to the node (see `services/pve/README.md`)."
    )
    return "\n".join(lines)


async def report_node_artifact_drift(
    runner: SSHRunner,
    send: Callable[[str, str], Awaitable[bool]],
    services_dir: Path = SERVICES_DIR,
) -> DriftReport:
    """Run the check once and hand any drift to ``send(subject, body)``.

    Never raises: a startup task that crashed here would cost more than the
    drift it exists to report.
    """
    try:
        report = await check_node_artifacts(runner, services_dir)
        if report.ok:
            logger.info("node artifacts match the image (%d files)", len(NODE_ARTIFACTS))
        else:
            logger.warning("node artifact drift: %s", report)
            await send("Node scripts are out of date", format_report(report))
        return report
    except Exception:  # noqa: BLE001 -- see docstring
        logger.exception("node artifact drift check failed")
        return DriftReport(unverifiable="check raised; see logs")
