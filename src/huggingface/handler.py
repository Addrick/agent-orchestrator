"""Tool handlers for HuggingFace model provisioning (DP-265).

Four tools behind the ``huggingface`` service binding:

- ``hf_search``      (read, HF):    gguf repos matching a query.
- ``hf_files``       (read, HF):    one repo's gguf files with byte size + sha256.
- ``install_model``  (WRITE → parked): download a gguf onto the pve node and
  template a **disabled** ``koboldcpp-<name>.service`` for it.
- ``install_status`` (read, node):  poll one install job.

Every handler returns a JSON-able dict; transport, validation and Hub failures
come back as ``{"status": "error", ...}`` rather than raised, so the model gets a
message instead of a tool crash.

**The security boundary is the node-side script, not this module.** Everything
here runs one command — ``/usr/local/sbin/derpr-model-install`` — over the same
forced-command-wrapped SSH key the proxmox tools use. The alternative, letting
the model drive ``curl`` / ``systemctl`` / file writes as separate verbs, would
widen that key to arbitrary node write for the sake of a slightly simpler
handler. One verb keeps it lean *and* narrow; see ``services/pve/README.md``.

**The download does not happen inside the tool loop.** A multi-GB fetch exceeds
any sane tool timeout, so the node runs it detached under ``systemd-run`` and
derpr polls ``install_status``. Deliberately *not* a background task inside
derpr: that re-opens the DP-304 interval-loop shutdown-contract problem for a
job the node is already supervising.

Two things the model never gets to decide:

1. **What bytes land.** ``install_model`` reads the file's size and sha256 from
   the Hub itself and passes them to the node, which refuses on any mismatch.
   The same two values are what the enricher puts on the approval card, so the
   human approves a *digest*, not a repo name.
2. **Where they land.** The node script owns the models directory and the unit
   template; nothing in the arguments is a path.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, Optional, TYPE_CHECKING

from config import global_config
from src.huggingface.client import HFClient, HFError, validate_file_path, validate_repo_id
from src.proxmox.ssh import SSHError, SSHRunner

if TYPE_CHECKING:
    from src.tools.tool_manager import ToolManager

logger = logging.getLogger(__name__)

#: The one node-side verb this whole feature is allowed to run. Absolute because
#: sshd's forced command inherits a minimal PATH.
_INSTALL_SCRIPT = "/usr/local/sbin/derpr-model-install"

#: A friendly model name → the unit stem ``koboldcpp-<name>.service``.
#: Narrower than the unit charset ``ProxmoxToolHandler._UNIT_RE`` accepts on
#: *discovery*, because this one is **minted** from a model-supplied string: a
#: name we create has to be one that discovery can later read back, and dots or
#: ``@`` in a unit stem mean something to systemd that we do not want chosen for
#: us. Matches the node script's own gate.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,48}$")

#: A minted job id. Same charset as ``_NAME_RE`` so the node can gate both with
#: one pattern, and so the id is legal in the ``modelinstall-<id>`` unit name.
_JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

#: Context the templated unit is written with when the caller names none.
#: Deliberately small. The unit lands **disabled** and a human tunes the number
#: against ``gpu_status`` before enabling it — DP-265 chose the conservative
#: fixed default over computing a budget here, on the grounds that today the
#: number is picked by hand with no check at all, so even the weak version is a
#: strict improvement.
_DEFAULT_CONTEXTSIZE = 8192
_MIN_CONTEXTSIZE = 512
_MAX_CONTEXTSIZE = 1048576

#: Keys derpr will surface out of the node's job JSON, and how to coerce them.
#: A whitelist rather than a passthrough: it is what keeps ``install_status``'s
#: ``produces_untrusted: False`` claim *enforced* instead of merely asserted, so
#: a future node-side change that starts echoing an HTTP error body cannot
#: quietly turn a trusted read into an injection surface.
_STATUS_FIELDS: Dict[str, type] = {
    "job_id": str,
    "state": str,
    "step": str,
    "reason": str,
    "repo": str,
    "file": str,
    "name": str,
    "unit": str,
    "sha256": str,
    "started": str,
    "finished": str,
    "size_bytes": int,
    "downloaded_bytes": int,
    "contextsize": int,
    "n_layer": int,
    "n_kv_head": int,
    "head_dim": int,
}

#: How much of any single node-supplied string is kept. A fixed-vocabulary
#: `reason` is short; a long one means the node script grew a passthrough and
#: the truncation is the backstop for that.
_MAX_STATUS_STR = 200

_GIB = 1024 ** 3


def _err(message: str) -> Dict[str, Any]:
    return {"status": "error", "message": message}


def _validate_name(name: str) -> str:
    value = str(name or "").strip().lower()
    if not _NAME_RE.match(value):
        raise HFError(
            f"invalid model name {name!r}; use lowercase letters, digits and "
            "dashes (it becomes the systemd unit stem koboldcpp-<name>.service)"
        )
    return value


def _validate_contextsize(contextsize: Any) -> int:
    if contextsize is None or contextsize == "":
        return _DEFAULT_CONTEXTSIZE
    try:
        value = int(contextsize)
    except (TypeError, ValueError):
        raise HFError(f"contextsize must be an integer, got {contextsize!r}")
    if not _MIN_CONTEXTSIZE <= value <= _MAX_CONTEXTSIZE:
        raise HFError(
            f"contextsize must be between {_MIN_CONTEXTSIZE} and "
            f"{_MAX_CONTEXTSIZE}, got {value}"
        )
    return value


class HuggingFaceToolHandler:
    """Owns the Hub client and shares the proxmox SSH transport.

    Both are injectable so tests drive the whole feature with no network and no
    node: a fake client returns canned Hub metadata, a fake runner records the
    single argv that crosses the SSH boundary.
    """

    def __init__(
        self,
        client: Optional[HFClient] = None,
        runner: Optional[SSHRunner] = None,
    ) -> None:
        self._hf = client or HFClient()
        self._ssh = runner or SSHRunner()

    def register(self, manager: "ToolManager") -> None:
        manager.register("hf_search", self._hf_search)
        manager.register("hf_files", self._hf_files)
        # The enricher is what puts repo / file / byte size / sha256 on the
        # approval card. Without it the card would show only what the model
        # typed, and the human would be approving a name rather than bytes.
        manager.register(
            "install_model", self._install_model, self._enrich_install_model
        )
        manager.register("install_status", self._install_status)

    # -- guards --------------------------------------------------------------

    @staticmethod
    def _enabled() -> bool:
        return bool(global_config.HF_TOOLS_ENABLED)

    @staticmethod
    def _disabled_error() -> Dict[str, Any]:
        return _err(
            "HuggingFace model tools are disabled (set HF_TOOLS_ENABLED=true and "
            "deploy services/pve/derpr-model-install on the node to enable)."
        )

    async def _run(self, argv: list[str]) -> Dict[str, Any]:
        """Run the one node-side verb, mapping transport/exit errors to dicts."""
        try:
            res = await self._ssh.run(argv)
        except SSHError as e:
            return _err(f"ssh failed: {e}")
        if res.returncode != 0:
            return {
                "status": "error",
                "message": f"node script exited {res.returncode}",
                "stderr": res.stderr,
                "stdout": res.stdout,
            }
        return {"status": "ok", "stdout": res.stdout, "stderr": res.stderr}

    # -- read tools ----------------------------------------------------------

    async def _hf_search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        logger.info("Tool hf_search: %r limit=%s", query, limit)
        if not self._enabled():
            return self._disabled_error()
        try:
            capped = max(1, min(int(limit), global_config.HF_SEARCH_LIMIT_MAX))
        except (TypeError, ValueError):
            capped = 10
        try:
            models = await self._hf.search_models(query, capped)
        except HFError as e:
            return _err(str(e))
        return {"status": "ok", "query": query, "models": models}

    async def _hf_files(self, repo: str) -> Dict[str, Any]:
        logger.info("Tool hf_files: %s", repo)
        if not self._enabled():
            return self._disabled_error()
        try:
            files = await self._hf.list_gguf_files(validate_repo_id(repo))
        except HFError as e:
            return _err(str(e))
        return {
            "status": "ok",
            "repo": repo,
            "files": [f.to_dict() for f in files],
            # Said out loud rather than left to inference: a repo with no gguf
            # is a normal answer, and "the list came back empty" must not read
            # as "the read failed" (or vice versa).
            "note": (
                "Sizes are bytes as HuggingFace reports them. A file with "
                "sha256: null cannot be installed — install_model refuses "
                "anything it cannot pin to a digest."
            ),
        }

    async def _install_status(self, job_id: str) -> Dict[str, Any]:
        logger.info("Tool install_status: %s", job_id)
        if not self._enabled():
            return self._disabled_error()
        value = str(job_id or "").strip().lower()
        if not _JOB_ID_RE.match(value):
            return _err(f"invalid job_id {job_id!r}")
        res = await self._run([_INSTALL_SCRIPT, "status", value])
        if res.get("status") != "ok":
            return res
        try:
            payload = json.loads(res.get("stdout") or "")
        except json.JSONDecodeError:
            return _err(
                f"job {value} returned no readable status; the job file may not "
                "exist yet (a just-started job writes it within a second or two)"
            )
        if not isinstance(payload, dict):
            return _err(f"job {value} returned a non-object status")
        return {"status": "ok", "job": _clean_status(payload)}

    # -- write tool (parked for confirmation) --------------------------------

    async def _enrich_install_model(self, **kwargs: Any) -> Optional[str]:
        """The one line a human reads before approving an install.

        Returns HF's *own* size and digest for the file, not the model's account
        of them. Failures are returned as loud text rather than raised: the
        ToolManager turns an enricher exception into ``None``, which would
        render a card that looks ordinary while having verified nothing.
        """
        repo = str(kwargs.get("repo") or "")
        file_path = str(kwargs.get("file") or "")
        try:
            entry = await self._hf.find_gguf_file(
                validate_repo_id(repo), validate_file_path(file_path)
            )
        except HFError as e:
            return f"⚠️ UNVERIFIED — HuggingFace lookup failed: {e}"
        gib = entry.size_bytes / _GIB
        return (
            f"{repo}/{entry.path} · {entry.size_bytes} bytes ({gib:.2f} GiB) · "
            f"sha256 {entry.sha256}"
        )

    async def _install_model(
        self,
        repo: str,
        file: str,
        name: str,
        contextsize: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Provision a gguf so it *becomes* a valid ``set_active_model`` target.

        Runs after a human approved the park. The Hub is re-read here rather
        than trusting the enricher's values, because the enricher's result is
        display text, not state — but that also means the digest enforced can
        differ from the digest approved if the repo changed in between. The
        returned ``sha256`` is the one actually enforced, so a swap between park
        and execution is visible in the tool result rather than silent.
        """
        logger.info(
            "Tool install_model: repo=%s file=%s name=%s ctx=%s",
            repo, file, name, contextsize,
        )
        if not self._enabled():
            return self._disabled_error()
        try:
            repo_id = validate_repo_id(repo)
            file_path = validate_file_path(file)
            unit_name = _validate_name(name)
            ctx = _validate_contextsize(contextsize)
            entry = await self._hf.find_gguf_file(repo_id, file_path)
        except HFError as e:
            return _err(str(e))

        job_id = f"{unit_name}-{uuid.uuid4().hex[:12]}"
        res = await self._run([
            _INSTALL_SCRIPT, "install",
            repo_id, entry.path, unit_name, str(ctx),
            str(entry.size_bytes), str(entry.sha256), job_id,
        ])
        if res.get("status") != "ok":
            return res
        return {
            "status": "ok",
            "job_id": job_id,
            "repo": repo_id,
            "file": entry.path,
            "name": unit_name,
            "unit": f"koboldcpp-{unit_name}.service",
            "contextsize": ctx,
            "size_bytes": entry.size_bytes,
            "sha256": entry.sha256,
            "state": "running",
            # Both halves matter to the model: the download is not finished when
            # this returns, and the unit will not be serving anything even when
            # it is. Saying so here is what stops "installed" being reported as
            # "active".
            "note": (
                "Download started on the node and continues after this call "
                f"returns — poll install_status(job_id='{job_id}'). The unit is "
                "written DISABLED: it will not start and will not take :5001. "
                "Putting it on :5001 is a separate, separately-approved "
                "set_active_model call, and the contextsize should be checked "
                "against gpu_status first."
            ),
        }


def _clean_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Whitelist + coerce the node's job JSON into the shape derpr publishes."""
    out: Dict[str, Any] = {}
    for key, kind in _STATUS_FIELDS.items():
        if key not in payload:
            continue
        raw = payload[key]
        if kind is int:
            try:
                out[key] = int(raw)
            except (TypeError, ValueError):
                continue
        else:
            out[key] = str(raw)[:_MAX_STATUS_STR]
    return out
