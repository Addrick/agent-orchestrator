"""Read-only HuggingFace Hub API client (DP-265).

Two metadata reads and nothing else: search for gguf repos, and list one repo's
files with the **byte size and sha256** the Hub reports for each. No download
happens here — the multi-GB fetch runs detached on the pve node (see
``services/pve/derpr-model-install``), because a download that outlives the tool
loop cannot be awaited inside it.

Why hand-rolled rather than ``huggingface_hub`` or HF's own MCP server: this
module needs ``lfs.oid`` and the byte size for the sha verify and the free-space
precheck, and DP-265 deliberately pins the tool surface to two reads on the one
persona that also holds node power. The decision and its re-litigation guard are
in the DP-265 task note.

**Every value returned by this module is attacker-authored.** A repo id, a
filename, and a repo's card text are all written by whoever uploaded the repo,
which is why the two read tools carry ``produces_untrusted: True``. Nothing here
interpolates a caller's string into a URL without validating it first — the
model chooses *which* repo to read, so a repo id is the one place a path
traversal could be steered from.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import aiohttp

from config import global_config

logger = logging.getLogger(__name__)

#: A Hub repo id: ``owner/name``. Anchored and segment-wise, so ``..`` cannot
#: appear as a whole segment and a leading ``/`` cannot make the joined URL
#: escape the API path. Validated *here* rather than trusting ``quote()``:
#: percent-encoding a traversal would produce a 404 instead of a refusal, and a
#: 404 reads to the model as "try another spelling".
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

#: A file path inside a repo. The character class matches the node-side script's
#: own gate (``services/pve/derpr-model-install``) on purpose — a value this
#: accepts and the node rejects would be a refusal a human has already approved.
_FILE_RE = re.compile(r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$")

#: sha256 as the Hub reports it in ``lfs.oid``.
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")

#: Cap on ``Link: rel="next"`` hops when walking a repo's file tree. Sharded
#: gguf repos genuinely page; an unbounded follow would let a repo decide how
#: long a tool call runs.
_MAX_TREE_PAGES = 10

_GIB = 1024 ** 3


class HFError(RuntimeError):
    """Raised when a Hub read cannot be attempted or comes back unusable."""


@dataclass
class HFFile:
    """One file in a repo, as the Hub describes it.

    ``sha256`` is ``lfs.oid`` and is **optional** — a small non-LFS file in the
    repo has no LFS pointer and therefore no Hub-published digest. That is not a
    problem to paper over: ``install_model`` refuses a file it cannot pin,
    because the whole verify step downstream is a comparison against this value.
    """

    path: str
    size_bytes: int
    sha256: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "size_gib": round(self.size_bytes / _GIB, 2),
            "sha256": self.sha256,
        }


def validate_repo_id(repo: str) -> str:
    """Return ``repo`` if it is a well-formed Hub repo id, else raise HFError."""
    value = str(repo or "").strip()
    if not _REPO_RE.match(value):
        raise HFError(
            f"invalid HuggingFace repo id {repo!r}; expected 'owner/name' using "
            "letters, digits, dot, dash or underscore"
        )
    return value


def validate_file_path(path: str) -> str:
    """Return ``path`` if it is a well-formed in-repo file path, else raise."""
    value = str(path or "").strip()
    if not _FILE_RE.match(value) or ".." in value.split("/"):
        raise HFError(
            f"invalid repo file path {path!r}; expected a relative path of "
            "letters, digits, dot, dash, underscore and '/'"
        )
    return value


class HFClient:
    """Metadata reads against the Hub API.

    One ``aiohttp`` request per call — no session is held open between calls.
    These are infrequent, human-paced lookups, and a long-lived session would
    outlive the event loop in tests for no throughput gain.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._base = (base_url or global_config.HF_API_BASE).rstrip("/")
        self._token = token if token is not None else global_config.HF_API_TOKEN
        self._timeout = (
            timeout if timeout is not None else global_config.HF_HTTP_TIMEOUT
        )

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """One GET returning parsed JSON, or HFError. Never raises aiohttp."""
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    url, params=params, headers=self._headers()
                ) as resp:
                    if resp.status != 200:
                        body = (await resp.text())[:200]
                        raise HFError(
                            f"HuggingFace returned {resp.status} for {url}: {body}"
                        )
                    return await resp.json(content_type=None)
        except HFError:
            raise
        except asyncio.TimeoutError as e:
            raise HFError(f"HuggingFace request timed out after {self._timeout:.0f}s") from e
        except aiohttp.ClientError as e:
            raise HFError(f"HuggingFace request failed: {e}") from e
        except ValueError as e:  # non-JSON body
            raise HFError(f"HuggingFace returned unparseable JSON: {e}") from e

    async def search_models(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """gguf repos matching ``query``, most-downloaded first.

        ``filter=gguf`` is applied server-side. It is a *hint*, not a guarantee
        that any particular file exists — the repo tag says the repo contains
        gguf somewhere, so the caller still has to list files before installing.
        """
        query = str(query or "").strip()
        if not query:
            raise HFError("search query must not be empty")
        rows = await self._get_json(
            f"{self._base}/api/models",
            params={
                "search": query,
                "filter": "gguf",
                "sort": "downloads",
                "direction": "-1",
                "limit": str(limit),
            },
        )
        if not isinstance(rows, list):
            raise HFError("HuggingFace search returned a non-list body")
        results: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            repo_id = row.get("id") or row.get("modelId")
            if not repo_id:
                continue
            results.append({
                "repo": str(repo_id),
                "downloads": row.get("downloads"),
                "likes": row.get("likes"),
                "gated": row.get("gated", False),
                "last_modified": row.get("lastModified"),
                "tags": [str(t) for t in (row.get("tags") or [])][:12],
            })
        return results

    async def list_gguf_files(self, repo: str, revision: str = "main") -> List[HFFile]:
        """Every ``.gguf`` in ``repo`` with its byte size and sha256.

        Walks the tree endpoint's ``Link: rel="next"`` pages up to
        ``_MAX_TREE_PAGES``; a repo with more shards than that is reported as
        truncated by the caller rather than silently half-listed.
        """
        repo = validate_repo_id(repo)
        url = (
            f"{self._base}/api/models/{quote(repo, safe='/')}"
            f"/tree/{quote(revision, safe='')}"
        )
        params: Optional[Dict[str, Any]] = {"recursive": "1"}
        files: List[HFFile] = []
        seen_cursors: set[str] = set()
        for _ in range(_MAX_TREE_PAGES):
            rows, cursor = await self._get_tree_page(url, params)
            for row in rows:
                entry = self._as_gguf_file(row)
                if entry is not None:
                    files.append(entry)
            if not cursor or cursor in seen_cursors:
                break
            seen_cursors.add(cursor)
            params = {"recursive": "1", "cursor": cursor}
        return files

    async def _get_tree_page(
        self, url: str, params: Optional[Dict[str, Any]]
    ) -> tuple[List[Any], Optional[str]]:
        """One page of the tree endpoint plus the next cursor, if any."""
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    url, params=params, headers=self._headers()
                ) as resp:
                    if resp.status == 404:
                        raise HFError(
                            f"no such HuggingFace repo or revision: {url}"
                        )
                    if resp.status != 200:
                        body = (await resp.text())[:200]
                        raise HFError(
                            f"HuggingFace returned {resp.status} for {url}: {body}"
                        )
                    rows = await resp.json(content_type=None)
                    cursor = _next_cursor(resp.headers.get("Link"))
        except HFError:
            raise
        except asyncio.TimeoutError as e:
            raise HFError(f"HuggingFace request timed out after {self._timeout:.0f}s") from e
        except aiohttp.ClientError as e:
            raise HFError(f"HuggingFace request failed: {e}") from e
        except ValueError as e:
            raise HFError(f"HuggingFace returned unparseable JSON: {e}") from e
        if not isinstance(rows, list):
            raise HFError("HuggingFace file tree returned a non-list body")
        return rows, cursor

    @staticmethod
    def _as_gguf_file(row: Any) -> Optional[HFFile]:
        """One tree row → HFFile, or None when it is not a usable gguf.

        The size read is ``lfs.size`` in preference to the top-level ``size``:
        for an LFS file both are the real byte count, but a repo that stores the
        *pointer* uncommitted reports the pointer's 130-odd bytes at top level.
        Taking the smaller of the two would size the free-space precheck against
        a number three orders of magnitude too small.
        """
        if not isinstance(row, dict) or row.get("type") != "file":
            return None
        path = str(row.get("path") or "")
        if not path.lower().endswith(".gguf"):
            return None
        raw_lfs = row.get("lfs")
        lfs: Dict[str, Any] = raw_lfs if isinstance(raw_lfs, dict) else {}
        raw_size: Any = lfs.get("size", row.get("size"))
        try:
            size = int(raw_size)
        except (TypeError, ValueError):
            return None
        oid = str(lfs.get("oid") or "")
        sha256 = oid if _SHA256_RE.match(oid) else None
        return HFFile(path=path, size_bytes=size, sha256=sha256)

    async def find_gguf_file(self, repo: str, file_path: str) -> HFFile:
        """The one file ``file_path`` names in ``repo``, or HFError.

        Refuses a file with no Hub-published sha256. Downstream the digest is
        the *only* thing tying the bytes that land on the node to the bytes a
        human approved, so "install it unverified" is not a degraded mode worth
        offering — it is the whole control.
        """
        file_path = validate_file_path(file_path)
        files = await self.list_gguf_files(repo)
        match = next((f for f in files if f.path == file_path), None)
        if match is None:
            available = [f.path for f in files][:20]
            raise HFError(
                f"{repo} has no gguf named {file_path!r}; it offers: {available}"
            )
        if not match.sha256:
            raise HFError(
                f"{repo}/{file_path} publishes no LFS sha256, so the downloaded "
                "bytes could not be verified against anything. Refusing."
            )
        return match


def _next_cursor(link_header: Optional[str]) -> Optional[str]:
    """The ``cursor`` query value of a ``Link: <...>; rel="next"`` header."""
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' not in part:
            continue
        start = part.find("<")
        end = part.find(">", start + 1)
        if start == -1 or end == -1:
            continue
        match = re.search(r"[?&]cursor=([^&>]+)", part[start + 1:end])
        if match:
            return match.group(1)
    return None
