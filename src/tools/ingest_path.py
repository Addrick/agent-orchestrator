# src/tools/ingest_path.py
"""`ingest_path` agent tool (DP-118).

Reads a markdown file or directory and ingests each file into the persona's
Hindsight bank as a standalone document. Idempotent via a per-bank sha256
cache: unchanged files are skipped unless `force=True` is passed.

Design: `docs/superpowers/specs/2026-05-18-ingest-path-tool-design.md`.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from config import global_config
from src.memory.backend.base import MemoryBackend, MemoryBackendError
from src.memory.date_extraction import LlmTagger, resolve_ingest_anchor
from src.tools.tool_manager import ToolManager
from src.tools.turn_context import get_turn_context

logger = logging.getLogger(__name__)

# Tools that mutate a Hindsight bank can hit transient 5xx during overlap
# windows; retry is cheap because retain is fire-and-forget.
_MAX_RETAIN_RETRIES = 3
_RETRY_BACKOFF_S = 5.0


class IngestPathHandler:
    """Handler for the `ingest_path` model-callable tool."""

    def __init__(
        self,
        memory_backend: MemoryBackend,
        cache_dir: Path,
        persona_lookup: Optional[Callable[[str], Any]] = None,
        date_tagger: Optional["LlmTagger"] = None,
    ) -> None:
        self.memory_backend = memory_backend
        self.cache_dir = cache_dir
        # persona_lookup(persona_name) -> Persona | None. Optional so unit tests
        # can wire a stub; production passes chat_system.personas.get.
        self._persona_lookup = persona_lookup
        # Optional LLM date-tagger fallback (DP-292 phase 2). None → regex-only
        # content-date extraction, file mtime as the fallback anchor.
        self._date_tagger = date_tagger

    def register(self, manager: ToolManager) -> None:
        manager.register("ingest_path", self._ingest_path)

    # ----- public tool entry point -----

    async def _ingest_path(
        self,
        path: str,
        glob: str = "**/*.md",
        bank: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        if not global_config.INGEST_PATH_ENABLED:
            return {"status": "disabled", "reason": "INGEST_PATH_ENABLED=0"}

        ctx = get_turn_context()
        if ctx is None:
            return {"status": "error", "reason": "no active turn context"}

        resolved_bank = self._resolve_bank(ctx.persona_name, bank)
        root = Path(path).expanduser().resolve()
        return await self.ingest_root(resolved_bank, root, glob, force)

    async def ingest_root(
        self,
        bank_id: str,
        root: Path,
        glob: str = "**/*.md",
        force: bool = False,
    ) -> Dict[str, Any]:
        """Walk `root`, retain matching files into `bank_id`.

        Turn-context-free core shared by the model-callable `_ingest_path`
        (which resolves bank + root first) and the DP-292 operator import
        panel (control-token gated, explicit bank + path). Idempotent via the
        per-bank sha256 cache — unchanged files are skipped unless `force`.
        Does NOT check INGEST_PATH_ENABLED: that flag gates the model tool;
        the operator path is authorized by DERPR_CONTROL_TOKEN instead.
        """
        if not root.exists():
            return {"status": "error", "reason": f"path not found: {root}"}

        candidates = self._collect_candidates(root, glob)
        if not candidates:
            return {
                "status": "ok", "bank": bank_id,
                "ingested": 0, "skipped": 0, "failed": 0,
                "reason": "no files matched glob",
            }

        resolved_bank = bank_id
        cache_path = self.cache_dir / f"{resolved_bank}.json"
        cache = self._load_cache(cache_path)

        # Determine display root for relpath keys: if user passed a file, use
        # its parent; if a dir, use the dir itself.
        display_root = root.parent if root.is_file() else root

        ingested = 0
        skipped = 0
        failed = 0
        sqlite_noop_warned = False

        for fpath in candidates:
            try:
                relpath = str(fpath.relative_to(display_root)).replace("\\", "/")
            except ValueError:
                relpath = str(fpath)

            try:
                data = fpath.read_bytes()
            except OSError as e:
                logger.warning("ingest_path: unreadable %s: %s", fpath, e)
                failed += 1
                continue

            sha = hashlib.sha256(data).hexdigest()
            entry = cache.get(relpath)
            if not force and entry and entry.get("sha256") == sha:
                skipped += 1
                continue

            try:
                content = data.decode("utf-8", errors="replace")
            except Exception as e:  # noqa: BLE001
                logger.warning("ingest_path: decode failed %s: %s", fpath, e)
                failed += 1
                continue

            mtime = datetime.fromtimestamp(fpath.stat().st_mtime, tz=timezone.utc)
            # Anchor to the date the content is about; mtime is the fallback
            # when the body carries no date (DP-292 phase 2).
            ts, date_tags, date_meta = await resolve_ingest_anchor(
                content, fallback_ts=mtime, llm_tagger=self._date_tagger,
            )
            metadata: Dict[str, str] = {
                "source_path": relpath,
                "sha256": sha,
                "file_mtime": mtime.isoformat(),
                **date_meta,
            }
            tags = ["ingest", "notes"] + date_tags

            success = await self._retain_with_retry(
                bank_id=resolved_bank,
                document_id=relpath,
                content=content,
                tags=tags,
                metadata=metadata,
                timestamp=ts,
            )
            if success is False:
                failed += 1
                continue
            # success may also be the sentinel "noop" if SQLite backend is in use.
            if success == "noop" and not sqlite_noop_warned:
                logger.warning(
                    "ingest_path: sqlite memory backend active; ingest was a noop "
                    "for bank=%s", resolved_bank,
                )
                sqlite_noop_warned = True

            ingested += 1
            cache[relpath] = {
                "sha256": sha,
                "mtime": mtime.isoformat(),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }

        self._save_cache(cache_path, cache)

        result: Dict[str, Any] = {
            "status": "ok",
            "bank": resolved_bank,
            "ingested": ingested,
            "skipped": skipped,
            "failed": failed,
        }
        if sqlite_noop_warned:
            result["note"] = "memory backend is sqlite; ingest was a noop"
        return result

    # ----- helpers -----

    def _resolve_bank(self, persona_name: str, bank_arg: Optional[str]) -> str:
        if bank_arg:
            return bank_arg
        if self._persona_lookup is not None:
            persona = self._persona_lookup(persona_name)
            override = getattr(persona, "get_ingest_bank", lambda: None)()
            if override:
                return str(override)
        return persona_name

    def _collect_candidates(self, root: Path, glob: str) -> list[Path]:
        if root.is_file():
            return [root]
        return sorted(p for p in root.glob(glob) if p.is_file())

    def _load_cache(self, cache_path: Path) -> Dict[str, Dict[str, str]]:
        if not cache_path.exists():
            return {}
        try:
            data: Dict[str, Dict[str, str]] = json.loads(cache_path.read_text(encoding="utf-8"))
            return data
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("ingest_path: cache load failed (%s); resetting", e)
            return {}

    def _save_cache(self, cache_path: Path, cache: Dict[str, Any]) -> None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        except OSError as e:
            logger.warning("ingest_path: cache save failed: %s", e)

    async def _retain_with_retry(
        self,
        *,
        bank_id: str,
        document_id: str,
        content: str,
        tags: list[str],
        metadata: Dict[str, str],
        timestamp: datetime,
    ) -> Any:
        """Returns True on success, "noop" if sqlite returned None, False on exhausted retries."""
        for attempt in range(_MAX_RETAIN_RETRIES):
            try:
                await self.memory_backend.retain_document(
                    bank_id, document_id, content,
                    tags=tags, metadata=metadata, timestamp=timestamp,
                )
                # Both Hindsight + sqlite return None; differentiate by class
                # so the tool can surface a "noop" note when sqlite is active.
                return "noop" if _backend_is_noop(self.memory_backend) else True
            except MemoryBackendError as e:
                # transient = retryable (HTTP 5xx on Hindsight) — classified by
                # the ABC-level error so this tool never imports a concrete
                # backend (DP-203).
                if e.transient and attempt + 1 < _MAX_RETAIN_RETRIES:
                    wait = _RETRY_BACKOFF_S * (attempt + 1)
                    logger.warning(
                        "ingest_path: transient backend error on %s (attempt %d): %s; retrying in %.1fs",
                        document_id, attempt + 1, e, wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.warning("ingest_path: retain failed for %s: %s", document_id, e)
                return False
            except Exception as e:  # noqa: BLE001
                logger.warning("ingest_path: retain crashed for %s: %s", document_id, e)
                return False
        return False


def _backend_is_noop(backend: MemoryBackend) -> bool:
    """True if the active backend is the sqlite (turn-history) backend.

    Used to translate `retain_document` returning None into a user-visible
    "noop" note. Hindsight backend also returns None on success, so we
    discriminate by class name rather than return value alone.
    """
    return backend.__class__.__name__ == "SqliteSemanticBackend"
