"""Per-session Claude Code -> Hindsight ingestion with idempotency tracking.

Modes:
  --session-id <uuid>   Ingest one specific session (used by SessionEnd hook).
                        If --session-id is omitted, reads JSON from stdin and
                        extracts `session_id` (Claude Code hook payload format).
                        An explicit session id means "this session ended", so
                        the still-active guard does not apply to it.
  --backfill            Scan the project dir for un-ingested sessions and POST
                        them all (used by SessionStart hook). Sessions whose
                        transcript was written to within the grace window are
                        assumed still running and are deferred to a later run.
  --recheck             Compare every already-ingested session against the text
                        the server actually holds and re-ingest the ones that
                        were captured mid-session. Repairs a corpus ingested
                        before the still-active guard existed.
  --mark-only           Seed state without POSTing.

State file: ~/.claude/hindsight_ingested.json

  {"version": 2,
   "sessions": {"<uuid>": {"ingested_at": iso, "chars": int, "first_turn": iso}},
   "updated_at": iso}

Version 1 (a bare `ingested_sessions` list) is migrated on read; migrated
entries carry no `chars`, so they are never re-ingested by the growth check --
only `--recheck`, which asks the server, can repair them.

Why the guard exists: `--backfill` globs the project dir, so a session that is
still being written gets POSTed with only the turns recorded so far and then
marked ingested forever. Measured 2026-08-20: 101 of 316 sessions had been
captured that way, one of them at 1,025 chars of an eventual 38,238.

Project scoping: by default only sessions for the current cwd's encoded project
folder are considered. Override with --project-dir <encoded-name> or
$CLAUDE_PROJECT_DIR (path is encoded internally).
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.recover_claudecode_hindsight import (  # noqa: E402
    BANK_ID,
    ingest_sessions,
    load_session,
)
from src.memory.backend.hindsight import HindsightBackend  # noqa: E402
from config.global_config import HINDSIGHT_URL  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cc_ingest")

STATE_FILE = Path.home() / ".claude" / "hindsight_ingested.json"
PROJECTS_ROOT = Path.home() / ".claude" / "projects"

# A transcript touched more recently than this is assumed to belong to a live
# session. 15 min is well past Claude Code's own flush cadence and well short
# of a session worth losing.
ACTIVE_GRACE_SECONDS = int(os.environ.get("HINDSIGHT_INGEST_GRACE", "900"))

# Parsed session tuple: (session_id, project, first_ts, content)
Session = Tuple[str, str, datetime, str]


def encode_project_dir(path: str) -> str:
    """Match Claude Code's encoding: drop ':' and replace separators with '-'.

    Example: C:\\Users\\Adam\\Programming\\Python\\derpr-python
    becomes  C--Users-Adam-Programming-Python-derpr-python
    """
    p = path.replace(":", "-").replace("\\", "-").replace("/", "-")
    return p


def load_state() -> Dict[str, Dict[str, Any]]:
    """Return {session_id: record}. Migrates the v1 list format in memory."""
    if not STATE_FILE.exists():
        return {}
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("State file unreadable (%s); treating as empty.", e)
        return {}
    sessions = data.get("sessions")
    if isinstance(sessions, dict):
        return sessions
    legacy = data.get("ingested_sessions") or []
    return {sid: {} for sid in legacy if isinstance(sid, str)}


def save_state(sessions: Dict[str, Dict[str, Any]]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "sessions": dict(sorted(sessions.items())),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def record_for(parsed: Session) -> Dict[str, Any]:
    _sid, _project, first_ts, content = parsed
    return {
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "first_turn": first_ts.isoformat(),
        "chars": len(content),
    }


def is_active(path: Path, grace: int = ACTIVE_GRACE_SECONDS) -> bool:
    """True if the transcript was written to recently enough to still be live."""
    try:
        return (time.time() - path.stat().st_mtime) < grace
    except OSError:
        return False


def resolve_project_dir(arg: Optional[str]) -> Path:
    # Accept absolute path, encoded folder name, or default to $CLAUDE_PROJECT_DIR / cwd.
    raw = arg or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    # If a bare encoded-name was passed (no separators, exists under PROJECTS_ROOT), use it as-is.
    if arg and ("/" not in arg and "\\" not in arg and ":" not in arg):
        candidate = PROJECTS_ROOT / arg
        if candidate.exists():
            return candidate
    return PROJECTS_ROOT / encode_project_dir(raw)


def read_session_id_from_stdin() -> Optional[str]:
    if sys.stdin.isatty():
        return None
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return None
    sid = payload.get("session_id")
    return sid if isinstance(sid, str) and sid else None


def collect_sessions(
    project_dir: Path,
    only_session_id: Optional[str],
    state: Dict[str, Dict[str, Any]],
) -> List[Session]:
    """Parse the sessions that should be POSTed now.

    A session is skipped when it is already ingested at its current length, or
    when it looks still-live -- unless it was named explicitly, which means the
    SessionEnd hook fired for it.
    """
    if not project_dir.exists():
        logger.warning("Project dir not found: %s", project_dir)
        return []

    explicit = only_session_id is not None
    if only_session_id:
        path = project_dir / f"{only_session_id}.jsonl"
        if not path.exists():
            logger.warning("Session jsonl not found: %s", path)
            return []
        candidates = [path]
    else:
        candidates = sorted(project_dir.glob("*.jsonl"))

    sessions: List[Session] = []
    for path in candidates:
        parsed = _parse_if_due(path, state.get(path.stem), explicit)
        if parsed is not None:
            sessions.append(parsed)
    sessions.sort(key=lambda s: s[2])
    return sessions


def _parse_if_due(
    path: Path,
    prior: Optional[Dict[str, Any]],
    explicit: bool,
) -> Optional[Session]:
    """Parse one transcript, or return None if it should not be POSTed now."""
    sid = path.stem
    if prior is not None and not explicit and prior.get("chars") is None:
        # v1-migrated record: no length to compare against. Leave it to
        # --recheck, which asks the server instead of guessing.
        return None
    if not explicit and is_active(path):
        logger.info("Session %s still active (mtime < %ds); deferring.",
                    sid[:8], ACTIVE_GRACE_SECONDS)
        return None
    parsed = load_session(path)
    if parsed is None:
        logger.info("Session %s skipped by load_session filter.", sid[:8])
        return None
    known = (prior or {}).get("chars")
    if isinstance(known, int):
        if len(parsed[3]) <= known:
            return None
        logger.info("Session %s grew %d -> %d chars; re-ingesting.",
                    sid[:8], known, len(parsed[3]))
    return parsed


async def server_documents_by_session(bank_id: str = BANK_ID) -> Dict[str, Dict[str, Any]]:
    """Map short session id (first 8 chars) -> document record on the server."""
    backend = HindsightBackend(url=HINDSIGHT_URL)
    out: Dict[str, Dict[str, Any]] = {}
    try:
        client = backend._get_client()
        offset = 0
        while True:
            page = await client.alist_documents(bank_id, limit=200, offset=offset)
            items = page.get("items") or []
            if not items:
                break
            for doc in items:
                doc_id = doc.get("id") or ""
                parts = doc_id.split(":")
                if len(parts) >= 3 and parts[1].startswith("session_"):
                    out[parts[1][len("session_"):]] = doc
            offset += len(items)
            if len(items) < 200:
                break
    finally:
        await backend.aclose()
    return out


async def delete_documents(doc_ids: List[str], bank_id: str = BANK_ID) -> int:
    backend = HindsightBackend(url=HINDSIGHT_URL)
    deleted = 0
    try:
        client = backend._get_client()
        for doc_id in doc_ids:
            try:
                await client.adelete_document(bank_id, doc_id)
                deleted += 1
            except Exception as e:  # noqa: BLE001 - a stale document is not fatal
                logger.warning("Could not delete document %s: %s", doc_id, e)
    finally:
        await backend.aclose()
    return deleted


def find_truncated(
    project_dir: Path,
    docs: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, Session, int, int]]:
    """Return (document_id, parsed, server_chars, transcript_chars) per short doc."""
    out: List[Tuple[str, Session, int, int]] = []
    for path in sorted(project_dir.glob("*.jsonl")):
        doc = docs.get(path.stem[:8])
        if doc is None or is_active(path):
            continue
        parsed = load_session(path)
        if parsed is None:
            continue
        have = int(doc.get("text_length") or 0)
        if len(parsed[3]) > have:
            out.append((doc["id"], parsed, have, len(parsed[3])))
    return out


async def recheck(
    project_dir: Path,
    state: Dict[str, Dict[str, Any]],
    dry_run: bool,
    limit: Optional[int],
) -> int:
    """Re-ingest sessions the server holds less text for than the transcript has.

    The server's `original_text` is the only trustworthy record of what was
    actually captured -- the state file only says that a POST happened.
    """
    docs = await server_documents_by_session()
    logger.info("Server holds %d session documents.", len(docs))

    stale = find_truncated(project_dir, docs)
    stale.sort(key=lambda s: s[3] - s[2], reverse=True)
    if limit:
        stale = stale[:limit]

    total_missing = sum(w - h for _d, _p, h, w in stale)
    logger.info("%d session(s) truncated on the server; %d chars missing.",
                len(stale), total_missing)
    for _doc_id, parsed, h, w in stale[:20]:
        logger.info("  %s  server=%d  transcript=%d  (+%d)", parsed[0][:8], h, w, w - h)
    if len(stale) > 20:
        logger.info("  ... and %d more", len(stale) - 20)
    if dry_run or not stale:
        return 0

    logger.info("Deleting %d stale document(s)...", len(stale))
    await delete_documents([d for d, _p, _h, _w in stale])

    enqueued = await ingest_sessions([p for _d, p, _h, _w in stale])
    by_sid = {p[0]: p for _d, p, _h, _w in stale}
    for sid in enqueued:
        state[sid] = record_for(by_sid[sid])
    save_state(state)
    logger.info("Re-ingested %d session(s).", len(enqueued))
    return 0


def mark_only(project_dir: Path, state: Dict[str, Dict[str, Any]]) -> int:
    """Seed state with every session jsonl in the project dir without POSTing.

    Used after the one-shot recover script to avoid duplicate ingestion on the
    first SessionStart.
    """
    new = 0
    for path in project_dir.glob("*.jsonl"):
        if path.stem not in state:
            state[path.stem] = {}
            new += 1
    save_state(state)
    logger.info("Marked sessions as ingested (%d new); total tracked: %d.",
                new, len(state))
    return 0


async def replace_superseded(sessions: List[Session], state: Dict[str, Dict[str, Any]]) -> None:
    """Delete the server documents for sessions being re-POSTed.

    The document id is derived from the session tag and the first turn's
    timestamp, both stable across a re-ingest, so without this the second POST
    would collide with the first instead of replacing it.
    """
    resend = [s for s in sessions if s[0] in state]
    if not resend:
        return
    docs = await server_documents_by_session()
    doc_ids = [docs[s[0][:8]]["id"] for s in resend if s[0][:8] in docs]
    if doc_ids:
        logger.info("Replacing %d superseded document(s).", len(doc_ids))
        await delete_documents(doc_ids)


async def main_async(args: argparse.Namespace) -> int:
    project_dir = resolve_project_dir(args.project_dir)
    logger.info("Project dir: %s", project_dir)

    session_id = args.session_id
    if not (args.backfill or args.mark_only or args.recheck) and not session_id:
        session_id = read_session_id_from_stdin()
        if not session_id:
            logger.error("No --session-id, no --backfill/--recheck/--mark-only, "
                         "and no session_id on stdin.")
            return 2

    state = load_state()

    if args.mark_only or args.recheck:
        if not project_dir.exists():
            logger.warning("Project dir not found: %s", project_dir)
            return 0
        if args.mark_only:
            return mark_only(project_dir, state)
        return await recheck(project_dir, state, args.dry_run, args.limit)

    sessions = collect_sessions(
        project_dir,
        only_session_id=session_id if not args.backfill else None,
        state=state,
    )
    if not sessions:
        logger.info("Nothing new to ingest.")
        return 0

    if args.dry_run:
        for sid, _project, first_ts, content in sessions:
            logger.info("DRY RUN would ingest %s (%s, %d chars)",
                        sid[:8], first_ts.isoformat(), len(content))
        return 0

    await replace_superseded(sessions, state)
    await post_and_record(sessions, state)
    return 0


async def post_and_record(sessions: List[Session], state: Dict[str, Dict[str, Any]]) -> None:
    logger.info("Ingesting %d session(s)...", len(sessions))
    enqueued = await ingest_sessions(sessions)
    by_sid = {s[0]: s for s in sessions}
    for sid in enqueued:
        state[sid] = record_for(by_sid[sid])
    if enqueued:
        save_state(state)
        logger.info("State updated: %d session(s) now tracked.", len(state))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", default=None,
                        help="Ingest just this session id. If omitted (and not --backfill), reads session_id from stdin JSON.")
    parser.add_argument("--backfill", action="store_true",
                        help="Ingest every un-ingested session in the project dir.")
    parser.add_argument("--recheck", action="store_true",
                        help="Re-ingest sessions the server holds a truncated copy of.")
    parser.add_argument("--mark-only", action="store_true",
                        help="Add every session in the project dir to the state file WITHOUT POSTing. Use to seed state after running the one-shot recover script.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be ingested without POSTing.")
    parser.add_argument("--limit", type=int, default=None,
                        help="With --recheck, repair at most N sessions (largest gap first).")
    parser.add_argument("--project-dir", default=None,
                        help="Encoded project folder name or absolute path. Defaults to $CLAUDE_PROJECT_DIR / cwd.")
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))
