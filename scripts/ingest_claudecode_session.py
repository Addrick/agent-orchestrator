"""Per-session Claude Code -> Hindsight ingestion with idempotency tracking.

Modes:
  --session-id <uuid>   Ingest one specific session (used by SessionEnd hook).
                        If --session-id is omitted, reads JSON from stdin and
                        extracts `session_id` (Claude Code hook payload format).
  --backfill            Scan the project dir for un-ingested sessions and POST
                        them all (used by SessionStart hook).

State file: ~/.claude/hindsight_ingested.json
  {"ingested_sessions": ["uuid1", "uuid2", ...], "updated_at": "<iso>"}

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
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.recover_claudecode_hindsight import (  # noqa: E402
    ingest_sessions,
    load_session,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cc_ingest")

STATE_FILE = Path.home() / ".claude" / "hindsight_ingested.json"
PROJECTS_ROOT = Path.home() / ".claude" / "projects"


def encode_project_dir(path: str) -> str:
    """Match Claude Code's encoding: drop ':' and replace separators with '-'.

    Example: C:\\Users\\Adam\\Programming\\Python\\derpr-python
    becomes  C--Users-Adam-Programming-Python-derpr-python
    """
    p = path.replace(":", "-").replace("\\", "-").replace("/", "-")
    return p


def load_state() -> Set[str]:
    if not STATE_FILE.exists():
        return set()
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return set(data.get("ingested_sessions", []))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("State file unreadable (%s); treating as empty.", e)
        return set()


def save_state(ingested: Set[str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ingested_sessions": sorted(ingested),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


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
    ingested: Set[str],
) -> List[Tuple[str, str, datetime, str]]:
    if not project_dir.exists():
        logger.warning("Project dir not found: %s", project_dir)
        return []

    if only_session_id:
        path = project_dir / f"{only_session_id}.jsonl"
        if not path.exists():
            logger.warning("Session jsonl not found: %s", path)
            return []
        candidates = [path]
    else:
        candidates = sorted(project_dir.glob("*.jsonl"))

    sessions: List[Tuple[str, str, datetime, str]] = []
    for path in candidates:
        sid = path.stem
        if sid in ingested:
            continue
        parsed = load_session(path)
        if parsed is None:
            logger.info("Session %s skipped by load_session filter.", sid[:8])
            continue
        sessions.append(parsed)
    sessions.sort(key=lambda s: s[2])
    return sessions


async def main_async(args: argparse.Namespace) -> int:
    project_dir = resolve_project_dir(args.project_dir)
    logger.info("Project dir: %s", project_dir)

    session_id = args.session_id
    if not args.backfill and not args.mark_only and not session_id:
        session_id = read_session_id_from_stdin()
        if not session_id:
            logger.error("No --session-id, no --backfill, no --mark-only, and no session_id on stdin.")
            return 2

    ingested = load_state()

    if args.mark_only:
        # Seed state with every session jsonl currently in the project dir
        # without POSTing. Used after the one-shot recover script to avoid
        # duplicate ingestion on the first SessionStart.
        if not project_dir.exists():
            logger.warning("Project dir not found: %s", project_dir)
            return 0
        sids = [p.stem for p in project_dir.glob("*.jsonl")]
        new = [s for s in sids if s not in ingested]
        ingested.update(sids)
        save_state(ingested)
        logger.info("Marked %d session(s) as ingested (%d new); total tracked: %d.",
                    len(sids), len(new), len(ingested))
        return 0

    sessions = collect_sessions(
        project_dir,
        only_session_id=session_id if not args.backfill else None,
        ingested=ingested,
    )
    if not sessions:
        logger.info("Nothing new to ingest.")
        return 0

    logger.info("Ingesting %d session(s)...", len(sessions))
    enqueued = await ingest_sessions(sessions)
    if enqueued:
        ingested.update(enqueued)
        save_state(ingested)
        logger.info("State updated: %d session(s) now tracked.", len(ingested))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", default=None,
                        help="Ingest just this session id. If omitted (and not --backfill), reads session_id from stdin JSON.")
    parser.add_argument("--backfill", action="store_true",
                        help="Ingest every un-ingested session in the project dir.")
    parser.add_argument("--mark-only", action="store_true",
                        help="Add every session in the project dir to the state file WITHOUT POSTing. Use to seed state after running the one-shot recover script.")
    parser.add_argument("--project-dir", default=None,
                        help="Encoded project folder name or absolute path. Defaults to $CLAUDE_PROJECT_DIR / cwd.")
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))
