# scripts/recover_claudecode_hindsight.py
"""One-shot ingestion of Claude Code transcripts into the `claudecode` bank.

Reads full session transcripts (user + assistant) from
`~/.claude/projects/<encoded-cwd>/*.jsonl`, not just user prompts from
`~/.claude/history.jsonl`. Each session becomes one Hindsight document.

Density-tuning choices baked in (see
memory/project/plans/hindsight_graph_density_tuning.md):
  - bigger retain_chunk_size (6000) → ~half the extraction calls
  - retain_extraction_mode=concise (explicit; matches server default)
  - narrow retain_mission + reflect_mission + observations_mission
  - junk-prompt filter on user turns
  - per-session document scope via `channel:session_<id>` tag
  - one retain item per session, let server chunk it

One-time script; no idempotency tracking. Re-run after `delete_bank` if
you need to start over.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.memory.backend.hindsight import HindsightBackend
from config.global_config import HINDSIGHT_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cc_recover")

BANK_ID = "claudecode"

# Density-tuning bank config. PATCHed after ensure_bank.
BANK_CONFIG = {
    "retain_chunk_size": 6000,
    "retain_extraction_mode": "concise",
    "enable_observations": True,
}

# ASCII-only — Hindsight 0.6.1 mangles utf-8 on PATCH config bodies.
RETAIN_MISSION = (
    "Extract ONLY durable signal from these Claude Code coding sessions: "
    "the user's stable preferences (style, tools, workflow), recurring "
    "project patterns, architectural decisions, and long-lived constraints. "
    "IGNORE: one-shot questions, slash commands, file paths, single-session "
    "debugging steps, error messages, conversational filler. "
    "Attribute preferences and goals to the user; attribute code suggestions "
    "and analyses to Claude."
)

REFLECT_MISSION = (
    "Reason over the user's Claude Code history to identify stable coding "
    "patterns, recurring project context, and durable preferences. Prefer "
    "synthesis across sessions over single-session detail."
)

OBSERVATIONS_MISSION = (
    "Consolidate the user's preferences, coding patterns, and project "
    "decisions into stable beliefs. Merge duplicates across sessions. "
    "Mark superseded preferences as historical rather than overwriting."
)

# Junk filter for user prompts. Only drops obvious noise — short conversational
# messages ("looks good", "yes please") are KEPT because they carry context
# about the user's coding flow and decisions. Drop only:
#   - empty text
#   - bare slash-command invocations with no prose ("/clear", "/caveman-commit")
#   - command-message / command-name XML wrappers from skill invocations
#   - <<autonomous-loop-...>> and similar harness sentinels
def is_junk_prompt(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if t.startswith("/") and "\n" not in t and len(t.split()) <= 2:
        return True
    if "<command-message>" in t or "<command-name>" in t:
        return True
    if t.startswith("<<") and t.endswith(">>"):
        return True
    return False


def _summarize_tool_uses(content: Any) -> str:
    """Render an assistant tool_use-only turn as `[tools: Read, Edit, Bash]`."""
    if not isinstance(content, list):
        return ""
    tools: List[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            name = block.get("name") or "?"
            tools.append(str(name))
    if not tools:
        return ""
    # Preserve order, dedupe consecutive duplicates for readability
    compact: List[str] = []
    for t in tools:
        if not compact or compact[-1] != t:
            compact.append(t)
    return f"[tools: {', '.join(compact)}]"


def _extract_text(content: Any) -> str:
    """Pull plain text out of a Claude Code message.content field.

    `content` is either a string or a list of blocks. We keep `text` blocks
    and drop tool_use/tool_result/image — those don't help fact extraction
    and would dominate the chunk-size budget.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return ""


def _parse_timestamp(row: Dict[str, Any]) -> Optional[datetime]:
    ts = row.get("timestamp")
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def load_session(
    path: Path, prune_log: Optional[List[Dict[str, Any]]] = None
) -> Optional[Tuple[str, str, datetime, str]]:
    """Read one session jsonl. Return (session_id, project, first_ts, content) or None.

    Skips files with <2 substantive turns or no user messages.
    If prune_log is provided, appends per-row drop reasons for diagnostics.
    """
    session_id = path.stem
    project = path.parent.name

    turns: List[Tuple[datetime, str, str]] = []  # (ts, speaker, text)
    user_turn_count = 0
    junk_prompts: List[str] = []
    empty_assistants = 0
    skipped_kinds: Dict[str, int] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            kind = row.get("type", "?")
            if kind not in ("user", "assistant"):
                skipped_kinds[kind] = skipped_kinds.get(kind, 0) + 1
                continue
            if row.get("isSidechain"):
                skipped_kinds["sidechain"] = skipped_kinds.get("sidechain", 0) + 1
                continue

            msg = row.get("message")
            if not isinstance(msg, dict):
                skipped_kinds[f"{kind}:no-message"] = skipped_kinds.get(f"{kind}:no-message", 0) + 1
                continue
            raw_content = msg.get("content")
            text = _extract_text(raw_content)
            if not text.strip():
                if kind == "assistant":
                    # Render tool_use-only turns as a terse `[tools: ...]` line
                    # instead of dropping. Preserves agent action signal.
                    tool_line = _summarize_tool_uses(raw_content)
                    if tool_line:
                        ts = _parse_timestamp(row) or datetime.now(timezone.utc)
                        turns.append((ts, "Claude", tool_line))
                    else:
                        empty_assistants += 1
                else:
                    skipped_kinds[f"{kind}:empty-text"] = skipped_kinds.get(f"{kind}:empty-text", 0) + 1
                continue

            ts = _parse_timestamp(row) or datetime.now(timezone.utc)

            if kind == "user":
                if is_junk_prompt(text):
                    junk_prompts.append(text.strip())
                    continue
                user_turn_count += 1
                turns.append((ts, "User", text))
            else:
                turns.append((ts, "Claude", text))

    if user_turn_count < 1 or len(turns) < 4:
        if prune_log is not None:
            prune_log.append({
                "kind": "session-dropped",
                "session": session_id[:8],
                "project": project,
                "reason": f"user_turns={user_turn_count}, total_turns={len(turns)} (need >=1 user, >=4 total)",
                "junk_prompts_in_session": junk_prompts,
                "empty_assistant_turns": empty_assistants,
                "skipped_kinds": skipped_kinds,
            })
        return None

    if prune_log is not None and (junk_prompts or empty_assistants):
        prune_log.append({
            "kind": "session-kept-with-pruning",
            "session": session_id[:8],
            "junk_prompts": junk_prompts,
            "empty_assistant_turns": empty_assistants,
        })

    turns.sort(key=lambda x: x[0])
    first_ts = turns[0][0]

    # Compact format: drop repeated speaker headers when same speaker continues.
    lines: List[str] = []
    last_speaker: Optional[str] = None
    for ts, speaker, text in turns:
        header = ts.strftime("%Y-%m-%d %H:%M:%S")
        if speaker == last_speaker:
            lines.append(f"\n{text}")
        else:
            lines.append(f"\n[{header}] {speaker}:\n{text}")
            last_speaker = speaker
    content = "".join(lines).strip()
    return session_id, project, first_ts, content


async def recover(
    dry_run: bool = False,
    limit: Optional[int] = None,
    show_dropped: bool = False,
    show_junk: bool = False,
) -> None:
    projects_root = Path.home() / ".claude" / "projects"
    if not projects_root.exists():
        logger.error("Projects dir not found: %s", projects_root)
        return

    session_files = sorted(projects_root.glob("*/*.jsonl"))
    logger.info("Found %d session files under %s", len(session_files), projects_root)

    collect_log = dry_run or show_dropped or show_junk
    prune_log: List[Dict[str, Any]] = [] if collect_log else None  # type: ignore[assignment]
    sessions: List[Tuple[str, str, datetime, str]] = []
    for path in session_files:
        parsed = load_session(path, prune_log=prune_log)
        if parsed is None:
            continue
        sessions.append(parsed)

    sessions.sort(key=lambda s: s[2])  # chronological
    if limit:
        sessions = sessions[:limit]
    logger.info("Kept %d sessions after filtering", len(sessions))

    if not sessions:
        logger.warning("Nothing to ingest.")
        return

    if dry_run:
        total_chars = sum(len(c) for _, _, _, c in sessions)
        logger.info("DRY RUN — would ingest %d sessions, %d total chars (~%d chunks @ %d)",
                    len(sessions), total_chars,
                    total_chars // BANK_CONFIG["retain_chunk_size"] + len(sessions),
                    BANK_CONFIG["retain_chunk_size"])
        for sid, proj, ts, content in sessions[:5]:
            logger.info("  %s  %s  %s  (%d chars)", ts.isoformat(), proj, sid[:8], len(content))
        if len(sessions) > 5:
            logger.info("  ... and %d more", len(sessions) - 5)

    if prune_log and (dry_run or show_dropped):
        dropped = [e for e in prune_log if e["kind"] == "session-dropped"]
        logger.info("=" * 60)
        logger.info("DROPPED SESSIONS (%d):", len(dropped))
        for e in dropped:
            logger.info("  %s/%s -- %s", e["project"][:40], e["session"], e["reason"])
            if e["junk_prompts_in_session"]:
                for jp in e["junk_prompts_in_session"][:3]:
                    snippet = jp.replace("\n", " ")[:80]
                    logger.info("      junk-prompt: %r", snippet)
                if len(e["junk_prompts_in_session"]) > 3:
                    logger.info("      ... %d more junk prompts", len(e["junk_prompts_in_session"]) - 3)
            if e["empty_assistant_turns"]:
                logger.info("      empty-assistant turns: %d", e["empty_assistant_turns"])
            if e["skipped_kinds"]:
                logger.info("      skipped-row-kinds: %s", e["skipped_kinds"])
        logger.info("=" * 60)

    if prune_log and (dry_run or show_junk):
        pruned = [e for e in prune_log if e["kind"] == "session-kept-with-pruning"]
        total_junk = sum(len(e.get("junk_prompts", [])) for e in pruned)
        total_empty = sum(e.get("empty_assistant_turns", 0) for e in pruned)
        logger.info("WITHIN-SESSION PRUNING in kept sessions:")
        logger.info("  junk user-prompts removed: %d (across %d sessions)", total_junk, len(pruned))
        logger.info("  empty assistant turns removed (no text, no tool_use): %d", total_empty)
        seen_samples = 0
        for e in pruned:
            for jp in e.get("junk_prompts", []):
                if seen_samples >= 15:
                    break
                snippet = jp.replace("\n", " ")[:80]
                logger.info("    sample junk: %r", snippet)
                seen_samples += 1
            if seen_samples >= 15:
                break

    if dry_run:
        return

    hindsight = HindsightBackend(url=HINDSIGHT_URL)
    try:
        logger.info("Ensuring '%s' bank exists...", BANK_ID)
        await hindsight.ensure_bank(
            bank_id=BANK_ID,
            enable_observations=True,
            retain_mission=RETAIN_MISSION,
            reflect_mission=REFLECT_MISSION,
            observations_mission=OBSERVATIONS_MISSION,
        )

        logger.info("Patching bank config: %s", BANK_CONFIG)
        client = hindsight._get_client()
        await client.apatch_bank_config(BANK_ID, BANK_CONFIG)

        logger.info("Enqueuing %d session documents...", len(sessions))
        for i, (session_id, project, first_ts, content) in enumerate(sessions, 1):
            await hindsight.retain_turn(
                bank_id=BANK_ID,
                role="conversation",
                content=content,
                timestamp=first_ts,
                scope_tags=[
                    f"channel:session_{session_id[:8]}",
                    f"project:{project}",
                ],
                source_persona="claudecode",
                metadata={
                    "session_id": session_id,
                    "project": project,
                    "session_start": first_ts.isoformat(),
                },
            )
            if i % 25 == 0:
                logger.info("  enqueued %d/%d", i, len(sessions))

        logger.info("All sessions enqueued. Draining workers (this waits for POST, not server-side extraction)...")
    finally:
        await hindsight.aclose()
    logger.info("Done. Watch Hindsight container logs for ingestion progress; check pending_consolidation to confirm settled.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Claude Code transcripts into Hindsight.")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report; don't POST.")
    parser.add_argument("--limit", type=int, default=None, help="Cap sessions for testing.")
    parser.add_argument("--show-dropped", action="store_true", help="Print which sessions were filtered out and why.")
    parser.add_argument("--show-junk", action="store_true", help="Print which user prompts were pruned within kept sessions.")
    args = parser.parse_args()
    asyncio.run(recover(
        dry_run=args.dry_run, limit=args.limit,
        show_dropped=args.show_dropped, show_junk=args.show_junk,
    ))
