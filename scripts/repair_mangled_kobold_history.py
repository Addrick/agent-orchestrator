#!/usr/bin/env python3
"""Diagnose and repair mangled instruct-template tags in User_Interactions.

Background (DP-126 audit)
-------------------------
The kobold engine portal (port 5003) rebuilds conversation history from the
DB and re-wraps each user turn in ``{{[INPUT]}}``/``{{[OUTPUT]}}`` placeholders
(see ``src/interfaces/kobold_export.py``). If a user row's *content* itself
already contains rendered instruct tags (``### Instruction:``, ``### Response:``,
``{{[INPUT]}}``, ``<|im_start|>`` ...), the export nests them — producing the
mangled blob seen in the UI and poisoning the LLM context on every turn.

Such rows are written when the logged "user turn" was the whole rendered
instruct prompt instead of the clean typed message (pre-sidecar engine code,
or passthrough-mode native ``/api/v1/generate``).

This script is read-only by default. It scans for the corruption signature and
reports it. With ``--repair`` it rewrites user rows to just the extracted last
user turn; with ``--delete`` it removes the offending rows entirely. Either
mutating mode writes a timestamped ``.bak`` copy of the DB first.

Usage
-----
    # Dry run (report only) against the default DB
    python scripts/repair_mangled_kobold_history.py

    # Point at a specific DB / persona
    python scripts/repair_mangled_kobold_history.py --db /path/to/user_memory.db --persona testr

    # Rewrite corrupted user rows to the clean last turn
    python scripts/repair_mangled_kobold_history.py --repair

    # Delete corrupted rows outright (good for throwaway test chatter)
    python scripts/repair_mangled_kobold_history.py --delete
"""

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime

# Tags whose presence inside a stored message indicates a rendered instruct
# prompt was logged instead of the raw user message. Kept in sync with
# KoboldEngineAdapter._extract_last_user_turn.
_USER_TAGS = [
    "### Instruction:",
    "<|im_start|>user",
    "<|start_header_id|>user<|end_header_id|>",
    "{{[INPUT]}}",
    "[INST]",
    "USER:",
    "User:",
    "Input:",
    "<|user|>",
]
_ASSISTANT_TAGS = [
    "### Response:",
    "<|im_start|>assistant",
    "<|start_header_id|>assistant<|end_header_id|>",
    "{{[OUTPUT]}}",
    "[/INST]",
    "ASSISTANT:",
    "Assistant:",
    "Output:",
    "<|assistant|>",
]
_CANDIDATES = list(zip(_USER_TAGS, _ASSISTANT_TAGS))
_ALL_TAGS = _USER_TAGS + _ASSISTANT_TAGS


def is_mangled(content: str) -> bool:
    """A row is mangled if its content contains any instruct-template tag."""
    if not content:
        return False
    return any(tag in content for tag in _ALL_TAGS)


def extract_last_user_turn(prompt: str) -> str:
    """Extract the last clean user turn from a rendered instruct prompt.

    Mirrors KoboldEngineAdapter._extract_last_user_turn so a repaired row
    matches what the live adapter would log going forward.
    """
    if not prompt:
        return ""
    s = prompt.rstrip()

    best_message = None
    best_idx = -1
    for user_tag, assistant_tag in _CANDIDATES:
        idx_assistant = s.rfind(assistant_tag)
        if idx_assistant == -1:
            continue
        idx_user = s[:idx_assistant].rfind(user_tag)
        if idx_user == -1:
            continue
        candidate = s[idx_user + len(user_tag):idx_assistant].strip()
        if not any(tag in candidate for tag in _ALL_TAGS):
            if idx_assistant > best_idx:
                best_idx = idx_assistant
                best_message = candidate
    if best_message is not None:
        return best_message

    # Fallback: last user_tag with nothing tag-like after it.
    max_user_idx = -1
    best_user_tag = None
    for user_tag in _USER_TAGS:
        idx_user = s.rfind(user_tag)
        if idx_user > max_user_idx:
            max_user_idx = idx_user
            best_user_tag = user_tag
    if best_user_tag is not None:
        candidate = s[max_user_idx + len(best_user_tag):].strip()
        if not any(tag in candidate for tag in _ALL_TAGS):
            return candidate

    return s


def _snippet(text: str, n: int = 100) -> str:
    one_line = " ".join((text or "").split())
    return one_line[:n] + ("…" if len(one_line) > n else "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=os.environ.get("MEMORY_DATABASE_FILE", "data/user_memory.db"),
                        help="Path to the SQLite DB (default: $MEMORY_DATABASE_FILE or data/user_memory.db)")
    parser.add_argument("--persona", default=None, help="Restrict to a single persona_name")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--repair", action="store_true",
                       help="Rewrite corrupted user rows to the extracted clean last turn")
    group.add_argument("--delete", action="store_true",
                       help="Delete corrupted rows entirely")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: database not found at {args.db}")
        return 1

    mutating = args.repair or args.delete

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    sql = ("SELECT interaction_id, persona_name, author_role, timestamp, content "
           "FROM User_Interactions WHERE content IS NOT NULL")
    params: list = []
    if args.persona:
        sql += " AND persona_name = ?"
        params.append(args.persona)
    sql += " ORDER BY interaction_id"
    cur.execute(sql, params)

    mangled = [r for r in cur.fetchall() if is_mangled(r["content"])]

    print(f"Connected to {args.db}")
    if args.persona:
        print(f"Persona filter: {args.persona}")
    print(f"Found {len(mangled)} mangled row(s).\n")

    if not mangled:
        conn.close()
        return 0

    for r in mangled:
        found = [t for t in _ALL_TAGS if t in r["content"]]
        print(f"  #{r['interaction_id']} [{r['author_role']}] persona={r['persona_name']} "
              f"ts={r['timestamp']} tags={found}")
        print(f"     before: {_snippet(r['content'])}")
        if r["author_role"] == "user":
            print(f"     clean : {_snippet(extract_last_user_turn(r['content']))}")
        print()

    if not mutating:
        print("DRY RUN — no changes made. Re-run with --repair or --delete to apply.")
        conn.close()
        return 0

    # Back up before mutating.
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = f"{args.db}.{ts}.bak"
    shutil.copy2(args.db, backup)
    print(f"Backup written to {backup}")

    changed = 0
    for r in mangled:
        if args.delete:
            cur.execute("DELETE FROM User_Interactions WHERE interaction_id = ?", (r["interaction_id"],))
            changed += 1
        else:  # repair
            if r["author_role"] != "user":
                # Only user rows have a meaningful "last user turn"; leave
                # assistant/system rows for manual review.
                continue
            cleaned = extract_last_user_turn(r["content"])
            cur.execute("UPDATE User_Interactions SET content = ? WHERE interaction_id = ?",
                        (cleaned, r["interaction_id"]))
            changed += 1

    conn.commit()
    conn.close()
    action = "Deleted" if args.delete else "Repaired"
    print(f"{action} {changed} row(s).")
    if args.repair:
        skipped = len(mangled) - changed
        if skipped:
            print(f"Skipped {skipped} non-user row(s) — review manually or use --delete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
