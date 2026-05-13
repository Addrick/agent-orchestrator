"""
Wipes all Memory_Summaries, Memory_Segments, and vec tables,
then resets User_Interactions.parent_summary_id to NULL so the
SqliteConsolidator will re-process everything from scratch on next deploy().

Usage:
    python -m scripts.reset_memory_summaries [--db PATH]

Defaults to the production database at data/user_memory.db.
Creates a timestamped backup before wiping.
"""

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "user_memory.db"


def reset(db_path: Path, *, skip_backup: bool = False) -> None:
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM Memory_Summaries")
    summary_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM Memory_Segments")
    segment_count = c.fetchone()[0]

    c.execute("PRAGMA table_info(User_Interactions)")
    ui_cols = {row['name'] for row in c.fetchall()}
    has_parent_col = 'parent_summary_id' in ui_cols

    linked_count = 0
    if has_parent_col:
        c.execute("SELECT COUNT(*) FROM User_Interactions WHERE parent_summary_id IS NOT NULL")
        linked_count = c.fetchone()[0]

    print(f"Database: {db_path}")
    print(f"  Summaries to delete: {summary_count}")
    print(f"  Segments to delete:  {segment_count}")
    print(f"  Interactions to unlink: {linked_count}")
    if not has_parent_col:
        print(f"  (parent_summary_id column not present — old schema, skip unlink)")

    if summary_count == 0 and segment_count == 0 and linked_count == 0:
        print("\nNothing to reset.")
        conn.close()
        return

    if not skip_backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = db_path.with_name(f"{db_path.stem}_backup_{ts}{db_path.suffix}")
        print(f"\nBacking up to: {backup_path}")
        conn.close()
        shutil.copy2(db_path, backup_path)
        conn = sqlite3.connect(str(db_path))

    print("\nWiping summaries, segments, and vec tables...")
    # vec tables may not exist on old schemas
    for vec_table in ("vec_Memory_Summaries", "vec_Message_Embeddings"):
        try:
            conn.execute(f"DELETE FROM {vec_table}")
        except sqlite3.OperationalError:
            pass
    conn.execute("DELETE FROM Memory_Summaries")
    conn.execute("DELETE FROM Memory_Segments")
    if has_parent_col:
        conn.execute("UPDATE User_Interactions SET parent_summary_id = NULL WHERE parent_summary_id IS NOT NULL")
    conn.commit()

    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM Memory_Summaries")
    assert c.fetchone()[0] == 0
    if has_parent_col:
        c.execute("SELECT COUNT(*) FROM User_Interactions WHERE parent_summary_id IS NOT NULL")
        assert c.fetchone()[0] == 0

    print("Done. SqliteConsolidator will re-segment and re-summarize on next deploy().")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset memory summaries for re-processing")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to user_memory.db")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup creation")
    args = parser.parse_args()
    reset(args.db, skip_backup=args.no_backup)
