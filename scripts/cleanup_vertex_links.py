import sqlite3
import os
import argparse
import sys
import sqlite_vec

# Add src to path so we can import utilities
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.message_utils import strip_vertex_links  # noqa: E402
from config.global_config import MEMORY_DATABASE_FILE  # noqa: E402

def cleanup_db(db_path, dry_run=True, invalidate_embeddings=True):
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Load sqlite-vec extension
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    
    cursor = conn.cursor()

    print(f"Connected to {db_path}")
    if dry_run:
        print("DRY RUN - No changes will be committed.")

    # 1. Process User_Interactions
    cursor.execute("SELECT interaction_id, content FROM User_Interactions WHERE content LIKE '%vertexaisearch.cloud.google.com%'")
    rows = cursor.fetchall()
    print(f"Found {len(rows)} rows in User_Interactions with Vertex links.")

    for row in rows:
        original = row['content']
        cleaned = strip_vertex_links(original)
        if original != cleaned:
            if not dry_run:
                cursor.execute("UPDATE User_Interactions SET content = ? WHERE interaction_id = ?", (cleaned, row['interaction_id']))
                if invalidate_embeddings:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Message_Embeddings'")
                    if cursor.fetchone():
                        cursor.execute("DELETE FROM Message_Embeddings WHERE interaction_id = ?", (row['interaction_id'],))
                    
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_Message_Embeddings'")
                    if cursor.fetchone():
                        cursor.execute("DELETE FROM vec_Message_Embeddings WHERE interaction_id = ?", (row['interaction_id'],))
            else:
                print(f"Would update interaction_id {row['interaction_id']}")
                # print(f"  Original: {original[:100]}...")
                # print(f"  Cleaned:  {cleaned[:100]}...")

    # 2. Process Interaction_Edit_History
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Interaction_Edit_History'")
    if cursor.fetchone():
        cursor.execute("SELECT edit_id, old_content FROM Interaction_Edit_History WHERE old_content LIKE '%vertexaisearch.cloud.google.com%'")
        rows = cursor.fetchall()
        print(f"Found {len(rows)} rows in Interaction_Edit_History with Vertex links.")

        for row in rows:
            original = row['old_content']
            cleaned = strip_vertex_links(original)
            if original != cleaned:
                if not dry_run:
                    cursor.execute("UPDATE Interaction_Edit_History SET old_content = ? WHERE edit_id = ?", (cleaned, row['edit_id']))
                    # Interaction_Edit_History embeddings are in Edit_History_Embeddings
                    if invalidate_embeddings:
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Edit_History_Embeddings'")
                        if cursor.fetchone():
                            cursor.execute("DELETE FROM Edit_History_Embeddings WHERE edit_id = ?", (row['edit_id'],))
                else:
                    print(f"Would update edit_id {row['edit_id']} in history")
    else:
        print("Table 'Interaction_Edit_History' does not exist, skipping.")

    if not dry_run:
        conn.commit()
        print("Changes committed successfully.")
    else:
        print("Dry run complete. No changes made.")

    conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cleanup Vertex AI links from the database.")
    parser.add_argument(
        "--db",
        default=MEMORY_DATABASE_FILE,
        help="Path to the database file (defaults to config.global_config.MEMORY_DATABASE_FILE).",
    )
    parser.add_argument("--no-dry-run", action="store_true", help="Perform actual updates instead of dry run.")
    parser.add_argument("--skip-embeddings", action="store_true", help="Do not invalidate embeddings for modified rows.")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)
    cleanup_db(db_path, dry_run=not args.no_dry_run, invalidate_embeddings=not args.skip_embeddings)
