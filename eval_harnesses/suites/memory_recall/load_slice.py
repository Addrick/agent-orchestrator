"""Materialize a frozen .sql slice into a temp SQLite DB for eval runs.

Pipeline:
    1. Create temp DB file (or use given path).
    2. Instantiate MemoryManager(db_path=tmp) and call create_schema() — gets
       the live migration path, never rots.
    3. Exec the slice .sql (INSERTs only; no schema).
    4. Call create_schema() AGAIN. The post-create sync block copies
       Memory_Summaries.embedding into vec_Memory_Summaries so KNN works.
    5. Close, return path.

Loader is idempotent against the same (slice.sql, schema) pair: rerunning
overwrites the temp DB cleanly.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional


def _slice_checksum(slice_sql: Path) -> str:
    return hashlib.sha256(slice_sql.read_bytes()).hexdigest()[:16]


def materialize_slice(
    slice_sql: Path,
    out_db: Optional[Path] = None,
    *,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Build a SQLite DB from a slice .sql. Returns the DB path.

    If `cache_dir` is given, the resulting DB is keyed by slice checksum
    and reused across runs (skip rebuild when slice hasn't changed).
    `out_db` overrides the path entirely (no checksum cache).
    """
    from src.memory.memory_manager import MemoryManager

    slice_sql = Path(slice_sql)
    if not slice_sql.exists():
        raise FileNotFoundError(f"slice SQL not found: {slice_sql}")

    if out_db is not None:
        db_path = Path(out_db)
        rebuild = True
    elif cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = _slice_checksum(slice_sql)
        db_path = cache_dir / f"slice_{slice_sql.stem}_{digest}.db"
        rebuild = not db_path.exists()
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db_path = Path(tmp.name)
        rebuild = True

    if rebuild:
        if db_path.exists():
            try:
                db_path.unlink()
            except OSError:
                pass
        mm = MemoryManager(db_path=str(db_path))
        try:
            mm.create_schema()
            with mm.transaction() as conn:
                conn.executescript(slice_sql.read_text(encoding="utf-8"))
            # Second create_schema() call re-runs the embedding sync block,
            # which now sees Memory_Summaries rows with embeddings and
            # populates vec_Memory_Summaries.
            mm.create_schema()
        finally:
            mm.close()

    return db_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="load_slice")
    ap.add_argument("--slice", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args()
    path = materialize_slice(args.slice, out_db=args.out, cache_dir=args.cache_dir)
    print(path)
