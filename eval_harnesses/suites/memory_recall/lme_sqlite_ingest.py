"""Headless LongMemEval ingest into derpr sqlite backend.

Mirrors the live process's two-stage memory pipeline so cross-backend
comparison vs Hindsight is fair:

    1. log_message every haystack turn into User_Interactions
       (channel=question_id so per-Q isolation is preserved at row level)
    2. SqliteConsolidator.deploy() — Phase 1 embed unembedded messages,
       Phase 2 segment by similarity + LLM-summarize → L0 Memory_Summaries
    3. MemoryConsolidator._run_global_consolidation() — cluster L0 by
       cosine ≥ 0.90, LLM-compress each cluster → L1 'Core Profile'

Notes:
    - Per-Q isolation via channel scope. One DB can hold many Qs without
      cross-talk because retrieval already filters by channel.
    - The session_id from LongMemEval is stashed in `tool_context` so
      session-id hit-rate graders can recover it after segmentation.
    - L1 consolidation is *opportunistic* — on a single Q's ~50 sessions
      you may get 0 clusters (similarity threshold 0.90 is strict). That's
      expected; sqlite eval should still grade against L0 in that case.

Usage:
    python -m eval_harnesses.suites.memory_recall.lme_sqlite_ingest \\
        --tier oracle --qid <qid> --db .eval_cache/lme_sqlite/<qid>.db
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from .lme_smoke import TIER_FILES, _to_iso, _pick_questions  # reuse helpers


EVAL_PERSONA = "eval_user"
EVAL_USER = "lme"


def _parse_iso(ts: Optional[str]) -> datetime:
    iso = _to_iso(ts) if ts else None
    if iso:
        # _to_iso emits trailing Z; fromisoformat in 3.11+ handles it
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return datetime.now(timezone.utc)


def _log_haystack(mm: Any, question: Dict[str, Any]) -> int:
    """Insert all haystack turns for one question. Returns row count."""
    qid = question["question_id"]
    sess_ids = question["haystack_session_ids"]
    sessions = question["haystack_sessions"]
    dates = question.get("haystack_dates", [None] * len(sessions))
    n = 0
    for sid, turns, sess_ts in zip(sess_ids, sessions, dates):
        base = _parse_iso(sess_ts)
        for i, turn in enumerate(turns):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            # Stash session_id so the eval grader can recover it from
            # Memory_Segments → User_Interactions later.
            mm.log_message(
                user_identifier=EVAL_USER,
                persona_name=EVAL_PERSONA,
                channel=qid,
                author_role=role,
                author_name=EVAL_USER if role == "user" else EVAL_PERSONA,
                content=content,
                timestamp=base.replace(microsecond=i),  # preserve turn order
                server_id=None,
                tool_context=sid,
            )
            n += 1
    return n


def _build_chat_system(mm: Any) -> Any:
    """Minimal ChatSystem for agent wiring. No personas needed for ingest."""
    from src.engine import TextEngine
    from src.bootstrap import create_chat_system
    # Empty persona map is fine — consolidator loads its own summarizer
    # persona via the agent_config / global default.
    with patch("src.bootstrap.load_personas_from_file", return_value={}):
        return create_chat_system(memory_manager=mm, text_engine=TextEngine())


async def _run_l0(chat_system: Any) -> None:
    from src.agents.sqlite_consolidator import SqliteConsolidator
    agent = SqliteConsolidator(chat_system=chat_system)
    # deploy() processes every channel with unprocessed messages and returns.
    await agent.deploy()


async def _run_l1(mm: Any, chat_system: Any) -> None:
    from src.memory.memory_consolidation import MemoryConsolidator
    consolidator = MemoryConsolidator(
        memory_manager=mm,
        text_engine=chat_system.text_engine,
        embedding_service=chat_system._embedding_service,
    )
    await consolidator._run_global_consolidation()


def _bank_stats(mm: Any) -> Dict[str, int]:
    with mm.transaction() as conn:
        c = conn.cursor()
        stats: Dict[str, int] = {}
        for table in ("User_Interactions", "Message_Embeddings", "Memory_Segments"):
            c.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = c.fetchone()[0]
        c.execute("SELECT summary_level, COUNT(*) FROM Memory_Summaries GROUP BY summary_level")
        for level, n in c.fetchall():
            stats[f"Memory_Summaries_L{level}"] = n
    return stats


def _clear_segment_failures(mm: Any) -> int:
    """Wipe Segment_Failures so the consolidator retries previously-failed
    ranges. Returns rows deleted."""
    with mm.transaction() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM Segment_Failures")
        n = c.fetchone()[0]
        c.execute("DELETE FROM Segment_Failures")
        return n


async def main(
    tier: str, qid: Optional[str], n: int, seed: int, db: Path,
    skip_l1: bool, resume: bool, clear_failures: bool,
) -> int:
    from src.memory.memory_manager import MemoryManager

    if resume:
        if not db.exists():
            raise SystemExit(f"--resume requires existing DB at {db}")
        print(f"Resuming against {db}")
        mm = MemoryManager(db_path=str(db))
        mm.create_schema()  # idempotent; re-verifies vec sync
        if clear_failures:
            n_cleared = _clear_segment_failures(mm)
            print(f"Cleared {n_cleared} Segment_Failures rows")
        print(f"Initial stats: {_bank_stats(mm)}")
    else:
        src = TIER_FILES[tier]
        if not src.exists():
            raise SystemExit(f"missing {tier} JSON at {src}")
        print(f"Loading {tier}...")
        data = json.loads(src.read_text(encoding="utf-8"))
        picked = _pick_questions(data, n, seed, qid)
        print(f"Picked {len(picked)} questions:")
        for q in picked:
            print(f"  {q['question_id']} [{q['question_type']}] "
                  f"sessions={len(q['haystack_sessions'])}")

        db.parent.mkdir(parents=True, exist_ok=True)
        if db.exists():
            print(f"Removing existing {db}")
            db.unlink()

        mm = MemoryManager(db_path=str(db))
        mm.create_schema()

        print(f"\n=== Stage 1: log_message ===")
        t0 = time.monotonic()
        total = 0
        for q in picked:
            total += _log_haystack(mm, q)
        t_log = time.monotonic() - t0
        print(f"Logged {total} turns in {t_log:.1f}s")

    chat_system = _build_chat_system(mm)

    print(f"\n=== Stage 2: SqliteConsolidator (embed + segment + L0 summarize) ===")  # noqa: E501
    t0 = time.monotonic()
    await _run_l0(chat_system)
    t_l0 = time.monotonic() - t0
    print(f"L0 stage: {t_l0:.1f}s")
    print(f"Stats: {_bank_stats(mm)}")

    if not skip_l1:
        print(f"\n=== Stage 3: MemoryConsolidator (L0 -> L1) ===")
        t0 = time.monotonic()
        await _run_l1(mm, chat_system)
        t_l1 = time.monotonic() - t0
        print(f"L1 stage: {t_l1:.1f}s")
        print(f"Stats: {_bank_stats(mm)}")

    print(f"\nDB at: {db}")
    mm.close()
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIER_FILES.keys()), default="oracle")
    ap.add_argument("--qid", default=None)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--skip-l1", action="store_true",
                    help="skip MemoryConsolidator (L0->L1) stage")
    ap.add_argument("--resume", action="store_true",
                    help="reuse existing DB; skip log_message stage")
    ap.add_argument("--clear-failures", action="store_true",
                    help="(resume mode) wipe Segment_Failures so failed ranges retry")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(
        args.tier, args.qid, args.n, args.seed, args.db, args.skip_l1,
        args.resume, args.clear_failures,
    )))
