"""Browse user_db summaries and emit scenario JSON stubs.

Complements `curate.py` (which probes Hindsight banks to refine locators
after a scenario exists). This tool is the upstream half: read the local
sqlite Memory_Summaries to discover candidate facts, then emit a stub
you paste into a fixtures file.

Two scenario shapes supported:

    sqlite     -> {"expected_segments": [...], "anti_segments": [...]}
                  graded by RetrievalHitsGrader against retrieve_relevant_summaries
    hindsight  -> {"expected_facts": [{key, source:"history", locators:[...]}]}
                  graded by SemanticRecallGrader against arecall

`--shape both` emits both blocks; you keep the one(s) you want to test.

Usage:
    # List recent summaries from a channel
    python -m eval_harnesses.suites.memory_recall.draft_scenarios list \\
        [--channel arbitr] [--level 0] [--limit 30] [--snippet 140]

    # Keyword search over summary content
    python -m eval_harnesses.suites.memory_recall.draft_scenarios search "surface" \\
        [--channel arbitr] [--limit 20]

    # Show one segment's summaries (and optionally raw turns)
    python -m eval_harnesses.suites.memory_recall.draft_scenarios show 909 [--with-turns]

    # Emit a scenario stub
    python -m eval_harnesses.suites.memory_recall.draft_scenarios draft \\
        --query "surface laptop charger reset" \\
        --segments 909,910 \\
        --noise-segments 907 \\
        --shape both \\
        [--id surface_smc_reset] [--bank test_persona]
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import textwrap
from typing import Any, Dict, List, Optional

DEFAULT_DB = "data/user_memory.db"


def _connect(db: str) -> sqlite3.Connection:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    return con


def _wrap(text: str, n: int) -> str:
    text = (text or "").replace("\n", " ").replace("\r", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def cmd_list(con: sqlite3.Connection, args: argparse.Namespace) -> int:
    where, params = ["1=1"], []
    if args.channel:
        where.append("seg.channel = ?")
        params.append(args.channel)
    if args.level is not None:
        where.append("ms.summary_level = ?")
        params.append(args.level)
    sql = f"""
        SELECT ms.summary_id, ms.segment_id, ms.summary_level,
               seg.channel, ms.created_at, ms.content
        FROM Memory_Summaries ms
        JOIN Memory_Segments seg ON ms.segment_id = seg.segment_id
        WHERE {' AND '.join(where)}
        ORDER BY ms.created_at DESC, ms.summary_id DESC
        LIMIT ?
    """
    params.append(args.limit)
    rows = con.execute(sql, params).fetchall()
    if not rows:
        print("(no rows)")
        return 0
    print(f"{'sid':>5}  {'seg':>5}  {'lvl':>3}  {'chan':<10}  {'created':<19}  content")
    for r in rows:
        print(
            f"{r['summary_id']:>5}  {r['segment_id']:>5}  "
            f"{r['summary_level']:>3}  {r['channel'][:10]:<10}  "
            f"{(r['created_at'] or '')[:19]:<19}  {_wrap(r['content'], args.snippet)}"
        )
    return 0


def cmd_search(con: sqlite3.Connection, args: argparse.Namespace) -> int:
    where = ["LOWER(ms.content) LIKE ?"]
    params: List[Any] = [f"%{args.term.lower()}%"]
    if args.channel:
        where.append("seg.channel = ?")
        params.append(args.channel)
    sql = f"""
        SELECT ms.summary_id, ms.segment_id, seg.channel, ms.content
        FROM Memory_Summaries ms
        JOIN Memory_Segments seg ON ms.segment_id = seg.segment_id
        WHERE {' AND '.join(where)}
        ORDER BY ms.summary_id DESC
        LIMIT ?
    """
    params.append(args.limit)
    rows = con.execute(sql, params).fetchall()
    if not rows:
        print("(no matches)")
        return 0
    for r in rows:
        print(f"sid={r['summary_id']:>4} seg={r['segment_id']:>4} chan={r['channel']:<8}  {_wrap(r['content'], 200)}")
    return 0


def cmd_show(con: sqlite3.Connection, args: argparse.Namespace) -> int:
    seg = con.execute("SELECT * FROM Memory_Segments WHERE segment_id = ?", (args.segment_id,)).fetchone()
    if not seg:
        print(f"segment {args.segment_id}: not found")
        return 1
    print(f"=== segment {args.segment_id} ({seg['channel']}, persona={seg['persona_name']}) ===")
    print(f"interactions: {seg['start_interaction_id']}..{seg['end_interaction_id']}  count={seg['message_count']}")
    print()
    sums = con.execute(
        "SELECT summary_id, summary_level, content, model_name, created_at "
        "FROM Memory_Summaries WHERE segment_id = ? ORDER BY summary_level",
        (args.segment_id,),
    ).fetchall()
    for s in sums:
        print(f"--- summary {s['summary_id']} (level {s['summary_level']}, {s['model_name']}, {s['created_at']}) ---")
        print(textwrap.indent(s["content"] or "", "    "))
        print()
    if args.with_turns:
        rows = con.execute(
            "SELECT interaction_id, author_role, author_name, substr(content,1,180) "
            "FROM User_Interactions WHERE interaction_id BETWEEN ? AND ? AND channel = ? "
            "ORDER BY interaction_id LIMIT 12",
            (seg["start_interaction_id"], seg["end_interaction_id"], seg["channel"]),
        ).fetchall()
        print("--- first 12 turns in id range ---")
        for r in rows:
            print(f"  [{r[0]}] {r[1]}/{r[2] or '-'}: {_wrap(r[3], 160)}")
    return 0


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:48] or "scenario"


def _segment_excerpt(con: sqlite3.Connection, seg_id: int) -> Optional[str]:
    """First substantive line from any summary level for this segment.

    Prefers level 0 (most specific) but falls back to higher levels so a
    consolidated-only segment still gets a locator candidate rather than
    a TODO marker.
    """
    rows = con.execute(
        "SELECT content FROM Memory_Summaries WHERE segment_id = ? "
        "ORDER BY summary_level ASC LIMIT 4",
        (seg_id,),
    ).fetchall()
    for row in rows:
        text = (row["content"] or "").strip()
        for line in text.splitlines():
            line = line.strip().lstrip("-*•").strip()
            if len(line) >= 20:
                return line[:160]
        if text:
            return text[:160]
    return None


def cmd_draft(con: sqlite3.Connection, args: argparse.Namespace) -> int:
    expected = [int(x) for x in args.segments.split(",") if x.strip()]
    noise = [int(x) for x in (args.noise_segments or "").split(",") if x.strip()]

    scenario: Dict[str, Any] = {
        "id": args.id or _slug(args.query),
        "description": args.description or f"recall scenario for: {args.query}",
        "user_request": args.query,
        "persona_name": args.persona,
        "channel": args.channel,
        "graders": [],
        "expectations": {},
    }

    if args.shape in ("sqlite", "both"):
        scenario["expectations"]["expected_segments"] = expected
        if noise:
            scenario["expectations"]["anti_segments"] = noise
        scenario["graders"].append("retrieval_hits")

    if args.shape in ("hindsight", "both"):
        expected_facts = []
        for sid in expected:
            loc = _segment_excerpt(con, sid)
            expected_facts.append({
                "key": f"seg_{sid}",
                "source": "history",
                "locators": [loc] if loc else ["TODO_FILL_LOCATOR"],
                "_seg_id_hint": sid,
            })
        noise_facts = []
        for sid in noise:
            loc = _segment_excerpt(con, sid)
            noise_facts.append({
                "key": f"noise_seg_{sid}",
                "source": "history",
                "locators": [loc] if loc else ["TODO_FILL_LOCATOR"],
                "_seg_id_hint": sid,
            })
        scenario["expectations"]["expected_facts"] = expected_facts
        if noise_facts:
            scenario["expectations"]["noise_facts"] = noise_facts
        scenario["graders"].append("semantic_recall")
        scenario["bank"] = args.bank or "TODO_BANK"
        scenario["k_sweep"] = [1, 3, 5, 10]
        scenario["thresholds"] = {"k_pass": 5, "recall": 0.5, "noise_rate": 0.25}

    json.dump(scenario, sys.stdout, indent=2)
    sys.stdout.write("\n")
    sys.stderr.write(
        "\nNOTE: review before pasting. Locator excerpts may be too long or\n"
        "non-unique against the live bank. Run `curate.py` against the target\n"
        "bank to verify each locator resolves to the expected memory(s) before\n"
        "committing. Strip `_seg_id_hint` fields if you want a clean fixture.\n"
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="draft_scenarios",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", default=DEFAULT_DB)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list summaries (newest first)")
    pl.add_argument("--channel", default=None)
    pl.add_argument("--level", type=int, default=None)
    pl.add_argument("--limit", type=int, default=30)
    pl.add_argument("--snippet", type=int, default=140)
    pl.set_defaults(func=cmd_list)

    ps = sub.add_parser("search", help="LIKE-search summary content")
    ps.add_argument("term")
    ps.add_argument("--channel", default=None)
    ps.add_argument("--limit", type=int, default=20)
    ps.set_defaults(func=cmd_search)

    psh = sub.add_parser("show", help="show a segment's summaries + metadata")
    psh.add_argument("segment_id", type=int)
    psh.add_argument("--with-turns", action="store_true")
    psh.set_defaults(func=cmd_show)

    pd = sub.add_parser("draft", help="emit a scenario JSON stub for chosen segments")
    pd.add_argument("--query", required=True)
    pd.add_argument("--segments", required=True, help="comma-separated expected segment_ids")
    pd.add_argument("--noise-segments", default=None)
    pd.add_argument("--shape", choices=["sqlite", "hindsight", "both"], default="both")
    pd.add_argument("--id", default=None)
    pd.add_argument("--description", default=None)
    pd.add_argument("--persona", default="default")
    pd.add_argument("--channel", default="arbitr")
    pd.add_argument("--bank", default=None)
    pd.set_defaults(func=cmd_draft)

    args = p.parse_args(argv)
    con = _connect(args.db)
    try:
        return args.func(con, args)
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
