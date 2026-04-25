"""
scripts/memory_diagnostics.py

Diagnostic tool for evaluating the long-term memory system's output.
Dumps segment boundaries, similarity scores, summary content, and
optionally runs an LLM judge for quality evaluation.

Usage
-----
  python -m scripts.memory_diagnostics                        # overview of all channels
  python -m scripts.memory_diagnostics --channel 114148...    # specific channel
  python -m scripts.memory_diagnostics --verbose              # full message content + scores
  python -m scripts.memory_diagnostics --threshold 0.25       # re-segment with alt threshold (dry run)
  python -m scripts.memory_diagnostics --reset-segments --channel 114...  # delete segments, keep embeddings
  python -m scripts.memory_diagnostics --judge gemini-2.5-flash  # LLM quality evaluation
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from config.global_config import MEMORY_DATABASE_FILE as _DEFAULT_DB_PATH
from memory.memory_manager import MemoryManager

logging.basicConfig(level=logging.WARNING, format='%(levelname)s  %(message)s')
logger = logging.getLogger(__name__)


def blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cosine_sim(a: bytes, b: bytes) -> float:
    va = blob_to_vector(a)
    vb = blob_to_vector(b)
    return float(np.dot(va, vb))


def relative_time(dt) -> str:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    delta = now - dt
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return "just now"
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 7:
        return f"{days}d ago"
    weeks = days // 7
    if weeks < 5:
        return f"{weeks}w ago"
    months = days // 30
    return f"{months}mo ago"


# --Overview ---------------------------------------------------------

def cmd_overview(mm: MemoryManager) -> None:
    """Print a high-level overview of all memory data."""
    conn = mm._get_connection()

    # Total messages
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM User_Interactions"
        " WHERE content IS NOT NULL AND content != ''"
    ).fetchone()
    total_messages = row['c']

    # Embedded messages
    row = conn.execute("SELECT COUNT(*) AS c FROM Message_Embeddings").fetchone()
    total_embedded = row['c']

    # Segments and summaries
    row = conn.execute("SELECT COUNT(*) AS c FROM Memory_Segments").fetchone()
    total_segments = row['c']
    row = conn.execute("SELECT COUNT(*) AS c FROM Memory_Summaries").fetchone()
    total_summaries = row['c']

    print("=== Memory System Overview ===")
    print(f"  Messages (non-empty):  {total_messages}")
    print(f"  Embedded:              {total_embedded}"
          f"  ({total_messages - total_embedded} remaining)")
    print(f"  Segments:              {total_segments}")
    print(f"  Summaries:             {total_summaries}")
    print()

    # Per-channel breakdown
    rows = conn.execute(
        "SELECT seg.channel, seg.persona_name, seg.server_id,"
        " COUNT(seg.segment_id) AS seg_count,"
        " SUM(seg.message_count) AS msg_count,"
        " MIN(seg.created_at) AS first_created,"
        " MAX(seg.created_at) AS last_created"
        " FROM Memory_Segments seg"
        " GROUP BY seg.channel, seg.persona_name, seg.server_id"
        " ORDER BY seg.channel, seg.persona_name"
    ).fetchall()

    if not rows:
        print("  No segments yet.")
        return

    print("  Channel / Persona                     Segments  Messages  Timespan")
    print("  " + "-" * 72)
    for r in rows:
        ch = r['channel']
        if len(ch) > 20:
            ch = ch[:17] + "..."
        pn = r['persona_name']
        if len(pn) > 15:
            pn = pn[:12] + "..."
        first = relative_time(r['first_created']) if r['first_created'] else "?"
        last = relative_time(r['last_created']) if r['last_created'] else "?"
        timespan = f"{first} -> {last}" if first != last else first
        print(f"  {ch:<20} / {pn:<15} {r['seg_count']:>8}"
              f"  {r['msg_count']:>8}  {timespan}")

    # Unprocessed channels
    print()
    unprocessed = conn.execute(
        "SELECT ui.channel, ui.persona_name, COUNT(*) AS c"
        " FROM User_Interactions ui"
        " LEFT JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id"
        " WHERE ui.content IS NOT NULL AND ui.content != ''"
        " AND me.embedding IS NULL"
        " GROUP BY ui.channel, ui.persona_name"
        " ORDER BY c DESC"
    ).fetchall()

    if unprocessed:
        print("  Unprocessed channels:")
        for r in unprocessed:
            ch = r['channel']
            if len(ch) > 20:
                ch = ch[:17] + "..."
            print(f"    {ch:<20} / {r['persona_name']:<15} {r['c']} messages")


# -- Embedding Similarity ------------------------------------------

def _print_embedding_similarity(conn, channel: str) -> None:
    """Print message-to-centroid similarity distribution for embedded messages."""
    rows = conn.execute(
        "SELECT me.embedding"
        " FROM User_Interactions ui"
        " JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id"
        " WHERE ui.channel = ?"
        " AND ui.content IS NOT NULL AND ui.content != ''"
        " ORDER BY ui.interaction_id ASC",
        (channel,)
    ).fetchall()

    if len(rows) < 2:
        return

    # Compute running centroid similarities (same algo as MemoryAgent)
    centroid = blob_to_vector(rows[0]['embedding']).copy()
    n = 1
    similarities: List[float] = []

    for row in rows[1:]:
        vec = blob_to_vector(row['embedding']).copy()
        sim = float(np.dot(centroid, vec))
        similarities.append(sim)
        n += 1
        centroid = centroid * ((n - 1) / n) + vec / n
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

    arr = np.array(similarities)
    print(f"\n  === Embedding Similarity (centroid-based, {len(rows)} messages) ===")
    print(f"  mean={arr.mean():.3f}  min={arr.min():.3f}  "
          f"max={arr.max():.3f}  std={arr.std():.3f}")
    print(f"  p5={np.percentile(arr, 5):.3f}  p25={np.percentile(arr, 25):.3f}  "
          f"median={np.median(arr):.3f}  p75={np.percentile(arr, 75):.3f}  "
          f"p95={np.percentile(arr, 95):.3f}")

    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    counts, _ = np.histogram(similarities, bins=bins)
    max_count = max(max(counts), 1)
    for i in range(len(counts)):
        bar = "#" * int(counts[i] * 40 / max_count)
        print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {counts[i]:>4} {bar}")
    print()


# -- Channel Detail ------------------------------------------------

def cmd_channel(mm: MemoryManager, channel: str, verbose: bool,
                limit: int) -> None:
    """Print detailed segment info for a specific channel."""
    conn = mm._get_connection()

    # Find all persona/server combos for this channel
    combos = conn.execute(
        "SELECT DISTINCT persona_name, server_id FROM Memory_Segments"
        " WHERE channel = ? ORDER BY persona_name",
        (channel,)
    ).fetchall()

    if not combos:
        # Check if there are unprocessed messages
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM User_Interactions"
            " WHERE channel = ? AND content IS NOT NULL AND content != ''",
            (channel,)
        ).fetchone()
        print(f"No segments for channel {channel}.")
        if row['c'] > 0:
            print(f"  ({row['c']} messages exist but are not yet processed)")

    for combo in combos:
        persona = combo['persona_name']
        server_id = combo['server_id']
        _print_channel_segments(
            conn, channel, persona, server_id, verbose, limit
        )

    # Always show embedding similarity distribution for the channel
    _print_embedding_similarity(conn, channel)


def _print_channel_segments(
    conn, channel: str, persona: str, server_id: Optional[str],
    verbose: bool, limit: int,
) -> None:
    """Print segments for a channel+persona+server_id combo."""
    query = (
        "SELECT seg.segment_id, seg.start_interaction_id, seg.end_interaction_id,"
        " seg.message_count, seg.created_at,"
        " seg.first_message_at, seg.last_message_at,"
        " ms.summary_id, ms.content AS summary, ms.model_name,"
        " ms.embedding AS summary_embedding"
        " FROM Memory_Segments seg"
        " LEFT JOIN Memory_Summaries ms ON seg.segment_id = ms.segment_id"
        " WHERE seg.channel = ? AND seg.persona_name = ?"
    )
    params: list = [channel, persona]

    if server_id is not None:
        query += " AND seg.server_id = ?"
        params.append(server_id)
    else:
        query += " AND seg.server_id IS NULL"

    query += " ORDER BY seg.start_interaction_id ASC"
    if limit:
        query += f" LIMIT {limit}"

    segments = conn.execute(query, params).fetchall()

    sid_label = f" (server: {server_id})" if server_id else ""
    print(f"\n=== {channel} / {persona}{sid_label} -- {len(segments)} segments ===\n")

    prev_summary_emb = None
    similarities: List[float] = []

    for seg in segments:
        seg_id = seg['segment_id']
        start = seg['start_interaction_id']
        end = seg['end_interaction_id']
        count = seg['message_count']
        summary = seg['summary']
        summary_emb = seg['summary_embedding']
        model = seg['model_name']
        first_msg = seg['first_message_at']
        last_msg = seg['last_message_at']

        # Show message time range if available, otherwise fall back to created_at
        if first_msg:
            time_str = relative_time(first_msg)
            if last_msg and last_msg != first_msg:
                time_str = f"{relative_time(first_msg)} -> {relative_time(last_msg)}"
        else:
            created = seg['created_at']
            time_str = relative_time(created) if created else "?"

        # Similarity to previous segment's summary
        sim_str = ""
        if prev_summary_emb is not None and summary_emb is not None:
            sim = cosine_sim(prev_summary_emb, summary_emb)
            similarities.append(sim)
            sim_str = f"  (sim to prev: {sim:.3f})"

        print(f"  +- Segment {seg_id}  |  IDs {start}-{end}"
              f"  |  {count} msgs  |  {time_str}  |  {model}{sim_str}")

        if summary:
            # Indent each fact line
            for line in summary.strip().split('\n'):
                print(f"  |  {line}")
        else:
            print("  |  (no summary)")

        if verbose:
            _print_segment_messages(conn, channel, persona, server_id,
                                    start, end, summary_emb)

        print(f"  +{'-' * 70}")
        prev_summary_emb = summary_emb

    # Distribution stats
    if similarities:
        arr = np.array(similarities)
        print(f"\n  Inter-segment similarity: "
              f"mean={arr.mean():.3f}  min={arr.min():.3f}  "
              f"max={arr.max():.3f}  std={arr.std():.3f}")

    # Segment size stats
    sizes = [seg['message_count'] for seg in segments]
    if sizes:
        arr = np.array(sizes)
        print(f"  Segment sizes:           "
              f"mean={arr.mean():.1f}  min={arr.min()}  "
              f"max={arr.max()}  total={arr.sum()}")
    print()


def _print_segment_messages(
    conn, channel: str, persona: str, server_id: Optional[str],
    start_id: int, end_id: int,
    summary_emb: Optional[bytes],
) -> None:
    """In verbose mode, print individual messages and their similarity scores."""
    query = (
        "SELECT ui.interaction_id, ui.author_role, ui.author_name,"
        " ui.content, ui.timestamp,"
        " me.embedding"
        " FROM User_Interactions ui"
        " LEFT JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id"
        " WHERE ui.channel = ? AND ui.persona_name = ?"
        " AND ui.interaction_id BETWEEN ? AND ?"
    )
    params: list = [channel, persona, start_id, end_id]

    if server_id is not None:
        query += " AND ui.server_id = ?"
        params.append(server_id)
    else:
        query += " AND ui.server_id IS NULL"

    query += " ORDER BY ui.interaction_id ASC"
    msgs = conn.execute(query, params).fetchall()

    print("  |")
    print("  |  -- Messages --")

    prev_emb = None
    for msg in msgs:
        iid = msg['interaction_id']
        role = msg['author_role']
        name = msg['author_name'] or '?'
        content = (msg['content'] or '').replace('\n', ' ')
        if len(content) > 120:
            content = content[:117] + "..."
        emb = msg['embedding']
        scores = []
        # Similarity to summary
        if emb and summary_emb:
            sim = cosine_sim(emb, summary_emb)
            scores.append(f"->sum:{sim:.3f}")
        # Similarity to previous message (centroid proxy)
        if emb and prev_emb:
            sim = cosine_sim(emb, prev_emb)
            scores.append(f"->prev:{sim:.3f}")

        score_str = f"  [{', '.join(scores)}]" if scores else ""
        print(f"  |  {iid:>6} [{role[0]}] {name}: {content}{score_str}")
        prev_emb = emb

    print("  |")


# -- Re-segment ----------------------------------------------------

def cmd_resegment(mm: MemoryManager, channel: str, threshold: float,
                  min_size: int) -> None:
    """Re-run segmentation with an alternate threshold (read-only, no DB writes)."""
    conn = mm._get_connection()

    # Get all embedded messages for this channel
    rows = conn.execute(
        "SELECT ui.interaction_id, ui.author_role, ui.author_name,"
        " ui.content, me.embedding"
        " FROM User_Interactions ui"
        " JOIN Message_Embeddings me ON ui.interaction_id = me.interaction_id"
        " WHERE ui.channel = ?"
        " AND ui.content IS NOT NULL AND ui.content != ''"
        " ORDER BY ui.interaction_id ASC",
        (channel,)
    ).fetchall()

    if not rows:
        print(f"No embedded messages in channel {channel}.")
        return

    print(f"\n=== Re-segmentation: threshold={threshold}, "
          f"min_size={min_size}, {len(rows)} messages ===\n")

    # Run segmentation algorithm (mirrors MemoryAgent._segment_by_similarity)
    centroid = None
    n = 0
    segments: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    cut_similarities: List[float] = []
    all_similarities: List[float] = []

    for row in rows:
        vec = blob_to_vector(row['embedding']).copy()

        if centroid is None:
            centroid = vec.copy()
            n = 1
            current.append(dict(row))
            continue

        sim = float(np.dot(centroid, vec))
        all_similarities.append(sim)

        if sim < threshold and len(current) >= min_size:
            cut_similarities.append(sim)
            segments.append({
                'start_id': current[0]['interaction_id'],
                'end_id': current[-1]['interaction_id'],
                'count': len(current),
            })
            current = [dict(row)]
            centroid = vec.copy()
            n = 1
        else:
            current.append(dict(row))
            n += 1
            centroid = centroid * ((n - 1) / n) + vec / n
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

    if current:
        segments.append({
            'start_id': current[0]['interaction_id'],
            'end_id': current[-1]['interaction_id'],
            'count': len(current),
        })

    # Print results
    for i, seg in enumerate(segments):
        print(f"  Segment {i+1:>3}: IDs {seg['start_id']:>6}-{seg['end_id']:>6}"
              f"  ({seg['count']} msgs)")

    sizes = [s['count'] for s in segments]
    arr_sizes = np.array(sizes)
    arr_sims = np.array(all_similarities) if all_similarities else np.array([0.0])

    print(f"\n  Segments:        {len(segments)}")
    print(f"  Segment sizes:   mean={arr_sizes.mean():.1f}  "
          f"min={arr_sizes.min()}  max={arr_sizes.max()}")
    print(f"  All similarities:  mean={arr_sims.mean():.3f}  "
          f"min={arr_sims.min():.3f}  max={arr_sims.max():.3f}  "
          f"std={arr_sims.std():.3f}")
    if cut_similarities:
        arr_cuts = np.array(cut_similarities)
        print(f"  Cut similarities:  mean={arr_cuts.mean():.3f}  "
              f"min={arr_cuts.min():.3f}  max={arr_cuts.max():.3f}")

    # Histogram-style distribution
    print("\n  Similarity distribution:")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    counts, _ = np.histogram(all_similarities, bins=bins)
    for i in range(len(counts)):
        bar = "#" * int(counts[i] * 40 / max(max(counts), 1))
        marker = " <-- threshold" if bins[i] <= threshold < bins[i + 1] else ""
        print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {counts[i]:>4} {bar}{marker}")
    print()


# -- Reset Segments ------------------------------------------------

def cmd_reset_segments(mm: MemoryManager, channel: str) -> None:
    """Delete all segments and summaries for a channel.

    Embeddings are preserved — messages return to 'embedded but unsegmented'
    state, so the next agent cycle will re-segment with the current config.
    """
    conn = mm._get_connection()

    # Count what we're about to delete
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM Memory_Segments WHERE channel = ?",
        (channel,)
    ).fetchone()
    seg_count = row['c']

    if seg_count == 0:
        print(f"No segments to reset for channel {channel}.")
        return

    row = conn.execute(
        "SELECT COUNT(*) AS c FROM Memory_Summaries ms"
        " JOIN Memory_Segments seg ON ms.segment_id = seg.segment_id"
        " WHERE seg.channel = ?",
        (channel,)
    ).fetchone()
    sum_count = row['c']

    print(f"Resetting channel {channel}:")
    print(f"  Deleting {sum_count} summaries and {seg_count} segments.")
    print(f"  Embeddings are preserved — messages will be re-segmented on next cycle.")

    with mm.transaction() as tx:
        # Delete summaries first (FK dependency)
        tx.execute(
            "DELETE FROM Memory_Summaries WHERE segment_id IN"
            " (SELECT segment_id FROM Memory_Segments WHERE channel = ?)",
            (channel,)
        )
        tx.execute(
            "DELETE FROM Memory_Segments WHERE channel = ?",
            (channel,)
        )

    print("  Done.")


# -- LLM Judge -----------------------------------------------------

_JUDGE_KEYS = ('SEGMENTATION', 'COMPLETENESS', 'ACCURACY', 'CONCISENESS')


def _parse_judge_response(text: str, seg_id: int) -> Dict[str, Any]:
    """Parse structured LLM judge output into a rating dict."""
    rating: Dict[str, Any] = {'segment_id': seg_id, 'raw': text}
    for line in text.split('\n'):
        for key in _JUDGE_KEYS:
            if line.startswith(f"{key}:"):
                try:
                    rating[key.lower()] = int(line.split(':')[1].strip()[0])
                except (ValueError, IndexError):
                    pass
        if line.startswith("NOTES:"):
            rating['notes'] = line.split(':', 1)[1].strip()
    return rating


def cmd_judge(mm: MemoryManager, channel: str, model: str,
              limit: int) -> None:
    """Run an LLM judge to evaluate segmentation and summary quality."""
    try:
        from google import genai
    except ImportError:
        print("Error: google-genai SDK not installed. Install with: pip install google-genai")
        return

    conn = mm._get_connection()

    # Get segments with their messages and summaries
    query = (
        "SELECT seg.segment_id, seg.start_interaction_id, seg.end_interaction_id,"
        " seg.message_count, ms.content AS summary"
        " FROM Memory_Segments seg"
        " JOIN Memory_Summaries ms ON seg.segment_id = ms.segment_id"
        " WHERE seg.channel = ?"
        " ORDER BY seg.start_interaction_id ASC"
    )
    if limit:
        query += f" LIMIT {limit}"

    segments = conn.execute(query, (channel,)).fetchall()

    if not segments:
        print(f"No summaries to evaluate for channel {channel}.")
        return

    print(f"\n=== LLM Judge ({model}) -- {len(segments)} segments ===\n")

    client = genai.Client()
    ratings: List[Dict[str, Any]] = []

    for seg in segments:
        seg_id = seg['segment_id']
        start = seg['start_interaction_id']
        end = seg['end_interaction_id']

        # Fetch source messages
        msgs = conn.execute(
            "SELECT interaction_id, author_role, author_name, content"
            " FROM User_Interactions"
            " WHERE interaction_id BETWEEN ? AND ?"
            " AND content IS NOT NULL AND content != ''"
            " ORDER BY interaction_id ASC",
            (start, end)
        ).fetchall()

        transcript = "\n".join(
            f"[{m['author_role']}] {m['author_name'] or '?'}: {m['content']}"
            for m in msgs
        )

        prompt = (
            "You are evaluating a memory system that segments conversations by "
            "topic and extracts facts from each segment.\n\n"
            "TRANSCRIPT:\n"
            f"{transcript}\n\n"
            "EXTRACTED FACTS:\n"
            f"{seg['summary']}\n\n"
            "Rate the following on a scale of 1-5:\n"
            "1. SEGMENTATION: Does this transcript look like a coherent single "
            "topic, or does it mix unrelated topics? (5 = coherent, 1 = mixed)\n"
            "2. COMPLETENESS: Are all important facts from the transcript "
            "captured? (5 = complete, 1 = many missing)\n"
            "3. ACCURACY: Are the extracted facts faithful to the transcript? "
            "No hallucinations? (5 = accurate, 1 = hallucinated)\n"
            "4. CONCISENESS: Are the facts concise without losing meaning? "
            "(5 = concise, 1 = verbose/redundant)\n\n"
            "Respond in exactly this format:\n"
            "SEGMENTATION: N\n"
            "COMPLETENESS: N\n"
            "ACCURACY: N\n"
            "CONCISENESS: N\n"
            "NOTES: one line of explanation"
        )

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            text = response.text.strip()
            rating = _parse_judge_response(text, seg_id)
            ratings.append(rating)

            seg_label = f"Segment {seg_id} (IDs {start}-{end})"
            scores = []
            for key in ('segmentation', 'completeness', 'accuracy', 'conciseness'):
                if key in rating:
                    scores.append(f"{key[:3]}={rating[key]}")
            notes = rating.get('notes', '')
            print(f"  {seg_label}: {', '.join(scores)}")
            if notes:
                print(f"    {notes}")

        except Exception as e:
            print(f"  Segment {seg_id}: ERROR -- {e}")

    # Aggregate scores
    if ratings:
        print(f"\n  -- Aggregate ({len(ratings)} segments) --")
        for key in ('segmentation', 'completeness', 'accuracy', 'conciseness'):
            vals = [r[key] for r in ratings if key in r]
            if vals:
                arr = np.array(vals, dtype=float)
                print(f"    {key:<14}: mean={arr.mean():.2f}  "
                      f"min={arr.min():.0f}  max={arr.max():.0f}")
    print()


# -- Main ----------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Memory system diagnostics and evaluation."
    )
    parser.add_argument(
        '--channel', type=str, default=None,
        help='Channel ID to inspect (omit for overview of all channels)'
    )
    parser.add_argument(
        '--limit', type=int, default=0,
        help='Max segments to display (0 = all)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Show individual messages with similarity scores'
    )
    parser.add_argument(
        '--threshold', type=float, default=None,
        help='Re-segment with this threshold (read-only, does not write to DB)'
    )
    parser.add_argument(
        '--min-size', type=int, default=3,
        help='Min segment size for re-segmentation (default: 3)'
    )
    parser.add_argument(
        '--judge', type=str, default=None, metavar='MODEL',
        help='Run LLM judge with this model (e.g., gemini-2.5-flash)'
    )
    parser.add_argument(
        '--reset-segments', action='store_true',
        help='Delete segments+summaries for --channel (keeps embeddings). '
             'Next agent cycle will re-segment with current config.'
    )
    parser.add_argument(
        '--db', type=str, default=None,
        help='Path to database file (default: production DB)'
    )

    args = parser.parse_args()

    load_dotenv()
    db_path = args.db or os.environ.get("MEMORY_DATABASE_FILE") or _DEFAULT_DB_PATH
    mm = MemoryManager(db_path=db_path)
    mm.create_schema()

    if args.reset_segments:
        if not args.channel:
            parser.error("--reset-segments requires --channel")
        cmd_reset_segments(mm, args.channel)
    elif args.threshold is not None:
        if not args.channel:
            parser.error("--threshold requires --channel")
        cmd_resegment(mm, args.channel, args.threshold, args.min_size)
    elif args.judge:
        if not args.channel:
            parser.error("--judge requires --channel")
        cmd_judge(mm, args.channel, args.judge, args.limit)
    elif args.channel:
        cmd_channel(mm, args.channel, args.verbose, args.limit)
    else:
        cmd_overview(mm)


if __name__ == '__main__':
    main()
