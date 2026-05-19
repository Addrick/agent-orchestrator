"""Sequential m-tier bank ingest with progress + overlap.

Ingests one qid at a time. Polls Hindsight's per-bank operations endpoint
for progress. When the current bank is "nearly drained" (pending+running
ops below `--overlap-threshold`), POSTs the next bank's retain so the LLM
worker pool stays saturated with no idle gap.

Usage:
    python -m eval_harnesses.suites.memory_recall.lme_ingest_queue \\
        --tier m --bank-prefix lme_m \\
        --qids q1,q2,q3 \\
        --state .eval_cache/lme_ingest_queue.state.json

Re-run with the same --state to resume: banks already at fact_count > 0
and idle (no pending ops) are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.global_config import HINDSIGHT_URL
from src.memory.backend.hindsight import (
    HINDSIGHT_API_PREFIX,
    HindsightAPIError,
    HindsightRESTClient,
)

from .lme_smoke import TIER_FILES, _session_to_text, _to_iso


POLL_INTERVAL_S = 15.0


def _fmt_dur(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"


async def _ops_counts(client: HindsightRESTClient, bank: str) -> Dict[str, int]:
    """Return {pending, running, completed} counts for a bank."""
    out = {}
    for status in ("pending", "running", "completed"):
        try:
            r = await client._request(
                "GET", f"{HINDSIGHT_API_PREFIX}/banks/{bank}/operations",
                params={"status": status, "limit": 1},
            )
            out[status] = (r or {}).get("total", 0)
        except HindsightAPIError:
            out[status] = -1
    return out


async def _bank_exists(client: HindsightRESTClient, bank: str) -> bool:
    try:
        await client._request("GET", f"{HINDSIGHT_API_PREFIX}/banks/{bank}/stats")
        return True
    except HindsightAPIError as e:
        if e.status_code == 404:
            return False
        raise


async def _bank_stats(client: HindsightRESTClient, bank: str) -> Dict[str, Any]:
    try:
        return await client._request(
            "GET", f"{HINDSIGHT_API_PREFIX}/banks/{bank}/stats"
        )
    except HindsightAPIError:
        return {}


def _build_items(q: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build retain items, deduping by document_id.

    ~90% of LongMemEval m-tier qids have repeated session_ids inside their
    own haystack (content is identical). Hindsight rejects dup document_ids
    in one batch, so we drop later duplicates.
    """
    qid = q["question_id"]
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    n_dups = 0
    for sid, turns, ts in zip(
        q["haystack_session_ids"], q["haystack_sessions"],
        q.get("haystack_dates", [None] * len(q["haystack_sessions"])),
    ):
        if sid in seen:
            n_dups += 1
            continue
        seen.add(sid)
        content = _session_to_text(turns)
        items.append({
            "content": content, "tags": [qid, "lme_m"],
            "timestamp": _to_iso(ts), "document_id": sid,
            "metadata": {"question_id": qid, "session_id": sid},
        })
    if n_dups:
        print(f"  [{qid}] deduped {n_dups} repeated session_ids", flush=True)
    return items


DEFAULT_RETAIN_MISSION = "Store user-assistant turns verbatim for recall eval."
DEFAULT_REFLECT_MISSION = "Surface turns relevant to the user's question."


async def _post_retain(
    client: HindsightRESTClient, bank: str, q: Dict[str, Any],
    retain_mission: str = DEFAULT_RETAIN_MISSION,
    reflect_mission: str = DEFAULT_REFLECT_MISSION,
) -> Dict[str, Any]:
    """Create bank if missing + fire async retain. Returns sizing info."""
    qid = q["question_id"]
    if not await _bank_exists(client, bank):
        await client.acreate_bank(
            bank,
            retain_mission=retain_mission,
            reflect_mission=reflect_mission,
        )
        print(f"  [{qid}] created bank '{bank}'", flush=True)
    items = _build_items(q)
    total_chars = sum(len(i["content"]) for i in items)
    await client.aretain(bank, items, async_=True)
    print(
        f"  [{qid}] retained {len(items)} sessions "
        f"(~{total_chars/1e6:.1f}M chars, ~{total_chars//4000}k tokens)",
        flush=True,
    )
    return {"sessions": len(items), "chars": total_chars}


def _load_state(path: Optional[Path]) -> Dict[str, Any]:
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"completed": [], "started_at": time.time(), "per_bank_seconds": {}}


def _save_state(path: Optional[Path], state: Dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _eta_remaining(
    state: Dict[str, Any], queued_left: int, current_elapsed: float,
    current_remaining: Optional[float] = None,
) -> str:
    """Rough ETA: avg observed bank duration × remaining banks.

    Bootstrap when no completed samples: estimate per-bank duration as
    current bank's projected total (elapsed + remaining). Less reliable
    until first bank finishes but better than 'unknown'.
    """
    durations = list(state.get("per_bank_seconds", {}).values())
    if durations:
        avg = sum(durations) / len(durations)
        cur_remaining = max(avg - current_elapsed, 0.0)
        total = cur_remaining + queued_left * avg
        return _fmt_dur(total)
    # Bootstrap: project from current bank
    if current_remaining is None:
        return "unknown"
    est_avg = current_elapsed + current_remaining
    return _fmt_dur(current_remaining + queued_left * est_avg) + "~"


async def _ingest_one(
    client: HindsightRESTClient,
    qid: str,
    q: Dict[str, Any],
    bank: str,
    queue_next: Optional[Dict[str, Any]],
    overlap_threshold: int,
    state: Dict[str, Any],
    state_path: Optional[Path],
    queued_left: int,
    retain_mission: str = DEFAULT_RETAIN_MISSION,
    reflect_mission: str = DEFAULT_REFLECT_MISSION,
) -> float:
    """Drive one bank to completion; fire next when close to drain. Returns seconds."""
    print(f"\n=== qid={qid} [{q['question_type']}] bank={bank} ===", flush=True)
    # Retry retain on transient 5xx (observed: dupe-500 during overlap window)
    sizing = None
    for attempt in range(5):
        try:
            sizing = await _post_retain(
                client, bank, q,
                retain_mission=retain_mission,
                reflect_mission=reflect_mission,
            )
            break
        except HindsightAPIError as e:
            if 500 <= e.status_code < 600 and attempt < 4:
                wait = 30 * (attempt + 1)
                print(f"  [{qid}] retain {e.status_code} (attempt {attempt+1}/5); "
                      f"sleeping {wait}s", file=sys.stderr, flush=True)
                await asyncio.sleep(wait)
                continue
            raise
    if sizing is None:
        raise RuntimeError(f"retain failed for {qid} after retries")

    t0 = time.monotonic()
    next_fired = queue_next is None  # nothing to fire
    last_completed = 0
    last_print = 0.0
    while True:
        ops = await _ops_counts(client, bank)
        pending, running, completed = ops["pending"], ops["running"], ops["completed"]
        busy = pending + running
        elapsed = time.monotonic() - t0
        rate = completed / max(elapsed / 60, 0.001)  # ops/min
        stats = await _bank_stats(client, bank)
        nc = stats.get("node_counts", {}) or {}
        facts = sum(nc.values()) if nc else 0

        # progress line — print every ~minute or on completion delta
        if elapsed - last_print >= 60 or completed != last_completed or busy == 0:
            eta_self_s = pending / max(rate, 0.01) * 60 if pending else 0.0
            eta_self = _fmt_dur(eta_self_s)
            eta_total = _eta_remaining(state, queued_left, elapsed, eta_self_s)
            print(
                f"  [{qid}] ops: {completed} done / {busy} busy ({pending}p+{running}r) "
                f"| facts: {facts} | elapsed {_fmt_dur(elapsed)} "
                f"| rate {rate:.1f}/m | eta bank {eta_self} | eta total {eta_total}",
                flush=True,
            )
            last_print = elapsed
            last_completed = completed

        # overlap-fire next bank
        if not next_fired and busy <= overlap_threshold and queue_next is not None:
            nq = queue_next["q"]
            nb = queue_next["bank"]
            print(
                f"  [{qid}] near-drain (busy={busy} ≤ {overlap_threshold}); "
                f"firing next [{nq['question_id']}]",
                flush=True,
            )
            try:
                await _post_retain(
                    client, nb, nq,
                    retain_mission=retain_mission,
                    reflect_mission=reflect_mission,
                )
            except Exception as e:
                print(f"  [WARN] failed to fire next {nq['question_id']}: {e}",
                      file=sys.stderr, flush=True)
            next_fired = True

        if busy == 0 and completed > 0:
            # retain queue drained; consolidation may continue in background
            elapsed_final = time.monotonic() - t0
            state["per_bank_seconds"][bank] = elapsed_final
            # small delay to let consolidator materialize a few counts
            await asyncio.sleep(5)
            stats = await _bank_stats(client, bank)
            facts = sum((stats.get("node_counts") or {}).values())
            pending_c = stats.get("pending_consolidation", 0)
            print(
                f"  [{qid}] RETAIN DRAINED in {_fmt_dur(elapsed_final)} | "
                f"facts so far: {facts} (consolidating: {pending_c}) | "
                f"moving on; consolidation continues in background",
                flush=True,
            )
            state.setdefault("completed", []).append(qid)
            _save_state(state_path, state)
            return elapsed_final

        await asyncio.sleep(POLL_INTERVAL_S)


async def main(
    tier: str, qids: List[str], bank_prefix: str,
    overlap_threshold: int, state_path: Optional[Path],
    skip_existing: bool,
    bank_suffix: str = "",
    retain_mission: str = DEFAULT_RETAIN_MISSION,
    reflect_mission: str = DEFAULT_REFLECT_MISSION,
) -> int:
    src = TIER_FILES[tier]
    if not src.exists():
        raise SystemExit(f"missing {tier} JSON at {src}")
    print(f"Loading {tier}...", flush=True)
    data = json.loads(src.read_text(encoding="utf-8"))
    by_id = {d["question_id"]: d for d in data}
    missing = [q for q in qids if q not in by_id]
    if missing:
        raise SystemExit(f"qids not in {tier}: {missing}")

    state = _load_state(state_path)
    already_done = set(state.get("completed", []))

    client = HindsightRESTClient(HINDSIGHT_URL, timeout=300.0)

    # Filter: skip qids whose bank already has facts and no pending ops
    queue: List[Dict[str, Any]] = []
    for qid in qids:
        if qid in already_done:
            print(f"skip {qid}: in state.completed", flush=True)
            continue
        bank = f"{bank_prefix}_{qid}{bank_suffix}"
        if skip_existing and await _bank_exists(client, bank):
            stats = await _bank_stats(client, bank)
            facts = sum((stats.get("node_counts") or {}).values())
            ops = await _ops_counts(client, bank)
            if facts > 0 and ops["pending"] + ops["running"] == 0:
                print(f"skip {qid}: bank exists with {facts} facts (idle)", flush=True)
                state.setdefault("completed", []).append(qid)
                _save_state(state_path, state)
                continue
        queue.append({"qid": qid, "q": by_id[qid], "bank": bank})

    print(f"\n=== queue: {len(queue)} banks to ingest ===", flush=True)
    for i, item in enumerate(queue):
        print(f"  {i+1:>2}. {item['qid']} [{item['q']['question_type']}] -> {item['bank']}",
              flush=True)
    if not queue:
        print("nothing to do.", flush=True)
        return 0

    run_t0 = time.monotonic()
    for i, item in enumerate(queue):
        nxt = queue[i + 1] if i + 1 < len(queue) else None
        await _ingest_one(
            client=client, qid=item["qid"], q=item["q"], bank=item["bank"],
            queue_next=nxt, overlap_threshold=overlap_threshold,
            state=state, state_path=state_path,
            queued_left=len(queue) - i - 1,
            retain_mission=retain_mission,
            reflect_mission=reflect_mission,
        )

    total = time.monotonic() - run_t0
    print(f"\n=== ALL DONE: {len(queue)} banks in {_fmt_dur(total)} ===", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIER_FILES.keys()), default="m")
    ap.add_argument("--qids", required=True, help="comma-separated question_ids")
    ap.add_argument("--bank-prefix", default="lme_m")
    ap.add_argument("--overlap-threshold", type=int, default=3,
                    help="fire next bank's retain when busy ops <= this")
    ap.add_argument("--state", type=Path, default=Path(".eval_cache/lme_ingest_queue.state.json"))
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="don't skip banks that already have facts")
    ap.add_argument("--bank-suffix", default="",
                    help="appended to bank name (e.g. '_v2a' for variant ingest)")
    ap.add_argument("--retain-mission", default=DEFAULT_RETAIN_MISSION,
                    help="retain_mission for create_bank (variant ingest lever)")
    ap.add_argument("--reflect-mission", default=DEFAULT_REFLECT_MISSION)
    ap.add_argument("--retain-mission-file", type=Path, default=None,
                    help="read retain_mission from file (overrides --retain-mission)")
    args = ap.parse_args()
    qlist = [q.strip() for q in args.qids.split(",") if q.strip()]
    retain_mission = args.retain_mission
    if args.retain_mission_file:
        retain_mission = args.retain_mission_file.read_text(encoding="utf-8").strip()
    raise SystemExit(asyncio.run(main(
        args.tier, qlist, args.bank_prefix, args.overlap_threshold,
        args.state, skip_existing=not args.no_skip_existing,
        bank_suffix=args.bank_suffix,
        retain_mission=retain_mission,
        reflect_mission=args.reflect_mission,
    )))
