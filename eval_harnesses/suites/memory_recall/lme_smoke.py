"""LongMemEval Oracle smoke probe.

Pick N questions, push their haystack sessions into a single Hindsight bank
with tags=[question_id], wait for queue drain, run tag-scoped recall per Q,
report wall-clock and session-id hit rate vs answer_session_ids.

Purpose: calibrate per-question ingest cost + sanity-check tag isolation
before committing to full Oracle ingest.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


_LME_TS_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})\s*\([^)]*\)\s*(\d{2}):(\d{2})")


def _to_iso(ts: Optional[str]) -> Optional[str]:
    """LongMemEval ts format: '2023/05/24 (Wed) 05:42' -> ISO."""
    if not ts:
        return None
    m = _LME_TS_RE.match(ts.strip())
    if not m:
        return None
    y, mo, d, hh, mm = m.groups()
    return f"{y}-{mo}-{d}T{hh}:{mm}:00Z"

from config.global_config import HINDSIGHT_URL
from src.memory.backend.hindsight import (
    HINDSIGHT_API_PREFIX,
    HindsightAPIError,
    HindsightRESTClient,
)

_HF_SNAP = (
    ".eval_cache/hf/datasets--xiaowu0162--longmemeval-cleaned/"
    "snapshots/98d7416c24c778c2fee6e6f3006e7a073259d48f"
)
TIER_FILES = {
    "oracle": Path(_HF_SNAP) / "longmemeval_oracle.json",
    "s": Path(_HF_SNAP) / "longmemeval_s_cleaned.json",
    "m": Path(_HF_SNAP) / "longmemeval_m_cleaned.json",
}
SMOKE_BANK_DEFAULT = "lme_smoke"


def _pick_questions(
    data: List[Dict[str, Any]],
    n: int,
    seed: int,
    qid: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """One per qtype until n, then fill randomly. If qid given, pick only that Q."""
    if qid:
        match = [d for d in data if d["question_id"] == qid]
        if not match:
            raise SystemExit(f"qid {qid} not found")
        return match
    rng = random.Random(seed)
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for d in data:
        by_type.setdefault(d["question_type"], []).append(d)
    picked: List[Dict[str, Any]] = []
    types = list(by_type.keys())
    rng.shuffle(types)
    for t in types:
        if len(picked) >= n:
            break
        picked.append(rng.choice(by_type[t]))
    remaining = [d for d in data if d not in picked]
    rng.shuffle(remaining)
    picked.extend(remaining[: max(0, n - len(picked))])
    return picked[:n]


def _session_to_text(turns: List[Dict[str, Any]]) -> str:
    parts = []
    for t in turns:
        role = t.get("role", "?").upper()
        parts.append(f"{role}: {t.get('content','')}")
    return "\n".join(parts)


async def _wait_drain(client: HindsightRESTClient, bank: str, timeout_s: float = 1800.0) -> float:
    """Poll list_operations until no pending/running. Returns wait seconds."""
    started = time.monotonic()
    while True:
        busy = False
        for status in ("pending", "running"):
            ops = await client._request(
                "GET", f"{HINDSIGHT_API_PREFIX}/banks/{bank}/operations",
                params={"status": status, "limit": 1},
            )
            if (ops or {}).get("total", 0) > 0:
                busy = True
                break
        if not busy:
            return time.monotonic() - started
        if time.monotonic() - started > timeout_s:
            raise TimeoutError(f"queue did not drain within {timeout_s}s")
        await asyncio.sleep(30)


async def main(
    n: int, seed: int, keep: bool, tier: str, bank: str, qid: Optional[str],
    drain_timeout_s: float,
) -> int:
    src = TIER_FILES[tier]
    if not src.exists():
        raise SystemExit(f"missing {tier} JSON at {src}; run hf download first")
    print(f"Loading {tier} from {src}...")
    data = json.loads(src.read_text(encoding="utf-8"))
    picked = _pick_questions(data, n, seed, qid)
    print(f"Picked {len(picked)} questions:")
    for q in picked:
        print(f"  {q['question_id']} [{q['question_type']}] sessions={len(q['haystack_sessions'])} "
              f"ans_sessions={len(q['answer_session_ids'])}")

    client = HindsightRESTClient(HINDSIGHT_URL, timeout=300.0)

    print(f"\n=== reset bank '{bank}' ===")
    try:
        await client.adelete_bank(bank)
    except HindsightAPIError as e:
        if e.status_code != 404:
            raise
    await client.acreate_bank(
        bank,
        retain_mission="Store user-assistant turns verbatim for recall eval.",
        reflect_mission="Surface turns relevant to the user's question.",
    )

    # Build retain items: one per session, content = concat turns.
    items: List[Dict[str, Any]] = []
    total_chars = 0
    for q in picked:
        qid = q["question_id"]
        sess_ids = q["haystack_session_ids"]
        sessions = q["haystack_sessions"]
        dates = q.get("haystack_dates", [None] * len(sessions))
        for sid, turns, ts in zip(sess_ids, sessions, dates):
            content = _session_to_text(turns)
            total_chars += len(content)
            items.append({
                "content": content,
                "tags": [qid, "lme_smoke"],
                "timestamp": _to_iso(ts),
                "document_id": sid,
                "metadata": {"question_id": qid, "session_id": sid},
            })
    print(f"\n=== retain {len(items)} sessions ({total_chars} chars, ~{total_chars//4} toks) ===")

    t0 = time.monotonic()
    # async retain — return immediately, we'll wait on queue drain.
    await client.aretain(bank, items, async_=True)
    retain_post = time.monotonic() - t0
    print(f"POST returned in {retain_post:.1f}s. Waiting for queue drain...")

    drain = await _wait_drain(client, bank, timeout_s=drain_timeout_s)
    total_ingest = time.monotonic() - t0
    print(f"Queue drained after {drain:.1f}s. Total ingest wall: {total_ingest:.1f}s")

    stats = await client._request("GET", f"{HINDSIGHT_API_PREFIX}/banks/{bank}/stats")
    node_counts = stats.get("node_counts", {}) or {}
    fact_count = sum(node_counts.values()) if node_counts else 0
    print(f"Bank node_counts: {node_counts} (total={fact_count})")
    print(f"  total_documents: {stats.get('total_documents')}")
    if fact_count:
        print(f"  per-fact wall: {total_ingest/fact_count:.2f}s ({fact_count*60/total_ingest:.1f} facts/min)")
        print(f"  per-1k-input-char: {total_ingest/total_chars*1000:.2f} s/kchar")
        print(f"  facts-per-1k-input-toks: {fact_count/(total_chars/4)*1000:.1f}")

    print("\n=== tag-scoped recall per question ===")
    hits_by_q = {}
    for q in picked:
        qid = q["question_id"]
        ans = set(q["answer_session_ids"])
        try:
            res = await client.arecall(bank, q["question"], tags=[qid], max_tokens=4096)
        except Exception as e:
            print(f"  {qid}: ERROR {type(e).__name__}: {e}")
            continue
        retrieved_sess = []
        for h in res or []:
            sid = (h.get("source") or {}).get("document_id") or (h.get("metadata") or {}).get("session_id")
            if sid:
                retrieved_sess.append(sid)
        retrieved_unique = list(dict.fromkeys(retrieved_sess))  # preserves order, dedup
        hit_set = set(retrieved_unique) & ans
        recall = len(hit_set) / max(len(ans), 1)
        hits_by_q[qid] = {
            "n_facts_returned": len(res or []),
            "expected_sessions": sorted(ans),
            "retrieved_unique_sessions": retrieved_unique,
            "hit_any": bool(hit_set),
            "recall": recall,
        }
        flag = "OK " if hit_set else "MISS"
        print(f"  {flag} {qid} ans={sorted(ans)} unique_retrieved={retrieved_unique} "
              f"recall={recall:.2f} ({len(res or [])} facts)")

    hit_rate = sum(1 for v in hits_by_q.values() if v["hit_any"]) / max(len(hits_by_q), 1)
    print(f"\n=== SUMMARY ===")
    print(f"Questions: {len(picked)}")
    print(f"Ingest wall: {total_ingest:.1f}s ({total_ingest/len(picked):.1f}s/Q)")
    print(f"Hit-any rate: {hit_rate:.0%}")
    print(f"Projected full Oracle (500 Qs): {total_ingest * 500 / len(picked) / 60:.1f} min")

    if not keep:
        print(f"\nDeleting bank '{bank}'...")
        await client.adelete_bank(bank)
    else:
        print(f"\nKeeping bank '{bank}' (--keep)")

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tier", choices=list(TIER_FILES.keys()), default="oracle")
    ap.add_argument("--bank", default=SMOKE_BANK_DEFAULT)
    ap.add_argument("--qid", default=None, help="pick a specific question_id (overrides --n)")
    ap.add_argument("--drain-timeout-s", type=float, default=1800.0)
    ap.add_argument("--keep", action="store_true", help="don't delete the smoke bank on exit")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(
        args.n, args.seed, args.keep, args.tier, args.bank, args.qid, args.drain_timeout_s,
    )))
