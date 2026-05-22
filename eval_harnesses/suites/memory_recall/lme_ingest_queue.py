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
from datetime import datetime, timedelta
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


def _fmt_wallclock(seconds_from_now: float) -> str:
    """Absolute local finish time, e.g. '2026-05-23 02:14'."""
    if seconds_from_now <= 0 or seconds_from_now != seconds_from_now:  # NaN guard
        return "?"
    return (datetime.now() + timedelta(seconds=seconds_from_now)).strftime("%Y-%m-%d %H:%M")


def _load_history(path: Optional[Path]) -> List[Dict[str, Any]]:
    """Load cross-run JSONL history of completed bank ingests."""
    if not path or not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _append_history(path: Optional[Path], entry: Dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _history_chars_per_sec(
    history: List[Dict[str, Any]],
    hindsight_url: str,
    model: Optional[str] = None,
) -> Optional[float]:
    """Average chars/sec from prior runs with matching backend (+model if given).

    Filters strictly so a 4090-local-llama history doesn't poison a Granite
    estimate. Returns None if no matching samples.
    """
    samples = []
    for h in history:
        if h.get("hindsight_url") != hindsight_url:
            continue
        if model is not None and h.get("model") != model:
            continue
        secs = h.get("seconds", 0)
        chars = h.get("chars", 0)
        if secs > 0 and chars > 0:
            samples.append(chars / secs)
    if not samples:
        return None
    return sum(samples) / len(samples)


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
    # Always call acreate_bank (it now handles 409 via PATCH internally).
    # Skipping when the bank already exists is what silently dropped the
    # v2a mission on 2026-05-17: an earlier failed attempt created the
    # bank with no mission, and subsequent runs took the early-return.
    existed = await _bank_exists(client, bank)
    await client.acreate_bank(
        bank,
        retain_mission=retain_mission,
        reflect_mission=reflect_mission,
    )
    action = "patched" if existed else "created"
    print(f"  [{qid}] {action} bank '{bank}'", flush=True)
    # Smoke verify: read back config and check mission landed. Mission
    # only affects extraction, not recall — so a silent miss now means
    # the whole ingest is wasted hours.
    cfg = await client._request(
        "GET", f"{HINDSIGHT_API_PREFIX}/banks/{bank}/config",
    )
    actual_mission = (cfg.get("config") or {}).get("retain_mission")
    if actual_mission != retain_mission:
        raise RuntimeError(
            f"[{qid}] retain_mission did not persist on {bank!r}. "
            f"Expected {retain_mission[:60]!r}..., "
            f"got {(actual_mission or '<null>')[:60]!r}. "
            f"Aborting ingest before wasting LLM cost."
        )
    print(f"  [{qid}] mission verified on {bank}", flush=True)
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


def _eta_remaining_seconds(
    state: Dict[str, Any], queued_left: int, current_elapsed: float,
    current_remaining: Optional[float] = None,
    history_avg_seconds: Optional[float] = None,
) -> tuple[Optional[float], bool]:
    """Estimate seconds remaining for the whole run. Returns (seconds, bootstrap).

    Priority: in-run completed bank durations > cross-run history avg >
    project from current bank's own remaining. `bootstrap=True` flags low-
    confidence estimates (rendered with a trailing '~').
    """
    durations = list(state.get("per_bank_seconds", {}).values())
    if durations:
        avg = sum(durations) / len(durations)
        cur_remaining = max(avg - current_elapsed, 0.0)
        return cur_remaining + queued_left * avg, False
    if history_avg_seconds is not None:
        cur_remaining = max(history_avg_seconds - current_elapsed, 0.0)
        return cur_remaining + queued_left * history_avg_seconds, True
    if current_remaining is None:
        return None, True
    est_avg = current_elapsed + current_remaining
    return current_remaining + queued_left * est_avg, True


def _fmt_eta(seconds: Optional[float], bootstrap: bool) -> str:
    if seconds is None:
        return "unknown"
    suffix = "~" if bootstrap else ""
    return f"{_fmt_dur(seconds)}{suffix} (eta {_fmt_wallclock(seconds)})"


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
    history_avg_seconds: Optional[float] = None,
    history_path: Optional[Path] = None,
    hindsight_url: str = "",
    llm_model: str = "",
    already_fired: Optional[set] = None,
) -> float:
    """Drive one bank to completion; fire next when close to drain. Returns seconds.

    `already_fired` (shared across iterations) tracks banks whose retain was
    POSTed during the previous bank's overlap window, so we don't double-fire.
    """
    print(f"\n=== qid={qid} [{q['question_type']}] bank={bank} ===", flush=True)
    sizing: Optional[Dict[str, Any]] = None
    # Resume-case detection: if the bank already has retain ops queued
    # (from a prior crashed/restarted run), don't re-POST aretain — it'd
    # duplicate the entire session set.
    remote_busy = 0
    if await _bank_exists(client, bank):
        ops_pre = await _ops_counts(client, bank)
        remote_busy = max(ops_pre.get("pending", 0), 0) + max(ops_pre.get("running", 0), 0)
    skip_post = (already_fired is not None and bank in already_fired) or remote_busy > 0
    if skip_post:
        items = _build_items(q)
        sizing = {"sessions": len(items),
                  "chars": sum(len(i["content"]) for i in items)}
        reason = ("overlap-fired earlier" if (already_fired and bank in already_fired)
                  else f"bank has {remote_busy} ops already queued")
        print(f"  [{qid}] skipping aretain ({reason}); resuming poll", flush=True)
    else:
        # Retry retain on transient 5xx (observed: dupe-500 during overlap window)
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
        # HS schema: prefer total_nodes (current); fall back to legacy node_counts.
        facts = stats.get("total_nodes")
        if facts is None:
            nc = stats.get("nodes_by_fact_type") or stats.get("node_counts") or {}
            facts = sum(nc.values()) if nc else 0
        docs_done = stats.get("total_documents", 0)

        # progress line — print every ~minute or on completion delta
        if elapsed - last_print >= 60 or completed != last_completed or busy == 0:
            eta_self_s = pending / max(rate, 0.01) * 60 if pending else 0.0
            eta_total_s, bootstrap = _eta_remaining_seconds(
                state, queued_left, elapsed, eta_self_s,
                history_avg_seconds=history_avg_seconds,
            )
            print(
                f"  [{qid}] ops: {completed} done / {busy} busy ({pending}p+{running}r) "
                f"| docs: {docs_done} | facts: {facts} | elapsed {_fmt_dur(elapsed)} "
                f"| rate {rate:.1f}/m | eta bank {_fmt_dur(eta_self_s)} "
                f"| eta total {_fmt_eta(eta_total_s, bootstrap)}",
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
                if already_fired is not None:
                    already_fired.add(nb)
            except Exception as e:
                print(f"  [WARN] failed to fire next {nq['question_id']}: {e}",
                      file=sys.stderr, flush=True)
            next_fired = True

        # Drain condition: ops queue empty AND at least one document extracted
        # (or completed status incremented, for legacy HS). `completed > 0`
        # alone was broken on current HS where batch_retain never enters the
        # 'completed' bucket — relying on docs_done from bank stats instead.
        if busy == 0 and (completed > 0 or docs_done > 0):
            elapsed_final = time.monotonic() - t0
            state["per_bank_seconds"][bank] = elapsed_final
            await asyncio.sleep(5)
            stats = await _bank_stats(client, bank)
            facts = stats.get("total_nodes")
            if facts is None:
                nc = stats.get("nodes_by_fact_type") or stats.get("node_counts") or {}
                facts = sum(nc.values()) if nc else 0
            docs_done = stats.get("total_documents", 0)
            pending_c = stats.get("pending_consolidation", 0)
            print(
                f"  [{qid}] RETAIN DRAINED in {_fmt_dur(elapsed_final)} | "
                f"docs: {docs_done} facts: {facts} (consolidating: {pending_c}) | "
                f"moving on; consolidation continues in background",
                flush=True,
            )
            state.setdefault("completed", []).append(qid)
            _save_state(state_path, state)
            _append_history(history_path, {
                "qid": qid, "qtype": q.get("question_type"),
                "bank": bank, "sessions": sizing["sessions"],
                "chars": sizing["chars"], "seconds": elapsed_final,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "hindsight_url": hindsight_url, "model": llm_model,
                "retain_mission": retain_mission[:200],
            })
            return elapsed_final

        await asyncio.sleep(POLL_INTERVAL_S)


async def main(
    tier: str, qids: List[str], bank_prefix: str,
    overlap_threshold: int, state_path: Optional[Path],
    skip_existing: bool,
    bank_suffix: str = "",
    retain_mission: str = DEFAULT_RETAIN_MISSION,
    reflect_mission: str = DEFAULT_REFLECT_MISSION,
    hindsight_url: Optional[str] = None,
    history_path: Optional[Path] = None,
    llm_model: str = "",
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

    url = hindsight_url or HINDSIGHT_URL
    print(f"hindsight: {url}  | model tag: {llm_model or '<unset>'}", flush=True)
    client = HindsightRESTClient(url, timeout=300.0)

    history = _load_history(history_path)
    history_avg_chars_per_sec = _history_chars_per_sec(
        history, url, model=llm_model or None,
    )
    if history_avg_chars_per_sec:
        print(f"history: {len(history)} entries total; "
              f"avg {history_avg_chars_per_sec:.0f} chars/sec for "
              f"{url} + {llm_model or '<any model>'}", flush=True)
    elif history:
        print(f"history: {len(history)} entries total; "
              f"no matching samples for {url} + {llm_model or '<any model>'}",
              flush=True)
    else:
        print(f"history: empty (file: {history_path})", flush=True)

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
    total_chars = 0
    for i, item in enumerate(queue):
        ic = sum(len(_session_to_text(t)) for t in item["q"]["haystack_sessions"])
        total_chars += ic
        print(f"  {i+1:>2}. {item['qid']} [{item['q']['question_type']}] -> {item['bank']} "
              f"(~{ic/1e6:.1f}M chars)", flush=True)
    if not queue:
        print("nothing to do.", flush=True)
        return 0

    # Up-front estimate from history (only when we have prior data)
    history_avg_seconds: Optional[float] = None
    if history_avg_chars_per_sec:
        est_total_s = total_chars / history_avg_chars_per_sec
        history_avg_seconds = est_total_s / len(queue)
        print(f"up-front estimate: ~{_fmt_dur(est_total_s)} "
              f"(finishes ~{_fmt_wallclock(est_total_s)}) "
              f"based on history avg {history_avg_chars_per_sec:.0f} chars/sec",
              flush=True)

    run_t0 = time.monotonic()
    overlap_fired: set = set()
    for i, item in enumerate(queue):
        nxt = queue[i + 1] if i + 1 < len(queue) else None
        await _ingest_one(
            client=client, qid=item["qid"], q=item["q"], bank=item["bank"],
            queue_next=nxt, overlap_threshold=overlap_threshold,
            state=state, state_path=state_path,
            queued_left=len(queue) - i - 1,
            retain_mission=retain_mission,
            reflect_mission=reflect_mission,
            history_avg_seconds=history_avg_seconds,
            history_path=history_path,
            hindsight_url=url,
            llm_model=llm_model,
            already_fired=overlap_fired,
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
    ap.add_argument("--hindsight-url", default=None,
                    help="override HINDSIGHT_URL (e.g. http://127.0.0.1:8888 for granite stack)")
    ap.add_argument("--history", type=Path,
                    default=Path(".eval_cache/lme_ingest_history.jsonl"),
                    help="append-only JSONL of completed bank ingests across runs")
    ap.add_argument("--llm-model", default="",
                    help="tag stored in history for matching prior runs (e.g. 'granite', "
                         "'us.anthropic.claude-sonnet-4-6'). Does NOT change what Hindsight uses.")
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
        hindsight_url=args.hindsight_url,
        history_path=args.history,
        llm_model=args.llm_model,
    )))
