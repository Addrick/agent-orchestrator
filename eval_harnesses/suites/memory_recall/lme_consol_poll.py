"""Poll v3a banks until consolidation drains.

Exits 0 once every listed bank has pending_consolidation == 0 and
fact_count > 0. Prints a status line per poll so progress is visible
in the tail of the background log.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any, Dict, List

from config.global_config import HINDSIGHT_URL
from src.memory.backend.hindsight import HindsightAPIError, HindsightRESTClient


HINDSIGHT_API_PREFIX = "/v1/default"


async def _stats(client: HindsightRESTClient, bank: str) -> Dict[str, Any]:
    try:
        return await client._request(
            "GET", f"{HINDSIGHT_API_PREFIX}/banks/{bank}/stats"
        )
    except HindsightAPIError as e:
        return {"_error": f"{e.status_code} {e}"}


async def main(banks: List[str], interval_s: float) -> int:
    client = HindsightRESTClient(HINDSIGHT_URL, timeout=60.0)
    t0 = time.monotonic()
    last_facts: Dict[str, int] = {b: -1 for b in banks}
    stuck_iters: Dict[str, int] = {b: 0 for b in banks}
    while True:
        ts = time.monotonic() - t0
        all_done = True
        line_parts: List[str] = []
        for bank in banks:
            s = await _stats(client, bank)
            if "_error" in s:
                line_parts.append(f"{bank}: ERR {s['_error']}")
                all_done = False
                continue
            nc = s.get("node_counts", {}) or {}
            facts = sum(nc.values()) if nc else s.get("total_nodes", 0)
            pending = s.get("pending_consolidation", 0)
            failed = s.get("failed_consolidation", 0)
            # stuck detection
            if facts == last_facts[bank] and pending > 0:
                stuck_iters[bank] += 1
            else:
                stuck_iters[bank] = 0
            last_facts[bank] = facts
            stuck_marker = f" STUCK@{stuck_iters[bank]}" if stuck_iters[bank] >= 3 else ""
            line_parts.append(
                f"{bank.split('_v3a')[0]}: facts={facts} pend={pending}"
                f" fail={failed}{stuck_marker}"
            )
            if pending > 0 or facts == 0:
                all_done = False
        print(f"[t+{ts/60:.1f}m] " + " | ".join(line_parts), flush=True)
        if all_done:
            print(f"\nAll banks drained at t+{ts/60:.1f}m", flush=True)
            return 0
        await asyncio.sleep(interval_s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", required=True,
                    help="comma-separated bank IDs to poll")
    ap.add_argument("--interval", type=float, default=300.0,
                    help="poll interval in seconds (default 300 = 5min)")
    args = ap.parse_args()
    bank_list = [b.strip() for b in args.banks.split(",") if b.strip()]
    raise SystemExit(asyncio.run(main(bank_list, args.interval)))
