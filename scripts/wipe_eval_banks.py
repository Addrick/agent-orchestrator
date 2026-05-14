"""List + delete Hindsight banks created by eval harnesses.

Targets banks whose `bank_id` matches one of the eval prefixes (default:
"eval_"). Dry-run by default; pass --apply to actually delete.

Usage:
    python -m scripts.wipe_eval_banks                  # dry-run
    python -m scripts.wipe_eval_banks --apply          # delete eval_* banks
    python -m scripts.wipe_eval_banks --prefix eval_backfill_ --apply
    python -m scripts.wipe_eval_banks --pattern '^eval_(backfill|ambient)_'
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.global_config import HINDSIGHT_URL
from src.memory.backend.hindsight import HindsightRESTClient


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="eval_", help="bank_id prefix to match")
    p.add_argument("--pattern", default=None, help="regex matched against bank_id (overrides --prefix)")
    p.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    p.add_argument("--url", default=HINDSIGHT_URL, help="Hindsight base URL")
    args = p.parse_args()

    matcher = re.compile(args.pattern) if args.pattern else None

    client = HindsightRESTClient(args.url, timeout=30.0)
    try:
        # No list method on the project client; hit the REST endpoint directly.
        resp = await client._request("GET", "/v1/default/banks")
        banks = resp.get("banks", []) if isinstance(resp, dict) else []
        targets = []
        for b in banks:
            bid = b.get("bank_id") or b.get("name") or ""
            if matcher:
                if matcher.search(bid):
                    targets.append(bid)
            elif bid.startswith(args.prefix):
                targets.append(bid)

        print(f"Matched {len(targets)} bank(s) (of {len(banks)} total)")
        for bid in targets:
            print(f"  - {bid}")

        if not targets:
            return 0
        if not args.apply:
            print("\nDRY-RUN. Pass --apply to delete.")
            return 0

        deleted = 0
        for bid in targets:
            try:
                await client.adelete_bank(bid)
                deleted += 1
                print(f"  deleted: {bid}")
            except Exception as e:
                print(f"  FAILED {bid}: {type(e).__name__}: {e}")
        print(f"\nDone. Deleted {deleted}/{len(targets)}.")
    finally:
        await client.client.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
