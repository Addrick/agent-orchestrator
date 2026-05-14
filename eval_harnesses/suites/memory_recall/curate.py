"""Interactive curation helper for memory_recall fact locators.

Given a bank + one or more candidate locator strings, prints every memory
in the bank whose text contains any locator (substring match), so a human
can iterate on locator/max_matches without editing fixtures blindly.

Usage:
    python -m eval_harnesses.suites.memory_recall.curate \\
        --bank claudecode \\
        --locator "Permission denied (publickey)" \\
        [--locator "publickey authentication"] \\
        [--max-tokens 800]

    # Or feed a JSON fact entry directly:
    python -m eval_harnesses.suites.memory_recall.curate \\
        --bank test_persona --fact-json '{"source":"seed","seed_key":"adam_city"}'

Exit code 0 if any match; 1 if zero matches; 2 if over max_matches default.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from config.global_config import HINDSIGHT_URL
from src.memory.backend.hindsight import HindsightRESTClient

from .resolver import DEFAULT_MAX_MATCHES, load_seed_data, resolve_facts

_HERE = Path(__file__).parent
_SEED_FILE = _HERE / "fixtures" / "test_persona_seed.json"


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


async def _curate(args) -> int:
    client = HindsightRESTClient(HINDSIGHT_URL)
    seed_data = load_seed_data(_SEED_FILE)

    if args.fact_json:
        fact = json.loads(args.fact_json)
        fact.setdefault("key", "_curate")
        fact["max_matches"] = args.max_matches
    else:
        if not args.locator:
            print("ERROR: provide --locator (one or more) or --fact-json", file=sys.stderr)
            return 2
        fact = {
            "key": "_curate",
            "source": "history",
            "locators": list(args.locator),
            "max_matches": args.max_matches,
        }

    print(f"Bank: {args.bank}")
    print(f"Fact: {json.dumps(fact, indent=2)}\n")

    result = await resolve_facts(
        args.bank, [fact], rest_client=client,
        seed_data=seed_data, recall_max_tokens=args.max_tokens,
    )

    matches: List[Dict[str, Any]] = result.matches.get("_curate", []) or []
    print(f"Found {len(matches)} match(es):")
    for i, m in enumerate(matches, 1):
        print(f"  [{i}] id={m['id']}")
        print(f"      locator: {m['locator']!r}")
        print(f"      text:    {_truncate(m['text'], args.text_chars)}")

    resolved = result.resolved.get("_curate")
    if resolved:
        print(f"\nRESOLVED: {len(resolved)} id(s) within max_matches={fact['max_matches']}")
        return 0
    diag = result.diagnostics.get("_curate", "")
    print(f"\nUNRESOLVED: {diag}")
    if "exceeds max_matches" in diag:
        return 2
    return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="curate",
        description="Probe a Hindsight bank for memories matching candidate locators.",
    )
    p.add_argument("--bank", required=True, help="bank name (e.g. test_persona, claudecode)")
    p.add_argument("--locator", action="append", default=[],
                   help="candidate locator substring; repeat for multiple")
    p.add_argument("--fact-json", help="full fact entry as JSON (overrides --locator)")
    p.add_argument("--max-matches", type=int, default=DEFAULT_MAX_MATCHES,
                   help=f"resolution cap (default {DEFAULT_MAX_MATCHES})")
    p.add_argument("--max-tokens", type=int, default=800,
                   help="max_tokens budget per arecall (default 800)")
    p.add_argument("--text-chars", type=int, default=180,
                   help="truncate displayed memory text to N chars (default 180)")
    args = p.parse_args(argv)
    return asyncio.run(_curate(args))


if __name__ == "__main__":
    sys.exit(main())
