"""Idempotent, checksum-gated Hindsight seeder for the test_persona bank.

Hindsight bank rebuilds are slow (LLM extraction). This script skips the
seed step entirely when nothing has changed.

Gate inputs (hashed together to form the freshness key):
    - test_persona_seed.json content
    - --bank name
    - extraction params (timestamps, tags) embedded in this script

State file: fixtures/.test_persona_seeded.json
    {"bank": "...", "checksum": "...", "items": N, "seeded_at": "..."}

Behavior:
    - If state.checksum matches and --force is not set: print "fresh, skipping" and exit 0
    - Otherwise: delete-then-recreate the bank, retain all seed items
      with a deterministic timestamp per item (so `mentioned_at` populates),
      then write a new state file.

Usage:
    python -m eval_harnesses.suites.memory_recall.seed_hindsight
    python -m eval_harnesses.suites.memory_recall.seed_hindsight --bank test_persona --force
    python -m eval_harnesses.suites.memory_recall.seed_hindsight --check
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from config.global_config import HINDSIGHT_URL
from src.memory.backend.hindsight import HindsightAPIError, HindsightRESTClient

_HERE = Path(__file__).parent
_SEED_FILE = _HERE / "fixtures" / "test_persona_seed.json"
_STATE_FILE = _HERE / "fixtures" / ".test_persona_seeded.json"

# Bumped whenever this script changes its retain payload shape so cached
# seedings get invalidated even if the seed JSON is byte-identical.
_SEEDER_VERSION = "v1"

# Anchor + per-item offset → deterministic distinct timestamps. Hindsight's
# mentioned_at populates from item.timestamp (per the project memory note).
_ANCHOR = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _checksum(bank: str, seed: Dict[str, Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    h.update(_SEEDER_VERSION.encode())
    h.update(b"\x00")
    h.update(bank.encode())
    h.update(b"\x00")
    h.update(json.dumps(seed, sort_keys=True, ensure_ascii=True).encode())
    return h.hexdigest()


def _read_state() -> Dict[str, Any]:
    if not _STATE_FILE.exists():
        return {}
    try:
        return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_state(state: Dict[str, Any]) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def _build_items(seed: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for i, (seed_key, entry) in enumerate(sorted(seed.items())):
        ts = (_ANCHOR + timedelta(hours=i)).isoformat()
        items.append({
            "content": entry["text"],
            "tags": list(entry.get("tags", [])),
            "timestamp": ts,
            "document_id": f"test_persona/{seed_key}",
            "metadata": {"seed_key": seed_key, "seeder_version": _SEEDER_VERSION},
        })
    return items


async def _seed(bank: str, seed: Dict[str, Dict[str, Any]], *, force: bool) -> int:
    checksum = _checksum(bank, seed)
    state = _read_state()
    if not force and state.get("bank") == bank and state.get("checksum") == checksum:
        print(
            f"test_persona bank seeded fresh (checksum={checksum[:12]}, "
            f"items={state.get('items')}, at={state.get('seeded_at')}). "
            f"Skipping. Use --force to override.",
            file=sys.stderr,
        )
        return 0

    client = HindsightRESTClient(HINDSIGHT_URL, timeout=60.0)
    items = _build_items(seed)

    print(f"Reseeding bank '{bank}' with {len(items)} items...", file=sys.stderr)
    try:
        await client.adelete_bank(bank)
    except HindsightAPIError as e:
        if e.status_code != 404:
            raise
    await client.acreate_bank(
        bank,
        retain_mission="Store persona facts verbatim for recall evaluation.",
        reflect_mission="Surface persona facts relevant to the query.",
    )
    # Sync retain so we know items are processed before scenarios run.
    # arecall doesn't return them until extraction completes.
    await client.aretain(bank, items, async_=False)

    state = {
        "bank": bank,
        "checksum": checksum,
        "items": len(items),
        "seeded_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_state(state)
    print(f"Done. checksum={checksum[:12]}", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="seed_hindsight", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", default="test_persona")
    p.add_argument("--seed-file", type=Path, default=_SEED_FILE)
    p.add_argument("--force", action="store_true",
                   help="reseed even if checksum unchanged")
    p.add_argument("--check", action="store_true",
                   help="print current state and exit (no network)")
    args = p.parse_args(argv)

    seed = json.loads(args.seed_file.read_text(encoding="utf-8"))

    if args.check:
        state = _read_state()
        expected = _checksum(args.bank, seed)
        print(json.dumps({
            "bank": args.bank,
            "current_seed_checksum": expected,
            "recorded_state": state,
            "fresh": (state.get("bank") == args.bank
                      and state.get("checksum") == expected),
        }, indent=2))
        return 0

    return asyncio.run(_seed(args.bank, seed, force=args.force))


if __name__ == "__main__":
    raise SystemExit(main())
