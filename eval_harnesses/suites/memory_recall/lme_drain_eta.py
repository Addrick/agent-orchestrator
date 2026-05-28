"""Measure consolidation drain rate and project a finish ETA.

Polls each bank's `pending_consolidation` on a hindsight server, computes
the drain rate (facts/min) from the first sample onward, and projects when
each bank — and the whole set — reaches zero pending.

Standalone (urllib only, no project imports) so it can point at any server.
Defaults target the granite/testing hindsight server at 10.0.0.70:8890.

Usage:
    # auto-discover banks, poll every 60s until drained
    python -m eval_harnesses.suites.memory_recall.lme_drain_eta

    # one-shot rate estimate from two quick samples, then exit
    python eval_harnesses/suites/memory_recall/lme_drain_eta.py \\
        --interval 30 --max-polls 2

    # explicit server + banks
    python .../lme_drain_eta.py --url http://10.0.0.70:8890 \\
        --banks lme_m_8fb83627_granite,lme_m_1c549ce4_granite
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

DEFAULT_URL = "http://10.0.0.70:8890"
API_PREFIX = "/v1/default"


def _get(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def discover_banks(base: str) -> List[str]:
    data = _get(f"{base}{API_PREFIX}/banks")
    banks = data.get("banks", data) if isinstance(data, dict) else data
    return [b["bank_id"] for b in banks]


def pending(base: str, bank: str) -> Optional[int]:
    try:
        s = _get(f"{base}{API_PREFIX}/banks/{bank}/stats")
        return int(s.get("pending_consolidation", 0))
    except Exception as e:  # network/parse — report, keep polling
        print(f"  ! {bank}: {e}", flush=True)
        return None


def _fmt_eta(minutes: float) -> str:
    if minutes <= 0:
        return "done"
    # project in UTC, then convert to the local system timezone for display
    finish = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).astimezone()
    if minutes < 90:
        span = f"{minutes:.0f}m"
    else:
        span = f"{minutes / 60:.1f}h"
    tz = finish.strftime("%Z") or "local"
    return f"{span} (~{finish.strftime('%H:%M')} {tz})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL, help="hindsight base URL")
    ap.add_argument("--banks", default="",
                    help="comma-separated bank IDs (default: auto-discover all)")
    ap.add_argument("--interval", type=float, default=60.0,
                    help="poll interval seconds (default 60)")
    ap.add_argument("--max-polls", type=int, default=0,
                    help="stop after N polls (0 = run until all drained)")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    banks = ([b.strip() for b in args.banks.split(",") if b.strip()]
             or discover_banks(base))
    print(f"server={base} banks={banks}\n", flush=True)

    # baseline anchor per bank: (t0_seconds, pending0)
    t0 = time.monotonic()
    base_pending: Dict[str, int] = {}
    last_pending: Dict[str, int] = {}
    polls = 0

    while True:
        elapsed_min = (time.monotonic() - t0) / 60.0
        polls += 1
        lines: List[str] = []
        etas: List[float] = []
        all_done = True

        for bank in banks:
            p = pending(base, bank)
            if p is None:
                all_done = False
                continue
            base_pending.setdefault(bank, p)
            drained = base_pending[bank] - p
            rate = drained / elapsed_min if elapsed_min > 0 else 0.0  # facts/min
            inst = ""
            if bank in last_pending:
                d = last_pending[bank] - p
                inst = f" inst={d / (args.interval / 60.0):+.0f}/m"
            last_pending[bank] = p

            short = bank.replace("lme_m_", "").replace("_granite", "")
            if p > 0:
                all_done = False
                if rate > 0:
                    eta_min = p / rate
                    etas.append(eta_min)
                    lines.append(f"{short}: pend={p} rate={rate:.0f}/m"
                                 f"{inst} eta={_fmt_eta(eta_min)}")
                else:
                    lines.append(f"{short}: pend={p} rate=0/m{inst} "
                                 f"eta=unknown")
            else:
                lines.append(f"{short}: drained")

        print(f"[t+{elapsed_min:.1f}m] " + " | ".join(lines), flush=True)

        if all_done:
            print(f"\nAll banks drained at t+{elapsed_min:.1f}m", flush=True)
            return 0
        if etas:
            print(f"           => slowest bank ETA {_fmt_eta(max(etas))}",
                  flush=True)
        if args.max_polls and polls >= args.max_polls:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
