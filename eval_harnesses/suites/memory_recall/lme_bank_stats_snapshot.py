"""CC-2: capture bank stats + config for every LME bank.

Pulls `/banks/{id}/stats` and `/banks/{id}/config` for the full LME bank
roster (baselines + variants) and writes one merged JSON snapshot. The
config endpoint includes `overrides`, which lets us see exactly what was
explicitly set per bank vs. server-default.

Usage:
    python -m eval_harnesses.suites.memory_recall.lme_bank_stats_snapshot \\
        --out .eval_cache/lme_results/bank_stats_2026-05-20.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from config.global_config import HINDSIGHT_URL
from src.memory.backend.hindsight import (
    HINDSIGHT_API_PREFIX,
    HindsightAPIError,
    HindsightRESTClient,
)


# 14 LME banks per docs/eval_results/lme.md + 3 v3a verbose variants
S_BASELINE = [
    "lme_s_1c549ce4", "lme_s_1c0ddc50", "lme_s_a3045048",
    "lme_s_cc539528", "lme_s_50635ada",
]
S_V2A = ["lme_s_1c0ddc50_v2a"]
S_V3A = ["lme_s_1c0ddc50_v3a", "lme_s_1c549ce4_v3a"]
M_BASELINE = [
    "lme_m_1c549ce4", "lme_m_91b15a6e", "lme_m_8fb83627", "lme_m_c9f37c46",
    "lme_m_gpt4_61e13b3c", "lme_m_gpt4_68e94287", "lme_m_6aeb4375_abs",
]
M_V2A = ["lme_m_91b15a6e_v2a"]
M_V3A = ["lme_m_91b15a6e_v3a"]

ALL_BANKS = {
    "s_baseline": S_BASELINE,
    "s_v2a": S_V2A,
    "s_v3a_verbose": S_V3A,
    "m_baseline": M_BASELINE,
    "m_v2a": M_V2A,
    "m_v3a_verbose": M_V3A,
}


async def _snapshot_bank(client: HindsightRESTClient, bank: str) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"bank_id": bank}
    try:
        stats = await client._request(
            "GET", f"{HINDSIGHT_API_PREFIX}/banks/{bank}/stats"
        )
        rec["stats"] = stats
    except HindsightAPIError as e:
        rec["stats_error"] = {"status": e.status_code, "body": str(e)}
    try:
        cfg = await client._request(
            "GET", f"{HINDSIGHT_API_PREFIX}/banks/{bank}/config"
        )
        rec["config"] = cfg.get("config")
        rec["overrides"] = cfg.get("overrides")
    except HindsightAPIError as e:
        rec["config_error"] = {"status": e.status_code, "body": str(e)}
    return rec


async def main(out_path: Path) -> int:
    client = HindsightRESTClient(HINDSIGHT_URL, timeout=60.0)
    snapshot: Dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hindsight_url": HINDSIGHT_URL,
        "groups": {},
    }
    for group, banks in ALL_BANKS.items():
        snapshot["groups"][group] = []
        for bank in banks:
            rec = await _snapshot_bank(client, bank)
            snapshot["groups"][group].append(rec)
            cfg = rec.get("config") or {}
            stats = rec.get("stats") or {}
            nc = stats.get("node_counts", {}) or {}
            facts = sum(nc.values()) if nc else stats.get("total_nodes", 0)
            print(
                f"  {bank}: facts={facts} "
                f"mode={cfg.get('retain_extraction_mode')} "
                f"chunk={cfg.get('retain_chunk_size')} "
                f"entities_free={cfg.get('entities_allow_free_form')} "
                f"labels={cfg.get('entity_labels')}",
                flush=True,
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", type=Path,
        default=Path(".eval_cache/lme_results/bank_stats_2026-05-20.json"),
    )
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.out)))
