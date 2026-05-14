from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List

from eval_harnesses.framework.fixtures import FixtureBundle
from eval_harnesses.framework.results import RunOutput
from eval_harnesses.framework.scenarios import Scenario
from eval_harnesses.framework.variants import MemoryVariant, PromptVariant

logger = logging.getLogger(__name__)


def _format_turns(turns: List[Dict[str, Any]], fmt: str) -> str:
    """Render a list of {speaker, text, ts?} dicts into the chosen layout.

    Layouts:
        date_speaker_header — backfill_hindsight.py format (`Date: ...\nSpeaker: ...\n---\n...`)
        inline              — `<Speaker>: <text>` per line
        bracketed           — `[Speaker] <text>` per line
        json_block          — JSON list per turn (one block)
    """
    if fmt == "date_speaker_header":
        parts = []
        for t in turns:
            ts = t.get("ts", "")
            header_ts = f"Date: {ts}\n" if ts else ""
            parts.append(f"{header_ts}Speaker: {t['speaker']}\n---\n{t['text']}")
        return "\n\n".join(parts)
    if fmt == "inline":
        return "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)
    if fmt == "bracketed":
        return "\n".join(f"[{t['speaker']}] {t['text']}" for t in turns)
    if fmt == "json_block":
        return json.dumps([{"speaker": t["speaker"], "text": t["text"]} for t in turns], indent=2)
    raise ValueError(f"unknown content_format: {fmt}")


async def ambient_attribution_driver(
    bundle: FixtureBundle,
    scenario: Scenario,
    mem_var: MemoryVariant,
    prompt_var: PromptVariant,
) -> RunOutput:
    """Provision an ambient-style bank, post a transcript, recall, return.

    Scenario shape (in scenario.expectations):
        turns: [{"speaker": "Alice", "text": "...", "ts": "2026-04-01T10:00:00Z"}],
        speakers: ["Alice", "Bob"],          # known speakers
        facts: [                              # what attribution must surface
            {"speaker": "Alice", "must_contain": ["wagyu", "discount"]},
            ...
        ]
    """
    from src.memory.backend.hindsight import HindsightRESTClient
    from config.global_config import HINDSIGHT_URL

    out = RunOutput()
    started = datetime.utcnow()

    fmt = mem_var.extra.get("content_format", "date_speaker_header")
    retain_mission = mem_var.extra.get("retain_mission")
    enable_observations = bool(mem_var.extra.get("enable_observations", True))

    suffix = uuid.uuid4().hex[:8]
    bank_id = f"eval_ambient_{scenario.id}_{mem_var.id}_{suffix}".lower()
    bank_id = re.sub(r"[^a-z0-9_]", "_", bank_id)[:63]

    timeout = float(mem_var.extra.get("http_timeout", 300.0))
    client = HindsightRESTClient(HINDSIGHT_URL, timeout=timeout)
    try:
        await client.acreate_bank(
            bank_id=bank_id,
            retain_mission=retain_mission,
            enable_observations=enable_observations,
        )
        turns = scenario.expectations.get("turns", [])
        content = _format_turns(turns, fmt)
        last_ts = next((t["ts"] for t in reversed(turns) if t.get("ts")), None)
        item: Dict[str, Any] = {
            "content": content,
            "document_id": f"transcript_{suffix}",
            "tags": ["channel:eval_ambient"],
            "update_mode": "replace",
            "strategy": "documents",
            "timestamp": last_ts or "unset",
        }
        retain_resp = await client.aretain(bank_id=bank_id, items=[item], async_=False)
        out.raw["retain_response"] = retain_resp
        out.raw["bank_id"] = bank_id
        out.raw["content_format"] = fmt

        wait = float(mem_var.extra.get("wait_seconds", 0.0))
        if wait:
            await asyncio.sleep(wait)

        # Probe recall with a query that should pull every speaker's statements.
        query = scenario.user_request or "Summarize what each participant said."
        results = await client.arecall(
            bank_id=bank_id,
            query=query,
            max_tokens=mem_var.extra.get("recall_max_tokens", 6000),
        )
        out.raw["recalled_items"] = results
        out.raw["recalled_text"] = [r.get("text", "") for r in results]
        out.response_text = "\n---\n".join(str(r.get("text", "")) for r in results)
    except Exception as e:
        out.error = f"{type(e).__name__}: {e}"
        logger.exception("ambient_attribution driver error")
    finally:
        if not mem_var.extra.get("keep_bank"):
            try:
                await client.adelete_bank(bank_id)
            except Exception as e:
                logger.warning("failed to delete eval bank %s: %s", bank_id, e)
        try:
            await client.client.aclose()
        except Exception:
            pass

    out.duration_s = (datetime.utcnow() - started).total_seconds()
    return out
