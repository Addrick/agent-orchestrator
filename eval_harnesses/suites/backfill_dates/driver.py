from __future__ import annotations

import asyncio
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from eval_harnesses.framework.fixtures import FixtureBundle
from eval_harnesses.framework.results import RunOutput
from eval_harnesses.framework.scenarios import Scenario
from eval_harnesses.framework.variants import MemoryVariant, PromptVariant

logger = logging.getLogger(__name__)

# Recognises ISO-ish dates in source content (used by header_inline /
# inline_dates strategies that extract a per-block anchor from the text).
_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b")


def _build_items(scenario: Scenario, strategy: str) -> List[Dict[str, Any]]:
    """Construct the items[] payload for /memories under a given strategy.

    Scenario shape (in scenario.expectations):
        documents: [
            {"text": "...", "doc_date": "2026-03-12T10:00:00Z", "id": "doc1"},
            ...
        ]

    Strategies:
        unset            — every item gets timestamp:"unset" (mirrors
                           hindsight_import.py bug)
        explicit         — timestamp = doc's stated date (correct path)
        header_inline    — no timestamp; doc text begins with "Date: <iso>"
        block_anchor     — timestamp = last date found in text via regex
                           (mirrors recover_claudecode_hindsight anchor logic
                           extended to inline-date sources)
    """
    docs: List[Dict[str, Any]] = scenario.expectations.get("documents", [])
    items: List[Dict[str, Any]] = []
    for doc in docs:
        text = doc["text"]
        doc_id = doc.get("id") or f"doc_{uuid.uuid4().hex[:8]}"
        item: Dict[str, Any] = {
            "content": text,
            "document_id": doc_id,
            "tags": doc.get("tags", []),
            "update_mode": "replace",
            "strategy": doc.get("retain_strategy", "documents"),
        }

        if strategy == "unset":
            item["timestamp"] = "unset"
        elif strategy == "explicit":
            item["timestamp"] = doc["doc_date"]
        elif strategy == "header_inline":
            # Prepend a Date: header so the LLM sees it; no timestamp set.
            item["content"] = f"Date: {doc['doc_date']}\n---\n{text}"
            item["timestamp"] = "unset"
        elif strategy == "block_anchor":
            matches = _ISO_DATE_RE.findall(text)
            anchor = matches[-1] if matches else doc.get("doc_date")
            if anchor and "T" not in anchor:
                anchor = f"{anchor}T00:00:00Z"
            item["timestamp"] = anchor or "unset"
        else:
            raise ValueError(f"unknown timestamp_strategy: {strategy}")
        items.append(item)
    return items


def _wait_seconds(memory_variant: MemoryVariant) -> float:
    return float(memory_variant.extra.get("wait_seconds", 0.0))


async def backfill_dates_driver(
    bundle: FixtureBundle,
    scenario: Scenario,
    mem_var: MemoryVariant,
    prompt_var: PromptVariant,
) -> RunOutput:
    """Provision a scratch bank, post items under one strategy, recall, return.

    The framework's FixtureBundle (ChatSystem, MemoryManager) is unused here
    — this suite talks straight to Hindsight. The bundle is still constructed
    so cleanup symmetry holds.
    """
    from src.memory.backend.hindsight import HindsightRESTClient
    from config.global_config import HINDSIGHT_URL

    out = RunOutput()
    started = datetime.utcnow()
    strategy = mem_var.extra.get("timestamp_strategy", "explicit")
    retain_mission = mem_var.extra.get("retain_mission")
    enable_observations = bool(mem_var.extra.get("enable_observations", True))

    suffix = uuid.uuid4().hex[:8]
    bank_id = f"eval_backfill_{scenario.id}_{mem_var.id}_{suffix}".lower()
    bank_id = re.sub(r"[^a-z0-9_]", "_", bank_id)[:63]

    timeout = float(mem_var.extra.get("http_timeout", 300.0))
    client = HindsightRESTClient(HINDSIGHT_URL, timeout=timeout)
    try:
        await client.acreate_bank(
            bank_id=bank_id,
            retain_mission=retain_mission,
            enable_observations=enable_observations,
        )
        items = _build_items(scenario, strategy)
        retain_resp = await client.aretain(bank_id=bank_id, items=items, async_=False)
        out.raw["retain_response"] = retain_resp
        out.raw["bank_id"] = bank_id
        out.raw["strategy"] = strategy

        # Some servers still finish work after a sync POST returns; allow opt-in wait.
        wait = _wait_seconds(mem_var)
        if wait:
            await asyncio.sleep(wait)

        results = await client.arecall(
            bank_id=bank_id,
            query=scenario.user_request,
            max_tokens=mem_var.extra.get("recall_max_tokens", 4000),
        )
        out.raw["recalled_items"] = results
        out.raw["mentioned_at"] = [r.get("mentioned_at") for r in results]
        out.raw["occurred_start"] = [r.get("occurred_start") for r in results]
        out.response_text = "\n".join(str(r.get("text", "")) for r in results)
    except Exception as e:
        out.error = f"{type(e).__name__}: {e}"
        logger.exception("backfill_dates driver error")
    finally:
        # Best-effort cleanup. Leave bank if --keep-banks set.
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
