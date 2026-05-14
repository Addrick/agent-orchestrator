from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from eval_harnesses.framework.fixtures import FixtureBundle
from eval_harnesses.framework.results import RunOutput
from eval_harnesses.framework.scenarios import Scenario
from eval_harnesses.framework.variants import MemoryVariant, PromptVariant

from .resolver import load_seed_data, resolve_facts

_SUITE_DIR = Path(__file__).parent
_SEED_FILE = _SUITE_DIR / "fixtures" / "test_persona_seed.json"

# Module-level caches keyed by (bank, scenario.id) so the cartesian
# product across variants doesn't re-resolve the same locators.
_RESOLVED_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_SEED_CACHE: Dict[str, Dict[str, Any]] = {}


def _seed() -> Dict[str, Any]:
    if "data" not in _SEED_CACHE:
        _SEED_CACHE["data"] = load_seed_data(_SEED_FILE)
    return _SEED_CACHE["data"]


async def _resolve_scenario(
    bank: str, scenario: Scenario, rest_client: Any
) -> Dict[str, Any]:
    """Resolve all expected + noise facts for a scenario. Cached per (bank, id)."""
    key = (bank, scenario.id)
    if key in _RESOLVED_CACHE:
        return _RESOLVED_CACHE[key]

    facts = list(scenario.expectations.get("expected_facts", []))
    facts += list(scenario.expectations.get("noise_facts", []))
    result = await resolve_facts(
        bank, facts, rest_client=rest_client, seed_data=_seed()
    )
    payload = {
        # JSON-safe: sets -> sorted lists. Grader normalizes back to sets.
        "resolved_ids": {k: sorted(v) for k, v in result.resolved.items()},
        "unresolved": list(result.unresolved),
        "diagnostics": dict(result.diagnostics),
        "matches": dict(result.matches),
    }
    _RESOLVED_CACHE[key] = payload
    return payload


def _hindsight_rest_client(memory_manager: Any) -> Any:
    """Pull the HindsightRESTClient out of the memory manager, if any.

    MemoryManager.backend may be a HindsightBackend (which exposes
    `_get_client()` -> HindsightRESTClient) or a SQLite backend (no client).
    Returns None when no live client is available.
    """
    backend = getattr(memory_manager, "backend", None)
    getter = getattr(backend, "_get_client", None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception:
        return None


async def recall_driver(
    bundle: FixtureBundle,
    scenario: Scenario,
    mem_var: MemoryVariant,
    prompt_var: PromptVariant,
) -> RunOutput:
    """Drive a recall-only eval. Bypasses generate_response entirely.

    For SQLite summary retrieval: calls retrieve_relevant_summaries() and
    populates retrieved_summary_ids for RetrievalHitsGrader.

    For Hindsight semantic recall: resolves curated fact locators to
    memory_ids (once per scenario), calls arecall(), and packs both the
    ranked hit list and the {fact_key -> memory_id} map into RunOutput
    for SemanticRecallGrader.
    """
    started = datetime.utcnow()
    mm = bundle.memory_manager
    out = RunOutput()

    try:
        if mem_var.sqlite_summaries:
            params = dict(mem_var.retrieval_params)
            hits = mm.retrieve_relevant_summaries(
                persona_name=scenario.persona_name,
                channel=scenario.channel,
                user_identifier=scenario.user_identifier,
                **params,
            )
            out.retrieved_summaries = hits or []
            out.retrieved_summary_ids = [
                h.get("segment_id")
                for h in (hits or [])
                if h.get("segment_id") is not None
            ]

        if mem_var.hindsight:
            bank = (
                mem_var.hindsight_params.get("bank")
                or scenario.meta.get("bank")
            )
            if not bank:
                out.error = "hindsight branch: no bank declared (variant or scenario.meta)"
            else:
                client = _hindsight_rest_client(mm)
                resolution = await _resolve_scenario(bank, scenario, client)
                out.raw["resolved_ids"] = resolution["resolved_ids"]
                out.raw["resolver"] = {
                    "unresolved": resolution["unresolved"],
                    "diagnostics": resolution["diagnostics"],
                    "matches": resolution["matches"],
                }

                if client is None:
                    out.error = "hindsight branch: no REST client on memory_manager.backend"
                else:
                    recall_kwargs = {
                        k: v
                        for k, v in mem_var.hindsight_params.items()
                        if k in ("max_tokens", "budget", "tags", "types")
                    }
                    hits = await client.arecall(
                        bank, scenario.user_request, **recall_kwargs
                    )
                    out.hindsight_hits = hits or []

    except NotImplementedError as e:
        out.error = f"NotImplementedError: {e}"
    except Exception as e:
        out.error = f"{type(e).__name__}: {e}"

    out.duration_s = (datetime.utcnow() - started).total_seconds()
    return out
