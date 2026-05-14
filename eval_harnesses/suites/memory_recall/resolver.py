"""Locator -> memory_id-set resolver for semantic recall scenarios.

A "fact" is a concept. The bank may store it as 1 row or several near-dupes.
Resolution returns the *set* of memory_ids whose text matches any of the
fact's locators. Grader treats any of those ids as proof the fact was
retrieved.

Fact entry shape:
    {
      "key":         "ssh_publickey",
      "source":      "history" | "seed",
      "locators":    ["Permission denied (publickey)", ...],   # history
      "seed_key":    "adam_birthday_date",                     # seed
      "max_matches": 5
    }

Resolution outcomes per fact:
    0 matches            -> unresolved (no match)
    > max_matches        -> unresolved (over-broad locator)
    1..max_matches       -> resolved: key -> set of ids
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_MAX_MATCHES = 5


@dataclass
class ResolutionResult:
    resolved: Dict[str, Set[str]] = field(default_factory=dict)
    unresolved: List[str] = field(default_factory=list)
    diagnostics: Dict[str, str] = field(default_factory=dict)
    # Per-fact match details, useful for --curate output and run debugging.
    matches: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)


def _fact_locators(
    fact: Dict[str, Any], seed_data: Optional[Dict[str, Dict[str, Any]]]
) -> Tuple[List[str], Optional[str]]:
    """Return (locator_strings, error). Seed facts contribute their text."""
    src = fact.get("source")
    if src == "history":
        if "locators" in fact:
            locs = [s for s in (fact["locators"] or []) if s]
        elif "locator" in fact:  # backwards-compat single-string form
            locs = [fact["locator"]] if fact["locator"] else []
        else:
            return [], "history fact missing 'locators'"
        if not locs:
            return [], "history fact has empty locators"
        return locs, None
    if src == "seed":
        key = fact.get("seed_key")
        if not key:
            return [], "seed fact missing 'seed_key'"
        if seed_data is None or key not in seed_data:
            return [], f"seed_key '{key}' not in seed_data"
        text = seed_data[key].get("text")
        if not text:
            return [], f"seed entry '{key}' has no text"
        return [text], None
    return [], f"unknown source '{src}'"


async def resolve_facts(
    bank_name: str,
    facts: List[Dict[str, Any]],
    *,
    rest_client: Any,
    seed_data: Optional[Dict[str, Dict[str, Any]]] = None,
    recall_max_tokens: int = 800,
) -> ResolutionResult:
    """Resolve fact locators to memory_id sets via Hindsight recall.

    rest_client.arecall(bank_id, query, *, max_tokens, ...) -> list of dicts
    with 'id' and 'text'. None client -> all facts unresolved.
    """
    out = ResolutionResult()
    if rest_client is None:
        out.unresolved = [f["key"] for f in facts]
        out.diagnostics["_global"] = "no rest_client provided"
        return out

    recall_cache: Dict[str, List[Dict[str, Any]]] = {}

    for fact in facts:
        key = fact.get("key")
        if not key:
            continue
        cap = int(fact.get("max_matches", DEFAULT_MAX_MATCHES))
        locators, err = _fact_locators(fact, seed_data)
        if err:
            out.unresolved.append(key)
            out.diagnostics[key] = err
            continue

        ids: Set[str] = set()
        per_locator: List[Dict[str, str]] = []
        recall_error: Optional[str] = None

        for loc in locators:
            if loc not in recall_cache:
                try:
                    recall_cache[loc] = await rest_client.arecall(
                        bank_name, loc, max_tokens=recall_max_tokens
                    )
                except Exception as e:
                    recall_error = f"recall failed: {type(e).__name__}: {e}"
                    recall_cache[loc] = []
            loc_lower = loc.lower()
            for hit in recall_cache[loc] or []:
                text = (hit.get("text") or "")
                if loc_lower in text.lower():
                    hid = hit.get("id")
                    if hid:
                        sid = str(hid)
                        if sid not in ids:
                            ids.add(sid)
                            per_locator.append({
                                "id": sid,
                                "locator": loc,
                                "text": text[:200],
                            })

        out.matches[key] = per_locator

        if recall_error and not ids:
            out.unresolved.append(key)
            out.diagnostics[key] = recall_error
            continue
        if len(ids) == 0:
            out.unresolved.append(key)
            out.diagnostics[key] = "no substring match for any locator"
            continue
        if len(ids) > cap:
            out.unresolved.append(key)
            out.diagnostics[key] = (
                f"{len(ids)} matches exceeds max_matches={cap} "
                f"(tighten locators or raise cap)"
            )
            continue
        out.resolved[key] = ids

    return out


def load_seed_data(path: str | Path) -> Dict[str, Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))
