# src/interfaces/_persona_patch.py
"""Shared PATCH helpers for the kobold engine adapter.

`kobold_engine_adapter.py` accepts persona PATCH requests over a fixed field
set. Centralizing the kobold-extras mapping + known-key list here keeps the
PATCH handling reusable and isolated (the legacy :5002 `kobold_adapter.py`
that also consumed these was retired in DP-200 finding A).
"""

from typing import Any, Callable, Dict, List, Tuple

from src.persona import Persona
from src.persona_fields import registry_patch_keys


# (key, coercer) pairs. Coercer raises ValueError/TypeError on bad input,
# which the caller appends to `rejected`. None / "clear" / "" → clear.
_KOBOLD_SAMPLER_EXTRAS: List[Tuple[str, Callable[[Any], Any]]] = [
    ("rep_pen", float),
    ("rep_pen_range", int),
    ("rep_pen_slope", float),
    ("min_p", float),
    ("typical", float),
    ("tfs", float),
]

# Persona keys accepted by the engine adapter's PATCH route. Core persona
# fields come from the registry (src/persona_fields.py) so the PATCH surface
# can never drift from the dev-command surface; the rest are route-specific:
# the history_messages/context_length pair, instruct_tags, and the kobold
# sampler extras above.
_KNOWN_PATCH_KEYS_ENGINE = registry_patch_keys() | {
    "history_messages",
    "context_length",
    "instruct_tags",
} | {key for key, _ in _KOBOLD_SAMPLER_EXTRAS}


def _apply_kobold_sampler_extras(
    persona: Persona, data: Dict[str, Any], rejected: List[str]
) -> None:
    """Write sampler extras from PATCH body into `provider_extras["kobold"]`.

    Missing keys are skipped. None / "clear" / empty string clears the entry.
    Coercion failures append to `rejected` and leave the prior value intact.
    """
    for key, coercer in _KOBOLD_SAMPLER_EXTRAS:
        if key not in data:
            continue
        raw = data[key]
        if raw is None or raw == "" or raw == "clear":
            persona.clear_provider_extra("kobold", key)
            continue
        try:
            persona.set_provider_extra("kobold", key, coercer(raw))
        except (ValueError, TypeError):
            rejected.append(key)


def get_kobold_extras_for_get(persona: Persona) -> Dict[str, Any]:
    """Build the `kobold_extras` block returned by GET /persona/{name}.

    Only includes keys actually set on the persona — absent keys are omitted
    so the portal can distinguish unset (use kobold-lite default) from set.
    """
    out: Dict[str, Any] = {}
    for key, _ in _KOBOLD_SAMPLER_EXTRAS:
        v = persona.get_provider_extra("kobold", key)
        if v is not None:
            out[key] = v
    return out
