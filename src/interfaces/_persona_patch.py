# src/interfaces/_persona_patch.py
"""Shared PATCH helpers for the kobold adapter pair.

Both `kobold_adapter.py` and `kobold_engine_adapter.py` accept persona PATCH
requests with the same field set. Centralizing the kobold-extras mapping +
known-key list here keeps the two adapter PATCH handlers in sync without
introducing a circular import.
"""

from typing import Any, Callable, Dict, List, Tuple

from src.persona import Persona


# Keys handled by the legacy adapter's PATCH route (kobold_adapter.py).
_KNOWN_PATCH_KEYS_LEGACY = {
    "prompt",
    "model_name",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "history_messages",
    "context_length",
    "memory_mode",
    "max_context_tokens",
    "instruct_tags",
    # kobold provider extras
    "rep_pen",
    "rep_pen_range",
    "rep_pen_slope",
    "min_p",
    "typical",
    "tfs",
}

# Engine adapter additionally accepts chat_template.
_KNOWN_PATCH_KEYS_ENGINE = _KNOWN_PATCH_KEYS_LEGACY | {"chat_template"}


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
