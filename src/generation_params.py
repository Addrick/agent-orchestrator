# src/generation_params.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _coerce_stop_sequences(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(v) for v in value]
    except TypeError:
        return []


def _coerce_provider_extras(value: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in value.items():
        if isinstance(v, dict):
            out[str(k)] = dict(v)
    return out


@dataclass
class GenerationParams:
    """Structured generation params shared across providers.

    Universal fields have consistent semantics across all providers. Provider-
    specific knobs (kobold samplers, anthropic thinking config, etc.) live
    under `provider_extras[provider_id]`. See plan
    `memory/project/plans/portal_engine_reintegration.md` Phase A.
    """

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    provider_extras: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "stop_sequences": list(self.stop_sequences),
            "seed": self.seed,
            "provider_extras": {k: dict(v) for k, v in self.provider_extras.items()},
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GenerationParams":
        if not data:
            return cls()
        return cls(
            temperature=_coerce_float(data.get("temperature")),
            top_p=_coerce_float(data.get("top_p")),
            top_k=_coerce_int(data.get("top_k")),
            max_tokens=_coerce_int(data.get("max_tokens")),
            stop_sequences=_coerce_stop_sequences(data.get("stop_sequences")),
            seed=_coerce_int(data.get("seed")),
            provider_extras=_coerce_provider_extras(data.get("provider_extras")),
        )

    def get_provider_extras(self, provider_id: str) -> Dict[str, Any]:
        """Return the extras block for a provider (empty dict if unset)."""
        return dict(self.provider_extras.get(provider_id, {}))
