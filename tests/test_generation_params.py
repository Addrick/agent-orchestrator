# tests/test_generation_params.py
"""Phase A — `GenerationParams` dataclass + Persona facade.

Plan: memory/project/plans/portal_engine_reintegration.md
"""

import pytest

from src.generation_params import GenerationParams
from src.persona import Persona


def test_defaults_are_none_and_empty():
    p = GenerationParams()
    assert p.temperature is None
    assert p.top_p is None
    assert p.top_k is None
    assert p.max_tokens is None
    assert p.stop_sequences == []
    assert p.seed is None
    assert p.provider_extras == {}


def test_to_dict_round_trip():
    p = GenerationParams(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_tokens=2048,
        stop_sequences=["</s>", "User:"],
        seed=42,
        provider_extras={"kobold": {"rep_pen": 1.05, "min_p": 0.05}},
    )
    d = p.to_dict()
    p2 = GenerationParams.from_dict(d)
    assert p2 == p


def test_from_dict_coerces_numeric_strings():
    p = GenerationParams.from_dict({
        "temperature": "0.5",
        "top_p": "0.9",
        "top_k": "32",
        "max_tokens": "1024",
        "seed": "7",
    })
    assert p.temperature == 0.5
    assert p.top_p == 0.9
    assert p.top_k == 32
    assert p.max_tokens == 1024
    assert p.seed == 7


def test_from_dict_drops_garbage_to_none():
    p = GenerationParams.from_dict({
        "temperature": "hot",
        "top_p": object(),
        "top_k": "many",
    })
    assert p.temperature is None
    assert p.top_p is None
    assert p.top_k is None


def test_from_dict_stop_sequences_string_wraps_to_list():
    p = GenerationParams.from_dict({"stop_sequences": "</s>"})
    assert p.stop_sequences == ["</s>"]


def test_from_dict_provider_extras_filters_non_dict_blocks():
    p = GenerationParams.from_dict({
        "provider_extras": {
            "kobold": {"rep_pen": 1.1},
            "openai": "not-a-dict",
        }
    })
    assert p.provider_extras == {"kobold": {"rep_pen": 1.1}}


def test_get_provider_extras_returns_copy():
    p = GenerationParams(provider_extras={"kobold": {"rep_pen": 1.05}})
    extras = p.get_provider_extras("kobold")
    extras["rep_pen"] = 9.0
    # mutation of returned dict must not bleed into the params object
    assert p.provider_extras["kobold"]["rep_pen"] == 1.05


def test_get_provider_extras_unknown_provider_empty():
    p = GenerationParams()
    assert p.get_provider_extras("anthropic") == {}


# --- Persona facade ---


@pytest.fixture
def base_kwargs():
    return {"persona_name": "p", "model_name": "m", "prompt": "x"}


def test_persona_flat_kwargs_populate_params(base_kwargs):
    """Legacy ctor signature still works — flat kwargs land in _params."""
    p = Persona(**base_kwargs, temperature=0.8, top_p=0.9, top_k=40, token_limit=512)
    gp = p.get_generation_params()
    assert gp.temperature == 0.8
    assert gp.top_p == 0.9
    assert gp.top_k == 40
    assert gp.max_tokens == 512
    # Legacy getters delegate.
    assert p.get_temperature() == 0.8
    assert p.get_top_p() == 0.9
    assert p.get_top_k() == 40
    assert p.get_response_token_limit() == 512


def test_persona_accepts_structured_params_dict(base_kwargs):
    """New shape: pass a `params` dict directly."""
    p = Persona(
        **base_kwargs,
        params={
            "temperature": 0.6,
            "top_p": 0.85,
            "top_k": 20,
            "max_tokens": 4096,
            "stop_sequences": ["</s>"],
            "seed": 13,
            "provider_extras": {"kobold": {"min_p": 0.05}},
        },
    )
    gp = p.get_generation_params()
    assert gp.temperature == 0.6
    assert gp.stop_sequences == ["</s>"]
    assert gp.seed == 13
    assert gp.provider_extras == {"kobold": {"min_p": 0.05}}
    assert p.get_response_token_limit() == 4096


def test_persona_flat_kwargs_override_structured_params(base_kwargs):
    """Explicit flat kwargs win when both are supplied."""
    p = Persona(
        **base_kwargs,
        params={"temperature": 0.5, "top_p": 0.5, "top_k": 5, "max_tokens": 1000},
        temperature=0.99,
        token_limit=2000,
    )
    assert p.get_temperature() == 0.99
    assert p.get_top_p() == 0.5  # not overridden
    assert p.get_response_token_limit() == 2000


def test_persona_setters_mutate_underlying_params(base_kwargs):
    p = Persona(**base_kwargs)
    p.set_temperature(0.42)
    p.set_top_p(0.77)
    p.set_top_k(7)
    p.set_response_token_limit(1500)
    gp = p.get_generation_params()
    assert gp.temperature == 0.42
    assert gp.top_p == 0.77
    assert gp.top_k == 7
    assert gp.max_tokens == 1500
