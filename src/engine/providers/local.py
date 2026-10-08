# src/engine/providers/local.py
"""Local provider: the model server on our own box (DP-244, DP-417).

Since DP-417 this is the OpenAI chat-completions API at ``LOCAL_LLM_URL``,
streamed by the same `stream_chat_completions` as the OpenAI provider. Both
local servers speak it — KoboldCPP (``/v1/chat/completions``, jinja on) and
Strata — so one transport serves either. It replaced the kobold-native
``/api/extra/generate/stream`` transport (`StreamEngine`), which rendered
derpr's own copies of each model's chat template and parsed a text
`<tool_call>` protocol out of the raw tokens. The server now applies the
model's own template and returns structured `tool_calls`.

What carried over, and how:
  - KoboldCPP-only samplers ride in ``extra_body``. KoboldCPP reads them from
    the same request dict as its native API; Strata ignores them.
  - The persona `chat_template` preset survives as a thinking switch only
    (`CHAT_TEMPLATE_THINKING`); `thinking_level` overrides it.
  - Cancelling is closing the stream. Both servers abort on disconnect, so
    the genkey ``/api/extra/abort`` call went with the native transport.
"""

import os
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Dict, List, Optional

import httpx
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI

from config import global_config
from src.generation_params import CHAT_TEMPLATE_THINKING, GenerationParams

from .base import Provider
from .openai import build_openai_params, stream_chat_completions

if TYPE_CHECKING:
    from src.engine.driver import TextEngine

#: KoboldCPP samplers with no chat-completions field. Sent in ``extra_body``
#: under their native names, which KoboldCPP's OpenAI route honours.
KOBOLD_SAMPLER_EXTRAS = ("rep_pen", "rep_pen_range", "rep_pen_slope", "min_p", "typical", "tfs")


def local_base_url() -> str:
    """The local model server's base URL, without a trailing ``/v1``."""
    raw = os.environ.get("LOCAL_LLM_URL", global_config.LOCAL_LLM_URL).rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[:-3]
    return raw


async def get_local_client(engine: "TextEngine") -> AsyncOpenAI:
    """The (lazily cached) client for the local server. No retries: a retried
    streaming request would start a second generation, and the native
    transport never retried either. The long read timeout is for prefill on
    a large prompt, which can run minutes before the first token."""
    if engine.local_client is None:
        engine.local_client = AsyncOpenAI(
            base_url=f"{local_base_url()}/v1",
            api_key="local",  # the SDK requires one; neither server checks it
            max_retries=0,
            timeout=httpx.Timeout(600.0, connect=15.0),
        )
    return engine.local_client


def params_from_legacy_dicts(
    persona_config: Dict[str, Any],
    local_inference_config: Optional[Dict[str, Any]],
) -> GenerationParams:
    """Bridge (persona_config dict + local_inference_config) callers into a
    GenerationParams. local_inference_config keys override the persona_config
    defaults; KoboldCPP-only knobs land in `provider_extras['kobold']`."""
    lic = local_inference_config or {}
    extras: Dict[str, Any] = {}
    for k in (*KOBOLD_SAMPLER_EXTRAS, "max_context_length"):
        if lic.get(k) is not None:
            extras[k] = lic[k]

    def _override(key: str, fallback_key: Optional[str] = None) -> Any:
        val = lic.get(key)
        if val is not None:
            return val
        return persona_config.get(fallback_key or key)

    params = GenerationParams(
        temperature=_override("temperature"),
        top_p=_override("top_p"),
        top_k=_override("top_k"),
        max_tokens=_override("max_tokens", "max_output_tokens"),
        provider_extras={"kobold": extras} if extras else {},
    )
    if lic.get("stop_sequence"):
        params.stop_sequences = list(lic["stop_sequence"])
    return params


def thinking_controls(persona_config: Dict[str, Any]) -> Dict[str, Any]:
    """The request fields that switch thinking, or {} for the server default.

    `thinking_level` wins and passes through as `reasoning_effort` — a budget
    on KoboldCPP (a fraction of max tokens), a level on Strata. Otherwise the
    `chat_template` preset (or ``KOBOLD_CHAT_TEMPLATE``) sets only the
    template's ``enable_thinking`` switch: KoboldCPP treats any
    `reasoning_effort` as a token budget and disables request batching for it,
    and both servers honour the switch alone."""
    level = persona_config.get("thinking_level")
    if level:
        return {
            "reasoning_effort": str(level),
            "chat_template_kwargs": {"enable_thinking": str(level).lower() != "none"},
        }
    preset = persona_config.get("chat_template") or os.environ.get("KOBOLD_CHAT_TEMPLATE")
    on = CHAT_TEMPLATE_THINKING.get(str(preset)) if preset else None
    if on is None:
        return {}
    return {"chat_template_kwargs": {"enable_thinking": on}}


def build_local_params(
    persona_config: Dict[str, Any],
    history_object: Dict[str, Any],
    params: GenerationParams,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Chat-completions request kwargs for the local server. Messages and tools
    are built exactly as for OpenAI; what differs is the knobs."""
    config = {
        **persona_config,
        "model_name": "local",  # neither server routes on it
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_output_tokens": params.max_tokens or persona_config.get("max_output_tokens"),
    }
    api_params = build_openai_params(config, history_object, tools)

    kobold = params.get_provider_extras("kobold")
    extra_body: Dict[str, Any] = {k: kobold[k] for k in KOBOLD_SAMPLER_EXTRAS if kobold.get(k) is not None}
    if params.top_k is not None:
        extra_body["top_k"] = params.top_k
    # KoboldCPP sizes its context per request; the persona's token budget
    # (not `context_length`, which is a turn count) caps it. Strata ignores it.
    ctx = kobold.get("max_context_length") or persona_config.get("max_context_tokens") or 2048
    extra_body["max_context_length"] = min(ctx, global_config.DEFAULT_MAX_CONTEXT_TOKENS)
    extra_body.update(thinking_controls(persona_config))
    api_params["extra_body"] = extra_body

    if params.stop_sequences:
        api_params["stop"] = list(params.stop_sequences)
    if params.seed is not None:
        api_params["seed"] = params.seed
    return api_params


async def stream_local(
    engine: "TextEngine",
    persona_config: Dict[str, Any],
    history_object: Dict[str, Any],
    params: GenerationParams,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Canonical local driver: one streaming chat-completions request."""
    client = await get_local_client(engine)
    api_params = build_local_params(persona_config, history_object, params, tools)
    # aclosing: closing this generator must close the HTTP stream (cancel).
    async with aclosing(stream_chat_completions(client, api_params, "Local model")) as events:
        async for ev in events:
            yield ev


class LocalProvider(Provider):
    """Local model server provider (``model_name == "local"``). No API rate
    limit (the box is ours), no image support, and the only provider that
    forwards ``local_inference_config``."""

    def __init__(self, engine: "TextEngine") -> None:
        self._engine = engine

    #: name of the engine seam method (back-compat for `_get_provider_route`).
    route_method_name = "_stream_local_response"

    def matches(self, model_name: str) -> bool:
        return model_name == "local"

    def limiters_for(self, model_name: str) -> List[AsyncLimiter]:
        return []

    async def stream(
        self,
        persona_config: Dict[str, Any],
        history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        local_inference_config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        # Route through the engine seam, and — unlike the API providers —
        # forward the local_inference_config channel.
        async with aclosing(self._engine._stream_local_response(
            persona_config, history_object, tools, local_inference_config
        )) as events:
            async for ev in events:
                yield ev
