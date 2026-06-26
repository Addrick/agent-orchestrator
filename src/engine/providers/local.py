# src/engine/providers/local.py
"""Local (kobold-native) provider (DP-244) — fourth provider extracted.

Unlike the API providers, the local stream logic does not live on ``TextEngine``
at all: it lives in the ``StreamEngine`` transport component, and
``TextEngine._stream_local_response`` is already a thin delegator to
``stream_engine.stream_local(...)``. So this slice is purely the ``Provider``
face — ``LocalProvider`` routes through that existing engine seam, forwarding the
``local_inference_config`` side-channel (GenerationParams / kobold provider_extras)
that only the local path consumes.
"""

from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

from aiolimiter import AsyncLimiter

from .base import Provider

if TYPE_CHECKING:
    from src.engine.driver import TextEngine


class LocalProvider(Provider):
    """Kobold-native local provider (``model_name == "local"``). No API rate
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
        # Route through the engine seam (which delegates to StreamEngine), and —
        # unlike the API providers — forward the local_inference_config channel.
        async for ev in self._engine._stream_local_response(
            persona_config, history_object, tools, local_inference_config
        ):
            yield ev
