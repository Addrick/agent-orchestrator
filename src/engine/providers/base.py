# src/engine/providers/base.py
"""The `Provider` ABC (DP-244).

Every LLM provider is packaged as one object. The driver
(`TextEngine._stream_response`) touches a provider through exactly this
contract:

  - `matches(model_name)`        — routing (replaces one `_get_provider_route`
                                    waterfall row),
  - `limiters_for(model_name)`   — the rate limiters to acquire (model-aware:
                                    Google splits limits by gemini/gemma family),
  - `ensure_supported(model_name)` — host/precondition guard run at routing time,
  - `stream(...)`                — the one streaming contract.

Image-support and 429-fallback policy stay on the driver
(`model_supports_images` / `_FALLBACK_MODELS`), not the provider — a provider
hook for them would be dead until the driver routes through it.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from aiolimiter import AsyncLimiter


class Provider(ABC):
    """One LLM provider, as a polymorphic object."""

    @abstractmethod
    def matches(self, model_name: str) -> bool:
        """True if this provider handles ``model_name``."""

    @abstractmethod
    def limiters_for(self, model_name: str) -> List[AsyncLimiter]:
        """Rate limiters to acquire for this model, in acquisition order."""

    def ensure_supported(self, model_name: str) -> None:
        """Host / precondition guard, run when the model is routed. Default no-op.
        Raise ``LLMCommunicationError`` to refuse the route."""

    #: Name of the engine seam method the driver routes this provider through
    #: (back-compat for ``_get_provider_route``). Every concrete provider sets it.
    route_method_name: str

    @abstractmethod
    def stream(
        self,
        persona_config: Dict[str, Any],
        history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        local_inference_config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Async generator emitting the unified event shape
        (``api_payload`` → ``text_delta``* → optional ``tool_calls`` → ``done``).
        Implementations are ``async def`` generators."""
        ...
