# src/engine/registry.py
"""Provider registry (DP-244) — the ordered list of `Provider` objects that
replaces the `_get_provider_route` string-prefix waterfall.

`resolve(model_name)` walks the providers in order, returns the first whose
`matches()` is true (after running its `ensure_supported` host guard), and
raises for an unknown model — byte-identical to the old waterfall, including
the `cc-*`-before-`claude` precedence.

Five of the six providers are thin `_EngineProvider` adapters that delegate to
the engine's existing `_stream_<provider>_response` methods (their own slices
come later, 244b–g). OpenAI is the fully-extracted proof in `providers.openai`.
"""

from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List, Optional

from aiolimiter import AsyncLimiter

from src.llm_errors import LLMCommunicationError

from .providers.base import Provider
from .providers.openai import OpenAIProvider
from .providers.anthropic import AnthropicProvider

if TYPE_CHECKING:
    from src.engine.driver import TextEngine


class _EngineProvider(Provider):
    """A thin adapter binding a `Provider` face onto one of the engine's
    existing `_stream_<provider>_response` methods. Transitional: each provider
    gets its own real class in a later DP-244 slice."""

    def __init__(
        self,
        engine: "TextEngine",
        *,
        method_name: str,
        matches: Callable[[str], bool],
        limiters: Callable[["TextEngine", str], List[AsyncLimiter]],
        supports_images: Callable[[str], bool] = lambda m: False,
        guard_name: Optional[str] = None,
        passes_local_config: bool = False,
    ) -> None:
        self._engine = engine
        self.route_method_name = method_name
        self._matches = matches
        self._limiters = limiters
        self._supports_images = supports_images
        self._guard_name = guard_name
        self._passes_local_config = passes_local_config

    def matches(self, model_name: str) -> bool:
        return self._matches(model_name)

    def limiters_for(self, model_name: str) -> List[AsyncLimiter]:
        return self._limiters(self._engine, model_name)

    def supports_images(self, model_name: str) -> bool:
        return self._supports_images(model_name)

    def ensure_supported(self, model_name: str) -> None:
        if self._guard_name is not None:
            # Looked up live so per-test monkeypatches of the guard take effect.
            getattr(self._engine, self._guard_name)()

    async def stream(
        self,
        persona_config: Dict[str, Any],
        history_object: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        local_inference_config: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        method = getattr(self._engine, self.route_method_name)
        if self._passes_local_config:
            async for ev in method(persona_config, history_object, tools, local_inference_config):
                yield ev
        else:
            async for ev in method(persona_config, history_object, tools):
                yield ev


def _google_limiters(engine: "TextEngine", model_name: str) -> List[AsyncLimiter]:
    """Google splits rate limits by model family (preserves the waterfall:
    gemma-4/gemma → gemma-4 RPM; gemini-3.1 → gemini-3 RPM; gemini → 2.5
    RPM+RPD)."""
    if "gemma" in model_name:
        return [engine._gemma_4_rpm_limiter]
    if "gemini-3.1" in model_name:
        return [engine._gemini_3_rpm_limiter]
    return [engine._gemini_25_rpm_limiter, engine._gemini_25_rpd_limiter]


class ProviderRegistry:
    """Ordered collection of providers with first-match routing."""

    def __init__(self, providers: List[Provider]) -> None:
        self._providers = providers

    def resolve(self, model_name: str) -> Provider:
        """Return the first provider matching ``model_name`` (running its host
        guard), or raise ``LLMCommunicationError`` for an unsupported model."""
        for provider in self._providers:
            if provider.matches(model_name):
                provider.ensure_supported(model_name)
                return provider
        raise LLMCommunicationError(f"Error: Model '{model_name}' is not supported.")


def build_registry(engine: "TextEngine") -> ProviderRegistry:
    """Construct the ordered provider list. Order preserves the old
    `_get_provider_route` waterfall — notably `cc-*` before `claude`."""
    providers: List[Provider] = [
        # cc-* must precede the `"claude" in model_name` match below.
        _EngineProvider(
            engine,
            method_name="_stream_cc_response",
            matches=lambda m: m.startswith("cc-"),
            limiters=lambda e, m: [e._cc_limiter],
            guard_name="_ensure_cc_supported",
        ),
        OpenAIProvider(engine),
        AnthropicProvider(engine),
        _EngineProvider(
            engine,
            method_name="_stream_google_response",
            matches=lambda m: "gemma" in m or "gemini" in m,
            limiters=_google_limiters,
            supports_images=lambda m: 'gemini' in m.lower() or 'gemma' in m.lower(),
        ),
        _EngineProvider(
            engine,
            method_name="_stream_agy_response",
            matches=lambda m: m.startswith("agy"),
            limiters=lambda e, m: [e._agy_limiter],
            guard_name="_ensure_agy_supported",
        ),
        _EngineProvider(
            engine,
            method_name="_stream_local_response",
            matches=lambda m: m == "local",
            limiters=lambda e, m: [],
            passes_local_config=True,
        ),
    ]
    return ProviderRegistry(providers)
