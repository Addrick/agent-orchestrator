# src/engine/registry.py
"""Provider registry (DP-244) — the ordered list of `Provider` objects that
replaces the `_get_provider_route` string-prefix waterfall.

`resolve(model_name)` walks the providers in order, returns the first whose
`matches()` is true (after running its `ensure_supported` host guard), and
raises for an unknown model — byte-identical to the old waterfall, including
the `cc-*`-before-`claude` precedence.

All six providers are now fully-extracted `Provider` objects (DP-244 244a–g):
each owns its logic in `providers/<name>.py` (agy/cc share `providers/_subprocess.py`).
"""

from typing import TYPE_CHECKING, List

from src.llm_errors import LLMCommunicationError

from .providers.base import Provider
from .providers.openai import OpenAIProvider
from .providers.anthropic import AnthropicProvider
from .providers.google import GoogleProvider
from .providers.local import LocalProvider
from .providers.agy import AgyProvider
from .providers.cc import CcProvider

if TYPE_CHECKING:
    from src.engine.driver import TextEngine


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
        CcProvider(engine),
        OpenAIProvider(engine),
        AnthropicProvider(engine),
        GoogleProvider(engine),
        AgyProvider(engine),
        LocalProvider(engine),
    ]
    return ProviderRegistry(providers)
