"""Central credential vault.

The single inventory of machine secrets. Resolves credential refs to values
via a pluggable ``source`` (default: ``os.environ.get``), so an encrypted-file
or keyring backend can be added later behind the same interface.

Secret values are never logged or printed.
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

from src.security.scrubber import SecretScrubber


class CredentialVault:
    """Resolves and inventories machine secrets."""

    KNOWN_REFS: Tuple[str, ...] = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_GENERATIVEAI_API_KEY",
        "ZAMMAD_API_KEY",
    )

    def __init__(
        self, source: Optional[Callable[[str], Optional[str]]] = None
    ) -> None:
        self._source: Callable[[str], Optional[str]] = (
            source if source is not None else os.environ.get
        )

    def get(self, ref: str) -> Optional[str]:
        """Resolve ``ref`` to its value, or None if unset."""
        return self._source(ref)

    def require(self, ref: str) -> str:
        """Resolve ``ref`` or raise ``KeyError`` if it is not set."""
        value = self.get(ref)
        if not value:
            raise KeyError(f"Credential '{ref}' is not set")
        return value

    def known_refs(self) -> Tuple[str, ...]:
        """Return the credential refs this vault manages."""
        return self.KNOWN_REFS

    def register_into(self, scrubber: SecretScrubber) -> int:
        """Register every resolved known secret value into ``scrubber``.

        Returns the number of refs that resolved to a non-empty value and were
        registered.
        """
        count = 0
        for ref in self.KNOWN_REFS:
            value = self.get(ref)
            if value:
                scrubber.register(value, ref)
                count += 1
        return count
