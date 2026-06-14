"""Process-global secret redactor (egress scrubber).

Redacts machine secrets from any string bound for the LLM context, audit
records, or the inspector. Two mechanisms:

1. Registered values — exact secret strings registered (e.g. via the vault),
   each redacted to ``[REDACTED:<ref>]``.
2. Pattern fallback — compiled regexes for common secret *shapes*, to catch
   unregistered leaks, redacted to ``[REDACTED:pattern]``.

The scrubber is a process-wide singleton (``get_scrubber``); secrets are a
process-level property so threading the instance through every constructor is
unnecessary. Tests reset it via ``reset_scrubber``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Pattern, Tuple

# Minimum length for a value to be registered. Shorter values risk redacting
# benign substrings, so they are ignored.
MIN_SECRET_LEN = 8

# Compiled regexes for common UNREGISTERED secret shapes. Order matters: the
# more specific Anthropic-style key must precede the generic ``sk-`` form so it
# matches first. All are labelled ``[REDACTED:pattern]``.
PATTERN_REDACTIONS: Tuple[Pattern[str], ...] = (
    # Anthropic-style API keys (more specific than the generic sk- form).
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    # OpenAI-style API keys.
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    # Zammad auth header form: ``Token token=<key>``.
    re.compile(r"Token token=[A-Za-z0-9._-]+"),
    # Generic bearer tokens.
    re.compile(r"Bearer [A-Za-z0-9._\-]{20,}"),
)

_PATTERN_LABEL = "[REDACTED:pattern]"


class SecretScrubber:
    """Redacts registered secret values and common secret shapes from data."""

    def __init__(self) -> None:
        # Map of secret value -> ref label. Keyed by value so registering the
        # same value twice dedupes; the newest ref for a value wins.
        self._secrets: Dict[str, str] = {}

    def register(self, value: str, ref: str) -> None:
        """Record a secret ``value`` under the label ``ref``.

        Ignored if ``value`` is None/empty or shorter than ``MIN_SECRET_LEN``.
        Registering an existing value updates its ref to the newest one.
        """
        if not value or len(value) < MIN_SECRET_LEN:
            return
        self._secrets[value] = ref

    def active_secret_count(self) -> int:
        """Number of currently registered secret values."""
        return len(self._secrets)

    def clear(self) -> None:
        """Forget all registered secret values."""
        self._secrets.clear()

    def scrub(self, obj: object) -> object:
        """Recursively redact secrets from ``obj``.

        Strings are redacted (registered values first, longest-first, then the
        pattern fallback). Dicts/lists/tuples are returned in the same shape
        with their values scrubbed (keys are left alone). Any other value is
        returned unchanged. Never raises.
        """
        if isinstance(obj, str):
            return self._scrub_str(obj)
        if isinstance(obj, dict):
            return {key: self.scrub(val) for key, val in obj.items()}
        if isinstance(obj, list):
            return [self.scrub(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self.scrub(item) for item in obj)
        return obj

    def _scrub_str(self, text: str) -> str:
        # Replace registered values longest-first so a secret that is a
        # substring of another secret cannot partially leak.
        if self._secrets:
            for value in sorted(self._secrets, key=len, reverse=True):
                if value in text:
                    text = text.replace(value, f"[REDACTED:{self._secrets[value]}]")
        # Pattern fallback for unregistered secret shapes.
        for pattern in PATTERN_REDACTIONS:
            text = pattern.sub(_PATTERN_LABEL, text)
        return text


_scrubber: Optional[SecretScrubber] = None


def get_scrubber() -> SecretScrubber:
    """Return the process-global scrubber, creating it on first use."""
    global _scrubber
    if _scrubber is None:
        _scrubber = SecretScrubber()
    return _scrubber


def reset_scrubber() -> None:
    """Replace the singleton with a fresh instance (for test isolation)."""
    global _scrubber
    _scrubber = SecretScrubber()


__all__: List[str] = [
    "SecretScrubber",
    "get_scrubber",
    "reset_scrubber",
    "MIN_SECRET_LEN",
    "PATTERN_REDACTIONS",
]
