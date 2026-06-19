# src/voice/intent.py
"""Intent routing for transcribed voice commands (DP-238).

``IntentRouter`` is the swap point between "we have text" and "do a thing". The
MVP ``KeywordTimerRouter`` is a cheap, local, regex matcher — deliberately NOT
the LLM. The voice path is always-listening, so routing every utterance through
the LLM would be noisy, slow, and costly. A later ``LLMIntentRouter`` can sit
behind this same ABC for arbitrary intents, fed only utterances this gate (or a
wake word) already accepted.

Duration parsing (``parse_duration``) is shared with the text-callable timer
tools so "10 minutes" means the same thing whether spoken or typed.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

# Spoken/typed unit words → seconds-per-unit.
_UNIT_SECONDS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
}

# Common spoken small-number words so "set a timer for ten minutes" parses
# without a digit. Keep small; large numbers are far more likely spoken as digits
# by the STT model anyway.
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "forty-five": 45, "forty five": 45, "sixty": 60, "ninety": 90,
}

_DURATION_RE = re.compile(
    r"(?P<amount>\d+(?:\.\d+)?|[a-z\- ]+?)\s*"
    r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TimerIntent:
    seconds: int
    label: Optional[str] = None


def _word_to_number(text: str) -> Optional[float]:
    text = text.strip().lower()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    # Take the trailing number-word (e.g. "for ten" → ten).
    if text in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[text])
    tokens = text.replace("-", " ").split()
    for tok in reversed(tokens):
        if tok in _NUMBER_WORDS:
            return float(_NUMBER_WORDS[tok])
    return None


def parse_duration(text: str) -> Optional[int]:
    """Parse the first ``<amount> <unit>`` in ``text`` to whole seconds.

    Handles digits ("10 minutes") and small spoken numbers ("ten minutes",
    "a minute"). Returns None when no recognizable duration is present."""
    if not text:
        return None
    match = _DURATION_RE.search(text)
    if not match:
        return None
    amount = _word_to_number(match.group("amount"))
    if amount is None or amount <= 0:
        return None
    per = _UNIT_SECONDS.get(match.group("unit").lower())
    if per is None:
        return None
    seconds = int(round(amount * per))
    return seconds if seconds > 0 else None


class IntentRouter(ABC):
    @abstractmethod
    async def route(self, text: str) -> Optional[TimerIntent]:
        """Return a parsed intent for ``text``, or None if it isn't a command."""


class KeywordTimerRouter(IntentRouter):
    """Match "set a timer for N units", gated by an optional wake word.

    When ``wake_word`` is set, the utterance must contain it (anywhere) to be
    considered — the cheap always-listening front door. ``require_timer_word``
    keeps it tight: the text must also mention "timer".
    """

    def __init__(
        self,
        *,
        wake_word: Optional[str] = None,
        require_timer_word: bool = True,
    ) -> None:
        self._wake = wake_word.lower().strip() if wake_word else None
        self._require_timer_word = require_timer_word

    async def route(self, text: str) -> Optional[TimerIntent]:
        if not text:
            return None
        low = text.lower()
        if self._wake and self._wake not in low:
            return None
        if self._require_timer_word and "timer" not in low and "alarm" not in low:
            return None
        seconds = parse_duration(low)
        if seconds is None:
            return None
        return TimerIntent(seconds=seconds, label=_extract_label(low))


def _extract_label(text: str) -> Optional[str]:
    """Pull a trailing label: "...for 10 minutes for the pasta" → "the pasta"."""
    # Greedy ``.*`` so we anchor on the LAST "for" ("...for 10 minutes for the
    # pasta" → "the pasta", not "10 minutes for the pasta").
    match = re.search(r".*\bfor\s+(.+)$", text)
    if not match:
        return None
    tail = match.group(1).strip()
    # Drop it when the "for" clause is just the duration restated
    # ("for 10 minutes") rather than a real label ("for the pasta").
    if parse_duration(tail) is not None and len(tail.split()) <= 3:
        return None
    return tail or None
