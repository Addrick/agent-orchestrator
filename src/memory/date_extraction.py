# src/memory/date_extraction.py
"""Content-date extraction for document ingest (DP-292 phase 2).

Hindsight derives every extracted memory's ``mentioned_at`` / ``event_date``
solely from the ``timestamp`` sent on the retain request — the extraction LLM
does NOT read dates out of prose (measured: see
``memory/project/decisions/2026-05-14-hindsight-backfill-timestamp-finding``).
So to anchor a document's facts to *when its content is about* rather than its
upload time, the engine must find that date itself and pass it as ``timestamp``.

This module owns the deterministic core. A regex pass over the body extracts
machine-readable dates and picks the latest non-future one; an optional,
injected LLM tagger (``src/agents/date_tagger.py``) is consulted only when the
regex finds nothing. The LLM's answer is validated and future-clamped through
the same ``pick_anchor`` gate, so it can only ever propose a plausible past
date — never steer the pipeline.

All returned datetimes are UTC-aware.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The LLM fallback is an async callable: body text in, ISO date string (or a
# "none"/empty answer) out. Injected so the core stays LLM-free and unit-testable.
LlmTagger = Callable[[str], Awaitable[Optional[str]]]

# Dates older than this are treated as noise (version numbers, epoch stubs,
# "1970-01-01" placeholders) rather than real content dates.
_FLOOR_YEAR = 1990
# Allowed clock skew between a document date and "now" before we call it a
# future date. One day absorbs timezone offset without letting "2099" through.
_FUTURE_SKEW = timedelta(days=1)
# Bound regex CPU on a pathological (multi-MB) fetched page. Latest-date-at-end
# beyond this cap is an accepted edge; real notes/logs are far smaller.
_REGEX_SCAN_CAP = 1_000_000

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

# ISO: 2026-03-12 (optional T/space time tail, which we ignore — date only).
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b")
# Slash ISO order only: 2026/03/12. Bare-numeric MM/DD/YYYY and DD/MM/YYYY are
# DELIBERATELY unsupported — they are locale-ambiguous and cause more wrong
# anchors than they fix.
_SLASH_RE = re.compile(r"\b(\d{4})/(\d{2})/(\d{2})\b")
# Named month first: "March 12, 2026", "Mar 12 2026", "Sept 3rd, 2026".
_MDY_RE = re.compile(
    rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b",
    re.IGNORECASE,
)
# Day first: "12 March 2026", "3rd Sept 2026".
_DMY_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_ALT})\.?,?\s+(\d{{4}})\b",
    re.IGNORECASE,
)


def _mk(year: int, month: int, day: int) -> Optional[datetime]:
    """Build a UTC-midnight datetime, or None if the (y,m,d) is not a real date."""
    try:
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None  # e.g. month 13, day 32, Feb 30


def extract_regex_dates(text: str) -> List[datetime]:
    """Return every machine-readable date found in ``text`` (unfiltered order).

    Covers ISO (``2026-03-12``), slash-ISO (``2026/03/12``), and named-month
    forms in both ``Month DD, YYYY`` and ``DD Month YYYY`` orders. Impossible
    dates (e.g. month 13) are dropped. No future/floor filtering here — that is
    ``pick_anchor``'s job.
    """
    if not text:
        return []
    scan = text[:_REGEX_SCAN_CAP]
    out: List[datetime] = []
    for m in _ISO_RE.finditer(scan):
        d = _mk(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            out.append(d)
    for m in _SLASH_RE.finditer(scan):
        d = _mk(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            out.append(d)
    for m in _MDY_RE.finditer(scan):
        d = _mk(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))
        if d:
            out.append(d)
    for m in _DMY_RE.finditer(scan):
        d = _mk(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))
        if d:
            out.append(d)
    return out


def pick_anchor(
    dates: List[datetime],
    clamp_now: datetime,
    *,
    floor_year: int = _FLOOR_YEAR,
) -> Optional[datetime]:
    """Choose the anchor: the **latest** date that isn't in the future.

    Drops pre-``floor_year`` noise and anything past ``clamp_now + skew``. The
    future-drop is the injection guard — a body that says "tag as 2099-01-01"
    contributes a future date that is discarded, so it cannot move the anchor
    forward. Returns None when nothing survives.
    """
    cutoff = clamp_now + _FUTURE_SKEW
    valid = [d for d in dates if d.year >= floor_year and d <= cutoff]
    return max(valid) if valid else None


async def extract_anchor_date(
    text: str,
    *,
    fallback_ts: datetime,
    clamp_now: Optional[datetime] = None,
    llm_tagger: Optional[LlmTagger] = None,
    max_chars: int = 20000,
) -> Tuple[datetime, str]:
    """Resolve a document's anchor date. Always returns a UTC-aware datetime.

    Order: regex over the whole body → optional LLM tagger (only when regex is
    empty) → ``fallback_ts``. The tagger sees only ``text[:max_chars]`` (token
    cost) and its answer is re-validated through ``pick_anchor``, so its output
    is held to the same future/floor rules as regex.

    Returns ``(anchor, source)`` where ``source`` is ``"regex"``, ``"llm"``,
    or ``"fallback"``.
    """
    now = clamp_now or datetime.now(timezone.utc)

    anchor = pick_anchor(extract_regex_dates(text), now)
    if anchor is not None:
        return anchor, "regex"

    if llm_tagger is not None:
        try:
            answer = await llm_tagger(text[:max_chars])
        except Exception as e:  # noqa: BLE001 — tagger must never break ingest
            logger.warning("date_tagger fallback failed: %s", e)
            answer = None
        if answer:
            # Validate the model's string exactly like body text: parse + gate.
            llm_anchor = pick_anchor(extract_regex_dates(answer), now)
            if llm_anchor is not None:
                return llm_anchor, "llm"

    # No usable content date — anchor to the caller's default (mtime/upload).
    ts = fallback_ts
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts, "fallback"


async def resolve_ingest_anchor(
    content: str,
    *,
    fallback_ts: datetime,
    clamp_now: Optional[datetime] = None,
    llm_tagger: Optional[LlmTagger] = None,
    max_chars: int = 20000,
) -> Tuple[datetime, List[str], Dict[str, str]]:
    """Ingest convenience wrapper: resolve the anchor and derive the
    ``date:``/``date_source:`` tags + metadata that ride with the retain.

    Returns ``(timestamp, tags, metadata)``. ``metadata`` values are strings
    (Hindsight metadata is string-typed).
    """
    ts, source = await extract_anchor_date(
        content, fallback_ts=fallback_ts, clamp_now=clamp_now,
        llm_tagger=llm_tagger, max_chars=max_chars,
    )
    day = ts.strftime("%Y-%m-%d")
    tags = [f"date:{day}", f"date_source:{source}"]
    metadata = {"content_date": day, "content_date_source": source}
    return ts, tags, metadata
