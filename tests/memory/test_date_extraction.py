"""Unit tests for DP-292 phase-2 content-date extraction.

Covers the deterministic regex core (formats, multi-date pick rule, future/
floor filtering), the async resolver's regex→LLM→fallback ordering, and the
injection guard: a body (or a compromised LLM) proposing a future date must
not steer the anchor.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.memory.date_extraction import (
    extract_anchor_date,
    extract_regex_dates,
    pick_anchor,
    resolve_ingest_anchor,
)

NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)


def _days(dates):
    return sorted(d.date().isoformat() for d in dates)


# ---------- extract_regex_dates ----------

def test_iso_dates():
    got = extract_regex_dates("event on 2026-03-12 and 2026-03-12T10:00:00 done")
    assert _days(got) == ["2026-03-12", "2026-03-12"]


def test_slash_iso_order():
    assert _days(extract_regex_dates("logged 2026/03/12 ok")) == ["2026-03-12"]


def test_named_month_both_orders():
    got = extract_regex_dates("March 12, 2026 ... 3rd Sept 2026 ... 12 April 2026")
    assert _days(got) == ["2026-03-12", "2026-04-12", "2026-09-03"]


def test_named_month_abbrev_and_ordinals():
    assert _days(extract_regex_dates("Mar 5th 2026")) == ["2026-03-05"]


def test_impossible_dates_dropped():
    # month 13 / day 32 / Feb 30 must not produce a datetime.
    assert extract_regex_dates("2026-13-01 2026-02-30 2026-04-31") == []


def test_ambiguous_numeric_slash_is_ignored():
    # Bare MM/DD/YYYY and DD/MM/YYYY are locale-ambiguous → deliberately unmatched.
    assert extract_regex_dates("dated 03/12/2026 and 12/03/2026") == []


def test_empty_text():
    assert extract_regex_dates("") == []


# ---------- pick_anchor ----------

def test_pick_latest_non_future():
    dates = extract_regex_dates("2026-04-01 2026-04-15 2026-04-09")
    assert pick_anchor(dates, NOW).date().isoformat() == "2026-04-15"


def test_future_date_dropped():
    dates = extract_regex_dates("2026-01-01 2099-01-01")
    assert pick_anchor(dates, NOW).date().isoformat() == "2026-01-01"


def test_all_future_returns_none():
    assert pick_anchor(extract_regex_dates("2099-01-01"), NOW) is None


def test_floor_year_noise_dropped():
    dates = extract_regex_dates("1970-01-01 2026-05-01")
    assert pick_anchor(dates, NOW).date().isoformat() == "2026-05-01"


def test_pick_anchor_empty():
    assert pick_anchor([], NOW) is None


# ---------- extract_anchor_date ----------

@pytest.mark.asyncio
async def test_regex_path_wins_and_ignores_tagger():
    async def tagger(_):
        raise AssertionError("tagger must not be called when regex finds a date")

    ts, source = await extract_anchor_date(
        "seen 2026-02-20 here", fallback_ts=NOW, clamp_now=NOW, llm_tagger=tagger,
    )
    assert source == "regex"
    assert ts.date().isoformat() == "2026-02-20"


@pytest.mark.asyncio
async def test_llm_fallback_used_when_regex_empty():
    async def tagger(_):
        return "2025-12-24"

    ts, source = await extract_anchor_date(
        "we met just before the holidays", fallback_ts=NOW, clamp_now=NOW,
        llm_tagger=tagger,
    )
    assert source == "llm"
    assert ts.date().isoformat() == "2025-12-24"


@pytest.mark.asyncio
async def test_llm_future_date_rejected_falls_back():
    """Injection guard: a compromised tagger returning a future date is dropped."""
    async def tagger(_):
        return "2099-01-01"

    ts, source = await extract_anchor_date(
        "no dates in body", fallback_ts=NOW, clamp_now=NOW, llm_tagger=tagger,
    )
    assert source == "fallback"
    assert ts == NOW


@pytest.mark.asyncio
async def test_llm_none_answer_falls_back():
    async def tagger(_):
        return None

    ts, source = await extract_anchor_date(
        "no dates", fallback_ts=NOW, clamp_now=NOW, llm_tagger=tagger,
    )
    assert source == "fallback"


@pytest.mark.asyncio
async def test_tagger_exception_falls_back():
    async def tagger(_):
        raise RuntimeError("engine down")

    ts, source = await extract_anchor_date(
        "no dates", fallback_ts=NOW, clamp_now=NOW, llm_tagger=tagger,
    )
    assert source == "fallback"


@pytest.mark.asyncio
async def test_no_tagger_regex_empty_falls_back():
    ts, source = await extract_anchor_date(
        "no dates at all", fallback_ts=NOW, clamp_now=NOW, llm_tagger=None,
    )
    assert source == "fallback"
    assert ts == NOW


@pytest.mark.asyncio
async def test_naive_fallback_ts_made_utc_aware():
    naive = datetime(2026, 6, 1, 12, 0, 0)
    ts, source = await extract_anchor_date("no dates", fallback_ts=naive, clamp_now=NOW)
    assert source == "fallback"
    assert ts.tzinfo is not None


# ---------- resolve_ingest_anchor ----------

@pytest.mark.asyncio
async def test_resolve_ingest_anchor_tags_and_metadata():
    ts, tags, meta = await resolve_ingest_anchor(
        "note dated 2026-03-12", fallback_ts=NOW, clamp_now=NOW,
    )
    assert ts.date().isoformat() == "2026-03-12"
    assert "date:2026-03-12" in tags
    assert "date_source:regex" in tags
    assert meta == {"content_date": "2026-03-12", "content_date_source": "regex"}
