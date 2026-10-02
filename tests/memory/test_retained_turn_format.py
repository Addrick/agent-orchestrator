# tests/memory/test_retained_turn_format.py
"""DP-402: the speaker/time header on what a live chat turn retains."""

from datetime import datetime, timedelta, timezone

import pytest

from config import global_config
from src.turn_persistence import format_retained_turn
from src.utils.timeutil import to_utc


@pytest.fixture(autouse=True)
def eastern(monkeypatch):
    monkeypatch.setattr(global_config, "LOCAL_TZ", "America/New_York")


def test_user_turn_is_stamped_in_local_time():
    ts = datetime(2026, 9, 26, 18, 5, tzinfo=timezone.utc)  # 14:05 EDT
    assert format_retained_turn("user", "Adam", "hi", ts) == "[2026-09-26 14:05] Adam: hi"


def test_local_tz_is_read_at_call_time(monkeypatch):
    monkeypatch.setattr(global_config, "LOCAL_TZ", "UTC")
    ts = datetime(2026, 9, 26, 18, 5, tzinfo=timezone.utc)
    assert format_retained_turn("user", "Adam", "hi", ts) == "[2026-09-26 18:05] Adam: hi"


def test_assistant_turn_has_speaker_and_no_timestamp():
    ts = datetime(2026, 9, 26, 18, 5, tzinfo=timezone.utc)
    out = format_retained_turn("assistant", "derpr", "hello", ts)
    assert out == "derpr: hello"


def test_naive_timestamp_is_utc_not_host_local():
    # DP-413: a naive timestamp is a stored one, and storage is UTC — it is
    # relabelled, never converted from whatever zone the host clock is in.
    naive = datetime(2026, 9, 26, 18, 5)
    assert to_utc(naive) == datetime(2026, 9, 26, 18, 5, tzinfo=timezone.utc)
    assert format_retained_turn("user", "Adam", "hi", naive) == "[2026-09-26 14:05] Adam: hi"


def test_aware_timestamp_keeps_its_instant():
    ts = datetime(2026, 9, 26, 14, 5, tzinfo=timezone(timedelta(hours=-4)))
    assert to_utc(ts) == datetime(2026, 9, 26, 18, 5, tzinfo=timezone.utc)
