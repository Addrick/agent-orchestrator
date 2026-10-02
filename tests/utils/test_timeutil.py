# tests/utils/test_timeutil.py
"""DP-413: storage is UTC, display is LOCAL_TZ, neither reads the host zone."""

from datetime import datetime, timedelta, timezone

import pytest

from config import global_config
from src.persona import Persona
from src.tools.tool_loop import build_wire_messages
from src.utils.timeutil import local_now, to_local, to_utc


@pytest.fixture(autouse=True)
def eastern(monkeypatch):
    monkeypatch.setattr(global_config, "LOCAL_TZ", "America/New_York")


def test_local_now_follows_local_tz(monkeypatch):
    assert local_now().tzname() in ("EDT", "EST")
    monkeypatch.setattr(global_config, "LOCAL_TZ", "Asia/Tokyo")
    assert local_now().utcoffset() == timedelta(hours=9)


def test_to_utc_relabels_naive_and_converts_aware():
    assert to_utc(datetime(2026, 10, 2, 1, 45)) == datetime(2026, 10, 2, 1, 45, tzinfo=timezone.utc)
    aware = datetime(2026, 10, 1, 21, 45, tzinfo=timezone(timedelta(hours=-4)))
    assert to_utc(aware) == datetime(2026, 10, 2, 1, 45, tzinfo=timezone.utc)


@pytest.mark.parametrize("stored", [
    datetime(2026, 10, 2, 1, 45),                       # naive UTC (CURRENT_TIMESTAMP)
    datetime(2026, 10, 2, 1, 45, tzinfo=timezone.utc),  # aware UTC (platform stamp)
    "2026-10-02 01:45:00",                              # SQLite text
    "2026-10-02T01:45:00+00:00",
    b"2026-10-02T01:45:00",
])
def test_to_local_renders_every_stored_shape_as_eastern(stored):
    # 01:45 UTC on Oct 2 is still Oct 1 in Eastern — the date itself differs.
    assert to_local(stored).strftime("%Y-%m-%d %H:%M %Z") == "2026-10-01 21:45 EDT"


def test_prompt_current_time_is_local_tz_not_host_clock(monkeypatch):
    monkeypatch.setattr(global_config, "LOCAL_TZ", "Asia/Tokyo")
    persona = Persona(persona_name="t", model_name="mock", prompt="p", inject_timestamp=True)
    system = build_wire_messages(persona, [])[0]["content"]
    assert system.startswith("[Current Time: ")
    assert " JST]" in system
