# src/utils/timeutil.py
"""One answer to "what time is it" and "what time was that" (DP-413).

Storage is UTC: aware datetimes, or naive ones that are UTC by fact (SQLite
CURRENT_TIMESTAMP, `MemoryManager._utc_stamp`). Anything shown to a person or a
model is LOCAL_TZ. Neither side reads the host clock's zone, so a container
that comes up in UTC cannot make two times disagree.
"""

from datetime import datetime, timezone
from typing import Union
from zoneinfo import ZoneInfo

from config import global_config


def local_now() -> datetime:
    """Now in LOCAL_TZ."""
    return datetime.now(ZoneInfo(global_config.LOCAL_TZ))


def to_utc(ts: datetime) -> datetime:
    """Aware UTC. A naive value is a stored timestamp, which is UTC."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def to_local(ts: Union[datetime, str, bytes]) -> datetime:
    """A stored timestamp (datetime or its ISO text) in LOCAL_TZ, for display."""
    if isinstance(ts, bytes):
        ts = ts.decode('utf-8')
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    return to_utc(ts).astimezone(ZoneInfo(global_config.LOCAL_TZ))
