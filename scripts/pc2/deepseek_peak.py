"""DeepSeek API peak windows in Beijing (Asia/Shanghai).

Beijing/China time historically sees elevated DeepSeek API latency/rate
limiting during local daytime "peak" hours. These helpers let batch
campaigns pause codegen work during those windows and resume once
off-peak.

Peak windows (local Asia/Shanghai time, [start, end)):
    09:00-12:00
    14:00-18:00
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

BEIJING = ZoneInfo("Asia/Shanghai")
# [start_hour, end_hour) local Beijing time.
PEAK_WINDOWS = ((9, 12), (14, 18))


def _to_beijing(now: datetime | None) -> datetime:
    """Normalize `now` (or the current time) to an aware Asia/Shanghai datetime."""
    if now is None:
        return datetime.now(BEIJING)
    if now.tzinfo is None:
        return now.replace(tzinfo=BEIJING)
    return now.astimezone(BEIJING)


def is_beijing_peak(now: datetime | None = None) -> bool:
    """Return True if `now` (default: current time) falls in a Beijing peak window."""
    dt = _to_beijing(now)
    h = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    for start, end in PEAK_WINDOWS:
        if start <= h < end:
            return True
    return False


def seconds_until_off_peak(now: datetime | None = None) -> float:
    """Seconds remaining until the current/enclosing peak window ends.

    Returns 0.0 if `now` is not in a peak window.
    """
    dt = _to_beijing(now)
    h = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    for start, end in PEAK_WINDOWS:
        if start <= h < end:
            end_dt = dt.replace(hour=end % 24, minute=0, second=0, microsecond=0)
            if end <= start:
                end_dt += timedelta(days=1)
            return max(0.0, (end_dt - dt).total_seconds())
    return 0.0


def sleep_hint_sec(now: datetime | None = None, *, max_sleep: float = 300.0) -> float:
    """Suggested sleep duration (seconds) while waiting out a peak window.

    Capped at `max_sleep` so callers can re-check periodically instead of
    sleeping through the entire remaining peak window in one shot.
    """
    rem = seconds_until_off_peak(now)
    if rem <= 0:
        return 0.0
    return min(rem, max_sleep)
