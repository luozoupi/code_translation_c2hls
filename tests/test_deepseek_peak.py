from datetime import datetime
from zoneinfo import ZoneInfo
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "pc2"))
from deepseek_peak import is_beijing_peak, seconds_until_off_peak

TZ = ZoneInfo("Asia/Shanghai")


def test_morning_peak():
    assert is_beijing_peak(datetime(2026, 7, 17, 9, 0, tzinfo=TZ))
    assert is_beijing_peak(datetime(2026, 7, 17, 11, 59, tzinfo=TZ))
    assert not is_beijing_peak(datetime(2026, 7, 17, 12, 0, tzinfo=TZ))


def test_afternoon_peak():
    assert is_beijing_peak(datetime(2026, 7, 17, 14, 0, tzinfo=TZ))
    assert is_beijing_peak(datetime(2026, 7, 17, 17, 59, tzinfo=TZ))
    assert not is_beijing_peak(datetime(2026, 7, 17, 18, 0, tzinfo=TZ))


def test_off_peak_night():
    assert not is_beijing_peak(datetime(2026, 7, 17, 2, 0, tzinfo=TZ))


def test_seconds_until_off_peak_positive_in_peak():
    assert seconds_until_off_peak(datetime(2026, 7, 17, 10, 0, tzinfo=TZ)) == 2 * 3600


def test_seconds_until_off_peak_zero_when_not_peak():
    assert seconds_until_off_peak(datetime(2026, 7, 17, 13, 0, tzinfo=TZ)) == 0.0


def test_naive_datetime_treated_as_beijing():
    assert is_beijing_peak(datetime(2026, 7, 17, 9, 30))
    assert not is_beijing_peak(datetime(2026, 7, 17, 13, 0))
