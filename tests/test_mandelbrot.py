import datetime

import pandas as pd

from risk_modeling.mandelbrot import _get_session_open, _resolve_day_baseline


def test_session_open_prefers_official_daily_open_over_premarket_bar():
    intraday_data = pd.DataFrame(
        {"Open": [753.50, 754.50], "Close": [754.00, 755.00]},
        index=pd.to_datetime(["2026-07-15 12:00", "2026-07-15 13:30"], utc=True),
    )
    daily_data = pd.DataFrame(
        {"Open": [754.24], "Close": [753.01]},
        index=pd.to_datetime(["2026-07-15"], utc=True),
    )

    session_open = _get_session_open(intraday_data, daily_data, datetime.date(2026, 7, 15))

    assert session_open == 754.24


def test_day_baseline_uses_prev_close_before_market_open():
    ts = pd.Timestamp("2026-07-15 09:29:00", tz="America/New_York")

    baseline = _resolve_day_baseline(100.0, 101.5, ts)

    assert baseline == 100.0


def test_day_baseline_uses_session_open_when_market_is_open():
    ts = pd.Timestamp("2026-07-15 09:31:00", tz="America/New_York")

    baseline = _resolve_day_baseline(100.0, 101.5, ts)

    assert baseline == 101.5