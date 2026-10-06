import datetime

import pandas as pd

from risk_modeling.mandelbrot import _get_session_open, _resolve_day_baseline, scan_market


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


def test_scan_market_returns_intraday_mean_reversion_average(monkeypatch):
    intraday_index = pd.date_range("2026-07-15 13:30", periods=60, freq="min", tz="America/New_York")
    intraday_prices = [100.0, 101.0, 102.0, 103.0] + [101.5] * 56
    intraday_data = pd.DataFrame(
        {
            "Open": intraday_prices,
            "Close": intraday_prices,
            "Volume": [1000] * 60,
        },
        index=intraday_index,
    )
    daily_data = pd.DataFrame(
        {
            "Open": [99.0] * 129 + [100.0],
            "High": [99.5] * 129 + [100.5],
            "Low": [97.5] * 129 + [98.5],
            "Close": [98.0] * 129 + [99.0],
            "Volume": [10000] * 130,
        },
        index=pd.date_range("2026-03-08", periods=130, freq="D", tz="UTC"),
    )

    monkeypatch.setattr(
        "risk_modeling.mandelbrot.calculate_hurst_vw",
        lambda prices, volumes, window: 0.50,
    )
    monkeypatch.setattr(
        "risk_modeling.mandelbrot.calculate_tail_index_robust",
        lambda returns, lookback: 2.0,
    )
    monkeypatch.setattr(
        "risk_modeling.mandelbrot.calculate_liquidity_signals",
        lambda data, benchmark: {"cmf": 0.0, "breadth_ratio": 1.0, "breadth_slope": 0.0, "rsi": 50.0},
    )
    monkeypatch.setattr(
        "risk_modeling.mandelbrot.get_data_persistent",
        lambda ticker, interval="1d", period=None: pd.DataFrame(),
    )

    result = scan_market(
        "TEST.TO",
        data_1m=intraday_data,
        data_1d=daily_data,
        check_new_data=False,
    )

    assert result["Mean Reversion Average"] == 101.5