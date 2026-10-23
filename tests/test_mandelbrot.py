import datetime

import numpy as np
import pandas as pd
import pytest

from risk_modeling.fragility import calculate_fragility_score as calculate_core_fragility_score
from risk_modeling.mandelbrot import (
    _fragility_config_for_ticker,
    _get_session_open,
    _resolve_day_baseline,
    calculate_stress_score,
    calculate_historical_pivot_data,
    scan_market,
)


def test_spcx_fragility_scores_when_history_is_below_63_observations():
    assert _fragility_config_for_ticker("SPCX", 91).min_observations == 91
    assert _fragility_config_for_ticker("SPCX", 50).min_observations == 50
    assert _fragility_config_for_ticker("SPCX", 200).min_observations == 126
    assert _fragility_config_for_ticker("SPY", 91).min_observations == 126

    history = np.linspace(-0.1, 0.1, 50)
    result = calculate_core_fragility_score(
        returns_1m=np.zeros(10),
        cmf_history=-history,
        breadth_slope_history=-history,
        hurst_history=np.full(50, 0.5),
        concentration_history=np.zeros(50),
        drawdown_accel_history=-history,
        config=_fragility_config_for_ticker("SPCX", len(history)),
    )

    assert np.isfinite(result["score"])


def test_stress_score_adapts_to_available_history_for_any_ticker():
    dates = pd.date_range("2026-01-01", periods=90, freq="B")
    changes = 0.001 * np.sin(np.arange(90) / 3) + 0.0005
    close = 100 * np.exp(np.cumsum(changes))
    daily = pd.DataFrame({"Close": close}, index=dates)
    vix = pd.DataFrame({"Close": 20 + np.arange(90) / 10}, index=dates)
    high_yield = pd.DataFrame(
        {"BAMLH0A0HYM2": 4 + np.arange(90) / 100},
        index=dates,
    )
    investment_grade = pd.DataFrame(
        {"BAMLC0A0CM": 1 + np.arange(90) / 200},
        index=dates,
    )

    result = calculate_stress_score(
        daily,
        daily,
        vix_df=vix,
        credit_spread_df=high_yield,
        ig_spread_df=investment_grade,
    )

    assert np.isfinite(result["score"])
    assert result["observations"]["minimum_required"] == 88

    with pytest.raises(
        ValueError,
        match="Insufficient aligned observations for stress scoring: .*required at least 20",
    ):
        calculate_stress_score(
            daily.tail(15),
            daily.tail(15),
            vix_df=vix,
            credit_spread_df=high_yield,
            ig_spread_df=investment_grade,
        )


def test_historical_pivots_and_after_close_earnings_levels():
    daily = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0, 13.0],
            "High": [11.0, 13.0, 14.0, 15.0],
            "Low": [9.0, 10.0, 11.0, 12.0],
            "Close": [10.0, 12.0, 12.0, 14.0],
        },
        index=pd.to_datetime(
            [
                "2026-01-01",
                "2026-01-02",
                "2026-01-05",
                "2026-01-06",
            ],
            utc=True,
        ),
    )
    earnings_date = pd.DatetimeIndex(
        ["2026-01-02 16:00"],
        tz="America/New_York",
    )

    history = calculate_historical_pivot_data(daily, earnings_date)

    assert history.loc["2026-01-02", "R1"] == pytest.approx(11.0)
    assert history.loc["2026-01-02", "S1"] == pytest.approx(9.0)
    assert history.loc["2026-01-05", "Earnings Event"]
    assert history.loc["2026-01-05", "Pre-Earnings High"] == 13.0
    assert history.loc["2026-01-05", "Pre-Earnings Low"] == 10.0


def test_historical_drawdowns_mark_only_troughs_at_five_percent_or_more():
    closes = [100.0, 95.0, 90.0, 92.0, 100.0, 95.0, 94.0, 93.0]
    daily = pd.DataFrame(
        {
            "Open": closes,
            "High": [value + 1 for value in closes],
            "Low": [value - 1 for value in closes],
            "Close": closes,
        },
        index=pd.date_range("2026-01-01", periods=len(closes), freq="D"),
    )

    history = calculate_historical_pivot_data(daily)
    troughs = history.loc[history["Drawdown Trough"]]

    assert troughs.index.tolist() == [
        pd.Timestamp("2026-01-03"),
        pd.Timestamp("2026-01-08"),
    ]
    assert troughs["Drawdown Trough %"].tolist() == pytest.approx([-10.0, -7.0])
    assert not history.loc["2026-01-02", "Drawdown Trough"]
    assert not history.loc["2026-01-04", "Drawdown Trough"]
    assert history.iloc[-1]["Drawdown"] == pytest.approx(-0.07)
    assert history.iloc[-1]["Drawdown Peak"] == pytest.approx(100.0)


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
    liquidity_benchmarks = []

    def capture_liquidity_benchmark(data, benchmark):
        liquidity_benchmarks.append(benchmark)
        return {"cmf": 0.0, "breadth_ratio": 1.0, "breadth_slope": 0.0, "rsi": 50.0}

    def unexpected_data_load(*args, **kwargs):
        raise AssertionError("scan should use the supplied benchmark data")

    stress_inputs = {
        "vix": pd.DataFrame({"Close": [20.0]}),
        "high_yield": pd.DataFrame({"BAMLH0A0HYM2": [3.0]}),
        "investment_grade": pd.DataFrame({"BAMLC0A0CM": [1.0]}),
    }
    stress_arguments = {}

    def capture_stress_inputs(data, benchmark, **kwargs):
        stress_arguments.update(kwargs)
        return None

    monkeypatch.setattr(
        "risk_modeling.mandelbrot.calculate_liquidity_signals",
        capture_liquidity_benchmark,
    )
    monkeypatch.setattr(
        "risk_modeling.mandelbrot.get_data_persistent",
        unexpected_data_load,
    )
    monkeypatch.setattr(
        "risk_modeling.mandelbrot.calculate_stress_score",
        capture_stress_inputs,
    )

    result = scan_market(
        "TEST.TO",
        data_1m=intraday_data,
        data_1d=daily_data,
        check_new_data=False,
        benchmark_data=daily_data,
        stress_data=stress_inputs,
    )

    assert result["Mean Reversion Average"] == 101.5
    assert result["Support (S1)"] == pytest.approx(97.16666666666667)
    assert result["Resistance (R1)"] == pytest.approx(99.16666666666667)
    assert liquidity_benchmarks == [daily_data]
    assert stress_arguments["vix_df"] is stress_inputs["vix"]
    assert stress_arguments["credit_spread_df"] is stress_inputs["high_yield"]
    assert stress_arguments["ig_spread_df"] is stress_inputs["investment_grade"]