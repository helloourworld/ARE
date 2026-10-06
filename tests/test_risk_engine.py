import numpy as np
import pandas as pd

from risk_modeling import rs_trend
from risk_modeling.risk_engine import AlphaRiskEngine


def test_annualized_shortfall_returns_nan_empirical_metric_without_21_sessions():
    engine = AlphaRiskEngine(tickers=["AAPL"], benchmark="SPY")
    short_history = pd.DataFrame(
        {"AAPL": [0.01, -0.02, 0.005]},
        index=pd.date_range("2026-07-10", periods=3, freq="D"),
    )
    engine.ingest_data = lambda: short_history

    result = engine.calculate_annualized_shortfall()

    assert not np.isnan(result["AAPL"]["Daily_ES"])
    assert np.isnan(result["AAPL"]["Empirical_Annual_ES"])


def test_get_rs_signals_excludes_benchmark_ticker(monkeypatch):
    index = pd.date_range("2024-01-01", periods=60, freq="D")
    data = pd.DataFrame(
        {
            "SPY": np.linspace(100, 110, 60),
            "AAPL": np.linspace(90, 120, 60),
        },
        index=index,
    )

    monkeypatch.setattr(rs_trend, "get_price_history_with_benchmark", lambda *args, **kwargs: data)

    result = rs_trend.get_rs_signals(["SPY", "AAPL"], benchmark="SPY")

    assert "SPY" not in result["Ticker"].tolist()
    assert "AAPL" in result["Ticker"].tolist()