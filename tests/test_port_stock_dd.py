import pandas as pd
import pytest

from alpha_research import Port_Stock_dd


def test_current_drawdown_is_unfiltered_and_matches_each_ticker(monkeypatch):
    prices = {
        "SPY": pd.DataFrame(
            {"Close": [100.0, 99.0]},
            index=pd.to_datetime(["2026-10-21", "2026-10-22"]),
        ),
        "QQQ": pd.DataFrame(
            {"Close": [200.0, 194.0]},
            index=pd.to_datetime(["2026-10-21", "2026-10-22"]),
        ),
    }
    monkeypatch.setattr(
        Port_Stock_dd,
        "get_data_persistent",
        lambda ticker, **_kwargs: prices[ticker],
    )
    monkeypatch.setattr(Port_Stock_dd, "get_current_price", lambda _ticker: None)
    monkeypatch.setattr(Port_Stock_dd, "get_current_quote", lambda _ticker: None)

    stats = Port_Stock_dd.calculate_drawdown_analysis(
        ["SPY", "QQQ"], threshold=0.02
    )["stats"].set_index("Ticker")

    assert stats.loc["SPY", "Current Drawdown"] == pytest.approx(-0.01)
    assert stats.loc["QQQ", "Current Drawdown"] == pytest.approx(-0.03)
    assert stats.loc["SPY", "Average Drawdown"] == pytest.approx(0.0)
    assert stats.loc["QQQ", "Average Drawdown"] == pytest.approx(-0.03)


def test_live_price_is_separate_from_cached_history():
    close = pd.Series(
        [140.94, 92.07],
        index=pd.to_datetime(["2026-06-22", "2026-10-29"]),
    )

    historical_drawdown = close / close.cummax() - 1
    live_drawdown = 64.72 / close.max() - 1

    assert close.iloc[-1] == 92.07
    assert historical_drawdown.iloc[-1] == pytest.approx(-0.3467432950191571, abs=1e-6)
    assert live_drawdown == pytest.approx(-0.5407975024833263, abs=1e-6)
