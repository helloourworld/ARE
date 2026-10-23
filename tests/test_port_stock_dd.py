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

    stats = Port_Stock_dd.calculate_drawdown_analysis(
        ["SPY", "QQQ"], threshold=0.02
    )["stats"].set_index("Ticker")

    assert stats.loc["SPY", "Current Drawdown"] == pytest.approx(-0.01)
    assert stats.loc["QQQ", "Current Drawdown"] == pytest.approx(-0.03)
