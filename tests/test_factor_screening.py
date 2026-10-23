import pytest

from alpha_research import factor_screening


def test_screen_uses_strict_thresholds_and_reports_each_ticker(monkeypatch):
    fundamentals = {
        "PASS.TO": {
            "shortName": "Passing Utility",
            "sector": "Utilities",
            "dividendYield": 0.05,
            "returnOnEquity": 0.16,
            "debtToEquity": 99.0,
        },
        "FAIL.TO": {
            "shortName": "Borderline Utility",
            "sector": "Utilities",
            "dividendYield": 0.04,
            "returnOnEquity": 0.15,
            "debtToEquity": 100.0,
        },
    }

    class FakeTicker:
        def __init__(self, ticker):
            self.info = fundamentals[ticker]

    monkeypatch.setattr(factor_screening.yf, "Ticker", FakeTicker)

    result = factor_screening.screen_canadian_utility_stocks(
        ["PASS.TO", "FAIL.TO", "PASS.TO"]
    ).set_index("Ticker")

    assert result.index.tolist() == ["PASS.TO", "FAIL.TO"]
    assert result.loc["PASS.TO", "Passes"]
    assert result.loc["PASS.TO", "Dividend Yield (%)"] == pytest.approx(5.0)
    assert not result.loc["FAIL.TO", "Passes"]
    assert "股息率" in result.loc["FAIL.TO", "Reason"]
    assert "ROE" in result.loc["FAIL.TO", "Reason"]
    assert "负债/权益" in result.loc["FAIL.TO", "Reason"]


def test_screen_reports_lookup_errors_and_rejects_invalid_thresholds(monkeypatch):
    class FailedTicker:
        @property
        def info(self):
            raise RuntimeError("request failed")

    monkeypatch.setattr(factor_screening.yf, "Ticker", lambda _ticker: FailedTicker())

    result = factor_screening.screen_canadian_utility_stocks(["FAIL.TO"])
    assert result.loc[0, "Error"] == "RuntimeError: request failed"
    assert not result.loc[0, "Passes"]

    with pytest.raises(ValueError, match="max_debt_to_equity_pct"):
        factor_screening.screen_canadian_utility_stocks(
            ["FAIL.TO"], max_debt_to_equity_pct=float("nan")
        )
