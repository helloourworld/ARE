import sys
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from ..data_pipeline import data_cache
except ImportError:
    try:
        from data_pipeline import data_cache
    except ImportError:
        package_root = Path(__file__).resolve().parent.parent
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from data_pipeline import data_cache


class DummyYF:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def download(self, ticker, period=None, interval=None, start=None, prepost=True, progress=False):
        self.calls.append({"ticker": ticker, "period": period,
                          "interval": interval, "start": start})
        return self.payload


def test_get_data_persistent_uses_existing_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()

    existing = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [
            9.0], "Close": [10.5], "Volume": [1000]},
        index=pd.to_datetime(["2024-01-01 09:30:00"]),
    )
    existing.to_csv(cache_dir / "cache_AAPL_1d.csv")

    monkeypatch.setattr(data_cache, "DATA_DIR", cache_dir)
    monkeypatch.setattr(data_cache, "REPO_ROOT", tmp_path)

    monkeypatch.setattr(data_cache, "yf", DummyYF(pd.DataFrame()))

    result = data_cache.get_data_persistent("AAPL", interval="1d", period="2y")

    assert not result.empty
    assert list(result.columns)[:5] == [
        "Open", "High", "Low", "Close", "Volume"]
    assert result.index.tz is None


def test_get_data_persistent_initial_daily_download_uses_two_years(tmp_path, monkeypatch):
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()
    yahoo = DummyYF(pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.5], "Volume": [1000]},
        index=pd.to_datetime(["2024-01-01"]),
    ))

    monkeypatch.setattr(data_cache, "DATA_DIR", cache_dir)
    monkeypatch.setattr(data_cache, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(data_cache, "yf", yahoo)
    monkeypatch.setattr(data_cache, "IB_FALLBACK_ENABLED", False)

    result = data_cache.get_data_persistent("NEW", interval="1d", period="5d")

    assert not result.empty
    assert yahoo.calls == [{
        "ticker": "NEW",
        "period": "2y",
        "interval": "1d",
        "start": None,
    }]


def test_get_daily_returns_includes_start_date_return_and_excludes_today(monkeypatch):
    dates = pd.date_range("2024-01-02", periods=4, freq="D")
    prices = pd.DataFrame(
        {"AAPL": [100.0, 102.0, 101.0, 103.0], "SPY": [200.0, 204.0, 202.0, 206.0]},
        index=dates,
    )

    monkeypatch.setattr(data_cache, "_load_close_series", lambda *args, **kwargs: prices)

    class FixedTimestamp(pd.Timestamp):
        @classmethod
        def now(cls, tz=None):
            return cls("2024-01-05", tz=tz)

    monkeypatch.setattr(data_cache.pd, "Timestamp", FixedTimestamp)

    returns = data_cache.get_daily_returns(["AAPL"], "SPY", "2024-01-03")

    assert returns.index.tolist() == dates[1:3].tolist()
    assert abs(returns.loc[dates[1], "AAPL"] - 0.02) < 1e-12
    assert abs(returns.loc[dates[2], "AAPL"] - (-1 / 102)) < 1e-12


def test_get_official_session_open_uses_matching_daily_ohlc_date(monkeypatch):
    daily_data = pd.DataFrame(
        {"Open": [749.84, 754.24]},
        index=pd.to_datetime(["2026-07-14", "2026-07-15"], utc=True),
    )
    calls = []

    def get_daily_data(*args, **kwargs):
        calls.append(kwargs)
        return daily_data

    monkeypatch.setattr(data_cache, "get_data_persistent", get_daily_data)

    open_price = data_cache.get_official_session_open("SPY", "2026-07-15")

    assert open_price == 754.24
    assert calls[0]["force_refresh"] is False


def test_get_daily_returns_refreshes_a_cache_that_does_not_reach_start_date(monkeypatch):
    cached_prices = pd.DataFrame(
        {"AAPL": [101.0], "SPY": [201.0]},
        index=pd.to_datetime(["2024-01-04"]),
    )
    refreshed_prices = pd.DataFrame(
        {"AAPL": [100.0, 101.0, 102.0], "SPY": [200.0, 201.0, 202.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    force_refresh_values = []

    def load_prices(*args, force_refresh, **kwargs):
        force_refresh_values.append(force_refresh)
        return refreshed_prices if force_refresh else cached_prices

    class FixedTimestamp(pd.Timestamp):
        @classmethod
        def now(cls, tz=None):
            return cls("2024-01-05", tz=tz)

    monkeypatch.setattr(data_cache, "_load_close_series", load_prices)
    monkeypatch.setattr(data_cache.pd, "Timestamp", FixedTimestamp)

    returns = data_cache.get_daily_returns(["AAPL"], "SPY", "2024-01-03")

    assert force_refresh_values == [False, True]
    assert returns.index.tolist() == refreshed_prices.index[1:].tolist()


def test_get_daily_returns_keeps_valid_symbols_when_one_column_is_all_nan(monkeypatch):
    dates = pd.date_range("2024-01-02", periods=4, freq="D")
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 102.0, 101.0, 103.0],
            "SPY": [200.0, 204.0, 202.0, 206.0],
            "BROKEN": [np.nan, np.nan, np.nan, np.nan],
        },
        index=dates,
    )

    monkeypatch.setattr(data_cache, "_load_close_series", lambda *args, **kwargs: prices)

    class FixedTimestamp(pd.Timestamp):
        @classmethod
        def now(cls, tz=None):
            return cls("2024-01-05", tz=tz)

    monkeypatch.setattr(data_cache.pd, "Timestamp", FixedTimestamp)

    returns = data_cache.get_daily_returns(["AAPL", "BROKEN"], "SPY", "2024-01-03")

    assert not returns.empty
    assert "BROKEN" not in returns.columns
    assert "AAPL" in returns.columns
    assert "SPY" in returns.columns


def test_normalize_daily_index_converts_legacy_2000_et_rows_to_session_dates():
    legacy = pd.DataFrame(
        {
            "Close": [975.80, 876.89, 889.60],
            "Open": [975.80, 975.80, 975.80],
        },
        index=pd.to_datetime(
            [
                "2026-07-14 20:00:00-04:00",
                "2026-07-15 20:00:00-04:00",
                "2026-07-15 00:00:00-04:00",
            ]
        ),
    )

    normalized = data_cache._normalize_yf_df(legacy, interval="1d")

    assert normalized.index.tz is None
    assert normalized.index.tolist() == [
        pd.Timestamp("2026-07-15"),
        pd.Timestamp("2026-07-16"),
    ]
