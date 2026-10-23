import sys

import numpy as np
import pandas as pd
import pytest

from alpha_research import price_forecasting as forecasting


def _ohlcv(index, close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.linspace(1000, 2000, len(close)),
        },
        index=index,
    )


def test_configured_forecast_tickers_combines_and_deduplicates_groups(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "universe:\n"
        "  core: [aapl, SPY]\n"
        "  benchmarks: [QQQ, SPY]\n"
        "external_tickers: [MSFT, AAPL]\n",
        encoding="utf-8",
    )

    assert forecasting._configured_forecast_tickers(config_path) == [
        "AAPL",
        "MSFT",
        "QQQ",
        "SPY",
    ]


def test_train_all_configured_calls_trainer_for_each_ticker(monkeypatch):
    trained_tickers = []
    monkeypatch.setattr(sys, "argv", [
        "price_forecasting",
        "--train-all-configured",
        "--train-profile",
        "lean",
    ])
    monkeypatch.setattr(
        forecasting, "_configured_forecast_tickers", lambda: ["AAPL", "SPY"]
    )

    def record_training(ticker, **kwargs):
        trained_tickers.append((ticker, kwargs["profile"]))
        return {"manifest_path": f"models/{ticker}_manifest.json"}

    monkeypatch.setattr(forecasting, "train_and_save_best_models", record_training)

    forecasting.main()

    assert trained_tickers == [("AAPL", "lean"), ("SPY", "lean")]


def test_forecast_models_available_checks_for_compatible_bundle(monkeypatch):
    monkeypatch.setattr(
        forecasting,
        "_load_saved_forecast_bundle",
        lambda ticker, output_dir=None: {"ticker": ticker} if ticker == "SAVED" else None,
    )

    assert forecasting.forecast_models_available("SAVED")
    assert not forecasting.forecast_models_available("MISSING")


@pytest.mark.parametrize("use_saved_model", [False, True])
def test_daily_forecast_uses_latest_complete_features(monkeypatch, use_saved_model):
    index = pd.bdate_range("2024-01-01", periods=500)
    close = 100 * np.exp(np.cumsum(np.random.default_rng(7).normal(0.0003, 0.01, len(index))))
    daily = _ohlcv(index, close)
    prediction_dates = []

    class ConstantRegressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y, sample_weight=None):
            return self

        def predict(self, X):
            prediction_dates.append(X.index[-1])
            return np.zeros(len(X))

    monkeypatch.setattr(forecasting, "_load_ohlcv", lambda **kwargs: daily)
    monkeypatch.setattr(forecasting, "_benchmark_ticker_for", lambda ticker: None)
    monkeypatch.setattr(forecasting, "HistGradientBoostingRegressor", ConstantRegressor)
    saved_bundle = {
        "daily_artifact": {
            "model": ConstantRegressor(),
            "feature_columns": forecasting.DAILY_FEATURE_COLUMNS,
            "target": "target_return",
        },
        "manifest": {
            "daily": {
                "walk_forward_metrics": {
                    "mae": 0.0,
                    "rmse": 0.0,
                    "mape": 0.0,
                    "directional_acc": 1.0,
                    "windows": 1.0,
                },
                "training_rows": 1,
            }
        },
    }
    monkeypatch.setattr(
        forecasting,
        "_load_saved_forecast_bundle",
        lambda ticker: saved_bundle if use_saved_model else None,
    )
    monkeypatch.setattr(
        forecasting,
        "_walk_forward_metrics",
        lambda **kwargs: {
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "directional_acc": 1.0,
            "windows": 1.0,
        },
    )

    forecasting.forecast_next_close_ml("TEST")

    expected_latest_date = forecasting._to_ny_datetime_index(daily).index[-1]
    assert prediction_dates[-1] == expected_latest_date


def test_daily_features_do_not_backfill_benchmark_prices():
    index = pd.bdate_range("2024-01-01", periods=80)
    asset = _ohlcv(index, np.linspace(100, 120, len(index)))
    benchmark = _ohlcv(index[5:], np.linspace(400, 420, len(index) - 5))

    features = forecasting._build_daily_features(asset, benchmark)

    assert np.isnan(features["Benchmark_Return_1D"].iloc[0])


def test_walk_forward_price_reference_reports_price_unit_errors():
    class ZeroRegressor:
        def fit(self, X, y, sample_weight=None):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    index = pd.RangeIndex(6)
    X = pd.DataFrame({"feature": np.arange(6)}, index=index)
    y = pd.Series(np.full(6, 0.01), index=index)

    metrics = forecasting._walk_forward_metrics(
        X,
        y,
        model_factory=ZeroRegressor,
        train_min=2,
        step=2,
        metric_price_ref=pd.Series(np.full(6, 100.0), index=index),
    )

    assert metrics["mae"] == 1.0
    assert metrics["rmse"] == 1.0


def test_intraday_training_targets_measure_future_session_extremes():
    index = pd.date_range("2026-10-01 09:30", periods=25, freq="5min")
    close = np.linspace(100.0, 102.0, len(index))
    intraday = _ohlcv(index, close)
    intraday.loc[index[12], "High"] = 105.0
    intraday.loc[index[18], "Low"] = 97.0

    features, targets = forecasting._build_intraday_training_data(intraday)

    current_close = close[5]
    assert targets.iloc[0]["high_to_go_return"] == pytest.approx(105.0 / current_close - 1.0)
    assert targets.iloc[0]["low_to_go_return"] == pytest.approx(97.0 / current_close - 1.0)
    assert len(features) == len(targets)


@pytest.mark.parametrize("use_saved_model", [False, True])
def test_intraday_forecast_returns_session_low_and_high(monkeypatch, use_saved_model):
    sessions = []
    for day_number, date in enumerate(pd.bdate_range("2026-09-14", periods=10)):
        index = pd.date_range(date + pd.Timedelta(hours=9, minutes=30), periods=30, freq="5min")
        close = np.linspace(100.0 + day_number, 101.0 + day_number, len(index))
        sessions.append(_ohlcv(index, close))
    intraday = pd.concat(sessions)

    class FixedRegressor:
        def __init__(self, value):
            self.value = value

        def fit(self, X, y, sample_weight=None):
            self.value = float(np.mean(y))
            return self

        def predict(self, X):
            return np.full(len(X), self.value)

    current_price = float(intraday["Close"].iloc[-1] + 0.1)
    bundle = {
        "intraday_artifact": {
            "model": FixedRegressor(0.01),
            "low_model": FixedRegressor(-0.02),
            "high_model": FixedRegressor(0.03),
            "feature_columns": [
                "open_to_now", "range_so_far", "price_vs_vwap",
                "cum_volume", "progress", "vol_so_far",
            ],
        },
        "manifest": {"intraday": {"walk_forward_metrics": {}, "training_rows": 240}},
    }
    monkeypatch.setattr(forecasting, "_load_ohlcv", lambda **kwargs: intraday)
    monkeypatch.setattr(
        forecasting,
        "_load_saved_forecast_bundle",
        lambda ticker: bundle if use_saved_model else None,
    )
    if use_saved_model:
        monkeypatch.setattr(
            forecasting,
            "_build_intraday_training_data",
            lambda _: pytest.fail("Saved models should skip training-data preparation"),
        )
    monkeypatch.setattr(forecasting, "_load_current_price_1m", lambda *args, **kwargs: current_price)
    monkeypatch.setattr(
        forecasting,
        "_walk_forward_metrics",
        lambda **kwargs: {
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "directional_acc": 1.0,
            "windows": 1.0,
        },
    )
    monkeypatch.setattr(
        forecasting,
        "HistGradientBoostingRegressor",
        lambda **kwargs: FixedRegressor(0.0),
    )

    result = forecasting.forecast_intraday_close("TEST")
    observed_session = intraday.iloc[-30:]

    assert result.predicted_low <= float(observed_session["Low"].min())
    assert result.predicted_low <= result.predicted_close
    assert result.predicted_high >= float(observed_session["High"].max())
    assert result.predicted_high >= result.predicted_close


def test_lean_training_profile_uses_sparse_single_candidate_search(monkeypatch, tmp_path):
    intraday_index = pd.date_range("2026-01-01", periods=300, freq="5min")
    intraday_features = pd.DataFrame(
        {"feature": np.linspace(0.0, 1.0, len(intraday_index))},
        index=intraday_index,
    )
    intraday_ohlcv = _ohlcv(
        intraday_index,
        np.linspace(100.0, 120.0, len(intraday_index)),
    )
    intraday_targets = pd.DataFrame(
        {
            "close_to_go_return": np.linspace(-0.01, 0.01, len(intraday_index)),
            "low_to_go_return": np.linspace(-0.02, 0.0, len(intraday_index)),
            "high_to_go_return": np.linspace(0.0, 0.02, len(intraday_index)),
        },
        index=intraday_index,
    )
    daily_ohlcv = _ohlcv(
        pd.bdate_range("2025-01-01", periods=200),
        np.linspace(100.0, 120.0, 200),
    )
    daily_index = forecasting._to_ny_datetime_index(daily_ohlcv).index
    daily_features = pd.DataFrame(
        {"feature": np.linspace(0.0, 1.0, len(daily_index))},
        index=daily_index,
    )
    search_calls = []

    class ConstantRegressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y, sample_weight=None):
            return self

    def record_search(**kwargs):
        search_calls.append(kwargs)
        return (
            {"max_depth": 4, "learning_rate": 0.04, "max_iter": 250},
            {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "directional_acc": 1.0, "windows": 1.0},
        )

    monkeypatch.setattr(
        forecasting,
        "_load_ohlcv",
        lambda ticker, interval, **kwargs: (
            daily_ohlcv if interval == "1d" else intraday_ohlcv
        ),
    )
    monkeypatch.setattr(
        forecasting,
        "_build_intraday_training_data",
        lambda _: (intraday_features, intraday_targets),
    )
    monkeypatch.setattr(
        forecasting,
        "_build_daily_features",
        lambda *_args, **_kwargs: daily_features,
    )
    monkeypatch.setattr(forecasting, "_benchmark_ticker_for", lambda _: None)
    monkeypatch.setattr(forecasting, "_search_best_hgb_params", record_search)
    monkeypatch.setattr(forecasting, "HistGradientBoostingRegressor", ConstantRegressor)
    monkeypatch.setattr(forecasting, "dump", lambda *_args, **_kwargs: None)

    forecasting.train_and_save_best_models(
        "TEST",
        output_dir=tmp_path,
        profile="lean",
        verbose=False,
    )

    assert [call["step"] for call in search_calls] == [500, 50]
    assert [len(call["max_depth_grid"]) for call in search_calls] == [1, 1]