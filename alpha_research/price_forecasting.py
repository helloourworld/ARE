"""
Practical price forecasting tools:

1) Machine-learning predictions
   - Intraday close forecast for the current session using intraday OHLCV state.
   - Next-day close forecast using daily technical indicators.

2) Monte Carlo simulation
   - Probabilistic close-price range using Geometric Brownian Motion (GBM).

Example:
    python -m alpha_research.price_forecasting --ticker AAPL
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

# Avoid loky physical-core detection warning on some Windows setups.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import pandas as pd
import yfinance as yf
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from data_pipeline.data_cache import get_data_persistent


MARKET_TIMEZONE = "America/New_York"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "models"
DAILY_FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_5",
    "SMA_20",
    "Daily_Return",
    "High_Low_Pct",
    "Open_Close_Pct",
]


@dataclass
class IntradayForecastResult:
    ticker: str
    current_price: float
    predicted_close: float
    predicted_return_to_close: float
    training_rows: int
    validation_metrics: Dict[str, float]


@dataclass
class DailyForecastResult:
    ticker: str
    latest_close: float
    predicted_next_close: float
    holdout_mae: float
    holdout_rmse: float
    holdout_mape: float
    holdout_directional_acc: float
    walk_forward_mae: float
    walk_forward_rmse: float
    walk_forward_mape: float
    walk_forward_directional_acc: float
    walk_forward_windows: int
    training_rows: int


@dataclass
class MonteCarloResult:
    ticker: str
    start_price: float
    horizon_days: float
    mu_annual: float
    sigma_annual: float
    mean_terminal_price: float
    quantiles: Dict[str, float]
    probability_above_start: float


@dataclass
class RiskRulesConfig:
    min_edge_pct: float = 0.0025
    min_prob_up: float = 0.60
    min_reward_to_risk: float = 1.30
    skip_edge_pct: float = 0.0015
    max_iqr_pct_for_small_edge: float = 0.02
    max_position_weight: float = 0.05
    max_single_trade_tail_loss_pct: float = 0.005
    daily_tail_risk_budget_pct: float = 0.015
    base_vol_target: float = 0.25
    high_vol_cutoff: float = 0.55
    high_vol_size_multiplier: float = 0.50


@dataclass
class TradeDecision:
    action: str
    recommended_weight: float
    recommended_notional: float
    tail_risk_notional: float
    edge_pct: float
    probability_up: float
    reward_to_risk: float
    reasons: Tuple[str, ...]


@dataclass
class DailyRiskBudgetTracker:
    portfolio_value: float
    daily_budget_pct: float = 0.015
    consumed_tail_risk_notional: float = 0.0

    @property
    def daily_budget_notional(self) -> float:
        return self.portfolio_value * self.daily_budget_pct

    @property
    def remaining_notional(self) -> float:
        return max(0.0, self.daily_budget_notional - self.consumed_tail_risk_notional)

    def can_allocate(self, additional_tail_risk_notional: float) -> bool:
        return additional_tail_risk_notional <= self.remaining_notional

    def register(self, additional_tail_risk_notional: float) -> None:
        self.consumed_tail_risk_notional += max(0.0, additional_tail_risk_notional)


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output columns to a flat OHLCV frame."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    out = out.loc[:, keep]
    out = out.dropna(how="all")
    return out


def _load_ohlcv_from_data_dir(ticker: str, interval: str) -> pd.DataFrame:
    """Read OHLCV directly from local cache CSV as a fallback path."""
    safe_ticker = str(ticker).replace("/", "_").replace("=", "_")
    csv_path = DATA_DIR / f"cache_{safe_ticker}_{interval}.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    return _flatten_yf_columns(df)


def _load_ohlcv(ticker: str, interval: str, period: str, prefer_pipeline: bool = True) -> pd.DataFrame:
    """Prefer project pipeline/cache data, then local CSV cache, then yfinance."""
    if prefer_pipeline:
        pipeline_df = get_data_persistent(ticker, interval=interval, period=period, force_refresh=False)
        ohlcv = _flatten_yf_columns(pipeline_df)
        if not ohlcv.empty:
            return ohlcv

    local_df = _load_ohlcv_from_data_dir(ticker, interval)
    if not local_df.empty:
        return local_df

    raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    return _flatten_yf_columns(raw)


def _to_ny_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize(MARKET_TIMEZONE)
    else:
        idx = idx.tz_convert(MARKET_TIMEZONE)
    out.index = idx
    return out


def _safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / denominator
    out[~np.isfinite(out)] = 0.0
    return out


def _compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}

    err = y_true_arr - y_pred_arr
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))

    denom = np.where(np.abs(y_true_arr) > 1e-12, np.abs(y_true_arr), np.nan)
    mape = float(np.nanmean(np.abs(err) / denom)) if np.any(np.isfinite(denom)) else float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _serialize_metrics(metrics: Dict[str, float]) -> Dict[str, float | None]:
    out: Dict[str, float | None] = {}
    for k, v in metrics.items():
        try:
            v_float = float(v)
        except (TypeError, ValueError):
            out[k] = None
            continue
        out[k] = v_float if np.isfinite(v_float) else None
    return out


def _rank_candidate(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    """Lower is better: RMSE, then MAE, then negative directional accuracy."""
    rmse = float(metrics.get("rmse", float("inf")))
    mae = float(metrics.get("mae", float("inf")))
    directional_acc = float(metrics.get("directional_acc", float("nan")))

    if not np.isfinite(rmse):
        rmse = float("inf")
    if not np.isfinite(mae):
        mae = float("inf")
    if not np.isfinite(directional_acc):
        directional_acc = -1.0

    return rmse, mae, -directional_acc


def _search_best_hgb_params(
    X: pd.DataFrame,
    y: pd.Series,
    train_min: int,
    step: int,
    direction_ref: pd.Series | None,
    max_depth_grid: Tuple[int, ...],
    learning_rate_grid: Tuple[float, ...],
    max_iter_grid: Tuple[int, ...],
    verbose: bool = False,
    label: str = "model",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    best_params: Dict[str, float] | None = None
    best_metrics: Dict[str, float] | None = None
    best_rank: Tuple[float, float, float] | None = None

    total = len(max_depth_grid) * len(learning_rate_grid) * len(max_iter_grid)
    candidate_idx = 0

    for max_depth in max_depth_grid:
        for learning_rate in learning_rate_grid:
            for max_iter in max_iter_grid:
                candidate_idx += 1
                params = {
                    "max_depth": int(max_depth),
                    "learning_rate": float(learning_rate),
                    "max_iter": int(max_iter),
                    "random_state": 42,
                }
                if verbose:
                    print(
                        f"[{label}] Candidate {candidate_idx}/{total}: "
                        f"depth={params['max_depth']}, lr={params['learning_rate']}, iter={params['max_iter']}",
                        flush=True,
                    )
                wf_metrics = _walk_forward_metrics(
                    X=X,
                    y=y,
                    model_factory=lambda p=params: HistGradientBoostingRegressor(**p),
                    train_min=train_min,
                    step=step,
                    direction_ref=direction_ref,
                )
                rank = _rank_candidate(wf_metrics)
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_params = params
                    best_metrics = wf_metrics
                    if verbose:
                        rmse, mae, neg_dir = rank
                        print(
                            f"[{label}] New best: rmse={rmse:.6f}, mae={mae:.6f}, dir_acc={-neg_dir:.2%}",
                            flush=True,
                        )

    if best_params is None or best_metrics is None:
        raise ValueError("Could not determine best parameters from walk-forward search.")

    return best_params, best_metrics


def train_and_save_best_models(
    ticker: str,
    intraday_period: str = "60d",
    intraday_interval: str = "5m",
    daily_period: str = "3y",
    output_dir: Path | None = None,
    profile: str = "full",
    use_pipeline: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """Train tuned intraday/daily models and persist artifacts for reuse."""
    profile_name = str(profile).strip().lower()
    if profile_name not in {"fast", "full"}:
        raise ValueError("profile must be either 'fast' or 'full'.")

    out_dir = Path(output_dir) if output_dir is not None else MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if profile_name == "fast":
        intra_depth_grid = (3, 4)
        intra_lr_grid = (0.04, 0.05)
        intra_iter_grid = (260, 360)

        daily_depth_grid = (4, 5)
        daily_lr_grid = (0.03, 0.04)
        daily_iter_grid = (350, 500)
    else:
        intra_depth_grid = (3, 4, 5)
        intra_lr_grid = (0.03, 0.05)
        intra_iter_grid = (300, 450, 600)

        daily_depth_grid = (4, 5, 6)
        daily_lr_grid = (0.03, 0.04, 0.05)
        daily_iter_grid = (450, 700, 900)

    # Intraday training set and parameter search.
    intraday_ohlcv = _load_ohlcv(
        ticker=ticker,
        interval=intraday_interval,
        period=intraday_period,
        prefer_pipeline=use_pipeline,
    )
    if intraday_ohlcv.empty:
        raise ValueError(f"No intraday data returned for {ticker}.")

    X_intra, y_intra = _build_intraday_training_data(intraday_ohlcv)
    if len(X_intra) < 200:
        raise ValueError(
            f"Not enough intraday training rows for {ticker}. Need at least 200, got {len(X_intra)}."
        )

    intra_params, intra_wf_metrics = _search_best_hgb_params(
        X=X_intra,
        y=y_intra,
        train_min=160,
        step=20,
        direction_ref=None,
        max_depth_grid=intra_depth_grid,
        learning_rate_grid=intra_lr_grid,
        max_iter_grid=intra_iter_grid,
        verbose=verbose,
        label=f"{ticker.upper()} intraday",
    )

    intraday_model = HistGradientBoostingRegressor(**intra_params)
    intraday_model.fit(X_intra, y_intra)

    intraday_artifact = {
        "model": intraday_model,
        "feature_columns": list(X_intra.columns),
        "target": "close_to_go_return",
        "interval": intraday_interval,
        "period": intraday_period,
        "params": intra_params,
    }
    intraday_path = out_dir / f"{ticker.upper()}_intraday_close_model.joblib"
    dump(intraday_artifact, intraday_path)

    # Daily training set and parameter search.
    daily_ohlcv = _load_ohlcv(
        ticker=ticker,
        interval="1d",
        period=daily_period,
        prefer_pipeline=use_pipeline,
    )
    if daily_ohlcv.empty:
        raise ValueError(f"No daily data returned for {ticker}.")

    daily_features = _build_daily_features(daily_ohlcv)
    daily_target = daily_ohlcv["Close"].shift(-1)
    daily_dataset = daily_features.copy()
    daily_dataset["target"] = daily_target
    daily_dataset = daily_dataset.dropna()

    if len(daily_dataset) < 150:
        raise ValueError(f"Not enough daily data for ML forecast on {ticker}. Need at least 150 rows.")

    X_daily = daily_dataset[DAILY_FEATURE_COLUMNS]
    y_daily = daily_dataset["target"]

    daily_params, daily_wf_metrics = _search_best_hgb_params(
        X=X_daily,
        y=y_daily,
        train_min=140,
        step=5,
        direction_ref=X_daily["Close"],
        max_depth_grid=daily_depth_grid,
        learning_rate_grid=daily_lr_grid,
        max_iter_grid=daily_iter_grid,
        verbose=verbose,
        label=f"{ticker.upper()} daily",
    )

    daily_model = HistGradientBoostingRegressor(**daily_params)
    daily_model.fit(X_daily, y_daily)

    daily_artifact = {
        "model": daily_model,
        "feature_columns": DAILY_FEATURE_COLUMNS,
        "target": "next_close",
        "interval": "1d",
        "period": daily_period,
        "params": daily_params,
    }
    daily_path = out_dir / f"{ticker.upper()}_daily_next_close_model.joblib"
    dump(daily_artifact, daily_path)

    manifest = {
        "ticker": ticker.upper(),
        "training_profile": profile_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "intraday_model_path": str(intraday_path),
        "daily_model_path": str(daily_path),
        "intraday": {
            "training_rows": int(len(X_intra)),
            "best_params": intra_params,
            "walk_forward_metrics": _serialize_metrics(intra_wf_metrics),
        },
        "daily": {
            "training_rows": int(len(X_daily)),
            "best_params": daily_params,
            "walk_forward_metrics": _serialize_metrics(daily_wf_metrics),
        },
    }
    manifest_path = out_dir / f"{ticker.upper()}_forecast_models_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "ticker": ticker.upper(),
        "intraday_model_path": str(intraday_path),
        "daily_model_path": str(daily_path),
        "manifest_path": str(manifest_path),
        "intraday_best_params": intra_params,
        "daily_best_params": daily_params,
        "intraday_wf_metrics": intra_wf_metrics,
        "daily_wf_metrics": daily_wf_metrics,
    }


def _walk_forward_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    train_min: int,
    step: int,
    direction_ref: pd.Series | None = None,
) -> Dict[str, float]:
    """Rolling train-forward evaluation to reduce optimistic single-split bias."""
    if len(X) <= train_min + 1:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape": float("nan"),
            "directional_acc": float("nan"),
            "windows": 0.0,
        }

    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    dir_true_all: list[float] = []
    dir_pred_all: list[float] = []
    windows = 0

    n = len(X)
    for train_end in range(train_min, n - 1, max(1, step)):
        test_end = min(train_end + step, n)
        if test_end <= train_end:
            continue

        model = model_factory()
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]
        if len(X_test) == 0:
            continue

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        y_true_part = y_test.to_numpy(dtype=float)
        y_pred_part = np.asarray(preds, dtype=float)
        y_true_all.extend(y_true_part.tolist())
        y_pred_all.extend(y_pred_part.tolist())

        if direction_ref is not None:
            ref_part = direction_ref.iloc[train_end:test_end].to_numpy(dtype=float)
            dir_true_all.extend(np.sign(y_true_part - ref_part).tolist())
            dir_pred_all.extend(np.sign(y_pred_part - ref_part).tolist())
        else:
            dir_true_all.extend(np.sign(y_true_part).tolist())
            dir_pred_all.extend(np.sign(y_pred_part).tolist())

        windows += 1

    metrics = _compute_regression_metrics(np.asarray(y_true_all, dtype=float), np.asarray(y_pred_all, dtype=float))
    if len(dir_true_all):
        metrics["directional_acc"] = float(
            np.mean(np.asarray(dir_true_all, dtype=float) == np.asarray(dir_pred_all, dtype=float))
        )
    else:
        metrics["directional_acc"] = float("nan")
    metrics["windows"] = float(windows)
    return metrics


def _load_current_price_1m(ticker: str) -> float | None:
    """Return the freshest known close from 1-minute bars."""
    try:
        one_min = get_data_persistent(ticker, interval="1m", period="2d", force_refresh=True)
        if one_min is not None and not one_min.empty and "Close" in one_min.columns:
            return float(one_min["Close"].dropna().iloc[-1])
    except Exception:
        return None
    return None


def _build_intraday_training_data(intraday_ohlcv: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Create point-in-time intraday features and close-to-go target."""
    if intraday_ohlcv.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    df = _to_ny_datetime_index(intraday_ohlcv)
    df = df.sort_index()
    df["session"] = df.index.date

    feature_rows = []
    targets = []

    for _, day in df.groupby("session"):
        day = day.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        if len(day) < 20:
            continue

        closes = day["Close"].to_numpy(dtype=float)
        highs = day["High"].to_numpy(dtype=float)
        lows = day["Low"].to_numpy(dtype=float)
        volumes = day["Volume"].fillna(0.0).to_numpy(dtype=float)

        cum_high = np.maximum.accumulate(highs)
        cum_low = np.minimum.accumulate(lows)
        cum_vol = np.cumsum(volumes)
        cum_vwap_num = np.cumsum(closes * volumes)
        vwap = _safe_div(cum_vwap_num, np.where(cum_vol == 0.0, np.nan, cum_vol))

        pct = pd.Series(closes).pct_change().fillna(0.0).to_numpy(dtype=float)
        running_vol = pd.Series(pct).expanding(min_periods=5).std().fillna(0.0).to_numpy(dtype=float)

        day_open = float(day["Open"].iloc[0])
        day_final_close = float(closes[-1])
        n = len(day)

        # Use only points before the final bar to avoid leakage.
        for i in range(5, n - 1):
            current_close = closes[i]
            progress = i / (n - 1)

            feature_rows.append(
                {
                    "open_to_now": (current_close / day_open) - 1.0,
                    "range_so_far": (cum_high[i] / max(cum_low[i], 1e-12)) - 1.0,
                    "price_vs_vwap": (current_close / max(vwap[i], 1e-12)) - 1.0,
                    "cum_volume": np.log1p(cum_vol[i]),
                    "progress": progress,
                    "vol_so_far": running_vol[i],
                }
            )
            targets.append((day_final_close / current_close) - 1.0)

    X = pd.DataFrame(feature_rows)
    y = pd.Series(targets, dtype=float)
    return X, y


def forecast_intraday_close(ticker: str, intraday_period: str = "60d", interval: str = "5m") -> IntradayForecastResult:
    """Forecast today's close from current intraday state using learned intraday patterns."""
    ohlcv = _load_ohlcv(ticker=ticker, interval=interval, period=intraday_period)
    if ohlcv.empty:
        raise ValueError(f"No intraday data returned for {ticker}.")

    X, y = _build_intraday_training_data(ohlcv)
    if len(X) < 200:
        raise ValueError(
            f"Not enough intraday training rows for {ticker}. Need at least 200, got {len(X)}."
        )

    split = int(len(X) * 0.8)
    split = min(max(split, 120), len(X) - 40)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model_eval = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400, random_state=42)
    model_eval.fit(X_train, y_train)

    pred_test = model_eval.predict(X_test)
    metrics = _compute_regression_metrics(y_test.to_numpy(dtype=float), pred_test)
    y_test_arr = y_test.to_numpy(dtype=float)
    pred_test_arr = np.asarray(pred_test, dtype=float)
    metrics["directional_acc"] = float(np.mean(np.sign(y_test_arr) == np.sign(pred_test_arr)))

    wf_metrics = _walk_forward_metrics(
        X=X,
        y=y,
        model_factory=lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=350, random_state=42),
        train_min=160,
        step=20,
        direction_ref=None,
    )
    metrics["wf_mae"] = wf_metrics["mae"]
    metrics["wf_rmse"] = wf_metrics["rmse"]
    metrics["wf_mape"] = wf_metrics["mape"]
    metrics["wf_directional_acc"] = wf_metrics["directional_acc"]
    metrics["wf_windows"] = wf_metrics["windows"]

    model = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=450, random_state=42)
    model.fit(X, y)

    live = _to_ny_datetime_index(ohlcv).sort_index()
    live["session"] = live.index.date
    today = live.groupby("session").tail(1).copy()
    if today.empty:
        raise ValueError("Could not determine current session data.")

    session_key = today["session"].iloc[-1]
    session = live[live["session"] == session_key].copy()
    if len(session) < 6:
        raise ValueError("Current session does not have enough intraday bars for inference.")

    closes = session["Close"].to_numpy(dtype=float)
    highs = session["High"].to_numpy(dtype=float)
    lows = session["Low"].to_numpy(dtype=float)
    volumes = session["Volume"].fillna(0.0).to_numpy(dtype=float)

    cum_high = np.maximum.accumulate(highs)
    cum_low = np.minimum.accumulate(lows)
    cum_vol = np.cumsum(volumes)
    cum_vwap_num = np.cumsum(closes * volumes)
    vwap = _safe_div(cum_vwap_num, np.where(cum_vol == 0.0, np.nan, cum_vol))

    pct = pd.Series(closes).pct_change().fillna(0.0)
    vol_so_far = float(pct.expanding(min_periods=5).std().fillna(0.0).iloc[-1])

    i = len(session) - 1
    current_close_session = float(closes[-1])
    current_close_1m = _load_current_price_1m(ticker)
    current_close = float(current_close_1m) if current_close_1m is not None else current_close_session
    session_open = float(session["Open"].iloc[0])

    x_live = pd.DataFrame(
        [
            {
                "open_to_now": (current_close / session_open) - 1.0,
                "range_so_far": (cum_high[i] / max(cum_low[i], 1e-12)) - 1.0,
                "price_vs_vwap": (current_close / max(vwap[i], 1e-12)) - 1.0,
                "cum_volume": np.log1p(cum_vol[i]),
                "progress": i / max(len(session) - 1, 1),
                "vol_so_far": vol_so_far,
            }
        ]
    )

    pred_ret = float(model.predict(x_live)[0])
    pred_close = current_close * (1.0 + pred_ret)

    return IntradayForecastResult(
        ticker=ticker,
        current_price=current_close,
        predicted_close=pred_close,
        predicted_return_to_close=pred_ret,
        training_rows=len(X),
        validation_metrics=metrics,
    )


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _build_daily_features(daily_ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = daily_ohlcv.copy().dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    df_feat = pd.DataFrame(index=df.index)
    df_feat["Open"] = df["Open"].astype(float)
    df_feat["High"] = df["High"].astype(float)
    df_feat["Low"] = df["Low"].astype(float)
    df_feat["Close"] = df["Close"].astype(float)
    df_feat["Volume"] = df["Volume"].astype(float)

    df_feat["SMA_5"] = df_feat["Close"].rolling(5).mean()
    df_feat["SMA_20"] = df_feat["Close"].rolling(20).mean()
    df_feat["Daily_Return"] = df_feat["Close"].pct_change(1)
    df_feat["High_Low_Pct"] = (df_feat["High"] - df_feat["Low"]) / df_feat["Low"].replace(0.0, np.nan)
    df_feat["Open_Close_Pct"] = (df_feat["Close"] - df_feat["Open"]) / df_feat["Open"].replace(0.0, np.nan)

    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    return df_feat[DAILY_FEATURE_COLUMNS]


def forecast_next_close_ml(ticker: str, period: str = "3y") -> DailyForecastResult:
    """Forecast tomorrow's close from daily technical indicators."""
    ohlcv = _load_ohlcv(ticker=ticker, interval="1d", period=period)
    if ohlcv.empty:
        raise ValueError(f"No daily data returned for {ticker}.")

    features = _build_daily_features(ohlcv)
    target = ohlcv["Close"].shift(-1)

    dataset = features.copy()
    dataset["target"] = target
    dataset = dataset.dropna()

    if len(dataset) < 150:
        raise ValueError(f"Not enough daily data for ML forecast on {ticker}. Need at least 150 rows.")

    X = dataset[DAILY_FEATURE_COLUMNS]
    y = dataset["target"]

    split = int(len(dataset) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.04, max_iter=600, random_state=42)
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred_test)) if len(X_test) else float("nan")
    metrics = _compute_regression_metrics(y_test.to_numpy(dtype=float), np.asarray(pred_test, dtype=float))
    if len(X_test):
        current_ref = X_test["Close"].to_numpy(dtype=float)
        true_dir = np.sign(y_test.to_numpy(dtype=float) - current_ref)
        pred_dir = np.sign(np.asarray(pred_test, dtype=float) - current_ref)
        directional_acc = float(np.mean(true_dir == pred_dir))
    else:
        directional_acc = float("nan")

    model_live = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.04, max_iter=700, random_state=42)
    model_live.fit(X, y)

    wf_metrics = _walk_forward_metrics(
        X=X,
        y=y,
        model_factory=lambda: HistGradientBoostingRegressor(max_depth=5, learning_rate=0.04, max_iter=500, random_state=42),
        train_min=140,
        step=5,
        direction_ref=X["Close"],
    )

    latest_x = X.iloc[[-1]]
    pred_next_close = float(model_live.predict(latest_x)[0])
    latest_close = float(ohlcv["Close"].iloc[-1])

    return DailyForecastResult(
        ticker=ticker,
        latest_close=latest_close,
        predicted_next_close=pred_next_close,
        holdout_mae=mae,
        holdout_rmse=metrics["rmse"],
        holdout_mape=metrics["mape"],
        holdout_directional_acc=directional_acc,
        walk_forward_mae=wf_metrics["mae"],
        walk_forward_rmse=wf_metrics["rmse"],
        walk_forward_mape=wf_metrics["mape"],
        walk_forward_directional_acc=wf_metrics["directional_acc"],
        walk_forward_windows=int(wf_metrics["windows"]),
        training_rows=len(X_train),
    )


def monte_carlo_gbm_close_range(
    ticker: str,
    start_price: float,
    daily_close_series: pd.Series,
    horizon_days: float = 1.0,
    n_sims: int = 20000,
) -> MonteCarloResult:
    """Estimate probabilistic terminal close range with one-step GBM."""
    clean = pd.Series(daily_close_series).dropna().astype(float)
    if len(clean) < 60:
        raise ValueError("Need at least 60 daily closes for stable GBM parameter estimation.")

    log_ret = np.log(clean / clean.shift(1)).dropna()
    mu_annual = float(log_ret.mean() * 252.0)
    sigma_annual = float(log_ret.std(ddof=1) * np.sqrt(252.0))

    dt = float(max(horizon_days, 1e-6) / 252.0)
    z = np.random.default_rng(42).standard_normal(int(n_sims))

    terminal = start_price * np.exp((mu_annual - 0.5 * sigma_annual * sigma_annual) * dt + sigma_annual * np.sqrt(dt) * z)

    q = {
        "p05": float(np.quantile(terminal, 0.05)),
        "p25": float(np.quantile(terminal, 0.25)),
        "p50": float(np.quantile(terminal, 0.50)),
        "p75": float(np.quantile(terminal, 0.75)),
        "p95": float(np.quantile(terminal, 0.95)),
    }

    return MonteCarloResult(
        ticker=ticker,
        start_price=float(start_price),
        horizon_days=float(horizon_days),
        mu_annual=mu_annual,
        sigma_annual=sigma_annual,
        mean_terminal_price=float(np.mean(terminal)),
        quantiles=q,
        probability_above_start=float(np.mean(terminal > start_price)),
    )


def _size_position_from_quantiles(
    current_price: float,
    mc: MonteCarloResult,
    portfolio_value: float,
    config: RiskRulesConfig,
) -> Tuple[float, float, float]:
    """Return (weight, notional, tail-loss notional) from MC quantiles and volatility."""
    if portfolio_value <= 0.0 or current_price <= 0.0:
        return 0.0, 0.0, 0.0

    sigma = max(mc.sigma_annual, 1e-8)
    vol_scalar = np.clip(config.base_vol_target / sigma, 0.20, 1.50)
    weight = min(config.max_position_weight, config.max_position_weight * float(vol_scalar))

    if mc.sigma_annual >= config.high_vol_cutoff:
        weight *= config.high_vol_size_multiplier

    notional = portfolio_value * weight
    downside_pct = max(0.0, (current_price - mc.quantiles["p05"]) / current_price)
    tail_risk_notional = notional * downside_pct

    single_trade_cap = portfolio_value * config.max_single_trade_tail_loss_pct
    if tail_risk_notional > single_trade_cap and downside_pct > 0.0:
        notional = single_trade_cap / downside_pct
        weight = notional / portfolio_value
        tail_risk_notional = single_trade_cap

    return float(weight), float(notional), float(tail_risk_notional)


def evaluate_intraday_trade_decision(
    intraday: IntradayForecastResult,
    mc: MonteCarloResult,
    portfolio_value: float,
    config: RiskRulesConfig | None = None,
    risk_tracker: DailyRiskBudgetTracker | None = None,
) -> TradeDecision:
    """Apply rule-based risk filter and output TRADE/REDUCE/SKIP with position size."""
    cfg = config or RiskRulesConfig()
    reasons = []

    current_price = intraday.current_price
    edge_pct = (intraday.predicted_close / current_price) - 1.0 if current_price > 0.0 else 0.0
    prob_up = mc.probability_above_start
    q25 = mc.quantiles["p25"]
    q75 = mc.quantiles["p75"]
    upside = max(0.0, q75 - current_price)
    downside = max(1e-9, current_price - q25)
    reward_to_risk = upside / downside
    iqr_pct = (q75 - q25) / max(current_price, 1e-9)

    if edge_pct < cfg.min_edge_pct:
        reasons.append("edge_below_threshold")
    if prob_up < cfg.min_prob_up:
        reasons.append("probability_below_threshold")
    if reward_to_risk < cfg.min_reward_to_risk:
        reasons.append("reward_to_risk_below_threshold")
    if abs(edge_pct) < cfg.skip_edge_pct and iqr_pct > cfg.max_iqr_pct_for_small_edge:
        reasons.append("small_edge_wide_distribution")

    if reasons:
        return TradeDecision(
            action="SKIP",
            recommended_weight=0.0,
            recommended_notional=0.0,
            tail_risk_notional=0.0,
            edge_pct=float(edge_pct),
            probability_up=float(prob_up),
            reward_to_risk=float(reward_to_risk),
            reasons=tuple(reasons),
        )

    weight, notional, tail_risk_notional = _size_position_from_quantiles(
        current_price=current_price,
        mc=mc,
        portfolio_value=portfolio_value,
        config=cfg,
    )

    if risk_tracker is not None:
        remaining = risk_tracker.remaining_notional
        if remaining <= 0.0:
            return TradeDecision(
                action="SKIP",
                recommended_weight=0.0,
                recommended_notional=0.0,
                tail_risk_notional=0.0,
                edge_pct=float(edge_pct),
                probability_up=float(prob_up),
                reward_to_risk=float(reward_to_risk),
                reasons=("daily_risk_budget_exhausted",),
            )

        if tail_risk_notional > remaining and tail_risk_notional > 0.0:
            scale = remaining / tail_risk_notional
            notional *= scale
            weight *= scale
            tail_risk_notional = remaining
            risk_tracker.register(tail_risk_notional)
            return TradeDecision(
                action="REDUCE",
                recommended_weight=float(weight),
                recommended_notional=float(notional),
                tail_risk_notional=float(tail_risk_notional),
                edge_pct=float(edge_pct),
                probability_up=float(prob_up),
                reward_to_risk=float(reward_to_risk),
                reasons=("scaled_to_fit_daily_risk_budget",),
            )

        risk_tracker.register(tail_risk_notional)

    return TradeDecision(
        action="TRADE",
        recommended_weight=float(weight),
        recommended_notional=float(notional),
        tail_risk_notional=float(tail_risk_notional),
        edge_pct=float(edge_pct),
        probability_up=float(prob_up),
        reward_to_risk=float(reward_to_risk),
        reasons=tuple(),
    )


def run_all(
    ticker: str,
    intraday_period: str = "60d",
    intraday_interval: str = "5m",
    portfolio_value: float | None = None,
    risk_tracker: DailyRiskBudgetTracker | None = None,
    rules_config: RiskRulesConfig | None = None,
) -> Dict[str, object]:
    intraday = forecast_intraday_close(ticker, intraday_period=intraday_period, interval=intraday_interval)
    daily = forecast_next_close_ml(ticker)

    daily_ohlcv = _load_ohlcv(ticker=ticker, interval="1d", period="3y")

    mc = monte_carlo_gbm_close_range(
        ticker=ticker,
        start_price=intraday.current_price,
        daily_close_series=daily_ohlcv["Close"],
        horizon_days=1.0,
    )

    results: Dict[str, object] = {
        "intraday_ml": intraday,
        "next_close_ml": daily,
        "gbm_monte_carlo": mc,
    }

    if portfolio_value is not None and portfolio_value > 0.0:
        cfg = rules_config or RiskRulesConfig()
        tracker = risk_tracker or DailyRiskBudgetTracker(
            portfolio_value=portfolio_value,
            daily_budget_pct=cfg.daily_tail_risk_budget_pct,
        )
        decision = evaluate_intraday_trade_decision(
            intraday=intraday,
            mc=mc,
            portfolio_value=portfolio_value,
            config=cfg,
            risk_tracker=tracker,
        )
        results["trade_decision"] = decision
        results["risk_budget_remaining"] = tracker.remaining_notional
        results["risk_budget_total"] = tracker.daily_budget_notional

    return results


def _print_summary(results: Dict[str, object]) -> None:
    intraday: IntradayForecastResult = results["intraday_ml"]
    daily: DailyForecastResult = results["next_close_ml"]
    mc: MonteCarloResult = results["gbm_monte_carlo"]

    print("\n=== Machine Learning: Intraday Close Forecast ===")
    print(f"Ticker: {intraday.ticker}")
    print(f"Current price: {intraday.current_price:.2f}")
    print(f"Predicted session close: {intraday.predicted_close:.2f}")
    print(f"Predicted return to close: {intraday.predicted_return_to_close * 100:.2f}%")
    print(f"Intraday training rows: {intraday.training_rows}")
    print(
        "Intraday holdout metrics: "
        f"MAE={intraday.validation_metrics.get('mae', float('nan')):.6f}, "
        f"RMSE={intraday.validation_metrics.get('rmse', float('nan')):.6f}, "
        f"MAPE={intraday.validation_metrics.get('mape', float('nan')):.4f}, "
        f"DirAcc={intraday.validation_metrics.get('directional_acc', float('nan')):.2%}"
    )

    print("\n=== Machine Learning: Next-Day Close Forecast ===")
    print(f"Latest close: {daily.latest_close:.2f}")
    print(f"Predicted next close: {daily.predicted_next_close:.2f}")
    print(f"Holdout MAE: {daily.holdout_mae:.4f}")
    print(f"Holdout RMSE: {daily.holdout_rmse:.4f}")
    print(f"Holdout MAPE: {daily.holdout_mape:.4f}")
    print(f"Holdout Directional Acc: {daily.holdout_directional_acc:.2%}")
    print(f"Walk-forward MAE: {daily.walk_forward_mae:.4f}")
    print(f"Walk-forward RMSE: {daily.walk_forward_rmse:.4f}")
    print(f"Walk-forward MAPE: {daily.walk_forward_mape:.4f}")
    print(f"Walk-forward Directional Acc: {daily.walk_forward_directional_acc:.2%}")
    print(f"Walk-forward windows: {daily.walk_forward_windows}")
    print(f"Daily training rows: {daily.training_rows}")

    print("\n=== Monte Carlo (GBM): Probabilistic Range ===")
    print(f"Start price: {mc.start_price:.2f}")
    print(f"Horizon (days): {mc.horizon_days:.2f}")
    print(f"Annualized drift (mu): {mc.mu_annual:.4f}")
    print(f"Annualized vol (sigma): {mc.sigma_annual:.4f}")
    print(f"Expected terminal price: {mc.mean_terminal_price:.2f}")
    print(
        "Quantiles: "
        f"5%={mc.quantiles['p05']:.2f}, "
        f"25%={mc.quantiles['p25']:.2f}, "
        f"50%={mc.quantiles['p50']:.2f}, "
        f"75%={mc.quantiles['p75']:.2f}, "
        f"95%={mc.quantiles['p95']:.2f}"
    )
    print(f"Probability terminal > start: {mc.probability_above_start * 100:.2f}%")

    decision = results.get("trade_decision")
    if isinstance(decision, TradeDecision):
        print("\n=== Risk Rules Decision ===")
        print(f"Action: {decision.action}")
        print(f"Edge: {decision.edge_pct * 100:.2f}%")
        print(f"Probability Up: {decision.probability_up * 100:.2f}%")
        print(f"Reward/Risk: {decision.reward_to_risk:.2f}")
        print(f"Recommended weight: {decision.recommended_weight * 100:.2f}%")
        print(f"Recommended notional: {decision.recommended_notional:.2f}")
        print(f"Tail-risk notional: {decision.tail_risk_notional:.2f}")
        if decision.reasons:
            print(f"Reasons: {', '.join(decision.reasons)}")

        remaining = results.get("risk_budget_remaining")
        total = results.get("risk_budget_total")
        if isinstance(remaining, (float, int)) and isinstance(total, (float, int)):
            print(f"Daily risk budget remaining: {float(remaining):.2f} / {float(total):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Practical ML + Monte Carlo close forecasting")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument("--intraday-period", type=str, default="60d", help="yfinance intraday period")
    parser.add_argument("--intraday-interval", type=str, default="5m", help="yfinance intraday interval")
    parser.add_argument("--daily-period", type=str, default="3y", help="yfinance daily period")
    parser.add_argument("--portfolio-value", type=float, default=0.0, help="Portfolio value for risk sizing")
    parser.add_argument(
        "--train-save-best",
        action="store_true",
        help="Train tuned models and save best artifacts for later reuse.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(MODEL_DIR),
        help="Directory to store trained model artifacts.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="When used with --train-save-best, skip immediate forecast run.",
    )
    parser.add_argument(
        "--train-profile",
        type=str,
        default="full",
        choices=["fast", "full"],
        help="Hyperparameter search profile: fast or full (default: full).",
    )
    parser.add_argument(
        "--train-skip-pipeline",
        action="store_true",
        help="Bypass data_pipeline cache during training and use local CSV/yfinance sources.",
    )
    parser.add_argument(
        "--train-quiet",
        action="store_true",
        help="Suppress per-candidate progress logs during hyperparameter search.",
    )
    parser.add_argument(
        "--daily-risk-budget-pct",
        type=float,
        default=0.015,
        help="Daily tail-risk budget as fraction of portfolio (default: 0.015)",
    )
    parser.add_argument(
        "--min-edge-pct",
        type=float,
        default=0.0025,
        help="Minimum edge threshold for trade decision (default: 0.0025)",
    )
    args = parser.parse_args()

    if args.train_save_best:
        training = train_and_save_best_models(
            ticker=args.ticker.upper(),
            intraday_period=args.intraday_period,
            intraday_interval=args.intraday_interval,
            daily_period=args.daily_period,
            output_dir=Path(args.models_dir),
            profile=args.train_profile,
            use_pipeline=not args.train_skip_pipeline,
            verbose=not args.train_quiet,
        )

        print("\n=== Model Training Complete ===")
        print(f"Ticker: {training['ticker']}")
        print(f"Intraday model: {training['intraday_model_path']}")
        print(f"Daily model: {training['daily_model_path']}")
        print(f"Manifest: {training['manifest_path']}")
        print(f"Intraday best params: {training['intraday_best_params']}")
        print(f"Daily best params: {training['daily_best_params']}")

        if args.train_only:
            return

    rules_config = RiskRulesConfig(
        min_edge_pct=args.min_edge_pct,
        daily_tail_risk_budget_pct=args.daily_risk_budget_pct,
    )

    results = run_all(
        ticker=args.ticker.upper(),
        intraday_period=args.intraday_period,
        intraday_interval=args.intraday_interval,
        portfolio_value=args.portfolio_value if args.portfolio_value > 0.0 else None,
        rules_config=rules_config,
    )
    _print_summary(results)


if __name__ == "__main__":
    main()
