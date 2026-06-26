from datetime import datetime, timedelta
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
import time
import os
from pathlib import Path
import pytz

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_path(file_name: str) -> Path:
    target_path = DATA_DIR / file_name
    legacy_paths = [Path(file_name), REPO_ROOT / file_name]
    for legacy_path in legacy_paths:
        if legacy_path.exists() and legacy_path != target_path:
            try:
                legacy_path.replace(target_path)
            except OSError:
                pass
    return target_path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Mandelbrot Framework Thresholds
HURST_BULLISH_MIN = 0.55      # Threshold for bullish persistence
HURST_UNSTABLE_MAX = 0.45     # Threshold for unstable/choppy market
TAIL_INDEX_SAFE = 1.7         # Normal tails (low crash risk)
TAIL_INDEX_RISKY = 1.55       # Elevated tail risk (fat tails)
VOL_THRESHOLD = 0.001         # 0.1% intraday move threshold for tail risk activation

# Analysis Window Parameters
# Data points for Hurst calculation (~5 trading days at 1m)
HURST_WINDOW_SIZE = 300
VOL_WINDOW_SIZE = 30          # Data points for volatility calculation
MIN_DATA_POINTS = 100         # Minimum intraday data points required

# Hurst Exponent Calculation Parameters
LAG_MIN = 10                  # Minimum lag size
LAG_DENSITY = 15              # Number of lag points for regression

# Tail Index Calculation Parameters
TAIL_PERCENTILE = 5           # Percentage (0.05) of returns to use as tail
TAIL_MIN_K = 15               # Minimum number of tail points
TAIL_BLOCK_SIZE = 300         # Window size for rolling Hill estimator
TAIL_STEP = 15                # Step size for rolling tail estimation


def get_data_persistent(ticker, interval="1m", period="2y"):
    """
    Persistent cache that handles 1m (7-day chunks) and 1d (multi-year) history.
    """
    file_path = _get_cache_path(f"cache_{ticker}_{interval}.parquet")
    now = datetime.now(pytz.utc)

    if interval == "1m":
        fetch_limit_days = 7
        max_history_days = 29
        default_period = "7d"
    else:
        fetch_limit_days = 365 * 10
        max_history_days = 365 * 50
        default_period = period

    if os.path.exists(file_path):
        local_df = pd.read_parquet(file_path)

        if local_df.index.tz is None:
            local_df.index = local_df.index.tz_localize('UTC')

        last_timestamp = local_df.index[-1]
        time_diff = now - last_timestamp
        threshold = timedelta(minutes=2) if interval == "1m" else timedelta(hours=24)

        if time_diff < threshold:
            return local_df

        lookback_limit = now - timedelta(days=fetch_limit_days)
        start_date = max(last_timestamp, lookback_limit)

        print(f"Updating {ticker} ({interval}) from {start_date.strftime('%Y-%m-%d')}...")
        new_data = yf.download(ticker, start=start_date,
                               interval=interval, prepost=True, progress=False)

        if not new_data.empty:
            if isinstance(new_data.columns, pd.MultiIndex):
                new_data.columns = new_data.columns.get_level_values(0)

            if new_data.index.tz is None:
                new_data.index = new_data.index.tz_localize('UTC')

            combined_df = pd.concat([local_df, new_data])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df.sort_index(inplace=True)

            combined_df.to_parquet(file_path)
            return combined_df

        return local_df

    else:
        print(f"First run for {ticker} ({interval}). Downloading {default_period}...")
        df = yf.download(ticker, period=default_period,
                         interval=interval, prepost=True, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')

        df.to_parquet(file_path)
        return df


def calculate_hurst(prices: np.ndarray, volumes: np.ndarray = None, window: int = HURST_WINDOW_SIZE) -> float:
    if len(prices) < window:
        return 0.5

    data = prices[-window:]
    vols = volumes[-window:] if volumes is not None else np.ones(len(data))
    returns = np.log(data[1:] / data[:-1])
    avg_vol = np.mean(vols)

    if avg_vol > 0:
        volume_weights = vols[1:] / avg_vol
        data_to_test = returns * volume_weights
    else:
        data_to_test = returns

    N = len(data_to_test)
    lags = np.unique(np.geomspace(
        LAG_MIN, N // 2, num=LAG_DENSITY).astype(int))

    RS = []
    valid_lags = []
    for lag in lags:
        num_blocks = N // lag
        rs_sub = []
        for i in range(num_blocks):
            block = data_to_test[i * lag: (i + 1) * lag]
            diff = block - np.mean(block)
            cum_sum = np.cumsum(diff)
            r = np.max(cum_sum) - np.min(cum_sum)
            s = np.std(block)
            if s > 1e-10:
                rs_sub.append(r / s)
        if rs_sub:
            RS.append(np.mean(rs_sub))
            valid_lags.append(lag)

    if len(RS) < 5:
        return 0.5
    res = linregress(np.log(valid_lags), np.log(RS))
    return float(res.slope)


def calculate_tail_index(returns: np.ndarray, volumes: np.ndarray = None, lookback: int = 500) -> float:
    if len(returns) < TAIL_BLOCK_SIZE:
        return 2.0

    active_mask = np.abs(returns) > 1e-6
    active_returns = returns[active_mask]

    if volumes is not None:
        active_volumes = volumes[1:][active_mask]

    if len(active_returns) < TAIL_BLOCK_SIZE:
        return 2.0

    subset_returns = active_returns[-lookback:]

    def hill_est_asymmetric(data: np.ndarray) -> float:
        left_tail_data = np.abs(data[data < 0])

        if len(left_tail_data) < TAIL_MIN_K:
            return 3.0

        k = max(int(len(left_tail_data) * (TAIL_PERCENTILE / 100)), TAIL_MIN_K)
        tails = np.sort(left_tail_data)[-k:]
        threshold = tails[0]

        if threshold <= 0:
            return 3.0

        return 1.0 / np.mean(np.log(tails / threshold))

    results = [
        hill_est_asymmetric(subset_returns[i: i + TAIL_BLOCK_SIZE])
        for i in range(0, len(subset_returns) - TAIL_BLOCK_SIZE, TAIL_STEP)
    ]

    if not results:
        return 2.0

    return float(np.percentile(results, 10))


def scan_now(ticker_symbol: str = "QQQ") -> str:
    print(f"--- Scanning {ticker_symbol} Mandelbrot Status ---")

    data = get_data_persistent(ticker_symbol, interval="1m")
    daily_data = get_data_persistent(ticker_symbol, interval="1d", period="2y")

    if daily_data.empty or data.empty:
        print("Error: insufficient market data returned from yfinance.")
        return "ERROR - No Data"

    daily_returns = daily_data["Close"].pct_change().dropna().values
    intraday_prices = data["Close"].dropna().values
    intraday_returns = data["Close"].pct_change().dropna().values
    intraday_volumes = data["Volume"].dropna().values

    if len(intraday_prices) < MIN_DATA_POINTS or len(intraday_volumes) < MIN_DATA_POINTS:
        print("Error: insufficient intraday price or volume data.")
        return "ERROR - Insufficient Data"

    last_price = float(intraday_prices[-1])

    hurst_value = calculate_hurst(
        intraday_prices[-HURST_WINDOW_SIZE:], intraday_volumes[-HURST_WINDOW_SIZE:], window=HURST_WINDOW_SIZE)
    tail_index = calculate_tail_index(
        intraday_returns, intraday_volumes, lookback=500)

    if np.isnan(hurst_value) or np.isnan(tail_index):
        print("Error: Invalid indicator values (NaN)")
        return "ERROR - Invalid Indicators"

    current_vol = np.std(
        intraday_prices[-VOL_WINDOW_SIZE:]) / np.mean(intraday_prices[-VOL_WINDOW_SIZE:])

    if tail_index < TAIL_INDEX_RISKY:
        regime = "5 - TAIL RISK (Active Danger)" if current_vol > VOL_THRESHOLD else "5 - DORMANT TAIL RISK (Watching)"
    elif hurst_value > HURST_BULLISH_MIN and tail_index > TAIL_INDEX_SAFE:
        regime = "1 - BULLISH PERSISTENCE (Trend is Real)"
    elif hurst_value > HURST_BULLISH_MIN and tail_index >= TAIL_INDEX_RISKY:
        regime = "2 - BEARISH PERSISTENCE (Downside Trend with Fat Tails)"
    elif hurst_value < HURST_UNSTABLE_MAX:
        regime = "4 - UNSTABLE (Mean Reverting/Choppy)"
    else:
        regime = "3 - NEUTRAL / RANDOM WALK"

    print(f"Current Price:  {last_price:.2f}")
    print(f"Hurst (Trend):  {hurst_value:.3f}")
    print(f"Tail Index:     {tail_index:.3f}")
    print(f"Intraday Vol:   {current_vol:.4f}")
    print(f"RESULT REGIME:  {regime}")
    print("-" * 40)

    return regime


if __name__ == "__main__":
    scan_now("GOOG")
    scan_now("MU")
    scan_now("SPY")
