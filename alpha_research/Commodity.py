import numpy as np
import pandas as pd
from pathlib import Path
import sys


def _find_repo_root(start_path: Path) -> Path:
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "enable_repo_root.py").exists():
            return candidate
    return start_path


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enable_repo_root import ensure_repo_root
from data_pipeline.data_cache import get_data_persistent, normalize_timestamp_for_index

REPO_ROOT = ensure_repo_root(REPO_ROOT)

FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]

def load_gold(start="2018-01-01"):
    df = get_data_persistent("GC=F", interval="1d", period="5y", force_refresh=False)
    if df.empty:
        return pd.DataFrame()
    start_ts = normalize_timestamp_for_index(start, df.index)
    df = df[df.index >= start_ts]
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)  # Drop multi-index if exists
    df.reset_index(inplace=True)
    return df

def compute_fibonacci(df, lookback=252):
    recent = df.tail(lookback)
    print(recent[["Date", "Close"]].tail())
    swing_low = recent["Close"].min()
    print(f"Swing Low: {swing_low:.2f}")
    swing_high = recent["Close"].max()
    print(f"Swing High: {swing_high:.2f}")

    fibs = {
        f"{int(level*100)}%": swing_high - level * (swing_high - swing_low)
        for level in FIB_LEVELS
    }
    return swing_low, swing_high, fibs

def trend_slope(df, window=60):
    y = df["Close"].tail(window).values
    x = np.arange(len(y))
    return np.polyfit(x, y, 1)[0]


if __name__ == "__main__":
    df = load_gold()
    low, high, fibs = compute_fibonacci(df)
    slope = trend_slope(df)
    
    print(f"Swing Low: {low:.2f}, Swing High: {high:.2f}")
    print("Fibonacci Levels:")
    for label, level in fibs.items():
        print(f"  {label}: {level:.2f}")
    print(f"Trend Slope: {slope:.4f}")