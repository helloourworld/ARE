import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.data_cache import get_data_persistent

FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]

def load_gold(start="2018-01-01"):
    df = get_data_persistent("GC=F", interval="1d", period="5y", force_refresh=False)
    if df.empty:
        return pd.DataFrame()
    df = df[df.index >= pd.to_datetime(start)]
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