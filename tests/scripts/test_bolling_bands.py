import json
import pandas as pd
import numpy as np
import sys
import pathlib
from datetime import datetime, timedelta

# Ensure repo root is on sys.path so local package imports work
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from risk_modeling.bolling_bands import get_hybrid_risk_signal
from data_pipeline.data_cache import get_data_persistent

# Synthesize 1-minute OHLCV for 6 hours (360 minutes)
# Try to fetch real 1-minute data; fall back to synthetic if unavailable
ticker = "SPY"
try:
    real = get_data_persistent(ticker, interval="1m")
    if not real.empty and len(real) >= 120:
        df = real[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        print(f"Using real 1-minute data for {ticker} ({len(df)} bars)")
    else:
        raise RuntimeError("Insufficient real data, falling back to synthetic")
except Exception as e:
    print(f"Real data fetch failed: {e}; using synthetic data")
    periods = 360
    start = pd.Timestamp(datetime.now().replace(second=0, microsecond=0) - timedelta(minutes=periods))
    idx = pd.date_range(start=start, periods=periods, freq='T')

    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(loc=0, scale=0.02, size=periods))
    volumes = np.random.randint(100, 1000, size=periods)

    opens = np.concatenate([[prices[0]], prices[:-1]])
    closes = prices
    highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 0.01, size=periods))
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 0.01, size=periods))

    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes,
    }, index=idx)

# Call the function
try:
    result = get_hybrid_risk_signal(df)
    print(json.dumps(result, indent=2, default=str))
except Exception as e:
    print(f"EXCEPTION: {e}")
