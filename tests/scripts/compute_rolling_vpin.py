import sys
import pathlib

repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import pandas as pd
from data_pipeline.data_cache import get_data_persistent
from risk_modeling.bolling_bands import compute_rolling_vpin

TICKER = "SPY"
OUT_CSV = repo_root / "data" / f"rolling_vpin_{TICKER}.csv"

print(f"Fetching 1-minute data for {TICKER}...")
df = get_data_persistent(TICKER, interval="1m")
if df is None or df.empty:
    print("No data available")
    sys.exit(1)

print(f"Computing rolling VPIN (250 minutes, 5-min buckets)...")
vpin_series = compute_rolling_vpin(df, window_minutes=250, resample_rule='5min')

if vpin_series.empty:
    print("No VPIN values computed")
    sys.exit(1)

vpin_series.to_csv(OUT_CSV, header=True)
print(f"Wrote rolling VPIN to {OUT_CSV}")
print(vpin_series.tail(5))
