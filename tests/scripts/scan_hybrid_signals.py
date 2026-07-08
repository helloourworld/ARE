import sys
import pathlib
import csv
from datetime import datetime

# Ensure repo root on path
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from data_pipeline.data_cache import get_data_persistent
from risk_modeling.bolling_bands import get_hybrid_risk_signal

# Default ticker list (modify as needed)
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "AMZN", "GOOG", "INTC", "MU"]
OUT_CSV = repo_root / "data" / "hybrid_signals.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

rows = []
for t in TICKERS:
    try:
        print(f"Scanning {t}...", flush=True)
        df = get_data_persistent(t, interval="1m")
        if df is None or df.empty or len(df) < 60:
            print(f"  Insufficient 1m data for {t}, skipping.")
            continue
        # Use a tuned VPIN window (minutes) so volumes are aggregated into 5-minute buckets
        result = get_hybrid_risk_signal(df, vpin_window_minutes=250)
        row = {
            "timestamp": str(df.index[-1]),
            "ticker": t,
            "signal": result.get("signal"),
            "reason": result.get("signal_reason"),
            "price": result.get("current_price"),
            "upper_band": result.get("upper_band"),
            "lower_band": result.get("lower_band"),
            "vpin": result.get("vpin"),
            "cvd_trend": result.get("cvd_trend"),
        }
        rows.append(row)
        print(f"  {t}: {row['signal']} | VPIN={row['vpin']}")
    except Exception as e:
        print(f"  Error scanning {t}: {e}")

# Write CSV
with open(OUT_CSV, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["timestamp","ticker","signal","reason","price","upper_band","lower_band","vpin","cvd_trend"]) 
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print(f"Wrote {len(rows)} signals to {OUT_CSV}")
