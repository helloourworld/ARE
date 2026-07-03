import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.data_cache import get_data_persistent

# 1. Configuration
ticker = "NVDA"
drift_days = 60  # The typical PEAD window is 30-90 days
surprise_threshold = 0.05  # We want a "Beat" of at least 5%

# 2. Download Historical Price Data
# We look at the last 2 years to see the recent AI boom drift
start_date = "2024-01-01"
df = get_data_persistent(ticker, interval="1d", period="2y", force_refresh=False)
if df.empty:
    raise SystemExit(f"No price data downloaded for {ticker} since {start_date}")
df = df[df.index >= pd.to_datetime(start_date)]
df = df[["Open", "High", "Low", "Close", "Volume"]]
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)  # Drop the 'Adj Close' multi-index if it exists 
df.index = pd.to_datetime(df.index)
print(f"Loaded {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")

# 3. Add our "Lines" (Moving Averages)
df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()   # Short term (1d)
df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean() # Long term (1m)
print(df.head())
# 4. Earnings Dates and Surprises (Manual sample for NVDA 2024/25/26)
# In a production bot, use an API like AlphaVantage or Zacks for this
earnings_data = [
    {"date": "2024-02-21", "surprise": 0.11},
    {"date": "2024-05-22", "surprise": 0.09},
    {"date": "2024-08-28", "surprise": 0.06},
    {"date": "2025-02-26", "surprise": 0.12},
    {"date": "2026-05-20", "surprise": 0.08} # Today's beat!
]
results = []
for event in earnings_data:
    e_date = pd.to_datetime(event['date'])
    if e_date not in df.index:
            print(f"Skipping {e_date.date()}: no trading data for earnings date")
            continue

    e_loc = df.index.get_loc(e_date)
    row = df.iloc[e_loc]
    is_bullish = row['Close'] > row['SMA_200']

    if event['surprise'] > surprise_threshold and is_bullish:
        # Entry: Close price the day AFTER earnings
        entry_idx = df.index.get_loc(e_date) + 1
        entry_price = df.iloc[entry_idx]['Close']
        
        # Exit: Close price 60 days later
        exit_idx = entry_idx + drift_days
        if exit_idx < len(df):
            exit_price = df.iloc[exit_idx]['Close']
            profit = (exit_price - entry_price) / entry_price
            results.append({"Date": e_date, "Profit": profit})

# 6. Display Results
summary = pd.DataFrame(results)
print(f"--- NVDA PEAD Strategy Results ---")
print(summary)
print(f"Average Profit per Drift: {summary['Profit'].mean():.2%}")