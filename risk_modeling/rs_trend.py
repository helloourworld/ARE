import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from data_pipeline import get_price_history_with_benchmark
except ImportError:
    from data_pipeline import get_price_history_with_benchmark

def calculate_professional_rs(ticker_data, benchmark_data, window=52):
    """
    Calculates Mansfield Relative Strength (CFA/Institutional Standard)
    ticker_data: Series of stock adjusted close
    benchmark_data: Series of benchmark adjusted close (e.g., SPY or XEQT.TO)
    window: 52 weeks (252 days) for long-term trend
    """
    # 1. Base Ratio
    base_ratio = ticker_data / benchmark_data
    
    # 2. Institutional Base (Simple Moving Average of the ratio)
    sma_ratio = base_ratio.rolling(window=window).mean()
    
    # 3. Mansfield RS Formula
    # This normalizes the ratio around a zero-line
    mrs = ((base_ratio / sma_ratio) - 1) * 100
    
    # 4. RS Momentum (Rate of Change of the RS line)
    # Professional desks look for a 'Hook' (RS crossing above 0)
    rs_slope = mrs.diff(5) # 5-day slope
    
    return mrs, rs_slope

# --- Integrated Execution ---
def get_rs_signals(tickers, benchmark='SPY'):
    data = get_price_history_with_benchmark(tickers, benchmark, period="2y", interval="1d")
    results = []
    
    for t in tickers:
        if t in data.columns:
            mrs, slope = calculate_professional_rs(data[t], data[benchmark])
            results.append({
                "Ticker": t,
                "RS_Score": mrs.iloc[-1],
                "RS_Trend": "Improving" if slope.iloc[-1] > 0 else "Decaying",
                "Institutional_Signal": "Accumulate" if mrs.iloc[-1] > 0 and slope.iloc[-1] > 0 else "Avoid"
            })
    
    return pd.DataFrame(results)

def calculate_mansfield_rs(ticker_data, benchmark_data, window=252):
    # 1. TEMPORAL ALIGNMENT: 
    # Use 'inner join' logic to ensure we only look at days BOTH were open
    combined = pd.concat([ticker_data, benchmark_data], axis=1).ffill()
    ticker_clean = combined.iloc[:, 0]
    bench_clean = combined.iloc[:, 1]

    # 2. RATIO CALCULATION
    ratio = ticker_clean / bench_clean
    
    # 3. SMA WITH MIN_PERIODS:
    # min_periods=1 tells Python: "Even if we don't have 252 days yet, show me the average 
    # of whatever we DO have." This makes the line consecutive from Day 1.
    sma_ratio = ratio.rolling(window=window, min_periods=1).mean()
    
    mrs = ((ratio / sma_ratio) - 1) * 100
    slope = mrs.diff(5) 
    
    return mrs, slope

def monitor_mean_reversion(rs_series, price_series):
    """
    Identifies if a Blue Quadrant stock is ready to revert to Green.
    rs_series: Series of Mansfield RS values
    price_series: Series of stock prices for 50-DMA support check
    """
    # 1. Calculate RS Volatility Bands (2 Standard Deviations)
    rs_ma = rs_series.ffill().rolling(window=20).mean()
    rs_std = rs_series.ffill().rolling(window=20).std()
    lower_band = rs_ma - (2 * rs_std)
    
    # 2. Check for "The Hook" 
    # (Slope turns positive while above the Zero Line)
    is_hooking = (rs_series.ffill().iloc[-1] > rs_series.ffill().iloc[-2]) and (rs_series.ffill().iloc[-1] > 0)
    
    # 3. Check for Band Touch
    is_oversold_rs = rs_series.ffill().iloc[-1] <= lower_band.ffill().iloc[-1]
    
    # 4. Check for 50-DMA Price Support
    price_50ma = price_series.ffill().rolling(window=50).mean()
    near_support = abs(price_series.ffill().iloc[-1] / price_50ma.ffill().iloc[-1] - 1) < 0.05
    # print(is_hooking, is_oversold_rs, near_support)
    if is_hooking and near_support:
        return "🪝REVERSION SIGNAL: Buy/Add. Consolidation complete."
    elif is_oversold_rs:
        return "🔥MONITOR: Extreme Weakness. Look for the Hook."
    else:
        return "NEUTRAL: Still Consolidating."

def calculate_rs_bollinger_bands(rs_series, window=20, num_std=2):
    """
    Calculates statistical bands for the RS Line to identify mean reversion.
    """
    sma = rs_series.ffill().rolling(window=window).mean().dropna() # Ensure alignment for plotting
    std = rs_series.ffill().rolling(window=window).std()
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    return sma, upper_band, lower_band,rs_series.dropna()

def detect_rs_hook(rs_series):
    """
    Institutional Hook Detector:
    Returns True if the RS line has printed a 'Higher Low' after a decline.
    """
    if len(rs_series) < 3:
        return False
        
    curr = rs_series.ffill().iloc[-1]
    prev = rs_series.ffill().iloc[-2]
    prev2 = rs_series.ffill().iloc[-3]
    
    # Logic: Today is higher than yesterday, AND yesterday was the local bottom
    is_turning_up = (curr > prev) and (prev <= prev2)
    
    return is_turning_up