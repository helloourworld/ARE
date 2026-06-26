import pandas as pd
import pandas_ta as ta
import yfinance as yf

def analyze_trend_integrity(ticker="SPY", benchmark="RSP"):
    # 1. Fetch Data (SPY and Equal-Weight RSP)
    spy = yf.download(ticker, period="1y", interval="1d")
    spy.columns = spy.columns.droplevel(1)  # Drop the multi-level column index if present
    rsp = yf.download(benchmark, period="1y", interval="1d")
    rsp.columns = rsp.columns.droplevel(1)  # Drop the multi-level column index if present

    # 2. Basic Trend (The 'Old' Logic)
    spy['EMA_21'] = ta.ema(spy['Close'], length=21)
    spy['EMA_50'] = ta.ema(spy['Close'], length=50)
    
    # 3. Money Flow Filter (Is the 'Big Money' buying?)
    # CMF > 0 means accumulation; CMF < 0 means distribution (selling)
    spy['CMF'] = ta.cmf(spy['High'], spy['Low'], spy['Close'], spy['Volume'], length=20)

    # 4. Breadth Filter (Is the rest of the market following?)
    # Ratio of Equal Weight vs Market Cap Weight
    rsp_close = rsp['Close']
    if isinstance(rsp_close, pd.DataFrame):
        # If yfinance returns multiple columns for Close, use the first series.
        if rsp_close.shape[1] > 1:
            print(f"WARNING: rsp['Close'] has {rsp_close.shape[1]} columns; using the first Close series.")
        rsp_close = rsp_close.iloc[:, 0]
    spy['Breadth_Ratio'] = rsp_close / spy['Close']
    spy['Breadth_Slope'] = ta.slope(spy['Breadth_Ratio'], length=5)

    # 5. Exhaustion Filter (Is it overextended?)
    spy['RSI'] = ta.rsi(spy['Close'], length=14)

    # Get the latest values
    last_price = spy['Close'].iloc[-1]
    last_ema50 = spy['EMA_50'].iloc[-1]
    last_cmf = spy['CMF'].iloc[-1]
    last_breadth_slope = spy['Breadth_Slope'].iloc[-1]
    last_rsi = spy['RSI'].iloc[-1]

    # --- THE IMPROVED LOGIC GATE ---
    
    status = ""
    action = ""
    confidence = "HIGH"

    # Define a "Fake Trend" (Price up, but internals down)
    is_price_above_ema = last_price > last_ema50
    is_money_exiting = last_cmf < 0
    is_breadth_weak = last_breadth_slope < 0

    if is_price_above_ema:
        if is_money_exiting and is_breadth_weak:
            status = "⚠️ WARNING: FAKE TREND (Distribution)"
            action = "LIGHTEN POSITIONS / DON'T BUY"
            confidence = "LOW (Internal Divergence)"
        elif is_money_exiting or is_breadth_weak:
            status = "NEUTRAL / WEAKENING TREND"
            action = "HOLD / RAISE STOPS"
            confidence = "MEDIUM"
        else:
            status = "✅ TREND IS REAL (Accumulation)"
            action = "BUY DIPS / HOLD"
            confidence = "HIGH"
    else:
        status = "🔴 BEARISH REGIME"
        action = "CASH / SHORT"

    # 6. Check for Exhaustion (The 'Blow-off Top' filter)
    if last_rsi > 75:
        status += " - OVEREXTENDED (Exhaustion)"
        action = "TAKE PROFITS"

    return {
        "Price": round(last_price, 2),
        "CMF (Money Flow)": round(last_cmf, 3),
        "Breadth Slope": round(last_breadth_slope, 5),
        "Status": status,
        "Recommendation": action,
        "Confidence": confidence
    }


def _normalize_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def calculate_liquidity_signals(asset_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> dict:
    asset_df = _normalize_yf_df(asset_df.copy())
    benchmark_df = _normalize_yf_df(benchmark_df.copy())

    benchmark_close = benchmark_df['Close']
    if isinstance(benchmark_close, pd.DataFrame):
        if benchmark_close.shape[1] > 1:
            print(f"WARNING: benchmark['Close'] has {benchmark_close.shape[1]} columns; using the first Close series.")
        benchmark_close = benchmark_close.iloc[:, 0]

    benchmark_close = benchmark_close.reindex(asset_df.index).ffill().bfill()
    asset_df['CMF'] = ta.cmf(asset_df['High'], asset_df['Low'], asset_df['Close'], asset_df['Volume'], length=20)
    asset_df['Breadth_Ratio'] = benchmark_close / asset_df['Close']
    asset_df['Breadth_Slope'] = ta.slope(asset_df['Breadth_Ratio'], length=5)
    asset_df['RSI'] = ta.rsi(asset_df['Close'], length=14)

    return {
        'cmf': float(asset_df['CMF'].iloc[-1]),
        'breadth_ratio': float(asset_df['Breadth_Ratio'].iloc[-1]),
        'breadth_slope': float(asset_df['Breadth_Slope'].iloc[-1]),
        'rsi': float(asset_df['RSI'].iloc[-1])
    }


if __name__ == "__main__":
    analysis = analyze_trend_integrity()
    for key, value in analysis.items():
        print(f"{key}: {value}")