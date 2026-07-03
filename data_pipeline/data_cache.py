import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Global cache tracking for multi-ticker operations
_MULTI_TICKER_CACHE = {}


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


def _normalize_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def get_data_persistent(ticker, interval="1d", period="2y", force_refresh=False):
    """Load OHLCV data from a local cache first, then refresh from Yahoo Finance as needed."""
    safe_ticker = str(ticker).replace("/", "_").replace("=", "_")
    file_path = _get_cache_path(f"cache_{safe_ticker}_{interval}.csv")
    now = datetime.now(ZoneInfo("America/Halifax"))

    try:
        if file_path.exists() and not force_refresh:
            local_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            local_df = _normalize_yf_df(local_df)

            if local_df.empty:
                return local_df

            last_ts = local_df.index[-2] # because the latest volume bar may be incomplete, we use the second to last timestamp for comparison
            wait_time = timedelta(minutes=2) if interval == "1m" else timedelta(hours=12)
            if now - last_ts < wait_time:
                return local_df

            start_date = max(last_ts, now - timedelta(days=7)) if interval == "1m" else last_ts
            if hasattr(start_date, "to_pydatetime"):
                start_date = start_date.to_pydatetime()

            new_data = yf.download(
                ticker,
                start=start_date,
                interval=interval,
                prepost=True,
                progress=False,
            )
            if not new_data.empty:
                new_data = new_data[1:]  # skip the first row to avoid duplicates
                new_data = _normalize_yf_df(new_data)
                combined = pd.concat([local_df, new_data])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                combined.to_csv(file_path)
                return combined
            return local_df
        p = "7d" if interval == "1m" else period
        print(f"initializing download for {ticker} with interval {interval} and period {p}")
        df = yf.download(ticker, period=p, interval=interval, prepost=True, progress=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df = _normalize_yf_df(df)
        df.to_csv(file_path)
        return df
    except Exception as e:
        print("Download/cache error:", repr(e))
        return pd.DataFrame()


# ============================================================================
# CONSOLIDATED DATA FETCHING INTERFACE - All methods route through cache
# ============================================================================

def get_daily_returns(tickers, benchmark, start_date):
    """
    Fetch daily price data and return percentage changes.
    Uses persistent cache with automatic updates.
    
    Args:
        tickers: List of ticker symbols or single ticker
        benchmark: Benchmark ticker
        start_date: Start date string (e.g., '2020-01-01')
    
    Returns:
        DataFrame of daily returns
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    
    all_tickers = list(set(tickers + [benchmark]))
    
    # Fetch all tickers through persistent cache
    price_data = {}
    for ticker in all_tickers:
        df = get_data_persistent(ticker, interval="1d", period="5y", force_refresh=False)
        if not df.empty:
            price_data[ticker] = df['Close']
    
    if not price_data:
        return pd.DataFrame()
    
    prices_df = pd.DataFrame(price_data)
    # Filter by start_date
    start = pd.to_datetime(start_date)
    prices_df = prices_df[prices_df.index >= start].dropna()
    
    return prices_df.pct_change(fill_method=None).dropna()


def get_price_history(tickers, period="2y", interval="1d"):
    """
    Fetch price history for technical analysis.
    Uses persistent cache with automatic updates.
    
    Args:
        tickers: List of ticker symbols or single ticker
        period: Time period (e.g., '2y', '1y', '3mo') - used for initial fetch
        interval: Data interval ('1d', '1wk', '1mo')
    
    Returns:
        DataFrame of closing prices
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    
    price_data = {}
    for ticker in tickers:
        df = get_data_persistent(ticker, interval=interval, period=period, force_refresh=False)
        if not df.empty:
            price_data[ticker] = df['Close']
    
    if not price_data:
        return pd.DataFrame()
    
    return pd.DataFrame(price_data)


def get_price_history_with_benchmark(tickers, benchmark, period="2y", interval="1d"):
    """
    Fetch price history including benchmark for RS analysis.
    Uses persistent cache with automatic updates.
    
    Args:
        tickers: List of ticker symbols or single ticker
        benchmark: Benchmark ticker
        period: Time period (e.g., '2y', '1y', '3mo')
        interval: Data interval ('1d', '1wk', '1mo')
    
    Returns:
        DataFrame of closing prices including benchmark
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    
    all_tickers = tickers + [benchmark]
    price_data = {}
    
    for ticker in all_tickers:
        df = get_data_persistent(ticker, interval=interval, period=period, force_refresh=False)
        if not df.empty:
            price_data[ticker] = df['Close']
    
    if not price_data:
        return pd.DataFrame()
    
    return pd.DataFrame(price_data)


def get_ohlcv_history(tickers, period="2y", interval="1d", force_refresh=False):
    """
    Fetch complete OHLCV data (not just Close).
    Uses persistent cache with automatic updates.
    
    Args:
        tickers: List of ticker symbols or single ticker
        period: Time period for data
        interval: Data interval ('1d', '1m', '1wk', etc.)
        force_refresh: Force refresh from Yahoo Finance
    
    Returns:
        DataFrame with OHLCV data
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    
    ohlcv_data = {}
    for ticker in tickers:
        df = get_data_persistent(ticker, interval=interval, period=period, force_refresh=force_refresh)
        if not df.empty:
            ohlcv_data[ticker] = df
    
    if not ohlcv_data:
        return pd.DataFrame()
    
    # For single ticker, return directly; for multiple, combine with MultiIndex
    if len(ohlcv_data) == 1:
        return list(ohlcv_data.values())[0]
    
    combined = pd.concat(ohlcv_data, axis=1)
    return combined


def get_premarket_data(tickers):
    """
    Fetch 1-minute intraday data with extended hours for pre-market gap analysis.
    Uses persistent cache with automatic updates.
    
    Args:
        tickers: List of ticker symbols or single ticker
    
    Returns:
        Tuple of (intraday_data, daily_history) or (None, None) on error
    """
    if not tickers:
        return None, None
    
    if isinstance(tickers, str):
        tickers = [tickers]
    
    try:
        # 1-minute data with pre/post market through cache
        intraday_data = {}
        for ticker in tickers:
            df = get_data_persistent(ticker, interval="1m", period="7d", force_refresh=False)
            if not df.empty:
                intraday_data[ticker] = df['Close']
        
        # 2-day daily data through cache
        daily_data = {}
        for ticker in tickers:
            df = get_data_persistent(ticker, interval="1d", period="2d", force_refresh=False)
            if not df.empty:
                daily_data[ticker] = df['Close']
        
        intraday_result = pd.DataFrame(intraday_data) if intraday_data else None
        daily_result = pd.DataFrame(daily_data) if daily_data else None
        
        return intraday_result, daily_result
    except Exception as e:
        print(f"Premarket data error: {e}")
        return None, None


def get_live_intraday(tickers, period="2d", force_refresh=True):
    """
    Fetch live intraday minute data for real-time monitoring.
    Always forces refresh to get latest data.
    
    Args:
        tickers: List of ticker symbols or single ticker
        period: Time period for data (default '2d')
        force_refresh: Always refresh for live data (default True)
    
    Returns:
        DataFrame of 1-minute interval closing prices
    """
    if not tickers:
        return None
    
    if isinstance(tickers, str):
        tickers = [tickers]
    
    try:
        price_data = {}
        for ticker in tickers:
            df = get_data_persistent(ticker, interval="1m", period=period, force_refresh=force_refresh)
            if not df.empty:
                price_data[ticker] = df['Close']
        
        return pd.DataFrame(price_data) if price_data else None
    except Exception as e:
        print(f"Live intraday error: {e}")
        return None


def force_refresh_ticker(ticker, interval="1d"):
    """
    Force refresh data for a specific ticker.
    Useful for manual updates.
    
    Args:
        ticker: Ticker symbol
        interval: Data interval
    
    Returns:
        Updated DataFrame or empty DataFrame on error
    """
    return get_data_persistent(ticker, interval=interval, period="2y", force_refresh=True)


    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    interval = sys.argv[2] if len(sys.argv) > 2 else "1m"
    period = sys.argv[3] if len(sys.argv) > 3 else "7d"
    force_refresh = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False

    df = get_data_persistent(ticker, interval, period, force_refresh)
    print(df.tail())        
