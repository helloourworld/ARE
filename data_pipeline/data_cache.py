"""
Persistent data cache for OHLCV history.

This module centralizes persistent price data access through a local CSV cache under data/.
It refreshes data from Yahoo Finance when needed and uses Interactive Brokers as a fallback
source for OHLCV history when the IB API is available.

Usage example:

    from data_pipeline.data_cache import get_price_history

    prices = get_price_history(["AAPL", "MSFT"], period="1y", interval="1d")
    print(prices.tail())
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Thread
import logging
import platform
import time

import pandas as pd
import yfinance as yf

try:
    from ibapi.client import EClient
    from ibapi.contract import Contract
    from ibapi.wrapper import EWrapper
except Exception:
    EClient = None
    EWrapper = None
    Contract = None

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

IB_CLIENT_AVAILABLE = EClient is not None and EWrapper is not None and Contract is not None
IS_WINDOWS = platform.system().lower().startswith("win")
IB_FALLBACK_ENABLED = IB_CLIENT_AVAILABLE and not IS_WINDOWS
IB_YF_SUFFIX_MAP = {
    "TO": {"primaryExchange": "TSE", "currency": "CAD"},
    "NE": {"primaryExchange": "AEQN", "currency": "CAD"},
}

logger = logging.getLogger(__name__)
STORAGE_TIMEZONE = "America/New_York"
DATE_ONLY_INTERVALS = {"1d", "1wk", "1mo"}


def _normalize_index_for_interval(index, interval: str):
    """Normalize index shape by interval.

    - Daily/weekly/monthly data is stored as session dates (tz-naive).
    - Intraday data is stored timezone-aware in New York time.
    """
    normalized_interval = str(interval).lower()
    dt_index = pd.DatetimeIndex(index)

    if normalized_interval in DATE_ONLY_INTERVALS:
        if dt_index.tz is not None:
            # Convert to UTC first so legacy 20:00-04 daily rows map to the next trading date.
            dt_index = dt_index.tz_convert("UTC").normalize().tz_localize(None)
        else:
            dt_index = dt_index.normalize()
        return pd.DatetimeIndex(dt_index)

    if dt_index.tz is None:
        dt_index = dt_index.tz_localize(STORAGE_TIMEZONE)
    elif str(dt_index.tz) != STORAGE_TIMEZONE:
        dt_index = dt_index.tz_convert(STORAGE_TIMEZONE)
    return pd.DatetimeIndex(dt_index)


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


def _normalize_yf_df(df: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()  # Always work with a copy to avoid SettingWithCopyWarning
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Coerce index to datetime, then normalize it for the requested interval.
    if not isinstance(df.index, pd.DatetimeIndex):
        coerced_index = pd.to_datetime(df.index, errors="coerce")
        valid_mask = ~coerced_index.isna()
        df = df.loc[valid_mask].copy()
        coerced_index = coerced_index[valid_mask]
        if len(coerced_index) == 0:
            return pd.DataFrame()
        df.index = pd.DatetimeIndex(coerced_index)

    df.index = _normalize_index_for_interval(df.index, interval)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    
    return df


def _normalize_ib_df(df: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    
    df = df.copy()  # Always work with a copy
    
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df = df.set_index("datetime")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    
    df.index = _normalize_index_for_interval(df.index, interval)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    return df.loc[:, [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]]


def normalize_timestamp_for_index(value, index):
    ts = pd.to_datetime(value)
    if getattr(index, "tz", None) is not None and index.tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize(STORAGE_TIMEZONE)
    return ts


def _load_cached_csv(path: Path, interval: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # Normalize index formatting/timezone on every read so legacy rows are repaired.
    df = _normalize_yf_df(df, interval=interval)
    if df.empty:
        return pd.DataFrame()
    return df


def _normalize_tickers(tickers):
    """Normalize ticker input to a list of symbols."""
    if isinstance(tickers, str):
        return [tickers]
    return list(tickers) if tickers else []


def _load_close_series(tickers, interval="1d", period="2y", force_refresh=False):
    """Load closing price series for symbols through the persistent cache.

    Each ticker is loaded via get_data_persistent, so this method inherits the
    local CSV refresh, IB fallback, and Yahoo Finance fetch behavior.
    """
    tickers = _normalize_tickers(tickers)
    if not tickers:
        return pd.DataFrame()

    price_data = {}
    for ticker in tickers:
        df = get_data_persistent(ticker, interval=interval, period=period, force_refresh=force_refresh)
        if not df.empty:
            price_data[ticker] = df["Close"]
    return pd.DataFrame(price_data)


def _cache_dataframe(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    df.to_csv(file_path)
    return df


def _refresh_from_yf(ticker: str, start_date, interval: str) -> pd.DataFrame:
    """Refresh data from Yahoo Finance with proper date handling."""
    # Convert timezone-aware Timestamp to string format that yfinance expects
    if hasattr(start_date, "strftime"):
        # It's a datetime/Timestamp object - convert to YYYY-MM-DD string
        start_date_str = start_date.strftime("%Y-%m-%d")
    else:
        start_date_str = str(start_date)
    
    return yf.download(
        ticker,
        start=start_date_str,
        interval=interval,
        prepost=True,
        progress=False,
    )


def _refresh_local_cache(local_df: pd.DataFrame, ticker: str, interval: str, file_path: Path) -> pd.DataFrame:
    last_ts = local_df.index[-2] # Use second-to-last index to avoid partial last row TODO
    
    daily_like = str(interval).lower() in DATE_ONLY_INTERVALS

    # Ensure last_ts is timezone-aware for staleness checks.
    if getattr(last_ts, "tzinfo", None) is None:
        if daily_like:
            last_ts = pd.Timestamp(last_ts).tz_localize(STORAGE_TIMEZONE)
        else:
            last_ts = pd.Timestamp(last_ts, tz=STORAGE_TIMEZONE)
    
    # Ensure comparison with timezone-aware now
    now_local = pd.Timestamp.now(tz=STORAGE_TIMEZONE)
    wait_time = timedelta(minutes=2) if interval == "1m" else timedelta(hours=12)
    
    if now_local - last_ts < wait_time:
        return local_df

    # Use the Timestamp object directly - _refresh_from_yf will handle conversion
    start_date = max(last_ts, now_local - pd.Timedelta(days=7)) if interval == "1m" else last_ts

    new_data = _refresh_from_yf(ticker, start_date, interval)
    if new_data.empty:
        return local_df

    new_data = new_data[1:]
    new_data = _normalize_yf_df(new_data, interval=interval)
    combined = pd.concat([local_df, new_data])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return _cache_dataframe(combined, file_path)


def _fetch_yf_initial(ticker: str, interval: str, period: str) -> pd.DataFrame:
    fetch_period = "7d" if interval == "1m" else period
    logger.debug(
        "Initializing Yahoo download | ticker=%s interval=%s period=%s",
        ticker,
        interval,
        fetch_period,
    )
    return yf.download(ticker, period=fetch_period, interval=interval, prepost=True, progress=False)


def _map_interval_to_ib_barsize(interval: str) -> str | None:
    return {
        "1m": "1 min",
        "2m": "2 mins",
        "3m": "3 mins",
        "5m": "5 mins",
        "10m": "10 mins",
        "15m": "15 mins",
        "30m": "30 mins",
        "1h": "1 hour",
        "4h": "4 hours",
        "1d": "1 day",
        "1wk": "1 week",
        "1mo": "1 month",
    }.get(str(interval).lower())


def _normalize_duration(period: str) -> str:
    if not period:
        return "2 Y"
    s = str(period).strip().lower()
    if s.endswith("mo"):
        return f"{int(s[:-2])} M"
    if s.endswith("wk"):
        return f"{int(s[:-2])} W"
    if s.endswith("d") and not s.endswith("wd"):
        return f"{int(s[:-1])} D"
    if s.endswith("y"):
        return f"{int(s[:-1])} Y"
    return str(period).upper()


def _build_ib_contract_from_ticker(ticker: str):
    """Map Yahoo-style ticker symbols to IB contract fields."""
    ticker_str = str(ticker).strip().upper()
    if not ticker_str:
        return None

    base_symbol = ticker_str
    mapping = {"primaryExchange": "NASDAQ", "currency": "USD"}

    if "." in ticker_str:
        maybe_symbol, maybe_suffix = ticker_str.rsplit(".", 1)
        if maybe_suffix in IB_YF_SUFFIX_MAP:
            base_symbol = maybe_symbol
            mapping = IB_YF_SUFFIX_MAP[maybe_suffix]
        else:
            return None

    contract = Contract()
    contract.symbol = base_symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.primaryExchange = mapping["primaryExchange"]
    contract.currency = mapping["currency"]
    return contract


if IB_CLIENT_AVAILABLE:
    class IBKRApp(EWrapper, EClient):
        """Simple Interactive Brokers wrapper for historical data requests."""

        def __init__(self):
            EClient.__init__(self, self)
            self.data = []
            self.finished = False
            self.request_error = None

        def historicalData(self, reqId, bar):
            self.data.append({
                "datetime": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            })

        def historicalDataEnd(self, reqId, start, end):
            self.finished = True

        def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
            # 2104/2106/2158 are IB connectivity status messages, not fatal request errors.
            if errorCode in {2104, 2106, 2158}:
                return
            self.request_error = (reqId, errorCode, errorString)
            if reqId in {1, -1}:
                self.finished = True
else:
    class IBKRApp:
        """Fallback placeholder when the IB API is not installed."""

        def __init__(self):
            raise RuntimeError("IB API is not available")

        def historicalData(self, reqId, bar):
            raise RuntimeError("IB API is not available")

        def historicalDataEnd(self, reqId, start, end):
            raise RuntimeError("IB API is not available")


def run_loop(app):
    """Run the IB API event loop in a separate thread."""
    app.run()


def get_data_from_ib(ticker: str, interval: str = "1d", period: str = "2y") -> pd.DataFrame:
    """Fetch historical OHLCV from Interactive Brokers as a backup source."""
    if IS_WINDOWS:
        print("IB fallback disabled on Windows; using cache/Yahoo Finance path.")
        return pd.DataFrame()

    if not IB_CLIENT_AVAILABLE:
        print("IB API unavailable: install ibapi to enable fallback.")
        return pd.DataFrame()

    # Skip IB fallback for symbols that are typically not valid IB stock contracts.
    if str(ticker).startswith("^") or any(ch in str(ticker) for ch in ["=", "/"]):
        return pd.DataFrame()

    bar_size = _map_interval_to_ib_barsize(interval)
    if bar_size is None:
        print(f"Unsupported IB interval: {interval}")
        return pd.DataFrame()

    duration = _normalize_duration(period)
    app = IBKRApp()
    try:
        app.connect("127.0.0.1", 4002, clientId=999)
    except Exception as exc:
        print(f"IB connect failed: {exc}")
        return pd.DataFrame()

    api_thread = Thread(target=run_loop, args=(app,))
    api_thread.daemon = True
    api_thread.start()
    time.sleep(1)

    contract = _build_ib_contract_from_ticker(ticker)
    if contract is None:
        print(f"Skipping IB fallback for {ticker}: unsupported Yahoo suffix for IB mapping.")
        try:
            app.disconnect()
        except Exception:
            pass
        return pd.DataFrame()

    # IB UTC notation requires a dash and no trailing timezone token.
    end_time = datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S")
    request_kwargs = {
        "reqId": 1,
        "contract": contract,
        "endDateTime": end_time,
        "durationStr": duration,
        "barSizeSetting": bar_size,
        "whatToShow": "TRADES",
        "useRTH": 0,
        "formatDate": 1,
        "keepUpToDate": 0,
        "chartOptions": [],
        "timezoneId": "America/New_York",
    }
    try:
        try:
            app.reqHistoricalData(**request_kwargs)
        except TypeError as exc:
            if "timezoneId" not in str(exc):
                raise
            # Some IB API versions don't support timezoneId.
            request_kwargs.pop("timezoneId", None)
            app.reqHistoricalData(**request_kwargs)
    except Exception as exc:
        print(f"IB historical request failed: {exc}")
        try:
            app.disconnect()
        except Exception:
            pass
        return pd.DataFrame()

    timeout = time.time() + 30
    while not app.finished and time.time() < timeout:
        time.sleep(0.5)

    if app.request_error is not None:
        req_id, code, message = app.request_error
        print(f"IB request error reqId={req_id} code={code}: {message}")
        try:
            app.disconnect()
        except Exception:
            pass
        return pd.DataFrame()

    try:
        app.disconnect()
    except Exception:
        pass

    if not app.data:
        return pd.DataFrame()

    df = pd.DataFrame(app.data)
    df = _normalize_ib_df(df, interval=interval)
    if df.empty:
        return pd.DataFrame()

    return df[~df.index.duplicated(keep="last")].sort_index()


def get_data_persistent(ticker, interval="1d", period="2y", force_refresh=False):
    """Load OHLCV data from the local cache, refreshing or fetching as needed.

    Workflow:
    1. Read cached CSV data if available.
    2. Refresh stale cache from Yahoo Finance.
    3. If no cache exists, try IB fallback first.
    4. Finally, download initial history from Yahoo Finance.
    """
    safe_ticker = str(ticker).replace("/", "_").replace("=", "_")
    file_path = _get_cache_path(f"cache_{safe_ticker}_{interval}.csv")

    try:
        if file_path.exists() and not force_refresh:
            local_df = _load_cached_csv(file_path, interval=interval)
            if local_df.empty:
                return local_df

            refreshed_df = _refresh_local_cache(local_df, ticker, interval, file_path)
            if not refreshed_df.empty:
                return refreshed_df
            return local_df

        if IB_FALLBACK_ENABLED:
            ib_df = get_data_from_ib(ticker, interval=interval, period=period)
            if not ib_df.empty:
                ib_df = _normalize_yf_df(ib_df, interval=interval)
                return _cache_dataframe(ib_df, file_path)

        yf_df = _fetch_yf_initial(ticker, interval, period)
        if yf_df is None or yf_df.empty:
            return pd.DataFrame()
        yf_df = _normalize_yf_df(yf_df, interval=interval)
        return _cache_dataframe(yf_df, file_path)
    except Exception as exc:
        print("Download/cache error:", repr(exc))
        return pd.DataFrame()


# ============================================================================
# CONSOLIDATED DATA FETCHING INTERFACE - All methods route through cache
# ============================================================================

__all__ = [
    "get_data_from_ib",
    "get_data_persistent",
    "get_daily_returns",
    "get_price_history",
    "get_price_history_with_benchmark",
    "get_ohlcv_history",
    "get_premarket_data",
    "get_live_intraday",
    "get_official_session_open",
    "force_refresh_ticker",
]

def get_daily_returns(tickers, benchmark, start_date):
    """
    Fetch completed-session daily close-to-close percentage changes.
    Uses persistent cache with automatic updates.
    
    Args:
        tickers: List of ticker symbols or single ticker
        benchmark: Benchmark ticker
        start_date: Start date string (e.g., '2020-01-01')
    
    Returns:
        DataFrame of daily returns
    """
    tickers = _normalize_tickers(tickers)
    all_tickers = list(set(tickers + [benchmark]))
    price_data = _load_close_series(all_tickers, interval="1d", period="5y", force_refresh=False)
    if price_data.empty:
        return pd.DataFrame()

    if isinstance(price_data.index, pd.DatetimeIndex):
        price_data.index = _normalize_index_for_interval(price_data.index, "1d")

    start_ts = normalize_timestamp_for_index(start_date, price_data.index)
    if price_data.index.min() > start_ts:
        price_data = _load_close_series(
            all_tickers, interval="1d", period="5y", force_refresh=True
        )
        if price_data.empty:
            return pd.DataFrame()
        if isinstance(price_data.index, pd.DatetimeIndex):
            price_data.index = _normalize_index_for_interval(price_data.index, "1d")

    returns = price_data.pct_change(fill_method=None)

    # Drop symbols with no valid returns instead of dropping all rows globally.
    returns = returns.loc[:, returns.notna().any(axis=0)]
    if returns.empty:
        return pd.DataFrame()

    # Yahoo's current daily bar changes until its trading session has closed.
    today = pd.Timestamp.now(tz=STORAGE_TIMEZONE).normalize()
    if getattr(returns.index, "tz", None) is None:
        today = today.tz_localize(None)
    returns = returns[returns.index.normalize() < today]
    returns = returns[returns.index >= start_ts]

    # Keep rows where the benchmark is available; remaining symbols align to it.
    if benchmark in returns.columns:
        returns = returns[returns[benchmark].notna()]

    returns = returns.dropna(how="all")
    return returns


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
    return _load_close_series(tickers, interval=interval, period=period, force_refresh=False)


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
    return _load_close_series(tickers + [benchmark], interval=interval, period=period, force_refresh=False)


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
    tickers = _normalize_tickers(tickers)
    if not tickers:
        return pd.DataFrame()

    ohlcv_data = {}
    for ticker in tickers:
        df = get_data_persistent(ticker, interval=interval, period=period, force_refresh=force_refresh)
        if not df.empty:
            ohlcv_data[ticker] = df
    
    if not ohlcv_data:
        return pd.DataFrame()
    
    if len(ohlcv_data) == 1:
        return list(ohlcv_data.values())[0]
    
    return pd.concat(ohlcv_data, axis=1)


def get_premarket_data(tickers):
    """
    Fetch 1-minute intraday data with extended hours for pre-market gap analysis.
    Uses persistent cache with automatic updates.
    
    Args:
        tickers: List of ticker symbols or single ticker
    
    Returns:
        Tuple of (intraday_data, daily_history) or (None, None) on error
    """
    tickers = _normalize_tickers(tickers)
    if not tickers:
        return None, None
    
    try:
        intraday_result = _load_close_series(tickers, interval="1m", period="7d", force_refresh=False)
        daily_result = _load_close_series(tickers, interval="1d", period="2d", force_refresh=False)
        return (
            intraday_result if not intraday_result.empty else None,
            daily_result if not daily_result.empty else None,
        )
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
    tickers = _normalize_tickers(tickers)
    if not tickers:
        return None
    
    try:
        result = _load_close_series(tickers, interval="1m", period=period, force_refresh=force_refresh)
        return result if not result.empty else None
    except Exception as e:
        print(f"Live intraday error: {e}")
        return None


def get_official_session_open(ticker, session_date):
    """Return a ticker's official daily opening price for a trading date."""
    try:
        daily_data = get_data_persistent(
            ticker, interval="1d", period="5d", force_refresh=False
        )
        if daily_data.empty or "Open" not in daily_data:
            return None

        target_date = pd.Timestamp(session_date).date()
        matching_rows = daily_data.loc[pd.DatetimeIndex(daily_data.index).date == target_date, "Open"].dropna()
        return float(matching_rows.iloc[-1]) if not matching_rows.empty else None
    except Exception as e:
        print(f"Official open data error for {ticker}: {e}")
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
