"""
Data utilities for ARE dashboard - consolidated yfinance data fetching.
Provides cached functions for various data types and timeframes.
All functions route through persistent cache in data_cache.py for automatic updates.
"""
import csv
from io import StringIO
import json
import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .data_cache import (
    get_daily_returns as _get_daily_returns,
    get_price_history as _get_price_history,
    get_price_history_with_benchmark as _get_price_history_with_benchmark,
    get_premarket_data as _get_premarket_data,
    get_live_intraday as _get_live_intraday,
)


def _build_requests_session(retries: int = 3, backoff_factor: float = 1.0):
    session = requests.Session()
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD", "OPTIONS"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(
        pool_connections=32,
        pool_maxsize=32,
        pool_block=False,
        max_retries=retry,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _fetch_10y_yield_from_fred(timeout_seconds: int = 25, retries: int = 3, backoff_factor: float = 1.0):
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    headers = {"User-Agent": "Mozilla/5.0"}
    session = _build_requests_session(retries=retries, backoff_factor=backoff_factor)

    response = session.get(url, timeout=(5, timeout_seconds), headers=headers)
    response.raise_for_status()

    csv_text = response.text.strip()
    if not csv_text:
        raise ValueError("FRED response returned no CSV data")

    reader = csv.reader(StringIO(csv_text))
    rows = [row for row in reader if row]
    if len(rows) < 2:
        raise ValueError("FRED CSV did not contain enough data rows for DGS10")

    values = []
    for row in reversed(rows[1:]):
        if len(row) >= 2 and row[1] not in (".", ""):
            values.append(float(row[1]))
            if len(values) == 2:
                break

    if len(values) < 2:
        raise ValueError("FRED CSV did not contain two non-missing DGS10 values")

    price, prev_close = values[0], values[1]
    delta = price - prev_close
    delta_pct = (delta / prev_close * 100) if prev_close != 0 else 0.0

    return {
        "val": price,
        "delta": delta,
        "delta_pct": delta_pct,
    }



@st.cache_data(ttl=900)
def get_10y_yield():
    """Fetch the 10-year U.S. Treasury yield from FRED.

    Resilient behavior:
    - retries with longer timeout
    - if still failing, falls back to last persisted local value
    """
    from pathlib import Path

    fred_cache_path = Path(__file__).resolve().parents[1] / "data" / "fred_dgs10_last.json"

    try:
        val = _fetch_10y_yield_from_fred(timeout_seconds=25, retries=3)

        # Persist last success for offline/timeout fallback
        import json

        fred_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fred_cache_path, "w", encoding="utf-8") as f:
            json.dump(val, f)

        return val
    except Exception as exc:
        # Fallback: read last successful value
        try:
            if fred_cache_path.exists():
                import json

                with open(fred_cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f) or None

                if cached and all(k in cached for k in ("val", "delta", "delta_pct")):
                    return cached
        except Exception:
            pass

        raise RuntimeError(
            f"Unable to fetch 10-year yield from FRED (and no valid fallback found): {exc}"
        ) from exc




@st.cache_data(ttl=3600)
def get_daily_returns(tickers, benchmark, start_date):
    """
    Fetch daily price data and return percentage changes.
    Cached for 1 hour. Routes through persistent cache.
    
    Args:
        tickers: List of ticker symbols
        benchmark: Benchmark ticker
        start_date: Start date string (e.g., '2020-01-01')
    
    Returns:
        DataFrame of daily returns
    """
    return _get_daily_returns(tickers, benchmark, start_date)


@st.cache_data(ttl=3600)
def get_price_history(tickers, period="2y", interval="1d"):
    """
    Fetch price history for technical analysis.
    Cached for 1 hour. Routes through persistent cache.
    
    Args:
        tickers: List of ticker symbols
        period: Time period (e.g., '2y', '1y', '3mo')
        interval: Data interval ('1d', '1wk', '1mo')
    
    Returns:
        DataFrame of closing prices
    """
    return _get_price_history(tickers, period, interval)


@st.cache_data(ttl=3600)
def get_price_history_with_benchmark(tickers, benchmark, period="2y", interval="1d"):
    """
    Fetch price history including benchmark for RS analysis.
    Cached for 1 hour. Routes through persistent cache.
    
    Args:
        tickers: List of ticker symbols
        benchmark: Benchmark ticker
        period: Time period (e.g., '2y', '1y', '3mo')
        interval: Data interval ('1d', '1wk', '1mo')
    
    Returns:
        DataFrame of closing prices including benchmark
    """
    return _get_price_history_with_benchmark(tickers, benchmark, period, interval)


@st.cache_data(ttl=600)
def get_premarket_data(tickers):
    """
    Fetch 1-minute intraday data with extended hours for pre-market gap analysis.
    Cached for 10 minutes. Routes through persistent cache.
    
    Args:
        tickers: List of ticker symbols
    
    Returns:
        Tuple of (intraday_data, daily_history) or None on error
    """
    return _get_premarket_data(tickers)


@st.cache_data(ttl=60)
def get_live_intraday(tickers, period="2d"):
    """
    Fetch live intraday minute data for real-time monitoring.
    Cached for 60 seconds (nearly live). Routes through persistent cache.
    
    Args:
        tickers: List of ticker symbols
        period: Time period for data (default '2d')
    
    Returns:
        DataFrame of 1-minute interval closing prices
    """
    return _get_live_intraday(tickers, period, force_refresh=True)

