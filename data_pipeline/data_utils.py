"""
Data utilities for ARE dashboard - consolidated yfinance data fetching.
Provides cached functions for various data types and timeframes.
All functions route through persistent cache in data_cache.py for automatic updates.
"""
import streamlit as st
from .data_cache import (
    get_daily_returns as _get_daily_returns,
    get_price_history as _get_price_history,
    get_price_history_with_benchmark as _get_price_history_with_benchmark,
    get_premarket_data as _get_premarket_data,
    get_live_intraday as _get_live_intraday,
)


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

