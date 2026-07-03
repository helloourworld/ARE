"""
data_pipeline module - Centralized data fetching and processing utilities.
All functions automatically update cached data while fetching.
"""

from .data_utils import (
    get_daily_returns,
    get_price_history,
    get_price_history_with_benchmark,
    get_premarket_data,
    get_live_intraday,
)

from .data_cache import (
    get_data_persistent,
    get_ohlcv_history,
    force_refresh_ticker,
)

__all__ = [
    # High-level Streamlit-cached functions
    "get_daily_returns",
    "get_price_history",
    "get_price_history_with_benchmark",
    "get_premarket_data",
    "get_live_intraday",
    # Lower-level persistent cache functions
    "get_data_persistent",
    "get_ohlcv_history",
    "force_refresh_ticker",
]
