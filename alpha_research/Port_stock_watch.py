"""
Market Watch Module

A professional market watch system for monitoring key market indicators, 
individual stocks, and generating periodic updates with visualizations.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import time
import schedule
import logging

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from .Port_Stock_dd import main
except ImportError:
    from Port_Stock_dd import main
from data_pipeline.data_cache import get_data_persistent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_market_data_from_cache(tickers, period="1y"):
    """Build the market-watch chart from the repository's persistent cache."""
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return None

    data = {}
    for ticker in tickers:
        history = get_data_persistent(ticker, interval="1d", period=period)
        if not history.empty and "Close" in history:
            data[ticker] = pd.to_numeric(history["Close"], errors="coerce")

    if not data:
        return None

    close_prices = pd.DataFrame(data).dropna(how="all")
    fig, ax = plt.subplots(figsize=(16, 7))
    for ticker in close_prices.columns:
        series = close_prices[ticker].dropna()
        if not series.empty:
            ax.plot(series.index, series.values, linewidth=1.3, label=ticker)
    ax.set_title("Market Watch")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", ncol=min(4, max(1, len(close_prices.columns))))
    fig.tight_layout()
    return fig


def fetch_market_data(tickers, start_date, end_date=None):
    """
    Fetch historical market data for given tickers and current market data
    
    Args:
        tickers (list): List of ticker symbols
        start_date (str): Start date for historical data
        end_date (str, optional): End date for historical data
        
    Returns:
        tuple: (DataFrame with historical data, dict with ticker names, dict with current prices)
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    # Fetch historical data
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    data = data.Close

    names = {}
    current_data = {}

    # Determine market session (EST times)
    now = datetime.now()
    market_open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    pre_market_time = now.replace(hour=4, minute=0, second=0, microsecond=0)

    is_market_open = market_open_time <= now <= market_close_time
    is_pre_market = pre_market_time <= now < market_open_time

    # Fetch current market data
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            current_price = ticker_obj.info.get("preMarketPrice") if is_pre_market else ticker_obj.history(
                period="1d").iloc[-1]['Close']
            
            if current_price:
                current_data[ticker] = current_price
                session_type = "Pre-market" if is_pre_market else ("Market" if is_market_open else "After-hours")
                logger.info(f"{session_type} {ticker}: ${current_price:.2f}")
            else:
                logger.warning(f"No current price available for {ticker}")
                current_data[ticker] = None

            # Get ticker name
            names[ticker] = ticker_obj.info.get('shortName', ticker)

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            current_data[ticker] = None
            names[ticker] = ticker

    return data, names, current_data


def plot_market_data(data, name, current_prices=None, title="Market Overview"):
    """
    Plot market data with enhanced visualization using interactive mode.
    
    Args:
        data (DataFrame): Historical price data
        name (dict): Mapping of ticker symbols to company names
        current_prices (dict, optional): Current prices for each ticker
        title (str): Title for the plot
        
    Returns:
        Figure: Matplotlib figure object
    """
    num_tickers = len(data.columns)
    num_rows = (num_tickers + 1) // 2  # Ceiling division
    num_cols = 2
    
    plt.close('all')
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle(title, fontsize=16)

    # Create subplot grid
    gs = fig.add_gridspec(num_rows, num_cols)

    # Iterate through the tickers and plot the data
    for i, ticker in enumerate(data.columns):
        row = i // num_cols
        col = i % num_cols
        ax = fig.add_subplot(gs[row, col])

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        # Plot historical data
        ax.plot(data[ticker], label=ticker, linewidth=1.5)

        # Plot real-time data point if available
        if current_prices and ticker in current_prices:
            realtime_price = current_prices.get(ticker)
            if realtime_price is not None:
                last_date = data[ticker].index[-1]
                ax.plot(last_date, realtime_price, 'ro', markersize=8, label='Real-time')
                ax.annotate(f'${realtime_price:.2f}',
                            xy=(last_date, realtime_price),
                            xytext=(5, 10),
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

        # Set subplot title and labels
        ax.set_title(f"{name.get(ticker, ticker)}", loc='left', fontsize=12, fontweight='bold')
        ax.set_ylabel("Price")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper left')

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Move the figure to the second screen and maximize it
    try:
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()  # Maximize the window
        if hasattr(mng.window, "move"):
            mng.window.move(1920, 0)  # Move to second screen
    except Exception as e:
        logger.warning(f"Could not move window to second screen: {e}")

    # Draw and flush events
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig


def calculate_correlations(data, display=False):
    """
    Calculate and optionally visualize correlations between assets.
    
    Args:
        data (DataFrame): Historical price data with 'Close' column
        display (bool): Whether to display the correlation heatmap
        
    Returns:
        DataFrame: Correlation matrix
    """
    if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
        close_prices = data['Close']
    else:
        close_prices = data
    
    correlation_matrix = close_prices.corr()

    if display:
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
        plt.title("Correlation Matrix of Key Assets")
        plt.tight_layout()
        plt.show()
    
    return correlation_matrix


class MarketWatch:
    """Professional market watch system with scheduled updates."""
    
    def __init__(self, market_indicators, start_date, dd_tickers=None):
        """
        Initialize market watch system.
        
        Args:
            market_indicators (list): List of ticker symbols to watch
            start_date (str): Start date for historical data (YYYY-MM-DD)
            dd_tickers (list, optional): Additional tickers for deep-dive analysis
        """
        self.market_indicators = market_indicators
        self.start_date = start_date
        self.dd_tickers = dd_tickers or ["AMD", "INTC", "GOOG", "MU", "SPY"]
        self.current_fig = None
        self.current_fig2 = None
    
    def update_market_watch(self):
        """Update the market watch data and plots."""
        try:
            data, names, current_prices = fetch_market_data(
                self.market_indicators, self.start_date)
            self.current_fig = plot_market_data(
                data, names, current_prices, title="Professional Market Watch")
            time.sleep(25)
            plt.close('all')
            logger.info("Market watch updated successfully")
        except Exception as e:
            logger.error(f"Error updating market watch: {e}")
    
    def update_market_dd(self):
        """Update the deep-dive market analysis plots."""
        try:
            self.current_fig2 = main(tickers=self.dd_tickers)
            time.sleep(20)
            plt.close('Figure 1')
            logger.info("Market deep-dive updated successfully")
        except Exception as e:
            logger.error(f"Error updating market deep-dive: {e}")
    
    def run(self, update_interval_watch=10, update_interval_dd=30):
        """
        Run the market watch with scheduled updates.
        
        Args:
            update_interval_watch (int): Update interval in minutes for market watch
            update_interval_dd (int): Update interval in minutes for deep-dive analysis
        """
        # Initial displays
        self.update_market_watch()
        self.update_market_dd()
        
        # Schedule updates
        schedule.every(update_interval_watch).minutes.do(self.update_market_watch)
        schedule.every(update_interval_dd).minutes.do(self.update_market_dd)
        
        logger.info(f"Market watch running. Updates every {update_interval_watch} minutes (watch) "
                   f"and {update_interval_dd} minutes (deep-dive)")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Exiting market watch...")
            plt.close('all')


def professional_market_watch(tickers, start_date, run_scheduler=False):
    """
    Simulate a professional market watch setup.
    
    Args:
        tickers (list): List of ticker symbols to monitor
        start_date (str): Start date for historical data (YYYY-MM-DD)
        run_scheduler (bool): Whether to run the scheduler (default: False for manual use)
        
    Returns:
        MarketWatch: Market watch instance
    """
    watch = MarketWatch(tickers, start_date)
    
    if run_scheduler:
        watch.run()
    else:
        watch.update_market_watch()
    
    return watch


if __name__ == "__main__":
    # Define key market indicators
    market_indicators = [
        'SPY', '^VIX', '^TNX', 'GC=F',
        'ES=F', 'USDCAD=X', 'CADCNY=X',
        '^IXIC', 'YM=F', 'NQ=F', 'ZN=F',
        'AAPL', 'ZT=F', '2YYK26.CBT'
    ]

    start_date = '2026-04-01'
    
    # Create and run market watch
    watch = MarketWatch(market_indicators, start_date)
    watch.run(update_interval_watch=10, update_interval_dd=30)
