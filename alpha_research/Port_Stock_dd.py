import sys
from pathlib import Path
import time
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from curl_cffi import requests
warnings.filterwarnings("ignore")

session = requests.Session(impersonate="chrome")


repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from data_pipeline.data_cache import get_data_persistent

def plot_multiple_stocks_from_cache(tickers, period="5y"):
    """Build the drawdown chart from the repository's persistent OHLCV cache."""
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return None

    fig, axes = plt.subplots(len(tickers), 2, figsize=(
        20, max(6, 4 * len(tickers))), squeeze=False)
    for idx, ticker in enumerate(tickers):
        ax_price, ax_dd = axes[idx]
        stock = get_data_persistent(ticker, interval="1d", period=period)
        if stock.empty or "Close" not in stock:
            ax_price.set_title(f"{ticker} (no cached data)")
            ax_price.axis("off")
            ax_dd.axis("off")
            continue

        close = pd.to_numeric(stock["Close"], errors="coerce").dropna()
        if close.empty:
            ax_price.set_title(f"{ticker} (no cached data)")
            ax_price.axis("off")
            ax_dd.axis("off")
            continue

        drawdown = close / close.cummax() - 1
        ax_price.plot(close.index, close.values,
                      color="steelblue", linewidth=1.1)
        ax_price.set_title(ticker)
        ax_price.set_ylabel("Price ($)")
        ax_price.grid(True, alpha=0.3)

        ax_dd.plot(drawdown.index, drawdown.values,
                   color="steelblue", linewidth=1.1)
        ax_dd.fill_between(drawdown.index, drawdown.values,
                           0, color="crimson", alpha=0.25)
        ax_dd.axhline(0, color="black", linewidth=0.8)
        ax_dd.set_ylabel("Drawdown")
        ax_dd.grid(True, alpha=0.3)
        ax_dd.set_title(
            f"Current {drawdown.iloc[-1]:.1%} | Max {drawdown.min():.1%}")

    fig.suptitle("Portfolio Drawdown Analysis", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


def calculate_drawdown_analysis(tickers, period="5y", threshold=0.02):
    """Return drawdown analysis after excluding drawdowns below ``threshold``."""
    threshold = max(0.0, float(threshold))
    stats_rows = []
    event_rows = []
    series_data = {}

    for ticker in dict.fromkeys(tickers):
        stock = get_data_persistent(ticker, interval="1d", period=period)
        if stock.empty or "Close" not in stock:
            continue

        close = pd.to_numeric(stock["Close"], errors="coerce").dropna()
        if close.empty:
            continue

        running_peak = close.cummax()
        drawdown = close / running_peak - 1
        # Keep only drawdowns meeting the threshold in the displayed series.
        filtered_drawdown = drawdown.where(drawdown <= -threshold)
        series_data[ticker] = filtered_drawdown.rename(ticker)

        # A drawdown event runs from the first close below its prior peak
        # through the first later close that reaches that peak again.
        in_event = False
        event_start = None
        event_peak_date = None
        event_peak_value = None
        event_start_position = None
        for position, (date, value) in enumerate(drawdown.items()):
            if not in_event and value <= -threshold:
                in_event = True
                event_start = date
                event_peak_date = running_peak.loc[:date].idxmax()
                event_peak_value = float(running_peak.loc[date])
                event_start_position = position

            if in_event and value >= 0:
                event_slice = drawdown.loc[event_start:date]
                trough_date = event_slice.idxmin()
                recovery_trading_days = position - event_start_position
                event_rows.append({
                    "Ticker": ticker,
                    "Peak Date": event_peak_date,
                    "Trough Date": trough_date,
                    "Recovery Date": date,
                    "Maximum Drawdown": float(event_slice.min()),
                    "Recovery Trading Days": recovery_trading_days,
                    "Recovery Days": (date - trough_date).days,
                    "Recovered": "Yes",
                })
                in_event = False

        if in_event:
            event_slice = drawdown.loc[event_start:]
            trough_date = event_slice.idxmin()
            event_rows.append({
                "Ticker": ticker,
                "Peak Date": event_peak_date,
                "Trough Date": trough_date,
                "Recovery Date": None,
                "Maximum Drawdown": float(event_slice.min()),
                "Recovery Trading Days": None,
                "Recovery Days": None,
                "Recovered": "No",
            })

    events = pd.DataFrame(event_rows)
    for ticker, drawdown in series_data.items():
        ticker_events = events[events["Ticker"] == ticker] if not events.empty else pd.DataFrame()
        recovered_events = ticker_events.dropna(subset=["Recovery Trading Days"])
        stats_rows.append({
            "Ticker": ticker,
            "Threshold": threshold,
            "Current Drawdown": float(filtered_drawdown.iloc[-1]),
            "Average Drawdown": float(filtered_drawdown[filtered_drawdown < 0].mean()) if (filtered_drawdown < 0).any() else 0.0,
            "Average Event Drawdown": float(ticker_events["Maximum Drawdown"].mean()) if not ticker_events.empty else None,
            "Maximum Drawdown": float(ticker_events["Maximum Drawdown"].min()) if not ticker_events.empty else None,
            "Average Recovery Days": float(recovered_events["Recovery Trading Days"].mean()) if not recovered_events.empty else None,
            "Maximum Recovery Days": int(recovered_events["Recovery Trading Days"].max()) if not recovered_events.empty else None,
            "Drawdown Events": int(len(ticker_events)),
            "Recovered Events": int(len(recovered_events)),
        })

    series = pd.DataFrame(series_data)
    stats = pd.DataFrame(stats_rows)
    return {"stats": stats, "series": series, "events": events}


def calculate_drawdown_stats(tickers, period="5y", threshold=0.02):
    """Backward-compatible summary-only interface."""
    return calculate_drawdown_analysis(tickers, period=period, threshold=threshold)["stats"]


def plot_drawdown_histogram(analysis):
    """Plot event maximum drawdowns as a histogram."""
    events = analysis["events"]
    values = events["Maximum Drawdown"].dropna() * 100 if not events.empty else pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    if not values.empty:
        ax.hist(values, bins=min(12, max(5, len(values))), color="crimson", alpha=0.75)
    ax.set_title("Drawdown Distribution by Event")
    ax.set_xlabel("Maximum Drawdown (%)")
    ax.set_ylabel("Event Count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_recovery_histogram(analysis):
    """Plot recovered-event durations as a histogram."""
    events = analysis["events"]
    values = events["Recovery Trading Days"].dropna() if not events.empty else pd.Series(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    if not values.empty:
        ax.hist(values, bins=min(12, max(5, len(values))), color="steelblue", alpha=0.75)
    ax.set_title("Recovery Duration Distribution")
    ax.set_xlabel("Recovery Trading Days")
    ax.set_ylabel("Event Count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def calculate_drawdown(ticker="SPY", start_date=None, window=180):
    """
    Calculate and visualize drawdown analysis for any stock
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format, defaults to 5 years ago
        window (int): Window size for finding local minima
    """
    # Set default start date to 5 years ago if not specified
    if start_date is None:
        start_date = pd.Timestamp.today() - pd.DateOffset(years=5)
    else:
        start_date = pd.Timestamp(start_date)

    end_date = pd.Timestamp.today()

    try:
        # Download stock data
        stock = yf.download(ticker,
                            start=start_date,
                            end=end_date,
                            progress=False,
                            interval='1d', session=session)

        if stock.empty:
            raise ValueError(f"No data received for {ticker}")

        print(
            f"Data range for {ticker}: {stock.index[0].date()} to {stock.index[-1].date()}")

        # Calculate rolling maximum
        rolling_max = stock['Close'].cummax()

        # Calculate drawdown
        drawdown = (stock['Close'] - rolling_max) / rolling_max

        # Find maximum drawdown and its date
        max_drawdown = drawdown.min().values[0]
        max_drawdown_date = drawdown.idxmin().values[0]

        # Calculate current drawdown
        current_drawdown = drawdown.iloc[-1].values[0]
        current_date = pd.to_datetime(drawdown.index[-1])

        # Print results
        print(f"\n{ticker} Drawdown Analysis:")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(
            f"Date of Maximum Drawdown: {pd.to_datetime(max_drawdown_date).date()}")

        # Find significant drawdown troughs (local minima)
        window = window  # Window size for finding local minima
        drawdown_series = drawdown.squeeze()
        troughs = []

        for i in range(window, len(drawdown_series) - window):
            if all(drawdown_series.iloc[i] <= drawdown_series.iloc[i-window:i]) and \
               all(drawdown_series.iloc[i] <= drawdown_series.iloc[i+1:i+window]):
                # Only mark significant drawdowns (>20%)
                if drawdown_series.iloc[i] < -0.20:
                    troughs.append(
                        (drawdown_series.index[i], drawdown_series.iloc[i]))

        # Create figure with two subplots stacked vertically
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(15, 10), height_ratios=[1, 1])
        fig.suptitle(f"{ticker} Price and Drawdown Analysis", fontsize=12)

        # Plot price chart on top subplot
        stock['Close'].plot(ax=ax1, color='blue')
        ax1.set_title('Price History')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True)

        # Plot current price point
        ticker_obj = yf.Ticker(ticker, session=session)
        print(ticker_obj.info.get("preMarketPrice"))
        current_price = ticker_obj.info.get("preMarketPrice") if ticker_obj.info.get(
            "preMarketPrice") else ticker_obj.info.get("last_price")

        price_annotation = f"Current\n${current_price:.2f}"
        if current_price > 1000:
            price_annotation = f"Current\n${current_price:,.0f}"
        elif current_price < 1:
            price_annotation = f"Current\n${current_price:.4f}"
        else:
            pass
        # Plot current price point
        ax1.plot(current_date, stock['Close'].iloc[-1], 'ro', markersize=10)
        ax1.annotate(price_annotation,
                     xy=(current_date, current_price),
                     xytext=(+40, 30),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5',
                               fc='green', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        drawdown.plot(ax=ax2)
        ax2.set_title('Historical Drawdown')
        ax2.set_ylabel('Drawdown')
        ax2.grid(True)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        ax2.fill_between(drawdown.index, drawdown.squeeze().values,
                         0, color='red', alpha=0.3)

        # Mark troughs on the chart
        for date, value in troughs:
            ax2.plot(date, value, 'v', color='black', markersize=10)
            ax2.annotate(f'{date.strftime("%Y-%m-%d")}\n{value:.1%}',
                         xy=(date, value),
                         xytext=(10, -20),
                         textcoords='offset points',
                         ha='left',
                         va='top',
                         bbox=dict(boxstyle='round,pad=0.5',
                                   fc='yellow', alpha=0.5),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        # Mark current drawdown
        ax2.plot(current_date, current_drawdown, 'ro', markersize=10)
        ax2.annotate(f'Current\n{current_drawdown:.1%}',
                     xy=(current_date, current_drawdown),
                     xytext=(-50, 30),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5',
                               fc='lightgreen', alpha=0.5),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.tight_layout()
        plt.show()
        return drawdown, max_drawdown

    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        return None, None


def main(tickers=["AAPL", "intc", "GOOG", "nvda", "SPY"]):
    """
    Plot multiple stocks drawdown analysis in a single figure
    """
    # Create figure with subplots
    fig, axes = plt.subplots(len(tickers), 2, figsize=(20, 24))
    axes = axes.flatten()

    for idx, ticker in enumerate(tickers):
        try:
            # Download and process data
            start_date = pd.Timestamp.today() - pd.DateOffset(years=5)
            stock = yf.download(ticker, start=start_date,
                                progress=False, session=session)

            if stock.empty:
                print(f"No data received for {ticker}")
                continue

            stock.columns = stock.columns.droplevel(1)
            # Calculate drawdown (ensure 1D series)
            rolling_max = stock['Close'].expanding().max()
            rolling_max2 = np.maximum.accumulate(stock['Close'])
            if not rolling_max.equals(rolling_max2):
                print(f"Warning: Rolling max mismatch for {ticker}")
            else:
                print(f"Rolling max matches for {ticker}")
            drawdown = pd.Series(
                ((stock['Close'] - rolling_max) / rolling_max) *
                100,  # Convert to percentage
                index=stock.index
            )

            # Get current values (ensure scalar values)
            ticker_obj = yf.Ticker(ticker, session=session)
            # print(ticker_obj.info.get("preMarketPrice") )
            current_price = ticker_obj.info.get("preMarketPrice") if ticker_obj.info.get(
                "preMarketPrice") else ticker_obj.history(
                period='1d').iloc[-1]['Close']
            current_drawdown = float(drawdown.iloc[-1])
            current_date = drawdown.index[-1]

            # Calculate max drawdown and its date (ensure scalar values)
            max_drawdown = float(drawdown.min())
            max_drawdown_date = drawdown.idxmin()

            # Plot price (left column)
            ax_price = axes[idx * 2]
            stock['Close'].plot(ax=ax_price, color='blue', linewidth=1.1)
            ax_price.set_title(f'{ticker}')
            ax_price.set_ylabel('Price ($)')
            ax_price.grid(True)

            # Add price annotation
            price_annotation = f"${current_price:.2f}"
            if current_price > 1000:
                price_annotation = f"${current_price:,.0f}"
            elif current_price < 1:
                price_annotation = f"${current_price:.4f}"

            ax_price.plot(current_date, current_price, 'ro', markersize=8)
            ax_price.annotate(price_annotation,
                              xy=(current_date, current_price),
                              xytext=(10, 10),
                              textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.5',
                                        fc='green', alpha=0.5),
                              arrowprops=dict(arrowstyle='->'))
            ax_price.set_xlabel('')
            # Plot drawdown (right column)
            ax_dd = axes[idx * 2 + 1]
            drawdown.plot(ax=ax_dd, color='blue', linewidth=1.1)
            ax_dd.set_title(f'')
            ax_dd.set_ylabel('Drawdown (%)')
            ax_dd.grid(True)

            # Add zero line
            ax_dd.axhline(y=0, color='black', linewidth=0.8)

            # Fill between line and zero
            ax_dd.fill_between(drawdown.index,
                               drawdown.values,
                               0,
                               color='red',
                               alpha=0.3)

            # Add current drawdown annotation
            ax_dd.plot(current_date, current_drawdown, 'ro', markersize=8)
            ax_dd.annotate(f'Current: {current_drawdown:.1f}%',
                           xy=(current_date, current_drawdown),
                           xytext=(10, 10),
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5',
                                     fc='lightgreen', alpha=0.5),
                           arrowprops=dict(arrowstyle='->'))

            # Add max drawdown annotation
            ax_dd.plot(max_drawdown_date, max_drawdown, 'rv', markersize=8)
            ax_dd.annotate(f'Max: {max_drawdown:.1f}%',
                           xy=(max_drawdown_date, max_drawdown),
                           xytext=(10, -20),
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5',
                                     fc='red', alpha=0.5),
                           arrowprops=dict(arrowstyle='->'))

            # Set y-axis limits
            ax_dd.set_ylim(min(max_drawdown * 1.1, current_drawdown * 1.1), 5)
            ax_dd.set_xlabel('')
            # Remove x-axis labels for all but the last row
            if idx < len(tickers) - 1:
                ax_price.set_xticklabels([])
                ax_dd.set_xticklabels([])

        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
            continue

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    time.sleep(10)  # Pause for 5 seconds before closing
    plt.close("all")  # Close the figure after displaying
    return fig


if __name__ == "__main__":
    # main()
    calculate_drawdown()