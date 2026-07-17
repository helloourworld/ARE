import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from io import StringIO


"""
market_breadth.py
------------------
Streamlit dashboard and helper functions to compute market breadth indicators
for the S&P 500. Breadth metrics here include:
- Advancers / Decliners / Unchanged counts
- Net Advancers and Advance-Decline (A/D) Line
- Percentage of stocks above 20/50/200 DMA
- 52-week new highs / new lows

Notes / assumptions:
- Input `close` is a DataFrame of daily closing prices (columns = tickers).
- Rolling windows (20/50/200/252) require at least that many rows to return
    non-NaN values; callers should choose an appropriate `period`.
- Uses `yfinance` for convenience; network failures should be handled by
    higher-level caching layers (see data_pipeline in repository).
"""


# -----------------------------
# Helper: get S&P 500 tickers
# -----------------------------
def get_sp500_tickers():
    # Read the S&P 500 constituents table from Wikipedia. Use requests with a
    # modern User-Agent to avoid basic 403 blocks, then parse the HTML with
    # pandas. Cache the result for a day to avoid repeated network calls.
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        import requests

        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
        # pandas will warn in future when passing literal HTML strings; wrap
        # the response text in a StringIO to provide a file-like object.
        tables = pd.read_html(StringIO(resp.text))
        df = tables[0]

        tickers = df["Symbol"].tolist()
        # Yahoo Finance uses BRK-B instead of BRK.B
        tickers = [t.replace(".", "-") for t in tickers]

        sectors = df[["Symbol", "Security", "GICS Sector"]].copy()
        sectors["Symbol"] = sectors["Symbol"].str.replace(".", "-", regex=False)

        return tickers, sectors
    except Exception:
        # If the automated fetch fails (network, 403, parsing), fall back to
        # a small curated subset so the dashboard and tests can still run.
        sample = [
            "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "JPM", "V", "JNJ",
            "WMT", "PG", "DIS", "MA", "BAC", "HD", "INTC", "CSCO",
        ]
        sectors = pd.DataFrame({
            "Symbol": sample,
            "Security": sample,
            "GICS Sector": ["Technology"] * len(sample),
        })
        return sample, sectors


# -----------------------------
# Helper: download price data
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers, period="2y"):
    data = yf.download(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True
    )

    # yfinance returns a MultiIndex when multiple tickers are requested
    # (e.g. ('Close','AAPL')). Normalize to a simple `close` DataFrame
    # where each column is a ticker and the values are adjusted close prices.
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].copy()

    # Remove tickers that have no data at all and forward-fill small gaps
    # which commonly occur around stock symbol changes or holidays.
    close = close.dropna(axis=1, how="all")
    close = close.ffill()

    return close


@st.cache_data(ttl=3600, show_spinner=False)
def download_etfs(period="2y"):
    etfs = ["SPY", "RSP", "QQQ", "IWM", "^VIX"]
    data = yf.download(
        etfs,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True
    )

    # Reuse same normalization logic as download_prices for ETF list
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data[["Close"]].copy()

    close = close.dropna(axis=1, how="all")
    close = close.ffill()

    return close

# -------- Append current price data --------
def append_current_prices(close_df, tickers):
    """Fetch current prices for all tickers and append as today's row if not already present."""
    try:
        today = pd.Timestamp.now().normalize()
        last_date = close_df.index[-1] if len(close_df) > 0 else None

        # Only fetch if today's data isn't already in the dataframe
        if last_date is None or last_date.normalize() < today:
            # Fetch current price (1d data from yesterday to today to catch market close)
            current_data = yf.download(
                tickers,
                start=(today - pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                end=(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            if isinstance(current_data.columns, pd.MultiIndex):
                current_close = current_data["Close"].copy()
            else:
                current_close = current_data[["Close"]].copy()

            # Align columns with historical data
            current_close = current_close.reindex(columns=close_df.columns, fill_value=np.nan)

            # Append only the most recent row if it's a new date
            if len(current_close) > 0:
                latest_current_row = current_close.iloc[-1:]
                latest_current_row.index = latest_current_row.index.normalize()

                if last_date is None or latest_current_row.index[0] > last_date.normalize():
                    close_df = pd.concat([close_df, latest_current_row])

        return close_df
    except Exception as e:
        # If current price fetch fails, return historical data as-is
        return close_df

# -----------------------------
# Breadth calculations
# -----------------------------
def calculate_breadth(close):
    # Compute daily returns; used to count advancers/decliners
    daily_ret = close.pct_change()

    # Count tickers that moved up/down/unchanged each day
    advancers = (daily_ret > 0).sum(axis=1)
    decliners = (daily_ret < 0).sum(axis=1)
    unchanged = (daily_ret == 0).sum(axis=1)

    # Net advancers and the cumulative A/D Line
    net_advancers = advancers - decliners
    ad_line = net_advancers.cumsum()

    # A/D ratio (advancers / decliners). Replace 0 with NaN to avoid inf values
    ad_ratio = advancers / decliners.replace(0, np.nan)

    # Moving averages for participation measures. These are full series to
    # allow computing percent above MA each day.
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    # Fraction of tickers above their moving averages on each date
    pct_above_20 = (close > ma20).sum(axis=1) / close.count(axis=1)
    pct_above_50 = (close > ma50).sum(axis=1) / close.count(axis=1)
    pct_above_200 = (close > ma200).sum(axis=1) / close.count(axis=1)

    # 52-week (approx. 252 trading days) rolling highs/lows to detect new extremes
    rolling_252_high = close.rolling(252).max()
    rolling_252_low = close.rolling(252).min()

    new_highs = (close >= rolling_252_high).sum(axis=1)
    new_lows = (close <= rolling_252_low).sum(axis=1)

    breadth = pd.DataFrame({
        "Advancers": advancers,
        "Decliners": decliners,
        "Unchanged": unchanged,
        "Net Advancers": net_advancers,
        "A/D Ratio": ad_ratio,
        "A/D Line": ad_line,
        "% Above 20DMA": pct_above_20,
        "% Above 50DMA": pct_above_50,
        "% Above 200DMA": pct_above_200,
        "52W New Highs": new_highs,
        "52W New Lows": new_lows
    })

    return breadth


def calculate_sector_breadth(close, sector_map):
    # For sector-level metrics we use the most recent available price and
    # simple cross-sectional comparisons to each sector's moving averages.
    latest_price = close.iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]

    records = []

    for sector in sector_map["GICS Sector"].unique():
        sector_tickers = sector_map.loc[
            sector_map["GICS Sector"] == sector, "Symbol"
        ].tolist()

        # Only consider tickers for which we have price data
        available = [t for t in sector_tickers if t in close.columns]

        if len(available) == 0:
            # skip sectors with no available price data in the matrix
            continue

        pct_50 = (latest_price[available] > ma50[available]).sum() / len(available)
        pct_200 = (latest_price[available] > ma200[available]).sum() / len(available)

        records.append({
            "Sector": sector,
            "Stocks Count": len(available),
            "% Above 50DMA": pct_50,
            "% Above 200DMA": pct_200
        })

    sector_df = pd.DataFrame(records)
    return sector_df.sort_values("% Above 200DMA", ascending=False)


def breadth_signal(latest, spy_series=None):
    """Compute a breadth health score from multiple internal indicators.

    Scoring rules (each criterion contributes ±1):
    1. % Above 50DMA   > 0.60 → +1   |  < 0.40 → -1
    2. % Above 200DMA  > 0.55 → +1   |  < 0.35 → -1  (tighter — 200DMA is structural)
    3. A/D Ratio       > 1.5  → +1   |  < 0.67 → -1
    4. 52W High/Low    net >  0  → +1  |  net < 0 → -1  (normalised by total)
    5. SPY trend       price > 50DMA > 200DMA → +1
                       price < 50DMA           → -1
                       Golden Cross (50 crossed above 200 recently) → +1 bonus
                       Death  Cross (50 crossed below 200 recently) → -1 bonus

    Returns (label, color, score, detail_lines).
    """
    pct_50  = latest["% Above 50DMA"]
    pct_200 = latest["% Above 200DMA"]
    ad_ratio = latest["A/D Ratio"]
    new_highs = latest["52W New Highs"]
    new_lows  = latest["52W New Lows"]

    score = 0
    details = []

    # 1. Participation — 50DMA
    if pct_50 > 0.60:
        score += 1
        details.append(f"✅ {pct_50:.1%} stocks above 50DMA (>60%)")
    elif pct_50 < 0.40:
        score -= 1
        details.append(f"❌ {pct_50:.1%} stocks above 50DMA (<40%)")
    else:
        details.append(f"➖ {pct_50:.1%} stocks above 50DMA (neutral)")

    # 2. Participation — 200DMA (structural)
    if pct_200 > 0.55:
        score += 1
        details.append(f"✅ {pct_200:.1%} stocks above 200DMA (>55%)")
    elif pct_200 < 0.35:
        score -= 1
        details.append(f"❌ {pct_200:.1%} stocks above 200DMA (<35%)")
    else:
        details.append(f"➖ {pct_200:.1%} stocks above 200DMA (neutral)")

    # 3. A/D Ratio
    if ad_ratio > 1.5:
        score += 1
        details.append(f"✅ A/D Ratio {ad_ratio:.2f} (>1.5 — buyers dominate)")
    elif ad_ratio < 0.67:
        score -= 1
        details.append(f"❌ A/D Ratio {ad_ratio:.2f} (<0.67 — sellers dominate)")
    else:
        details.append(f"➖ A/D Ratio {ad_ratio:.2f} (neutral)")

    # 4. 52W High-Low Net
    hl_net = int(new_highs) - int(new_lows)
    hl_total = max(int(new_highs) + int(new_lows), 1)
    hl_ratio = hl_net / hl_total
    if hl_ratio > 0.15:
        score += 1
        details.append(f"✅ 52W Highs dominate: +{hl_net} net ({new_highs:.0f} H / {new_lows:.0f} L)")
    elif hl_ratio < -0.15:
        score -= 1
        details.append(f"❌ 52W Lows dominate: {hl_net} net ({new_highs:.0f} H / {new_lows:.0f} L)")
    else:
        details.append(f"➖ 52W Highs/Lows balanced: {hl_net:+d} net")

    # 5. SPY price vs moving averages + cross detection
    if spy_series is not None and len(spy_series) >= 201:
        spy_50  = spy_series.rolling(50).mean()
        spy_200 = spy_series.rolling(200).mean()
        latest_price = spy_series.iloc[-1]
        s50  = spy_50.iloc[-1]
        s200 = spy_200.iloc[-1]

        if latest_price > s50 and s50 > s200:
            score += 1
            details.append(f"✅ SPY {latest_price:.2f} > 50DMA {s50:.2f} > 200DMA {s200:.2f}")
        elif latest_price < s50:
            score -= 1
            details.append(f"❌ SPY {latest_price:.2f} below 50DMA {s50:.2f}")
        else:
            details.append(f"➖ SPY {latest_price:.2f} — mixed vs MAs")

        # Golden / Death Cross: did the 50DMA cross the 200DMA in the last 10 sessions?
        cross_window = min(10, len(spy_50.dropna()))
        recent_50  = spy_50.dropna().iloc[-cross_window:]
        recent_200 = spy_200.dropna().iloc[-cross_window:]
        diff = (recent_50 - recent_200)
        if len(diff) >= 2:
            if diff.iloc[-1] > 0 and diff.iloc[0] <= 0:
                score += 1
                details.append("🌟 Golden Cross detected (50DMA crossed above 200DMA)")
            elif diff.iloc[-1] < 0 and diff.iloc[0] >= 0:
                score -= 1
                details.append("💀 Death Cross detected (50DMA crossed below 200DMA)")

    if score >= 3:
        return "Healthy", "green", score, details
    elif score <= -2:
        return "Weak / Unstable", "red", score, details
    else:
        return "Neutral / Mixed", "orange", score, details


# -----------------------------
# App layout
# -----------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Market Breadth Dashboard", layout="wide")
    st.title("S&P 500 Market Breadth Dashboard")

    st.caption(
        "This dashboard checks whether the S&P 500 rally is broad-based or concentrated in a few large-cap names."
    )

    with st.sidebar:
        st.header("Settings")

        period = st.selectbox(
            "Historical period",
            ["1y", "2y", "5y"],
            index=1
        )

        show_raw = st.checkbox("Show raw breadth table", value=False)

        refresh = st.button("Refresh data")

        if refresh:
            st.cache_data.clear()
            st.rerun()


    # -----------------------------
    # Load data
    # -----------------------------
    tickers, sector_map = get_sp500_tickers()
    close = download_prices(tickers, period=period)
    etf_close = download_etfs(period=period)

    # Append current prices (today's latest data)
    close = append_current_prices(close, tickers)
    etf_close = append_current_prices(etf_close, ["SPY", "RSP", "QQQ", "IWM", "^VIX"])

    breadth = calculate_breadth(close)
    sector_breadth = calculate_sector_breadth(close, sector_map)

    latest = breadth.dropna().iloc[-1]
    latest_date = breadth.dropna().index[-1].date()

    spy_series = etf_close["SPY"].dropna() if "SPY" in etf_close.columns else None
    signal, signal_color, score, details = breadth_signal(latest, spy_series)


    # -----------------------------
    # Top summary
    # -----------------------------
    st.subheader(f"Latest Breadth Reading: {latest_date}")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Breadth Signal",
        signal
    )

    col2.metric(
        "A/D Ratio",
        f"{latest['A/D Ratio']:.2f}"
    )

    col3.metric(
        "% Above 50DMA",
        f"{latest['% Above 50DMA']:.1%}"
    )

    col4.metric(
        "% Above 200DMA",
        f"{latest['% Above 200DMA']:.1%}"
    )

    col5, col6, col7, col8 = st.columns(4)

    col5.metric(
        "Advancers",
        int(latest["Advancers"])
    )

    col6.metric(
        "Decliners",
        int(latest["Decliners"])
    )

    col7.metric(
        "52W New Highs",
        int(latest["52W New Highs"])
    )

    col8.metric(
        "52W New Lows",
        int(latest["52W New Lows"])
    )


    # -----------------------------
    # Interpretation box
    # -----------------------------
    st.markdown("### Interpretation")

    if signal == "Healthy":
        st.success(
            f"Market breadth looks healthy (score {score:+d}). A large share of S&P 500 stocks are participating in the move."
        )
    elif signal == "Weak / Unstable":
        st.error(
            f"Market breadth looks weak (score {score:+d}). The index may be supported by a narrow group of large-cap stocks."
        )
    else:
        st.warning(
            f"Market breadth is mixed (score {score:+d}). Some internal indicators are supportive, while others show caution."
        )
    with st.expander("Signal breakdown"):
        for line in details:
            st.markdown(f"- {line}")


    # -----------------------------
    # Charts
    # -----------------------------
    st.markdown("---")
    st.subheader("Breadth Charts")

    chart1, chart2 = st.columns(2)

    with chart1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(breadth.index, breadth["A/D Line"])
        ax.set_title("Advance-Decline Line")
        ax.set_ylabel("Cumulative Net Advancers")
        ax.grid(True)
        st.pyplot(fig)

    with chart2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(breadth.index, breadth["% Above 50DMA"], label="% Above 50DMA")
        ax.plot(breadth.index, breadth["% Above 200DMA"], label="% Above 200DMA")
        ax.axhline(0.60, linestyle="--", alpha=0.5)
        ax.axhline(0.40, linestyle="--", alpha=0.5)
        ax.set_title("Percentage of Stocks Above Moving Averages")
        ax.set_ylabel("Percentage")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


    chart3, chart4 = st.columns(2)

    with chart3:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(breadth.index, breadth["52W New Highs"], label="New Highs")
        ax.bar(breadth.index, -breadth["52W New Lows"], label="New Lows")
        ax.set_title("52-Week New Highs vs New Lows")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    with chart4:
        if "RSP" in etf_close.columns and "SPY" in etf_close.columns:
            ratio = etf_close["RSP"] / etf_close["SPY"]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(ratio.index, ratio)
            ax.set_title("RSP/SPY Ratio: Equal Weight vs Cap Weight")
            ax.set_ylabel("RSP / SPY")
            ax.grid(True)
            st.pyplot(fig)

            latest_ratio = ratio.dropna().iloc[-1]
            ratio_50ma = ratio.rolling(50).mean().dropna().iloc[-1]

            if latest_ratio > ratio_50ma:
                st.success("RSP/SPY is above its 50DMA: broader participation is improving.")
            else:
                st.warning("RSP/SPY is below its 50DMA: mega-cap concentration may be increasing.")
        else:
            st.warning("RSP or SPY data unavailable.")


    # -----------------------------
    # Sector breadth
    # -----------------------------
    st.markdown("---")
    st.subheader("Sector Breadth")

    sector_display = sector_breadth.copy()
    sector_display["% Above 50DMA"] = sector_display["% Above 50DMA"].map("{:.1%}".format)
    sector_display["% Above 200DMA"] = sector_display["% Above 200DMA"].map("{:.1%}".format)

    st.dataframe(
        sector_display,
        width='stretch',
        hide_index=True
    )


    # -----------------------------
    # SPY confirmation
    # -----------------------------
    st.markdown("---")
    st.subheader("SPY Price Confirmation")

    if "SPY" in etf_close.columns:
        spy = etf_close["SPY"].dropna()
        spy_ma50 = spy.rolling(50).mean()
        spy_ma200 = spy.rolling(200).mean()

        latest_spy = spy.iloc[-1]
        latest_spy_50 = spy_ma50.iloc[-1]
        latest_spy_200 = spy_ma200.iloc[-1]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(spy.index, spy, label="SPY")
        ax.plot(spy_ma50.index, spy_ma50, label="50DMA")
        ax.plot(spy_ma200.index, spy_ma200, label="200DMA")
        ax.set_title("SPY with 50DMA and 200DMA")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)

        annotation_text = (
            f"Latest: {latest_spy:.2f}\n"
            f"50DMA: {latest_spy_50:.2f}\n"
            f"200DMA: {latest_spy_200:.2f}"
        )
        ax.annotate(
            annotation_text,
            xy=(spy.index[-1], latest_spy),
            xytext=(-150, 60),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="angle3", color="gray"),
            fontsize=10,
            horizontalalignment="right",
            verticalalignment="bottom",
        )

        st.pyplot(fig)

        if latest_spy > latest_spy_50 > latest_spy_200:
            st.success("SPY trend is technically strong: price > 50DMA > 200DMA.")
        elif latest_spy < latest_spy_50:
            st.warning("SPY is below its 50DMA: short-term momentum is weakening.")
        elif latest_spy < latest_spy_200:
            st.error("SPY is below its 200DMA: long-term trend risk is elevated.")
    else:
        st.warning("SPY data unavailable.")


    # -----------------------------
    # Raw data
    # -----------------------------
    if show_raw:
        st.markdown("---")
        st.subheader("Raw Breadth Data")
        st.dataframe(breadth.tail(100), width='stretch')