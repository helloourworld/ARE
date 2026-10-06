import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import sys


def _find_repo_root(start_path: Path) -> Path:
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "enable_repo_root.py").exists():
            return candidate
    return start_path


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enable_repo_root import ensure_repo_root
from data_pipeline import get_price_history, get_10y_yield
from fundamental.Economic import get_indicator_history, compute_macro_health_summary

REPO_ROOT = ensure_repo_root(REPO_ROOT)

# --- SYSTEM CONFIG ---
st.set_page_config(page_title="ARE Macro Terminal", layout="wide")


def render_fundamental_dashboard():
    st.title("Fundamental Macro Monitor")
    st.caption("Core macro and recession data pulled from FRED. Each chart shows the recent trend for one key economic signal.")

    try:
        macro_df = get_indicator_history(years=10)
    except Exception as exc:
        st.error(f"Unable to load economic data: {exc}")
        return

    if macro_df.empty:
        st.warning("No economic series returned from FRED. Check the API key or network connection.")
        return

    numeric_cols = [c for c in macro_df.columns if pd.api.types.is_numeric_dtype(macro_df[c])]
    if not numeric_cols:
        st.warning("No numeric series available for charting.")
        return

    latest_row = macro_df[macro_df.notna().any(axis=1)].tail(1).copy()
    if latest_row.empty:
        st.warning("No recent data available for the selected macro series.")
        return

    latest_values = latest_row.iloc[-1].to_dict()
    if not any(pd.notna(v) for v in latest_values.values()):
        st.warning("No recent data available for the selected macro series.")
        return

    latest_valid_date = latest_row.index[-1].strftime("%Y-%m-%d")
    st.caption(
        f"Latest valid macro data: {latest_valid_date}. Some FRED series update on different schedules, so a few indicators may lag by a month or more."
    )

    health_summary = compute_macro_health_summary(macro_df)
    st.subheader("Macro health score")
    col1, col2, col3 = st.columns([1.2, 1.2, 2.5])
    col1.metric("Macro score", f"{health_summary['score']:.1f}/100")
    col2.metric("Regime", health_summary["label"])
    col3.empty()
    st.progress(min(1.0, max(0.0, health_summary["score"] / 100)))
    st.caption(f"Research read: {health_summary['label']} macro backdrop with the current FRED/Chicago mix.")

    scorecard_data = []
    for label, value in latest_values.items():
        if pd.isna(value):
            continue
        scorecard_data.append({"Indicator": label, "Latest": float(value)})

    if scorecard_data:
        score_df = pd.DataFrame(scorecard_data)
        score_df["Trend"] = score_df["Latest"].apply(
            lambda x: "Higher" if x > score_df["Latest"].median() else "Lower"
        )
        st.subheader("Macro scorecard")
        st.dataframe(score_df.sort_values("Indicator").reset_index(drop=True), use_container_width=True)

    st.divider()
    st.subheader("Macro trend heatmap")
    heatmap_df = macro_df[numeric_cols].dropna(how="all").copy()
    if not heatmap_df.empty:
        heatmap_df = heatmap_df.tail(24)
        standard_deviation = heatmap_df.std(ddof=0).replace(0, 1)
        heatmap_df = (heatmap_df - heatmap_df.mean()) / standard_deviation
        styled_heatmap = heatmap_df.style.background_gradient(
            cmap="RdYlGn", vmin=-3, vmax=3
        ).format("{:.2f}")
        st.dataframe(styled_heatmap, use_container_width=True)

    st.divider()
    cols = st.columns(2)
    for idx, col in enumerate(numeric_cols):
        with cols[idx % 2]:
            st.subheader(col)
            series = macro_df[[col]].dropna()
            if series.empty:
                st.write("No observations")
                continue
            st.line_chart(series)
            latest = series.iloc[-1].iloc[0]
            st.caption(f"Latest: {latest:.3f} ({series.index[-1].date()})")

    st.divider()
    st.subheader("Latest values")
    latest_table = macro_df.dropna().tail(1).T
    latest_table.columns = ["value"]
    st.dataframe(latest_table, use_container_width=True)


MULTIPLE_UNIVERSE = {
    "Mega-cap technology": ["AAPL", "GOOG", "MSFT", "AMZN", "NVDA", "META"],
    "Broad market": ["SPY", "QQQ", "IWM", "DIA"],
    "Semiconductors": ["AMD", "AVGO", "INTC", "MU", "MRVL"],
    "Energy and materials": ["XOM", "CVX", "COP", "GLD", "SLV"],
}


@st.cache_data(ttl=3600)
def fetch_multiple_data(tickers):
    """Return current valuation multiples without failing the whole screen on one symbol."""
    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            info = {}

        def number(key):
            value = info.get(key)
            return float(value) if isinstance(value, (int, float)) and pd.notna(value) else None

        rows.append({
            "Ticker": ticker,
            "Name": info.get("shortName", ticker),
            "Price": number("currentPrice") or number("regularMarketPrice"),
            "Market Cap": number("marketCap"),
            "P/E (TTM)": number("trailingPE"),
            "P/E (Fwd)": number("forwardPE"),
            "Price / Sales": number("priceToSalesTrailing12Months"),
            "Price / Book": number("priceToBook"),
            "EV / EBITDA": number("enterpriseToEbitda"),
            "Dividend Yield": number("dividendYield"),
        })
    return pd.DataFrame(rows)


def render_multiples_dashboard():
    st.title("Market Multiples")
    st.caption("A compact valuation directory for selected market groups. Quotes and multiples are sourced from Yahoo Finance and may be delayed.")

    group = st.sidebar.selectbox("Universe", list(MULTIPLE_UNIVERSE))
    search = st.text_input("Search ticker or company", placeholder="Try NVDA or Microsoft").strip().lower()
    tickers = MULTIPLE_UNIVERSE[group]
    multiples = fetch_multiple_data(tuple(tickers))
    if search:
        multiples = multiples[
            multiples["Ticker"].str.lower().str.contains(search)
            | multiples["Name"].str.lower().str.contains(search)
        ]

    if multiples.empty:
        st.warning("No matching securities found.")
        return

    st.subheader(f"{group} valuation screen")
    display = multiples.copy()
    display["Market Cap"] = display["Market Cap"].map(
        lambda value: f"${value / 1e9:,.1f}B" if pd.notna(value) else "n/a"
    )
    for column in ["Price", "P/E (TTM)", "P/E (Fwd)", "Price / Sales", "Price / Book", "EV / EBITDA"]:
        display[column] = display[column].map(lambda value: f"{value:,.2f}" if pd.notna(value) else "n/a")
    display["Dividend Yield"] = display["Dividend Yield"].map(
        lambda value: f"{value:.2%}" if pd.notna(value) else "n/a"
    )
    st.dataframe(display.set_index("Ticker"), use_container_width=True, hide_index=False)

    selected = st.selectbox("Security detail", multiples["Ticker"].tolist())
    selected_row = multiples.loc[multiples["Ticker"] == selected].iloc[0]
    st.divider()
    st.subheader(f"{selected} · {selected_row['Name']}")
    metric_columns = st.columns(5)
    for column, label in zip(metric_columns, ["P/E (TTM)", "P/E (Fwd)", "Price / Sales", "Price / Book", "EV / EBITDA"]):
        value = selected_row[label]
        column.metric(label, f"{value:.2f}" if pd.notna(value) else "n/a")

    try:
        history = get_price_history([selected], period="5y", interval="1d")
        if selected in history.columns and not history[selected].dropna().empty:
            st.line_chart(history[selected].dropna().rename("Close"))
    except Exception as exc:
        st.info(f"Price history unavailable for {selected}: {exc}")


VIEW = st.sidebar.radio("Menu", ["Macro Terminal", "Fundamental", "Multiples"], index=0)
if VIEW in ["Macro Terminal", "Fundamental"]:
    render_fundamental_dashboard()
    st.stop()
if VIEW == "Multiples":
    render_multiples_dashboard()
    st.stop()

# --- LIVE DATA INGESTION ENGINE ---
@st.cache_data(ttl=3600)  # Automates update: Refreshes every 3600 seconds (1 hour)
def fetch_macro_data():
    # Ticker Mapping: BZ=F (Brent), GC=F (Gold), ^VIX (Volatility), USDCAD=X (FX)
    ticker_map = {
        "BZ=F": "Brent Crude",
        "GC=F": "Gold",
        "^VIX": "VIX Index",
        "USDCAD=X": "USD/CAD",
        "AMD": "AMD"
    }
    
    tickers = list(ticker_map.keys())
    # Download 5 days to calculate delta/change
    data = get_price_history(tickers, period="5d", interval="1d")
    stats = {}
    for t_id, name in ticker_map.items():
        if t_id not in data.columns:
            raise ValueError(f"Missing ticker data for {t_id} ({name})")

        series = data[t_id].dropna()
        if len(series) < 2:
            raise ValueError(f"Not enough data points for {t_id} ({name}) to compute delta")

        current_val = series.iloc[-1]
        prev_val = series.iloc[-2]
        delta = current_val - prev_val
        delta_pct = (delta / prev_val) * 100
            
        stats[name] = {
            "val": current_val,
            "delta": delta,
            "delta_pct": delta_pct
        }

    stats["US 10Y Yield"] = get_10y_yield()
    return stats

# --- RUN AUTO-UPDATE ---
try:
    live_stats = fetch_macro_data()
    st.sidebar.success(f"Last Sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
except Exception as e:
    st.error(f"Data Pipeline Error: {e}")
    st.stop()

# --- HEADER: THE ALPHA PULSE ---
st.title("Alpha Risk Engine: Institutional Macro Terminal")
st.caption(f"Strategy: Global Equity Alpha | Analyst: CFA Lead")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("AMD", 
              f"${live_stats['AMD']['val']:.2f}", 
              f"{live_stats['AMD']['delta_pct']:+.2f}%")
    st.caption("Post-Earnings Momentum")

with m2:
    st.metric("US 10Y Yield", 
              f"{live_stats['US 10Y Yield']['val']:.3f}%", 
              f"{live_stats['US 10Y Yield']['delta']:+.3f}")
    st.caption("Discount Rate Benchmark")

with m3:
    st.metric("Brent Crude", 
              f"${live_stats['Brent Crude']['val']:.2f}", 
              f"{live_stats['Brent Crude']['delta_pct']:+.2f}%", 
              delta_color="inverse")
    st.caption("Inflation Stress Point")

with m4:
    st.metric("VIX Index", 
              f"{live_stats['VIX Index']['val']:.2f}", 
              f"{live_stats['VIX Index']['delta_pct']:+.2f}%", 
              delta_color="inverse")
    st.caption("Market Volatility (Fear)")

# --- SECTION 1: FX & GOLD ---
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Currency Radar")
    st.metric("USD/CAD", 
              f"{live_stats['USD/CAD']['val']:.4f}", 
              f"{live_stats['USD/CAD']['delta_pct']:+.2f}%")
    st.write("**Consultant Note:** Direct US equity (MSFT/GOOG) value increases as this rises.")

with c2:
    st.subheader("Safe Haven Audit")
    st.metric("Gold Spot", 
              f"${live_stats['Gold']['val']:,.2f}", 
              f"{live_stats['Gold']['delta_pct']:+.2f}%")
    st.write("**Status:** Diversification benefit active via KILO.TO.")

with c3:
    st.subheader("Market Regime")
    if live_stats['VIX Index']['val'] < 18:
        st.success("Regime: Risk-On / Low Vol")
    else:
        st.warning("Regime: High Variance / Risk-Off")

# --- SECTION 2: LIVE CHARTING ---
st.divider()
st.subheader("Post-Earnings Drift: Tech Leaders")

@st.cache_data(ttl=3600)
def get_chart_data():
    # Comparing current AI leaders
    tks = ["AMD", "GOOG", "MSFT", "SNDK"]
    d = get_price_history(tks, period="1mo", interval="1d")
    # Normalize to 100 for comparison
    return (d / d.dropna().iloc[0] * 100)

chart_df = get_chart_data()
st.line_chart(chart_df)

# --- FOOTER: GOVERNANCE ---
st.divider()
if st.button("Generate GIPS-Compliant Daily Summary"):
    st.write(f"**ARE Summary for {datetime.now().date()}:**")
    st.write(f"Portfolio exposure is currently {live_stats['VIX Index']['val'] < 18 and 'Optimized' or 'Defensive'}. "
             f"Yield pressure on tech is {live_stats['US 10Y Yield']['val'] > 4.4 and 'High' or 'Moderate'}.")