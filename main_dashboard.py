import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

REPO_ROOT = ensure_repo_root(REPO_ROOT)

# --- SYSTEM CONFIG ---
st.set_page_config(page_title="ARE Macro Terminal", layout="wide")

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