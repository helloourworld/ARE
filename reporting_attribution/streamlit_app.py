# Standard library
import datetime
import logging
import os
from pathlib import Path
import sys

# Avoid loky physical-core detection warning on some Windows setups.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


def _find_repo_root(start_path: Path) -> Path:
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "enable_repo_root.py").exists():
            return candidate
    return start_path


# Repository startup helper
REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enable_repo_root import ensure_repo_root
REPO_ROOT = ensure_repo_root(REPO_ROOT)

# Third-party libraries
import appdirs as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import statsmodels.api as sm
import streamlit.components.v1 as components
import yfinance as yf
import yaml
from sklearn.covariance import LedoitWolf
from pypfopt import (
    black_litterman,
    risk_models,
    BlackLittermanModel,
    EfficientFrontier,
    objective_functions,
)

# Local application modules
from frontier_plots import plot_institutional_frontier
from data_pipeline import get_daily_returns, get_price_history, get_price_history_with_benchmark, get_premarket_data, get_live_intraday, get_data_persistent, get_official_session_open
from risk_modeling import AlphaRiskEngine, calculate_mansfield_rs, monitor_mean_reversion, calculate_rs_bollinger_bands, get_rs_signals, detect_rs_hook
from risk_modeling.mandelbrot import scan_market
from risk_modeling.market_breadth import (
    get_sp500_tickers, download_prices, download_etfs,
    calculate_breadth, calculate_sector_breadth, breadth_signal, append_current_prices,
)
from risk_modeling.bolling_bands import compute_rolling_vpin, compute_rolling_cvd
from data_pipeline.data_cache import DATA_DIR
from alpha_research.price_forecasting import run_all, RiskRulesConfig
from alpha_research.Port_Stock_dd import (
    calculate_drawdown_analysis,
    calculate_drawdown_stats,
    plot_drawdown_histogram,
    plot_multiple_stocks_from_cache,
    plot_recovery_histogram,
)
from alpha_research.Port_stock_watch import plot_market_data_from_cache
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Apply default request headers for all requests sessions so yfinance uses headers
DEFAULT_REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

_original_requests_session_request = requests.Session.request


def _requests_session_request_with_headers(self, method, url, *args, **kwargs):
    headers = kwargs.get('headers', {}) or {}
    if not isinstance(headers, dict):
        headers = dict(headers)
    merged_headers = {**DEFAULT_REQUEST_HEADERS, **headers}
    kwargs['headers'] = merged_headers
    return _original_requests_session_request(self, method, url, *args, **kwargs)


requests.Session.request = _requests_session_request_with_headers

# Create a valid path in your Windows Temp directory
cache_path = os.path.join(
    os.environ['TEMP'], 'yfinance') if os.name == 'nt' else ad.user_cache_dir("yfinance")
if not os.path.exists(cache_path):
    os.makedirs(cache_path)
yf.set_tz_cache_location(cache_path)

# --- Capture mandelbrot WARNING logs into session state ---
_MANDELBROT_HANDLER_TAG = "are_mandelbrot_session_handler"


class _SessionStateLogHandler(logging.Handler):
    """Forwards WARNING+ records from risk_modeling.mandelbrot to st.session_state."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            if "mandelbrot_warnings" not in st.session_state:
                st.session_state["mandelbrot_warnings"] = []
            msg = self.format(record)
            warnings_list = st.session_state["mandelbrot_warnings"]
            # Prevent consecutive duplicate warnings caused by reruns/duplicate emits.
            if warnings_list and warnings_list[-1].endswith(f"| {msg}"):
                return
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            warnings_list.append(f"{ts} | {msg}")
        except Exception:
            pass

_mandelbrot_logger = logging.getLogger("risk_modeling.mandelbrot")
# Keep exactly one session-state warning handler across Streamlit reruns.
_mandelbrot_logger.handlers = [
    h for h in _mandelbrot_logger.handlers
    if getattr(h, "_are_tag", None) != _MANDELBROT_HANDLER_TAG
]
_handler = _SessionStateLogHandler(level=logging.WARNING)
_handler.setFormatter(logging.Formatter("%(message)s"))
_handler._are_tag = _MANDELBROT_HANDLER_TAG
_mandelbrot_logger.addHandler(_handler)

# Local modules

# Prefer setting PYTHONPATH or using a package structure with __init__.py files.

# --- CONSTANTS ---
PORTFOLIO_VALUE = 10_000
RS_WINDOW = 50
RS_LOOKBACK_WINDOW = 200
ANNUAL_TRADING_DAYS = 252
BUBBLE_Z_THRESHOLD = 2.5
BUBBLE_PCT_FALLBACK = 50.0


def get_regime_icon(regime: str) -> str:
    regime_text = (regime or "").upper()
    if "1 - BULLISH" in regime_text:
        return "🟢"
    if "2 - BEARISH" in regime_text:
        return "🔴"
    if "3 - NEUTRAL" in regime_text or "RANDOM WALK" in regime_text:
        return "🟡"
    if "4 - UNSTABLE" in regime_text:
        return "⚠️"
    if "5 - TAIL RISK" in regime_text:
        return "🚨"
    if regime_text.strip():
        return "🔎"
    return "❔"


def get_cvd_icon(cvd_value) -> str:
    try:

        if "UP" in cvd_value.upper() or "⬆️" in cvd_value or float(cvd_value) > 0:
            return "⬆️"
        if "DOWN" in cvd_value.upper() or "⬇️" in cvd_value or float(cvd_value) < 0:
            return "⬇️"
        if "FLAT" in cvd_value.upper() or "→" in cvd_value:
            return "→"
        # Parse fallback: numeric slope sign
        slope = float(cvd_value.split()[0])
        return "⬆️" if slope > 0 else "⬇️" if slope < 0 else "→"
    except Exception:
        pass
    return "→"


def normalize_timestamp_for_index(value, index):
    ts = pd.to_datetime(value)
    if getattr(index, "tz", None) is not None and index.tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def get_reference_price(ticker: str, series: pd.Series):
    if series is None or series.empty:
        return None, "Unknown"

    idx = series.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    est_index = idx.tz_convert("America/New_York")
    last_ts = est_index[-1]
    current_date = last_ts.date()
    premarket = last_ts.time() < datetime.time(9, 30)

    prev_close_vals = [i for i, d in enumerate(est_index.date) if d < current_date]
    # prev_close = series.iloc[prev_close_vals[-1]] if prev_close_vals else None
    _, hist = get_premarket_data(selected_benchmark)

    # 1. Previous Day Close
    if premarket:
        prev_close = hist[selected_benchmark].iloc[-1]
    else:
        prev_close = hist[selected_benchmark].iloc[-2]
        
    today_open_mask = [d == current_date and t >= datetime.time(9, 30)
                       for d, t in zip(est_index.date, est_index.time)]
    today_open = get_official_session_open(ticker, current_date) if any(today_open_mask) else None

    if premarket or today_open is None:
        return prev_close, "Previous Close"
    return today_open, "Official Open"


def render_metric_with_threshold(name: str, value: float, threshold_text: str, color: str, precision: int = 3):
    formatted_value = f"{value:.{precision}f}" if isinstance(value, (float, int)) else value
    st.markdown(
        f"**{name}:** <span style='color:{color}; font-weight:bold'>{formatted_value}</span> "
        f"<span style='color:#888;'>({threshold_text})</span>",
        unsafe_allow_html=True
    )


def format_hurst_color(value: float) -> str:
    if value >= 0.60:
        return "green"
    if value >= 0.53:
        return "orange"
    return "red"


def format_tail_color(value: float) -> str:
    if value >= 1.70:
        return "green"
    if value >= 1.55:
        return "orange"
    return "red"


def format_vpin_color(value: float) -> str:
    if value >= 0.75:
        return "red"
    if value >= 0.60:
        return "orange"
    return "green"


def format_vol_color(value: float) -> str:
    vol_threshold = 0.0015
    if value >= vol_threshold * 2:
        return "red"
    if value >= vol_threshold:
        return "orange"
    return "green"


def render_headline_benchmark_alert() -> None:
    """Show a global top banner when live benchmark move exceeds threshold."""
    change_pct = st.session_state.get("live_benchmark_change_pct")
    benchmark = st.session_state.get("live_benchmark_symbol")
    baseline = st.session_state.get("live_benchmark_baseline")

    if not isinstance(change_pct, (float, int)):
        return

    if change_pct > 1.5:
        label = benchmark or "Benchmark"
        basis = f" vs {baseline}" if isinstance(baseline, str) and baseline else ""
        st.warning(
            f"🚨 **Market Momentum Alert:** {label} is up **{float(change_pct):.2f}%**{basis}. "
            "Upside can become fragile in fast melt-up conditions; tighten stops and avoid chasing late entries."
        )


# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Alpha Risk Engine (ARE)", page_icon="🤖", layout="wide")

@st.fragment(run_every="60s")
def update_browser_tab_title():
    try:
        monitor_list = ['SPY']
        live_data = get_live_intraday(monitor_list, period="1d")

        bench_series = live_data['SPY'].dropna()
        bench_current = bench_series.iloc[-1]
        # st.text(f"Current SPY Price: ${bench_current:,.2f}")
        # JavaScript to dynamically update the browser tab title
        html_script = f"""
            10 {bench_current:,.2f}
        """
        st.set_page_config(page_title=html_script, page_icon="🤖", layout="wide")
        # st.title(html_script, height=0, width=0)
    except Exception:
        st.error("Error updating browser tab title. Please check your internet connection or data source.")
# Run browser title updater in background
update_browser_tab_title()

# --- LOAD CONFIGURATION ---
def load_config():
    config_path = REPO_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


cfg = load_config()

# --- SIDEBAR: DYNAMIC UNIVERSE SELECTION ---
st.sidebar.header("ARE Control Panel")

# Flatten the universe categories into one list for the selector
master_universe = (
    cfg['universe']['core'] +
    cfg['universe']['active_growth'] +
    cfg['universe']['passive_em'] +
    cfg['universe']['intl_hedged'] +
    cfg['universe']['us_tech'] +
    cfg['universe']['cad_value']

)

selected_benchmark = st.sidebar.selectbox(
    "Reference Benchmark",
    options=cfg['universe']['benchmarks'],
    index=cfg['universe']['benchmarks'].index(cfg['defaults']['benchmark'])
)

# --- SIDEBAR: SELECTION PANEL ---
st.sidebar.header("ARE Selection Panel")

# 1. Initialize 'external_tickers' in session state if it doesn't exist
if 'external_tickers' not in st.session_state:
    # Start with any pre-defined tickers in config
    st.session_state['external_tickers'] = cfg['external_tickers']

# 2. Add External Ticker Input
with st.sidebar.expander("➕ Add External Symbol", expanded=False):
    new_ticker = st.text_input(
        "Enter Ticker (e.g. MU, ARM, XCHP.TO)", key="ticker_input").upper()
    if st.button("Add to Universe"):
        if new_ticker and new_ticker not in st.session_state['external_tickers']:
            # Validate ticker existence with yfinance before adding
            try:
                check = yf.Ticker(new_ticker).fast_info
                st.session_state['external_tickers'].append(new_ticker.upper())
                st.sidebar.success(f"Added {new_ticker}")
                # Refresh app to update multiselect
                st.rerun()
            except Exception as e:
                print(e)
                st.error("Invalid Ticker Symbol")

# 3. Combine Core Universe with External Tickers
master_universe = list(master_universe + st.session_state['external_tickers'])

# 4. Clear External Tickers (Housekeeping)
if st.sidebar.button("Clear External Tickers"):
    st.session_state['external_tickers'] = []
    st.rerun()

# 5. Multiselect for Active Analysis
selected_tickers = st.sidebar.multiselect(
    "Select Universe for Attribution",
    options=sorted(master_universe),
    default=cfg['defaults']['selected_portfolio']
)

# VPIN / CVD controls
st.sidebar.header("VPIN / CVD Controls")
smoothing_window = st.sidebar.slider("Smoothing window (bars)", min_value=1, max_value=21, value=3, step=1)
vpin_window_minutes = st.sidebar.slider("VPIN lookback window (minutes)", min_value=5, max_value=1440, value=250, step=5)
vpin_bucket_count = st.sidebar.slider("VPIN bucket count", min_value=10, max_value=200, value=50, step=5)

# Use start_date from config
returns = get_daily_returns(
    selected_tickers,
    selected_benchmark,
    cfg['defaults']['start_date']
)
if returns.empty:
    st.error(
        "No data available for the selected tickers and date range. Please adjust your selection.")
    st.stop()

# Keep a NaN-free returns matrix for risk modeling and regressions.
# Instead of dropping assets with incomplete history, forward-fill NaN values
incomplete_cols = [col for col in returns.columns if returns[col].isna().any()]
if incomplete_cols:
    returns[incomplete_cols] = returns[incomplete_cols].ffill().bfill().fillna(0)
    # st.info(
    #     f"⚠️ Forward-filled missing returns for: {', '.join(sorted(incomplete_cols))} (using ffill → bfill → 0)"
    # )

if selected_benchmark not in returns.columns:
    st.error(
        f"Benchmark {selected_benchmark} has incomplete return history for this window. Please choose another benchmark or start date."
    )
    st.stop()

selected_tickers = [
    ticker for ticker in selected_tickers if ticker in returns.columns and ticker != selected_benchmark
]
if not selected_tickers:
    st.error(
        "No analyzable assets remain. Please adjust your selection."
    )
    st.stop()

returns = returns[selected_tickers + [selected_benchmark]].dropna(how='any')
if returns.empty:
    st.error(
        "No overlapping complete return history remains after cleaning missing values. Please adjust your selection or start date."
    )
    st.stop()

# --- DISPLAY METADATA ---
# st.title(cfg['metadata']['report_title'])
# st.caption(
#     f"Analyst: {cfg['metadata']['analyst_name']} | Strategy: {cfg['metadata']['strategy_id']}")
render_headline_benchmark_alert()

# Access parameters for math
rf = cfg['parameters']['risk_free_rate']
st.write(f"Risk-Free Rate (Annualized Proxy): {rf:.1%}")

# Display returns data preview
with st.expander("📊 Returns Data Preview"):
    st.dataframe(returns.tail(3).style.format("{:.2%}"), width='stretch')
    st.caption(
        f"Data shape: {returns.shape[0]} periods × {returns.shape[1]} assets | Starting: {returns.index[0].date()}")


market_caps = {ticker: yf.Ticker(ticker).info.get(
    'marketCap', 0) for ticker in selected_tickers+[selected_benchmark]}


# --- APP TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab_dd, tab_watch, *rest = st.tabs(
    ["📈 MARKET BREADTH", "⚖️LIVE MARKET", "📊 MARKET RISK", "📡 ALPHA PERSISTENCE", "🔮 FORECAST PORTAL",
     "🧠 FACTOR ATTRIBUTION", "🛡️ RISK REPORT", "📶 REBALANCING (BL)", "🧭 EFFICIENT FRONTIER", "🧪 SCENARIO STRESS TEST",
     "DD", "Watch"])

# =============================================================================
# TAB 1: MARKET BREADTH PORTAL
# =============================================================================
with tab1:
    st.header("📈 S&P 500 Market Breadth Portal")
    st.caption(
        "Checks whether the S&P 500 rally is broad-based or concentrated in a few large-cap names."
    )

    mb_period = st.selectbox("Historical period", ["1y", "2y", "5y"], index=1, key="mb_period")
    if st.button("🔄 Refresh Breadth Data", key="mb_refresh"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Loading S&P 500 price data…"):
        mb_tickers, mb_sector_map = get_sp500_tickers()
        mb_close   = download_prices(mb_tickers, period=mb_period)
        mb_etf     = download_etfs(period=mb_period)

        # Append current prices (today's latest data)
        mb_close = append_current_prices(mb_close, mb_tickers)
        mb_etf   = append_current_prices(mb_etf, ["SPY", "RSP", "QQQ", "IWM", "^VIX"])

    mb_breadth      = calculate_breadth(mb_close)
    mb_sector_df    = calculate_sector_breadth(mb_close, mb_sector_map)
    mb_latest       = mb_breadth.dropna().iloc[-1]
    mb_latest_date  = mb_breadth.dropna().index[-1].date()
    mb_spy_series   = mb_etf["SPY"].dropna() if "SPY" in mb_etf.columns else None
    mb_signal, mb_color, mb_score, mb_details = breadth_signal(mb_latest, mb_spy_series)

    # ── Summary metrics ──────────────────────────────────────────────────────
    st.subheader(f"Latest Breadth Reading: {mb_latest_date}")
    _color_map = {"green": "normal", "orange": "off", "red": "inverse"}
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Signal",          mb_signal)
    c2.metric("Score",           f"{mb_score:+d} / 5")
    c3.metric("A/D Ratio",       f"{mb_latest['A/D Ratio']:.2f}")
    c4.metric("% > 50DMA",       f"{mb_latest['% Above 50DMA']:.1%}")
    c5.metric("% > 200DMA",      f"{mb_latest['% Above 200DMA']:.1%}")
    c6.metric("Advancers",       int(mb_latest["Advancers"]))
    c7.metric("52W New Highs",   int(mb_latest["52W New Highs"]))
    c8.metric("52W New Lows",    int(mb_latest["52W New Lows"]))

    # ── Interpretation ───────────────────────────────────────────────────────
    _msg = {
        "Healthy":        f"✅ Market breadth is **healthy** (score {mb_score:+d}). Broad participation supports the rally.",
        "Weak / Unstable":f"🚨 Market breadth is **weak** (score {mb_score:+d}). Rally may be narrow / fragile.",
        "Neutral / Mixed":f"⚠️ Market breadth is **mixed** (score {mb_score:+d}). Watch for confirmation.",
    }
    if mb_color == "green":
        st.success(_msg[mb_signal])
    elif mb_color == "red":
        st.error(_msg[mb_signal])
    else:
        st.warning(_msg[mb_signal])

    with st.expander("Signal breakdown"):
        for _d in mb_details:
            st.markdown(f"- {_d}")

    st.divider()

    # ── Charts row 1: A/D Line + % Above MAs ─────────────────────────────────
    st.subheader("Breadth Charts")
    _r1c1, _r1c2 = st.columns(2)

    with _r1c1:
        _fig = go.Figure()
        _fig.add_trace(go.Scatter(
            x=mb_breadth.index, y=mb_breadth["A/D Line"],
            mode="lines", name="A/D Line", line=dict(color="#1f77b4")
        ))
        _fig.update_layout(title="Advance-Decline Line", yaxis_title="Cumulative Net Advancers",
                           height=350, margin=dict(t=40, b=20))
        st.plotly_chart(_fig, width="stretch")

    with _r1c2:
        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=mb_breadth.index, y=mb_breadth["% Above 50DMA"],
                                  mode="lines", name="% Above 50DMA", line=dict(color="#2ca02c")))
        _fig.add_trace(go.Scatter(x=mb_breadth.index, y=mb_breadth["% Above 200DMA"],
                                  mode="lines", name="% Above 200DMA", line=dict(color="#d62728")))
        _fig.add_hline(y=0.60, line_dash="dash", line_color="gray", annotation_text="60%")
        _fig.add_hline(y=0.40, line_dash="dash", line_color="gray", annotation_text="40%")
        _fig.update_layout(title="% of Stocks Above Moving Averages", yaxis_title="Fraction",
                           yaxis_tickformat=".0%", height=350, margin=dict(t=40, b=20))
        st.plotly_chart(_fig, width="stretch")

    # ── Charts row 2: 52W Highs/Lows + RSP/SPY ───────────────────────────────
    _r2c1, _r2c2 = st.columns(2)

    with _r2c1:
        _fig = go.Figure()
        _fig.add_trace(go.Bar(x=mb_breadth.index, y=mb_breadth["52W New Highs"],
                              name="New Highs", marker_color="#2ca02c"))
        _fig.add_trace(go.Bar(x=mb_breadth.index, y=-mb_breadth["52W New Lows"],
                              name="New Lows", marker_color="#d62728"))
        _fig.update_layout(barmode="overlay", title="52-Week New Highs vs New Lows",
                           yaxis_title="Count", height=350, margin=dict(t=40, b=20))
        st.plotly_chart(_fig, width="stretch")

    with _r2c2:
        if "RSP" in mb_etf.columns and "SPY" in mb_etf.columns:
            _ratio = (mb_etf["RSP"] / mb_etf["SPY"]).dropna()
            _ratio_50 = _ratio.rolling(50).mean()
            _fig = go.Figure()
            _fig.add_trace(go.Scatter(x=_ratio.index, y=_ratio,
                                      mode="lines", name="RSP/SPY", line=dict(color="#1f77b4")))
            _fig.add_trace(go.Scatter(x=_ratio_50.index, y=_ratio_50,
                                      mode="lines", name="50DMA", line=dict(color="orange", dash="dash")))
            _fig.update_layout(title="RSP/SPY: Equal-Weight vs Cap-Weight",
                               yaxis_title="Ratio", height=350, margin=dict(t=40, b=20))
            st.plotly_chart(_fig, width="stretch")
            _latest_r = _ratio.iloc[-1]; _50ma_r = _ratio_50.dropna().iloc[-1]
            if _latest_r > _50ma_r:
                st.success("RSP/SPY above 50DMA → broader participation is improving.")
            else:
                st.warning("RSP/SPY below 50DMA → mega-cap concentration may be increasing.")

    st.divider()

    # ── SPY Price Confirmation ────────────────────────────────────────────────
    st.subheader("SPY Price Confirmation")
    if mb_spy_series is not None and len(mb_spy_series) >= 50:
        _spy = mb_spy_series
        _spy_50  = _spy.rolling(50).mean()
        _spy_200 = _spy.rolling(200).mean()
        _cur_price = _spy.iloc[-1]
        _cur_50    = _spy_50.dropna().iloc[-1]
        _cur_200   = _spy_200.dropna().iloc[-1]
        _cur_date  = _spy.index[-1]

        # Price summary row
        _sc1, _sc2, _sc3 = st.columns(3)
        _sc1.metric("SPY Current Price", f"${_cur_price:.2f}")
        _sc2.metric("50DMA",  f"${_cur_50:.2f}",  delta=f"{(_cur_price/_cur_50-1)*100:+.2f}% vs price")
        _sc3.metric("200DMA", f"${_cur_200:.2f}", delta=f"{(_cur_price/_cur_200-1)*100:+.2f}% vs price")

        _fig = go.Figure()
        _fig.add_trace(go.Scatter(x=_spy.index, y=_spy.values,
                                  mode="lines", name="SPY", line=dict(color="#1f77b4", width=1.5)))
        _fig.add_trace(go.Scatter(x=_spy_50.index, y=_spy_50.values,
                                  mode="lines", name="50DMA", line=dict(color="orange", dash="dash")))
        _fig.add_trace(go.Scatter(x=_spy_200.index, y=_spy_200.values,
                                  mode="lines", name="200DMA", line=dict(color="red", dash="dot")))

        # Current price: horizontal reference line + marker
        _fig.add_hline(y=_cur_price, line_dash="dot", line_color="#1f77b4", opacity=0.5,
                       annotation_text=f"  Current ${_cur_price:.2f}",
                       annotation_position="top left",
                       annotation_font=dict(color="#1f77b4", size=12))
        _fig.add_trace(go.Scatter(
            x=[_cur_date], y=[_cur_price],
            mode="markers+text",
            marker=dict(color="#1f77b4", size=10, symbol="circle"),
            text=[f"  ${_cur_price:.2f}"],
            textposition="top right",
            textfont=dict(size=11, color="#1f77b4"),
            name="Current Price",
            showlegend=True,
        ))

        _fig.update_layout(
            title="SPY with 50DMA and 200DMA",
            yaxis_title="Price (USD)",
            height=430,
            margin=dict(t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(_fig, width="stretch")

        # Trend interpretation
        if _cur_price > _cur_50 > _cur_200:
            st.success(f"SPY trend strong: price ${_cur_price:.2f} > 50DMA ${_cur_50:.2f} > 200DMA ${_cur_200:.2f}")
        elif _cur_price < _cur_200:
            st.error(f"SPY below 200DMA (${_cur_200:.2f}) — long-term trend risk elevated.")
        elif _cur_price < _cur_50:
            st.warning(f"SPY below 50DMA (${_cur_50:.2f}) — short-term momentum weakening.")
        else:
            st.info(f"SPY ${_cur_price:.2f} above 200DMA but below 50DMA — trend transitioning.")
    else:
        st.warning("SPY data unavailable.")

    st.divider()

    # ── Sector Breadth ────────────────────────────────────────────────────────
    st.subheader("Sector Breadth")
    _sd = mb_sector_df.copy()
    _sd["% Above 50DMA"]  = _sd["% Above 50DMA"].map("{:.1%}".format)
    _sd["% Above 200DMA"] = _sd["% Above 200DMA"].map("{:.1%}".format)
    st.dataframe(_sd, width="stretch", hide_index=True)

    if st.checkbox("Show raw breadth table (last 100 rows)", key="mb_raw"):
        st.dataframe(mb_breadth.tail(100), width="stretch")

# =============================================================================
# TAB 2: LIVE MARKET EXECUTION TERMINAL
# =============================================================================
# Module-level variables for monitoring
all_monitor_tickers = []

with tab2:
    st.header("🎛️ Live Market Execution Terminal")
    st.markdown("""
    **Objective:** Real-time monitoring of Price vs. Benchmark.
    Use this to identify 'Slippage' and 'Relative Strength' during intraday surges.
    """)
    
    # Tactical Consultant Action
    st.subheader("Consultant's Intraday Audit")
    benchmark_change = st.session_state.get("live_benchmark_change_pct", 0)
    
    if benchmark_change > 1.5:
        st.warning(
            "🚨 **Momentum Risk Alert:** Benchmark is up more than 1.5% from baseline. "
            "Melt-up conditions can reverse sharply; avoid late entries and tighten risk controls."
        )
    elif benchmark_change < -1.5:
        st.error(
            "📉 **Drawdown Alert:** Benchmark is down more than 1.5% from baseline. "
            "Prioritize capital protection, reduce gross exposure, and focus on liquidity quality."
        )
    else:
        st.info("Regime: Intraday move is within normal variance; keep execution disciplined and size by risk.")
    
    st.divider()
    if not selected_tickers:
        st.warning("Please select tickers in the sidebar to monitor prices.")
    else:
        # 1. Fetch Data for Selected Universe + Benchmark
        globals()['all_monitor_tickers'] = list(
            set(selected_tickers + st.session_state.external_tickers + [selected_benchmark]))

        # Fetch live prices (refreshes every 60 seconds)
        try:
            monitor_list = globals().get('all_monitor_tickers', [])
            live_data = get_live_intraday(monitor_list, period="2d")

            # 2. Process Metrics
            price_report = []
            benchmark_change = 0

            # Calculate Benchmark change first for relative comparison
            bench_series = live_data[selected_benchmark].dropna()
            bench_current = live_data[selected_benchmark].dropna().iloc[-1]
            bench_ref_price, bench_ref_label = get_reference_price(selected_benchmark, bench_series)
            benchmark_change = (bench_current / bench_ref_price - 1) * 100 if bench_ref_price is not None else float('nan')
            st.session_state["live_benchmark_change_pct"] = float(benchmark_change) if not pd.isna(benchmark_change) else None
            st.session_state["live_benchmark_symbol"] = selected_benchmark
            st.session_state["live_benchmark_baseline"] = bench_ref_label

            for t in monitor_list:
                if t == selected_benchmark:
                    continue  # Skip benchmark in the individual report
                ticker_series = live_data[t].dropna()
                if ticker_series.empty:
                    continue
                current_p = ticker_series.iloc[-1]
                ref_price, _ = get_reference_price(t, ticker_series)
                prev_p = ref_price if ref_price is not None else ticker_series.iloc[0]
                change_abs = current_p - prev_p
                change_pct = (change_abs / prev_p) * 100 if prev_p else float('nan')
                rel_perf = change_pct - benchmark_change  # Alpha check

                price_report.append({
                    "Ticker": t,
                    "Current Price": round(current_p, 2),
                    "Day Change (%)": round(change_pct, 2) if not pd.isna(change_pct) else float('nan'),
                    "Rel. to Bench (%)": round(rel_perf, 2) if not pd.isna(rel_perf) else float('nan'),
                    "Status": "🔥 Outperforming" if rel_perf > 0 else "❄️ Lagging"
                })

            # 3. Visualization: Metrics Row
            st.subheader(f"System Pulse vs. {selected_benchmark}")
            col1, col2, col3 = st.columns(3)
            col1.metric(
                f"Benchmark: {selected_benchmark}",
                f"{bench_current:.2f}",
                f"{benchmark_change:.2f}% ({bench_ref_label})" if not pd.isna(benchmark_change) else "N/A"
            )
            col2.metric(
                "Reference Basis",
                f"{bench_ref_label}",
                f"{bench_ref_price:.2f}" if bench_ref_price is not None else "N/A"
            )
            col3.write(
                f"**Last Update:** {datetime.datetime.now().strftime('%H:%M:%S')} EST")

            # 4. Display Professional Price Table

            df_live = pd.DataFrame(price_report).sort_values(
                by="Day Change (%)", ascending=False)
            df_live.dropna(subset=['Current Price'], inplace=True)

            def style_live_report(val):
                if isinstance(val, float):
                    color = 'green' if val > 0 else 'red'
                    return f'color: {color}'
                return ''

            st.dataframe(
                df_live.style.map(style_live_report, subset=[
                    'Day Change (%)', 'Rel. to Bench (%)']),
                width='stretch',
                hide_index=True
            )

        except Exception as e:
            st.error(f"Execution Terminal Error: {e}")


# =============================================================================
# TAB 3: MARKET RISK & FRAGILITY PORTAL
# =============================================================================    
with tab3: 
    st.subheader("Risk Alert Portal")
    st.write(f"**Monitor List:** {', '.join(sorted(monitor_list))}")

    save_plots = st.checkbox("Save Volume plots", value=False)
    if st.button("Scan All Tickers & List Signals"):
        import io
        from contextlib import redirect_stdout

        st.info(
            f"Scanning all {len(monitor_list)} tickers for regime signals...")

        all_signals = []
        for ticker in sorted(monitor_list):
            try:
                result = scan_market(ticker)
            except Exception as e:
                print(f"Error scanning {ticker}: {e}")
                continue

            if isinstance(result, dict) and result.get("Regime"):
                cvd_icon = get_cvd_icon(result.get("CVD Trend", "N/A"))
                all_signals.append({
                    # --- METADATA & TIME ---
                    "Time": result.get("Bar Time", "N/A"),
                    "Ticker": ticker,
                    
                    # --- PRICE ACTION ---
                    "Price": f"{result.get('Price', 0.0):.2f}",
                    "Open": f"{result.get('Open', 0.0):.2f}",
                    "Day % Net": f"{result.get('Day % Net', 0.0):.2f}%",
                    "Day %": f"{result.get('Day %', 0.0):.2f}%",
                    
                    # --- REGIME CLASSIFICATION ---
                    "Signal/Regime": f"{result.get('Regime')} {get_regime_icon(result.get('Regime'))}",
                    # "Tail Quality": result.get('Tail Quality', 'N/A'),
                    
                    # --- VERDICT & ACTION ---
                    "Verdict": result.get('Verdict', ''),
                    "Suggestion": result.get('Suggestion', ''),
                    "Reason": result.get('Reason', ''),
                    
                    # --- FRACTAL METRICS ---
                    "Hurst": f"{result.get('Hurst', 0.0):.3f}",
                    "Tail Index": f"{result.get('Tail Index', 0.0):.3f}",
                    
                    # --- RISK ASSESSMENT ---
                    "Fragility": result.get('Fragility Alert', ''),
                    
                    # --- FLOW & LIQUIDITY ---
                    "Hybrid Signal": result.get('Hybrid Signal', 'N/A'),
                    "Hybrid VPIN": f"{result.get('Hybrid VPIN', 0.0):.3f}",                    
                    "Hybrid CVD": f"{get_cvd_icon(result.get("Hybrid CVD Trend", "N/A"))} {result.get('Hybrid CVD Trend', 'N/A')}",
                    "VPIN": f"{result.get('VPIN', 0.0):.3f}",
                    "CVD Trend": f"{cvd_icon} {result.get('CVD Trend', 'N/A')}",
                    "CVD Threshold": f"{result.get('CVD Threshold', 0.0):.3f}",

                    "Intraday Vol": f"{result.get('Intraday Vol', 0.0):.4f}"

                })

                # Optionally save plots per ticker
                if save_plots:
                    try:
                        intraday = get_data_persistent(ticker, interval="1m", period="7d", force_refresh=True)
                        if intraday is not None and not intraday.empty:
                            # Get last 2 consecutive days of data
                            intraday = intraday.sort_index().tail(2880)  # 1440 minutes per day * 2
                            
                            if intraday.empty:
                                continue
                            
                            # Extract volume data
                            volume_data = intraday[['Volume']].copy()
                            volume_data.index = pd.to_datetime(volume_data.index, errors="coerce")
                            volume_data = volume_data[~volume_data.index.isna()].sort_index()
                            if volume_data.index.tz is None:
                                volume_data.index = volume_data.index.tz_localize("America/New_York")
                            else:
                                volume_data.index = volume_data.index.tz_convert("America/New_York")

                            # Keep regular-session bars with real traded volume only.
                            volume_data["Volume"] = pd.to_numeric(volume_data["Volume"], errors="coerce")
                            session_mask = (
                                (volume_data.index.dayofweek < 5)
                                & (volume_data.index.time >= datetime.time(9, 30))
                                & (volume_data.index.time < datetime.time(16, 0))
                                & (volume_data["Volume"] > 0)
                            )
                            volume_data = volume_data.loc[session_mask].dropna(subset=["Volume"])

                            if not volume_data.empty:
                                first_quartile = volume_data["Volume"].quantile(0.25)
                                third_quartile = volume_data["Volume"].quantile(0.75)
                                iqr = third_quartile - first_quartile
                                if iqr > 0:
                                    upper_outlier_limit = third_quartile + 3 * iqr
                                    volume_data = volume_data[
                                        volume_data["Volume"] <= upper_outlier_limit
                                    ]
                            
                            if volume_data.empty:
                                continue
                            
                            # Create plot
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=volume_data.index,
                                y=volume_data['Volume'],
                                name='Volume',
                                line=dict(color='blue')
                            ))
                            fig.update_layout(
                                title=f"{ticker} - Volume (1m) - Last 2 Days",
                                xaxis_title="Time",
                                yaxis_title="Volume",
                                hovermode='x unified'
                            )
                            out_dir = DATA_DIR / 'hybrid_plots'
                            out_dir.mkdir(parents=True, exist_ok=True)
                            out_file = out_dir / f"{ticker}_volume.html"
                            fig.write_html(str(out_file))
                    except Exception as e:
                        print(f"Failed saving plot for {ticker}: {e}")

        # Display signals table
        df_signals = pd.DataFrame(all_signals).sort_values(by=["Signal/Regime", "Day % Net"], ascending=False)
        st.subheader("📊 All Risk Alert Signals")
        st.dataframe(
            df_signals,
            hide_index=True,
            width='stretch',
            column_config={"Ticker": st.column_config.TextColumn("Ticker", pinned=True)},
        )
        
        
        # Summary statistics
        st.subheader("🎯 Signal Summary")
        signal_counts = df_signals["Signal/Regime"].value_counts()
        st.bar_chart(signal_counts)
    
    
    st.divider()
    alert_ticker = st.selectbox(
        "Select Ticker for Regime Scan",
        options=sorted(set(monitor_list)),
        index=sorted(set(monitor_list)).index(selected_benchmark)
        if selected_benchmark in monitor_list else 0,
    )
    
    if st.button("Run"):
        result = scan_market(alert_ticker)

        if not isinstance(result, dict):
            st.error(f"Scan failed: {result}")
        else:
            scan_status = str(result.get("Scan Status", "RUN")).upper()
            latest_bar_time = result.get("Bar Time", "N/A")
            if scan_status == "WAIT":
                st.warning(f"Scan Status: WAIT (no new 1m bar). Latest bar: {latest_bar_time}")
            else:
                st.success(f"Scan Status: {scan_status}. Latest bar: {latest_bar_time}")

            st.write(f"**Price:** {result['Price']:.2f}")
            st.write(f"**Daily Return (Session Baseline):** {result['Day %']:.2f}%")
            st.write(f"**Daily Return (Net):** {result.get('Day % Net', float('nan')):.2f}%")
            render_metric_with_threshold(
                "Hurst (Trend)",
                result['Hurst'],
                "> 0.60 = Strong Trend, < 0.45 = Mean Reversion",
                format_hurst_color(result['Hurst']),
                precision=3
            )
            render_metric_with_threshold(
                "Tail Index",
                result['Tail Index'],
                ">= 1.70 = Stable, < 1.55 = Tail Risk",
                format_tail_color(result['Tail Index']),
                precision=3
            )
            render_metric_with_threshold(
                "VPIN",
                result['VPIN'],
                "< 0.40 = Low Toxicity, > 0.75 = Toxic",
                format_vpin_color(result['VPIN']),
                precision=3
            )
            if isinstance(result.get('CVD Threshold'), (float, int)):
                st.write(f"**Calibrated CVD Threshold:** {result.get('CVD Threshold', 0.0):.3f}")
            st.write(f"**Tail Quality:** {result.get('Tail Quality', 'N/A')}")
            render_metric_with_threshold(
                "Hybrid Signal VPIN",
                result.get('Hybrid VPIN', 0.0),
                "< 0.45 = Accumulate bias, > 0.70 = Exit bias",
                format_vpin_color(result.get('Hybrid VPIN', 0.0)),
                precision=3
            )
            st.write(f"**Hybrid Signal:** {result.get('Hybrid Signal', 'N/A')} | {result.get('Hybrid Reason', '')}")
            st.write(f"**Hybrid CVD Trend:** {result.get('Hybrid CVD Trend', 'N/A')}")
            render_metric_with_threshold(
                "Intraday Volatility",
                result['Intraday Vol'],
                "Threshold: 0.0015",
                format_vol_color(result['Intraday Vol']),
                precision=5
            )
            cvd_icon = get_cvd_icon(result['CVD Trend'])
            st.write(f"**CVD Trend:** {result['CVD Trend']} {cvd_icon}")

            st.markdown("**Judgment & Suggestion**")
            st.write(
                f"**Regime:** {get_regime_icon(result['Regime'])} {result['Regime']}")
            st.write(f"**Verdict:** {result['Verdict']}")
            st.write(f"**Suggestion:** {result['Suggestion']}")
            st.write(f"**Reason:** {result['Reason']}")
            if result.get("Fragility Alert"):
                st.error(f"⚠️ {result['Fragility Alert']} | Score: {result['Fragility Score']:.2f}")

   
    st.divider()
    st.markdown(
        """**Regime Icon Legend:** 🟢 Bullish | 🔴 Bearish | 🟡 Neutral | ⚠️ Unstable | 🚨 Tail Risk | 🔎 Other  
        **CVD Trend Icons:** 📈 Up | ⬇️ Down | → Flat"""
    )
    with st.expander("Evaluation Legend", expanded=False):
                st.markdown(
                        """
                        - **Tail Quality**
                            - **Tail-Stable:** Tail index is in the safer zone for trend-following.
                            - **Tail-Caution:** Trend exists, but fat-tail risk is elevated; reduce size.
                            - **Tail-Risk:** Jump/gap risk is dominant; protect capital first.
                        - **Calibrated CVD Threshold**
                            - Dynamic divergence sensitivity for the current tape.
                            - Higher values mean only stronger CVD slopes are treated as meaningful divergence.
                        """
                )

# =============================================================================
# TAB 4: ALPHA PERSISTENCE (RS SIGNALS)
# =============================================================================  
with tab4:
    st.header("📡 Relative Strength (RS) Audit")
    st.markdown("""
    **Objective:** Identify 'Institutional Footprints'.
    We look for assets with **RS Score > 0** (Outperforming) and **Positive Slope** (Accumulating).
    """)

    # 1. Setup Universe & Benchmark
    # We use user-selected benchmark, and always include SPY as institutional reference.
    rs_benchmark = st.selectbox("RS Reference Benchmark", [
                                "XWD.TO", "XEQT.TO", "SPY"], index=2)
    st.caption("Institutional baseline is always tracked vs SPY in addition to the selected benchmark.")

    # Combined Universe from your SD and Managed accounts
    rs_universe = list(set(cfg['defaults']['selected_portfolio'] +
                       [ticker.upper() for ticker in st.session_state['external_tickers']]))

    # 2. Process RS Signals
    rs_results = []

    # Fetch 2 years of data for the 52-week SMA
    rs_data = get_price_history_with_benchmark(
        rs_universe, rs_benchmark, period="2y", interval="1d")

    # Always include SPY context, even when user selects another benchmark.
    spy_ref_series = None
    if rs_benchmark == "SPY" and "SPY" in rs_data.columns:
        spy_ref_series = rs_data["SPY"].copy()
    else:
        spy_ref_df = get_data_persistent("SPY", interval="1d", period="2y")
        if spy_ref_df is not None and not spy_ref_df.empty and "Close" in spy_ref_df.columns:
            spy_ref_series = spy_ref_df["Close"].copy()
            spy_ref_series.index = pd.to_datetime(spy_ref_series.index, errors="coerce")
            rs_index = pd.to_datetime(rs_data.index, errors="coerce")
            spy_ref_series = spy_ref_series[~spy_ref_series.index.isna()].sort_index()
            rs_index_series = pd.Series(rs_index, index=rs_data.index)
            rs_data = rs_data.loc[~rs_index_series.isna()].copy()
            rs_data.index = rs_index_series[~rs_index_series.isna()].values
            spy_ref_series = spy_ref_series.reindex(rs_data.index).ffill().bfill()

    for t in rs_universe:
        mrs_series, slope_series = calculate_mansfield_rs(
            rs_data[t], rs_data[rs_benchmark])

        # ratio = rs_data[t] / rs_data[rs_benchmark]

        # sma_ratio = ratio.rolling(window=window).mean()
        # mrs = ((ratio / sma_ratio) - 1) * 100

        # # 5-day slope to determine momentum of the RS line
        # slope = mrs.diff(5)

        current_score = mrs_series.ffill().iloc[-1]
        # print(f"Current RS Score for {t}: {current_score:.2f}")
        current_slope = slope_series.ffill().iloc[-1]

        if spy_ref_series is not None:
            mrs_spy, slope_spy = calculate_mansfield_rs(rs_data[t], spy_ref_series)
            current_score_spy = mrs_spy.ffill().iloc[-1]
            current_slope_spy = slope_spy.ffill().iloc[-1]
        else:
            current_score_spy = float("nan")
            current_slope_spy = float("nan")

        def get_signal_logic(current_score, current_slope):
            # Qualitative Signal Logic
            if current_score > 0 and current_slope > 0:
                signal = "Strong Accumulation|Hold/Buy. High Alpha persistence."
                color = "green"
            elif current_score > 0 and current_slope <= 0:
                signal = "Consolidating Alpha|Hold/Trim. Monitor for mean-reversion."
                color = "blue"
            elif current_score <= 0 and current_slope > 0:
                signal = "Early Recovery|Speculative Buy. Watch for RS-Zero cross."
                color = "orange"
            else:
                signal = "Institutional Avoid|Sell/Avoid. Opportunity cost is too high."
                color = "red"
            return signal, color
        signal, color = get_signal_logic(current_score, current_slope)

        # Add mean-reversion monitoring (e.g., if MRS is above 20% but slope turns negative, it may signal an impending reversal)
        reversion_status = monitor_mean_reversion(mrs_series, rs_data[t])

        # bollinger band detector for mean reversion and hook detection
        # 1. Run Detectors
        has_hook = detect_rs_hook(mrs_series)
        sma_t, upper_t, lower_t, rs_series_t = calculate_rs_bollinger_bands(
            mrs_series)

        is_near_lower_band = mrs_series.ffill(
        ).iloc[-1] <= (lower_t.ffill().iloc[-1] * 1.02)  # Within 2% of band

        # 2. Determine Hook Status
        hook_status = ""
        if has_hook and mrs_series.ffill().iloc[-1] > 0:
            hook_status = "🪝 BULLISH HOOK (Re-entry)"
        elif has_hook and mrs_series.ffill().iloc[-1] <= 0:
            hook_status = "⚓ RECOVERY HOOK (Spec Buy)"
        elif mrs_series.ffill().iloc[-1] > upper_t.ffill().iloc[-1]:
            hook_status = "🔥 PARABOLIC"
        else:
            hook_status = "Steady"

        # Dynamic bubble alert: compare extension above 200DMA versus its own 252-day extension volatility.
        price_200ma = rs_data[t].ffill().rolling(
            window=RS_LOOKBACK_WINDOW).mean()
        dist_from_200ma = (rs_data[t].ffill().iloc[-1] /
                           price_200ma.ffill().iloc[-1] - 1) * 100

        extension = rs_data[t].ffill() - price_200ma
        extension_std = extension.rolling(window=252, min_periods=126).std()

        ext_now = float(extension.ffill().iloc[-1]) if not extension.empty else float("nan")
        ext_std_now = float(extension_std.ffill().iloc[-1]) if not extension_std.empty else float("nan")

        if np.isfinite(ext_now) and np.isfinite(ext_std_now) and ext_std_now > 1e-9:
            extension_z = ext_now / ext_std_now
        else:
            extension_z = float("nan")

        if np.isfinite(extension_z):
            if extension_z >= BUBBLE_Z_THRESHOLD and dist_from_200ma > 0:
                bubble_alert = (
                    f"🚨 BURRY ALERT: Z={extension_z:.2f}σ | Overextended: {dist_from_200ma:.2f}%"
                )
            else:
                bubble_alert = f"Safe: Z={extension_z:.2f}σ | 200MA: {dist_from_200ma:.2f}%"
        elif dist_from_200ma > BUBBLE_PCT_FALLBACK:
            bubble_alert = f"🚨 BURRY ALERT (Fallback): Overextended: {dist_from_200ma:.2f}%"
        else:
            bubble_alert = f"Safe: Z=N/A | >200MA: {dist_from_200ma:.2f}%"

        rs_results.append({
            "Ticker": t,
            "RS Score": round(current_score, 2),
            "RS Trend": round(current_slope, 2),
            "RS Score vs SPY": round(current_score_spy, 2) if pd.notna(current_score_spy) else np.nan,
            "RS Trend vs SPY": round(current_slope_spy, 2) if pd.notna(current_slope_spy) else np.nan,
            "Institutional Signal": signal,
            "Mean Reversion Alert": reversion_status,
            "Hook Alert": hook_status,
            "Bubble Alert": bubble_alert
        })

        # breakpoint()
    # 3. RS Ranking Table
    rs_df = pd.DataFrame(rs_results).sort_values(
        by="RS Score", ascending=False)

    def color_signal(val):
        color = 'red' if 'Avoid' in val else 'green' if 'Accumulation' in val else 'orange' if 'Early' in val else 'blue'
        return f'color: {color}; font-weight: bold'

    def style_hook(val):
        if '🪝' in val:
            return 'background-color: #004d00; color: white; font-weight: bold'
        if '⚓' in val:
            return 'background-color: #4d2600; color: white; font-weight: bold'
        if '🔥' in val:
            return 'color: #ff4d4d; font-weight: bold'
        return ''

    st.subheader("Cross-Sectional RS Ranking")
    st.table(rs_df.style.map(
        color_signal, subset=['Institutional Signal']).map(style_hook, subset=['Hook Alert']))

    # 4. Visualizing the RS Quadrant
    st.subheader("RS Momentum Quadrant")
    fig_quad = px.scatter(
        rs_df, x="RS Score", y="RS Trend", text="Ticker",
        color="Institutional Signal",
        labels={
            "RS Score": "Outperformance (Mansfield)", "RS Trend": "Momentum (5D Slope)"},
        title=f"Relative Strength Quadrant vs {rs_benchmark}"
    )
    # Add quadrant lines
    fig_quad.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_quad.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_quad.update_traces(textposition='top center')
    fig_quad.update_layout(template="plotly_dark")
    st.plotly_chart(fig_quad, width='stretch')

    st.divider()

    # 2. Visual Analysis: RS Bollinger Band Panel
    st.subheader("Statistical Reversion Monitor")
    target_t = st.selectbox("Select Asset to Monitor Bands",
                            rs_universe, index=rs_universe.index("GOOG"))

    # Recalculate for specific ticker
    mrs_t, slope_t = calculate_mansfield_rs(
        rs_data[target_t], rs_data[rs_benchmark])
    sma_t, upper_t, lower_t, rs_series_t = calculate_rs_bollinger_bands(mrs_t)
    # # check the data first:
    # print(f"Upper Band for {target_t}:\n{upper_t}")
    # print(f"Lower Band for {target_t}:\n{lower_t}")
    # print(f"Current MRS for {target_t}: {mrs_t.tail()}")
    # print(f"Current SMA for {target_t}: {rs_series_t.tail()}")
    fig_bands = go.Figure()

    # Add Shaded Area for Bands
    fig_bands.add_trace(go.Scatter(x=sma_t.index, y=upper_t,
                        line=dict(width=0), showlegend=False))
    fig_bands.add_trace(go.Scatter(x=sma_t.index, y=lower_t, line=dict(width=0),
                                   fill='tonexty', fillcolor='rgba(100, 100, 100, 0.2)', name="Statistical Range"))

    # Add RS Line
    fig_bands.add_trace(go.Scatter(x=mrs_t.index, y=mrs_t,
                        name="Mansfield RS", line=dict(color='#00ffcc', width=2)))
    # mark the latest point
    fig_bands.add_trace(go.Scatter(x=[mrs_t.index[-1]], y=[mrs_t.iloc[-1]], mode='markers+text',
                                   name="Current RS", text=[f"{mrs_t.iloc[-1]:.2f}"], textposition="top center",
                                   marker=dict(color='yellow', size=10, symbol='star')))
    # Add SMA (Center Line)
    fig_bands.add_trace(go.Scatter(x=sma_t.index, y=sma_t, name="RS 20D MA", line=dict(
        color='orange', dash='dot'), connectgaps=True))
    # Add Zero Line
    fig_bands.add_trace(go.Scatter(x=sma_t.index, y=[
                        0]*len(mrs_t), name="Institutional Floor (0)", line=dict(color='red', width=1)))

    fig_bands.update_layout(
        title=f"Statistical RS Bands: {target_t} vs {rs_benchmark}",
        yaxis_title="RS Score (%)",
        template="plotly_dark",
        hovermode="x unified"
    )
    st.plotly_chart(fig_bands, width='stretch')

    # 3. Consultant's Interpretation of the Bands
    st.info(f"""
    **How to read the Bands for {target_t}:**
    * **Touch Upper Band (Top):** Alpha is likely 'maxed out'. This is the **TRIM** signal.
    * **Touch Lower Band (Bottom):** Alpha is statistically 'exhausted'. If the stock is Blue (Weakening), look for a **HOOK** here to re-enter.
    * **The Zero Line:** If RS is within bands but crosses 0, the regime has changed.
    
    Studies show the top 20% of RS stocks continue to outperform over the following 3–6 months (ex. Post-Earnings Drift).
    """)

# =============================================================================
# TAB 5: FORECAST PORTAL (ML + MONTE CARLO + ACTIONS)
# =============================================================================
with tab5:
    st.header("🔮 Intraday Forecast & Action Portal")
    st.caption("Machine-learning forecasts + Monte Carlo probabilistic range + rule-based trade action.")

    forecast_universe = sorted(set(selected_tickers + [selected_benchmark] + st.session_state.get('external_tickers', [])))
    if not forecast_universe:
        st.warning("No symbols available. Select tickers in the sidebar first.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            fc_ticker = st.selectbox("Ticker", options=forecast_universe, index=0, key="fc_ticker")
        with c2:
            fc_intraday_period = st.selectbox("Intraday Training Period", options=["30d", "60d"], index=1, key="fc_period")
        with c3:
            fc_intraday_interval = st.selectbox("Intraday Interval", options=["1m", "2m", "5m", "15m"], index=2, key="fc_interval")

        r1, r2, r3 = st.columns(3)
        with r1:
            fc_portfolio_value = st.number_input("Portfolio Value", min_value=1_000.0, value=100_000.0, step=5_000.0)
        with r2:
            fc_daily_budget = st.slider("Daily Tail-Risk Budget %", min_value=0.10, max_value=5.00, value=1.50, step=0.05) / 100.0
        with r3:
            fc_min_edge = st.slider("Min Edge %", min_value=0.05, max_value=2.00, value=0.25, step=0.05) / 100.0

        if st.button("Run Forecast & Action", key="run_forecast_action"):
            try:
                rules_cfg = RiskRulesConfig(
                    min_edge_pct=fc_min_edge,
                    daily_tail_risk_budget_pct=fc_daily_budget,
                )
                with st.spinner(f"Running forecast for {fc_ticker}..."):
                    forecast_results = run_all(
                        ticker=fc_ticker,
                        intraday_period=fc_intraday_period,
                        intraday_interval=fc_intraday_interval,
                        portfolio_value=fc_portfolio_value,
                        rules_config=rules_cfg,
                    )

                intraday = forecast_results["intraday_ml"]
                daily = forecast_results["next_close_ml"]
                mc = forecast_results["gbm_monte_carlo"]
                decision = forecast_results.get("trade_decision")

                st.subheader(f"Forecast Snapshot: {fc_ticker}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Price (1m)", f"{intraday.current_price:.2f}")
                m2.metric("Predicted Session Close", f"{intraday.predicted_close:.2f}", f"{intraday.predicted_return_to_close * 100:.2f}%")
                m3.metric("Predicted Next Close", f"{daily.predicted_next_close:.2f}", f"Bias {daily.market_context_bias_pct * 100:.2f}%")
                m4.metric("Terminal Confidence", f"{mc.terminal_confidence * 100:.2f}%")

                with st.expander("Model Validation Metrics", expanded=True):
                    v1, v2, v3, v4 = st.columns(4)
                    v1.metric("Daily Holdout MAE", f"{daily.holdout_mae:.4f}")
                    v2.metric("Daily Holdout RMSE", f"{daily.holdout_rmse:.4f}")
                    v3.metric("Daily Holdout MAPE", f"{daily.holdout_mape:.2%}")
                    v4.metric("Daily Directional Acc", f"{daily.holdout_directional_acc:.2%}")

                    w1, w2, w3, w4 = st.columns(4)
                    w1.metric("Daily WF MAE", f"{daily.walk_forward_mae:.4f}")
                    w2.metric("Daily WF RMSE", f"{daily.walk_forward_rmse:.4f}")
                    w3.metric("Daily WF MAPE", f"{daily.walk_forward_mape:.2%}")
                    w4.metric("Daily WF Dir Acc", f"{daily.walk_forward_directional_acc:.2%}")
                    st.caption(f"Daily walk-forward windows: {daily.walk_forward_windows}")
                    st.caption(f"Market-context bias from SPY overlay: {daily.market_context_bias_pct * 100:.2f}%")

                    intraday_metrics = getattr(intraday, "validation_metrics", {}) or {}
                    i1, i2, i3, i4 = st.columns(4)
                    i1.metric("Intraday Holdout MAE", f"{float(intraday_metrics.get('mae', float('nan'))):.6f}")
                    i2.metric("Intraday Holdout RMSE", f"{float(intraday_metrics.get('rmse', float('nan'))):.6f}")
                    i3.metric("Intraday Holdout MAPE", f"{float(intraday_metrics.get('mape', float('nan'))):.2%}")
                    i4.metric("Intraday Directional Acc", f"{float(intraday_metrics.get('directional_acc', float('nan'))):.2%}")

                    iw1, iw2, iw3, iw4 = st.columns(4)
                    iw1.metric("Intraday WF MAE", f"{float(intraday_metrics.get('wf_mae', float('nan'))):.6f}")
                    iw2.metric("Intraday WF RMSE", f"{float(intraday_metrics.get('wf_rmse', float('nan'))):.6f}")
                    iw3.metric("Intraday WF MAPE", f"{float(intraday_metrics.get('wf_mape', float('nan'))):.2%}")
                    iw4.metric("Intraday WF Dir Acc", f"{float(intraday_metrics.get('wf_directional_acc', float('nan'))):.2%}")
                    st.caption(f"Intraday walk-forward windows: {int(float(intraday_metrics.get('wf_windows', 0.0)))}")

                q_df = pd.DataFrame([
                    {"Quantile": "5%", "Price": mc.quantiles["p05"]},
                    {"Quantile": "25%", "Price": mc.quantiles["p25"]},
                    {"Quantile": "50%", "Price": mc.quantiles["p50"]},
                    {"Quantile": "75%", "Price": mc.quantiles["p75"]},
                    {"Quantile": "95%", "Price": mc.quantiles["p95"]},
                ])

                c_left, c_right = st.columns([1, 2])
                with c_left:
                    st.markdown("**Monte Carlo Quantile Range**")
                    st.dataframe(q_df.style.format({"Price": "{:.2f}"}), hide_index=True, width='stretch')
                with c_right:
                    fig_q = go.Figure()
                    fig_q.add_trace(go.Scatter(
                        x=q_df["Quantile"],
                        y=q_df["Price"],
                        mode="lines+markers",
                        name="Quantile Curve",
                        line=dict(color="#1f77b4", width=2),
                    ))
                    fig_q.add_hline(
                        y=intraday.current_price,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text=f"Current {intraday.current_price:.2f}",
                        annotation_position="top left",
                    )
                    fig_q.update_layout(title="GBM Probabilistic Close Range", yaxis_title="Price", height=320)
                    st.plotly_chart(fig_q, width='stretch')

                st.caption(f"P(terminal > start): {mc.probability_above_start * 100:.2f}% | Terminal confidence: {mc.terminal_confidence * 100:.2f}%")

                if decision is not None:
                    st.subheader("Action Engine")
                    if decision.action == "TRADE":
                        st.success(f"Action: {decision.action}")
                    elif decision.action == "REDUCE":
                        st.warning(f"Action: {decision.action}")
                    else:
                        st.error(f"Action: {decision.action}")

                    a1, a2, a3, a4 = st.columns(4)
                    a1.metric("Edge", f"{decision.edge_pct * 100:.2f}%")
                    a2.metric("Reward / Risk", f"{decision.reward_to_risk:.2f}")
                    a3.metric("Suggested Weight", f"{decision.recommended_weight * 100:.2f}%")
                    a4.metric("Tail-Risk Notional", f"{decision.tail_risk_notional:.2f}")

                    budget_total = forecast_results.get("risk_budget_total")
                    budget_remaining = forecast_results.get("risk_budget_remaining")
                    if isinstance(budget_total, (float, int)) and isinstance(budget_remaining, (float, int)):
                        st.caption(f"Daily risk budget remaining: {budget_remaining:.2f} / {budget_total:.2f}")

                    if getattr(decision, "reasons", None):
                        st.markdown("**Decision Reasons**")
                        for reason in decision.reasons:
                            st.write(f"- {reason}")

            except Exception as e:
                st.error(f"Forecast portal error: {e}")
    
# =============================================================================
# TAB 6: FACTOR ATTRIBUTION
# ============================================================================= 
with tab6:
    st.header(f"🧠 Factor-Based Alpha Analysis since {cfg['defaults']['start_date']}")
    target_stock = st.selectbox("Analyze Asset Alpha", options=globals().get('all_monitor_tickers', []), index=0)
    if target_stock not in returns.columns:
        temp_ret = get_daily_returns(
                target_stock,
                selected_benchmark,
                cfg['defaults']['start_date']
            )
        temp_ret = temp_ret.dropna(how='any')
        y = temp_ret[target_stock] - rf / 252  # Daily excess return
        X = temp_ret[selected_benchmark] - rf / \
            252  # Daily excess return of benchmark
    else:
        # Simple Factor Proxy (Market Excess)
        y = returns[target_stock] - rf / 252  # Daily excess return
        X = returns[selected_benchmark] - rf / \
            252  # Daily excess return of benchmark
    
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"{target_stock} Alpha (Daily)",
                  f"{model.params.iloc[0]:.3%}")
        st.write("**Interpretation:** Return not explained by the benchmark.")
    with col2:
        st.metric(f"{target_stock} Beta", f"{model.params.iloc[1]:.2f}")
        st.write("**Interpretation:** Systematic risk sensitivity.")

# =============================================================================
# TAB 7: RISK REPORT
# ============================================================================= 
# --- ROBUST RISK ENGINE ---
def get_robust_metrics(returns):
    # Ledoit-Wolf Shrinkage to fix the Rho > 1 problem
    lw = LedoitWolf().fit(returns)
    cov_matrix = pd.DataFrame(
        lw.covariance_, index=returns.columns, columns=returns.columns)

    # Calculate Correlation from Shrunk Covariance
    std_dev = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_dev, std_dev)
    return cov_matrix, corr_matrix

with tab7:
    st.header("🛡️ Risk Report")
    shrunk_cov, shrunk_corr = get_robust_metrics(returns)
    col_a, col_b, col_c = st.columns([2, 1, 1])

    with col_a:
        st.subheader("Robust Correlation Heatmap")
        st.info(
            "Note: Using Ledoit-Wolf Shrinkage to prevent Implied Correlation > 1.0.")
        fig_corr = px.imshow(shrunk_corr, text_auto=".2f", aspect="auto",
                             color_continuous_scale='RdBu_r', origin='lower')
        st.plotly_chart(fig_corr, width='stretch')

    with col_b:
        st.subheader("Tail Risk (CVaR)")
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = returns.apply(lambda x: x[x <= x.quantile(0.05)].mean())
        cvar_df = pd.DataFrame(cvar_95, columns=['Expected Shortfall (5%)']).sort_values(
            by='Expected Shortfall (5%)')
        st.table(cvar_df.style.format("{:.2%}"))
        st.warning(
            "CVaR represents the average loss in the worst 5% of scenarios.")
    with col_c:
        st.subheader("Volatility & Beta")
        vol_beta_df = pd.DataFrame({
            'Volatility (Annualized)': returns.std() * np.sqrt(ANNUAL_TRADING_DAYS),
            'Beta vs Benchmark': returns.corr()[selected_benchmark]
        }).sort_values(by='Beta vs Benchmark', ascending=False)
        st.table(vol_beta_df.style.format(
            {"Volatility (Annualized)": "{:.2%}", "Beta vs Benchmark": "{:.2f}"}))
        st.info(
            "Volatility is annualized. Beta indicates sensitivity to benchmark movements.")
    st.subheader("Annualized Shortfall (Tail Risk)")
    st.info("Theoretical uses √252; Empirical uses rolling 1-month clusters.")

    shortfall_data = AlphaRiskEngine(tickers=selected_tickers, benchmark=selected_benchmark).calculate_annualized_shortfall(
        confidence_level=cfg['parameters']['confidence_level'])
    # Convert to DataFrame for display
    df_es = pd.DataFrame(shortfall_data).T
    st.table(
        df_es[['Theoretical_Annual_ES', 'Empirical_Annual_ES']].style.format("{:.2%}"))

    st.warning("**Analyst Insight:** If Empirical ES is significantly larger than Theoretical ES, "
               "it indicates 'Volatility Clustering' in the asset (Common in PSI & MSFT).")
   
# =============================================================================
# TAB 8: REBALANCING (BL)
# ============================================================================= 
with tab8:
    st.header("⚖️ Institutional Rebalancing: Black-Litterman Model")
    st.markdown("""
    **Analytical Framework:** We blend Market Equilibrium (Priors) with your specific Analyst Views (the Alpha).
    This prevents the model from over-allocating based on noisy historical data.
    """)

    # 1. INPUT: Current Portfolio State
    st.subheader("1. Current Holdings")

    # Pre-populate with your tickers
    holdings_data = []
    total_market_value = 0

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        current_cash = st.number_input(
            "Current Cash Balance (CAD/USD)", value=5000.0)

    for ticker in selected_tickers:
        col1, col2 = st.columns(2)
        with col1:
            shares = st.number_input(
                f"Current Shares: {ticker}", value=10, key=f"shares_{ticker}")
        with col2:
            price = returns[ticker].iloc[-1]  # Get latest price from data
            mkt_val = shares * price
            total_market_value += mkt_val
            st.write(f"Current Market Value: ${mkt_val:,.2f}")
            holdings_data.append(
                {'Ticker': ticker, 'Shares': shares, 'Price': price, 'Value': mkt_val})

    portfolio_total = total_market_value + current_cash
    st.info(
        f"**Total Portfolio Net Asset Value (NAV): ${portfolio_total:,.2f}**")

    # 2. INPUT: Analyst Views (The CFA Work)
    st.subheader("2. Inject Analyst Views")
    st.write("Express your views as *Expected Annual Return %*.")

    views_dict = {}
    for ticker in selected_tickers:
        # Default to a neutral market return (e.g. 7%)
        view = st.slider(
            f"Expected Return for {ticker} (%)", -50, 50, 7, key=f"view_{ticker}")
        views_dict[ticker] = view / 100

    # 3. THE BLACK-LITTERMAN MATH
    # Calculate Market Priors (Implied Returns)
    # In a real setup, we'd use market caps. Here we use an Equilibrium proxy.
    cov_matrix = shrunk_cov  # From Tab 2 (Ledoit-Wolf)

    # Black-Litterman Model
    # We use the mean returns as the 'prior' and inject your 'views'
    bl = BlackLittermanModel(
        cov_matrix, pi="market", market_caps=market_caps, absolute_views=views_dict)
    rets = bl.bl_returns()
    ef = EfficientFrontier(rets, cov_matrix)
    ef.add_objective(objective_functions.L2_reg,
                     gamma=0.1)  # Smooths weights
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    if st.button("Calculate Optimal Weights"):

        # 4. OUTPUT: Trade Execution List
        st.subheader("3. Execution Plan")

        rebalance_list = []
        for ticker in selected_tickers:
            # market_caps = {t: yf.Ticker(t).info.get('marketCap', 0) for t in selected_tickers}
            target_weight = cleaned_weights[ticker]
            target_value = portfolio_total * target_weight
            current_val = next(
                item['Value'] for item in holdings_data if item['Ticker'] == ticker)
            price = next(item['Price']
                         for item in holdings_data if item['Ticker'] == ticker)

            trade_value = target_value - current_val
            trade_shares = trade_value / price

            # Identify "In-Kind" Alert
            # If the trade involves selling a massive winner, flag it for tax review
            tax_alert = "🚨 TAX REVIEW" if trade_shares < 0 and (
                returns[ticker].pct_change().sum() > 0.5) else "✅"

            rebalance_list.append({
                'Ticker': ticker,
                'Market Cap': f"${market_caps[ticker]:,.0f}",
                'Target %': f"{target_weight:.2%}",
                'Target Value': f"${target_value:,.2f}",
                'Trade Action': "BUY" if trade_shares > 0 else "SELL",
                'Shares': round(abs(trade_shares), 2),
                'Tax Warning': tax_alert
            })

        rebalance_df = pd.DataFrame(rebalance_list)
        st.table(rebalance_df)

        st.success("Strategy generated using Black-Litterman Optimization.")
        st.warning(
            "Ensure the 'SELL' orders in Tax Review are not triggered 'In-Cash' if significant capital gains exist.")


# =============================================================================
# TAB 9: EFFICIENT FRONTIER
# ============================================================================= 
with tab9:
    st.header("🧭 Efficient Frontier Analytics")
    st.markdown("""
    **Consultant's View:** Assets below the white line are 'Dominated.'
    Your Optimized Portfolio (The Star) is positioned to maximize return for your chosen **Risk Aversion (λ=3)**.
    """)

    # Generate Plot
    # Assuming 'mu' (CAPM Returns) and 'shrunk_cov' (Ledoit-Wolf) are already defined
    # rets from Black-Litterman, cov from Tab 2, weights from optimization
    fig_frontier = plot_institutional_frontier(
        rets, shrunk_cov, cleaned_weights)
    st.plotly_chart(fig_frontier, width='stretch')

 # =============================================================================

# =============================================================================
# TAB 10: STRATEGIC SCENARIO
# ============================================================================= 
with tab10:
      
         st.divider()
         def get_current_weights():
             if 'weights_dict' in st.session_state:
                 return st.session_state['weights_dict']
             return {k: v['target'] for k, v in cfg['constraints'].items()} | {t: 0.05 for t in selected_tickers if t not in cfg['constraints']}
     
         # Now, in any tab, you just call:
         current_weights = get_current_weights()
     
         st.header("🧪 Strategic Scenario Analysis & Contagion Audit")
         st.markdown("""
         **Analytical Framework:** We utilize the **Conditional Linear Regression** method.
         By shocking a 'Primary Factor,' we estimate the impact on all other assets using their
         **Robust Correlation** sensitivities.
         """)
     
         # 1. Scenario Selection
         col_scen1, col_scen2 = st.columns([1, 1])
     
         with col_scen1:
             scenario_type = st.selectbox(
                 "Select Macro Stress Scenario",
                 [
                     "Custom Manual Shock",
                     "AI Infrastructure Meltdown (SNDK/NVDA -35%)",
                     "Geopolitical Escalation (Energy/Gold Spike)",
                     "CAD Debt Crisis (Financials/XDIV -15%)",
                     "US Tech Regime Change (MSFT/GOOG -20%)"
                 ]
             )
     
         # 2. Define Scenario Parameters
         # Map scenario to primary ticker and its shock magnitude
         scenario_map = {
             "AI Infrastructure Meltdown (SNDK/NVDA -35%)": {"primary": "SNDK", "shock": -0.35},
             "Geopolitical Escalation (Energy/Gold Spike)": {"primary": "KILO.TO", "shock": 0.15},
             "CAD Debt Crisis (Financials/XDIV -15%)": {"primary": "XDIV.TO", "shock": -0.15},
             "US Tech Regime Change (MSFT/GOOG -20%)": {"primary": "MSFT", "shock": -0.20}
         }
     
         if scenario_type == "Custom Manual Shock":
             target_asset = st.selectbox("Select Asset to Shock", selected_tickers)
             shock_magnitude = st.slider(
                 "Magnitude of Shock (%)", -50, 50, -10) / 100
         else:
             target_asset = scenario_map[scenario_type]["primary"]
             # Allow user to check/override the pre-set shock
             shock_magnitude = st.number_input(
                 f"Shock for {target_asset} (%)", value=scenario_map[scenario_type]["shock"]*100) / 100
     
         # 3. CONTAGION MATH: E(Ri | Rj = shock)
         # R_i_impact = Beta_(i,j) * Shock_j
         # Beta_(i,j) = (Cov(i,j) / Var(j))
     
         impact_results = []
     
         # We use the 'shrunk_cov' matrix we calculated in the Risk Engine
         for asset in selected_tickers:
             if asset == target_asset:
                 impact = shock_magnitude
             else:
                 # Calculate the sensitivity (Beta) of 'asset' to 'target_asset'
                 cov_ij = shrunk_cov.loc[asset, target_asset]
                 var_j = shrunk_cov.loc[target_asset, target_asset]
                 beta_sensitivity = cov_ij / var_j
     
                 # Apply a 'decay factor' if correlation is low (Institutional Caution)
                 impact = beta_sensitivity * shock_magnitude
     
             impact_results.append({
                 "Ticker": asset,
                 "Estimated Impact (%)": impact,
                 # Based on optimized weights
                 "Dollar Impact": PORTFOLIO_VALUE * (current_weights[asset] * impact)
             })
     
         impact_df = pd.DataFrame(impact_results)
         total_portfolio_impact = impact_df["Dollar Impact"].sum() / 4800
     
         # 4. VISUALIZATION
         st.divider()
         m1, m2 = st.columns(2)
         m1.metric("Total Portfolio Shock Impact", f"{total_portfolio_impact:+.2%}")
         m2.metric(f"Est. NAV Change (${PORTFOLIO_VALUE:,} Principal)",
                   f"${PORTFOLIO_VALUE * total_portfolio_impact:+,.2f} CAD")
     
         # Bar chart of individual asset contagion
         fig_impact = px.bar(
             impact_df, x="Ticker", y="Estimated Impact (%)",
             color="Estimated Impact (%)",
             color_continuous_scale="RdYlGn",
             title=f"Contagion Map: Response to {target_asset} {shock_magnitude:+.0%} Shock"
         )
         fig_impact.update_layout(template="plotly_dark")
         st.plotly_chart(fig_impact, width='stretch')
     
         # 5. Analyst Commentary
         st.subheader("Consultant's Scenario Audit")
         if scenario_type.startswith("AI Infrastructure"):
             st.info(f"""
             **Skeptic's Hedge Verified:** Because your portfolio holds **CLSE** (Long/Short) and **Gold**,
             the contagion from a tech crash is dampened. While {target_asset} drops 35%,
             the portfolio only loses {abs(total_portfolio_impact):.2%}, demonstrating structural resilience.
             """)
         elif scenario_type.startswith("Geopolitical"):
             st.success(f"""
             **Crisis Alpha:** A spike in Gold serves as a positive tail-wind.
             Note that **REMD.NE** (Emerging Markets) may show negative contagion due to risk-off sentiment
             in Taiwan/Korea foundries.
             """)

# =============================================================================
# TAB: PORTFOLIO DRAWDOWN
# =============================================================================
with tab_dd:
    st.header("Portfolio Drawdown")

    dd_tickers = st.multiselect(
        "Drawdown tickers", options=sorted(set(globals().get('all_monitor_tickers', [])+cfg['universe']['benchmarks'])),
        default=selected_benchmark, key="dd_tickers")
    if dd_tickers:
        with st.spinner("Loading cached drawdown history..."):
            dd_fig = plot_multiple_stocks_from_cache(dd_tickers)
        if dd_fig is None:
            st.warning(
                "No cached daily data is available for the selected tickers.")
        else:
            st.pyplot(dd_fig, width="stretch")
            plt.close(dd_fig)
        
        dd_threshold_pct = st.slider(
        "Minimum drawdown threshold (%)",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.5,
        key="dd_threshold_pct",
        help="Exclude drawdown observations and events smaller than this threshold.",
    )    
        if st.button("Calculate Drawdown Stats", key="dd_stats"):
            with st.spinner("Calculating drawdown statistics..."):
                dd_analysis = calculate_drawdown_analysis(
                    dd_tickers, threshold=dd_threshold_pct / 100
                )
                dd_stats = dd_analysis["stats"]
            if dd_stats.empty:
                st.warning("No cached daily data is available for the selected tickers.")
            else:
                st.subheader("Drawdown Statistics")
                st.dataframe(
                    dd_stats.style.format({
                        "Current Drawdown": "{:.2%}",
                        "Average Drawdown": "{:.2%}",
                        "Average Event Drawdown": "{:.2%}",
                        "Maximum Drawdown": "{:.2%}",
                        "Average Recovery Days": "{:.1f}",
                    }),
                    width="stretch",
                    hide_index=True,
                )
                st.subheader("Drawdown Series")
                st.dataframe(
                    dd_analysis["series"].tail(250).style.format("{:.2%}"),
                    width="stretch",
                )
                st.subheader("Drawdown and Recovery Distributions")
                hist_col1, hist_col2 = st.columns(2)
                with hist_col1:
                    dd_hist = plot_drawdown_histogram(dd_analysis)
                    st.pyplot(dd_hist, width="stretch")
                    plt.close(dd_hist)
                with hist_col2:
                    recovery_hist = plot_recovery_histogram(dd_analysis)
                    st.pyplot(recovery_hist, width="stretch")
                    plt.close(recovery_hist)
                with st.expander("Drawdown event detail"):
                    st.dataframe(dd_analysis["events"], width="stretch", hide_index=True)
    else:
        st.info("Select at least one ticker to display drawdown history.")

# =============================================================================
# TAB: MARKET WATCH
# =============================================================================
with tab_watch:
    st.header("Market Watch")
    watch_tickers = st.multiselect(
        "Watch tickers", options=sorted(set(globals().get('all_monitor_tickers', [])+cfg['universe']['benchmarks'])),
        default=selected_benchmark, key="watch_tickers")
    if watch_tickers:
        with st.spinner("Loading cached market history..."):
            watch_fig = plot_market_data_from_cache(watch_tickers)
        if watch_fig is None:
            st.warning(
                "No cached daily data is available for the selected tickers.")
        else:
            st.pyplot(watch_fig, width="stretch")
            plt.close(watch_fig)
    else:
        st.info("Select at least one ticker to display the market watch.")


# --- FOOTER: DECISION LOG ---
st.divider()
st.subheader("Decision Log Entry")

# Show any warnings captured from risk_modeling.mandelbrot this session
_captured_warnings = st.session_state.get("mandelbrot_warnings", [])
if _captured_warnings:
    st.markdown("**⚠️ Risk Engine Warnings (this session):**")
    for _w in _captured_warnings:
        st.warning(_w)
    if st.button("Clear Warnings"):
        st.session_state["mandelbrot_warnings"] = []
        st.rerun()

note = st.text_area("Record today's rationale (GIPS Governance):",
                    placeholder="e.g., Retained GIL despite volatility due to Hanes synergy targets.")
if st.button("Save Entry"):
    # Also flush any captured warnings into the file
    _warnings_to_save = st.session_state.get("mandelbrot_warnings", [])
    with open("governance_ips/decision_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n{pd.Timestamp.now()}: {note}")
        for _w in _warnings_to_save:
            f.write(f"\n  [AUTO-WARNING] {_w}")
    st.success("Entry saved to /governance_ips/")
