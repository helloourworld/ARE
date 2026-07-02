# Standard library
import datetime
import os
import sys
from pathlib import Path

# Ensure the repository root is on PYTHONPATH for local imports.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Repository startup helper
from enable_repo_root import ensure_repo_root
ensure_repo_root(REPO_ROOT)

# Third-party libraries
import appdirs as ad
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import statsmodels.api as sm
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
from data_pipeline import get_daily_returns, get_price_history, get_price_history_with_benchmark, get_premarket_data, get_live_intraday
from risk_modeling import AlphaRiskEngine, calculate_mansfield_rs, monitor_mean_reversion, calculate_rs_bollinger_bands, get_rs_signals, detect_rs_hook
from risk_modeling.mandelbrot import scan_market

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
# Local modules

# Prefer setting PYTHONPATH or using a package structure with __init__.py files.

# --- CONSTANTS ---
PORTFOLIO_VALUE = 10_000
RS_WINDOW = 50
RS_LOOKBACK_WINDOW = 200
ANNUAL_TRADING_DAYS = 252


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


def get_cvd_icon(cvd_value: str) -> str:
    try:
        if isinstance(cvd_value, str):
            if "UP" in cvd_value.upper() or "⬆️" in cvd_value:
                return "⬆️"
            if "DOWN" in cvd_value.upper() or "⬇️" in cvd_value:
                return "⬇️"
            if "FLAT" in cvd_value.upper() or "→" in cvd_value:
                return "→"
            # Parse fallback: numeric slope sign
            slope = float(cvd_value.split()[0])
            return "⬆️" if slope > 0 else "⬇️" if slope < 0 else "→"
    except Exception:
        pass
    return "→"


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


# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Alpha Risk Engine (ARE)", layout="wide")


# --- LOAD CONFIGURATION ---
def load_config():
    with open("config.yaml", "r") as f:
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

# --- DISPLAY METADATA ---
st.title(cfg['metadata']['report_title'])
st.caption(
    f"Analyst: {cfg['metadata']['analyst_name']} | Strategy: {cfg['metadata']['strategy_id']}")

# Access parameters for math
rf = cfg['parameters']['risk_free_rate']
st.write(f"Risk-Free Rate (Annualized Proxy): {rf:.1%}")

# Display returns data preview
with st.expander("📊 Returns Data Preview"):
    st.dataframe(returns.tail(3), width='stretch')
    st.caption(
        f"Data shape: {returns.shape[0]} periods × {returns.shape[1]} assets | Starting: {returns.index[0].date()}")


market_caps = {ticker: yf.Ticker(ticker).info.get(
    'marketCap', 0) for ticker in selected_tickers+[selected_benchmark]}

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


shrunk_cov, shrunk_corr = get_robust_metrics(returns)

# --- APP TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    ["Performance Attribution", "Risk Report", "Scenario Stress Test", "Factor Attribution", "Efficient Frontier", "CURRENCY EXPOSURE & FX SENSITIVITY", "Rebalancing & Execution",
     "Relative Strength Signals", "Market Trend"])

# --- TAB 1: ALPHA ATTRIBUTION ---
with tab1:
    st.header("Factor-Based Alpha Analysis")
    target_stock = st.selectbox("Analyze Asset Alpha", selected_tickers)

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

# --- TAB 2: THE RISK REPORT ---
with tab2:
    st.header("Institutional Risk Report")

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
               "it indicates 'Volatility Clustering' in the asset (Common in SNDK and MSFT).")
# --- TAB 3: SCENARIO STRESS TEST (Institutional Contagion Model) ---
with tab3:
    def get_current_weights():
        if 'weights_dict' in st.session_state:
            return st.session_state['weights_dict']
        return {k: v['target'] for k, v in cfg['constraints'].items()} | {t: 0.05 for t in selected_tickers if t not in cfg['constraints']}

    # Now, in any tab, you just call:
    current_weights = get_current_weights()

    st.header("Strategic Scenario Analysis & Contagion Audit")
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

# --- TAB 4: REBALANCING & EXECUTION (Black-Litterman) ---
with tab4:
    st.header("Institutional Rebalancing: Black-Litterman Model")
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

# --- STREAMLIT INTEGRATION ---
with tab5:
    st.header("Efficient Frontier Analytics")
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


# --- TAB 6: CURRENCY EXPOSURE & FX SENSITIVITY ---
with tab6:
    st.header("Global Currency Exposure Audit")
    st.markdown("""
    **Analytical Note:** This portfolio utilizes an **Unhedged Strategy**.
    We capture the 'Currency Alpha' during periods of CAD weakness.
    """)

    # 1. Define Asset-Level Currency Exposure (Look-through)
    # As an analyst, we define how much of each ticker's NAV is tied to USD
    fx_map = {
        'MSFT': 1.0,      # Direct USD
        'GOOG': 1.0,      # Direct USD
        'CLSE': 1.0,      # Direct USD (US-listed)
        'KILO.TO': 1.0,   # Gold is USD-denominated (Unhedged)
        'XAW.TO': 0.65,   # Global ex-CA (Approx 65% US exposure)
        'CLML.TO': 0.70,  # Global Quality (Approx 70% US exposure)
        'XDIV.TO': 0.0,   # Pure CAD (Canadian Banks/Utilities)
        'SPY': 1.0,       # Benchmark
        'XIU.TO': 0.0     # Benchmark
    }

    # 2. Calculate Portfolio-Wide USD Exposure
    # We use the weights from the Optimization/Input section
    if not selected_tickers:
        st.warning("Please select tickers in the sidebar.")
        usd_exposure = 0
    else:
        weights_dict = {t: 1/len(selected_tickers)
                        for t in selected_tickers}  # Placeholder weights

        usd_exposure = sum(weights_dict[t] * fx_map.get(t, 0)
                           for t in selected_tickers)
    cad_exposure = 1.0 - usd_exposure

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Currency Breakdown")
        fx_pie_data = pd.DataFrame({
            "Currency": ["USD (Direct & Indirect)", "CAD (Domestic)"],
            "Exposure": [usd_exposure, cad_exposure]
        })
        fig_fx = px.pie(fx_pie_data, values='Exposure', names='Currency',
                        color_discrete_sequence=['#1f77b4', '#ff7f0e'],
                        hole=0.4)
        st.plotly_chart(fig_fx, width='stretch')

    with col2:
        st.subheader("FX Sensitivity Analysis")
        fx_move = st.slider("Simulate USD/CAD Move (%)", -15.0, 15.0, 5.0)

        # Calculate impact on CAD-denominated NAV
        portfolio_impact = usd_exposure * (fx_move / 100)
        nav_change = PORTFOLIO_VALUE * portfolio_impact

        st.metric("Portfolio Impact (CAD Value)",
                  f"{portfolio_impact:+.2%}",
                  f"${nav_change:+,.2f} CAD")

        st.write(f"""
        **Consultant's Comment:**
        A {fx_move}% rise in the USD increases your total FHSA value by ${abs(nav_change):,.2f}
        regardless of stock price movement. This provides a 'Natural Hedge' if Canadian
        equities (**XDIV.TO**) drop due to domestic economic weakness.
        """)

    # 3. Currency-Adjusted Beta (Advanced Metric)
    st.divider()
    st.subheader("Institutional FX Observation")
    if usd_exposure > 0.5:
        st.success(
            f"**High USD Convexity:** {usd_exposure:.1%} of your wealth is protected against CAD depreciation.")
    else:
        st.warning(
            f"**CAD Home Bias:** Your portfolio is highly sensitive to the Canadian dollar.")

# --- TAB 7: SECTOR ROTATION (FINN vs XCHP) ---
with tab7:
    st.header("Relative Strength Audit: Application vs. Infrastructure")
    st.markdown("""
    **Analytical Thesis:** Are we at a 'Semiconductor Peak'?
    We compare the **Infrastructure (XCHP)** to the **Transaction Layer (FINN.NE)**.
    """)

    # 1. Fetch Data
    tickers = ["FINN.NE", "XCHP.TO"]
    data = get_price_history(tickers, period="2y", interval="1d")

    # 2. Calculate Ratio
    # We use a base-100 normalization to see the divergence clearly
    ratio = data["FINN.NE"] / data["XCHP.TO"]

    # 3. Statistical Z-Score (The 'Extreme' indicator)
    # Moving Average and Standard Deviation of the ratio
    ma = ratio.rolling(window=RS_WINDOW).mean()
    std = ratio.rolling(window=RS_WINDOW).std()
    z_score = (ratio - ma) / std

    # 4. Plotting the RS Ratio
    fig_rs = go.Figure()

    fig_rs.add_trace(go.Scatter(x=ratio.index, y=ratio,
                     name="FINN/XCHP Ratio", line=dict(color='#00ffcc')))
    fig_rs.add_trace(go.Scatter(x=ma.index, y=ma,
                     name="50-Day Mean", line=dict(dash='dash', color='gray')))

    fig_rs.update_layout(
        title="Relative Strength: Fintech vs. Semiconductors",
        yaxis_title="Price Ratio",
        template="plotly_dark"
    )
    st.plotly_chart(fig_rs, width='stretch')

    # 5. The "Rotation Alert"
    st.subheader("Statistical Regime Signal")
    current_z = z_score.iloc[-1]

    col_z1, col_z2 = st.columns(2)
    with col_z1:
        st.metric("Ratio Z-Score (50D)", f"{current_z:.2f}")

    with col_z2:
        if current_z < -2.0:
            st.error("🚨 SIGNAL: Fintech Extremely Undervalued vs. Semis")
            st.write("**Consultant's View:** The gap is at a 2-Standard Deviation extreme. Institutional rotation into FINN is mathematically probable.")
        elif current_z > 2.0:
            st.warning("⚠️ SIGNAL: Fintech Overextended vs. Semis")
        else:
            st.info("Regime: Momentum in Semis remains within historical bounds.")

 # --- TAB 8: ALPHA PERSISTENCE (RS SIGNALS) ---
with tab8:
    st.header("Institutional Relative Strength (RS) Audit")
    st.markdown("""
    **Objective:** Identify 'Institutional Footprints'.
    We look for assets with **RS Score > 0** (Outperforming) and **Positive Slope** (Accumulating).

    Studies show the top 20% of RS stocks continue to outperform over the following 3–6 months (Post-Earnings Drift).
    """)

    # 1. Setup Universe & Benchmark
    # We use XEQT.TO as the 'Global Beta' benchmark for RS comparison
    rs_benchmark = st.selectbox("RS Reference Benchmark", [
                                "XWD.TO", "XEQT.TO", "SPY"], index=2)

    # Combined Universe from your SD and Managed accounts
    rs_universe = list(set(cfg['defaults']['selected_portfolio'] +
                       [ticker.upper() for ticker in st.session_state['external_tickers']]))

    # 2. Process RS Signals
    rs_results = []

    # Fetch 2 years of data for the 52-week SMA
    rs_data = get_price_history_with_benchmark(
        rs_universe, rs_benchmark, period="2y", interval="1d")

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

        # bubble alert: If the price is more than 50% above the 200-day moving average,
        # it may be overextended and at risk of a sharp pullback.
        price_200ma = rs_data[t].ffill().rolling(
            window=RS_LOOKBACK_WINDOW).mean()
        dist_from_200ma = (rs_data[t].ffill().iloc[-1] /
                           price_200ma.ffill().iloc[-1] - 1) * 100
        # print(dist_from_200ma)
        if dist_from_200ma > 50:
            bubble_alert = f"🚨 BURRY ALERT: Overextended: {dist_from_200ma:.2f}%"
        else:
            bubble_alert = "Safe"

        rs_results.append({
            "Ticker": t,
            "RS Score": round(current_score, 2),
            "RS Trend": round(current_slope, 2),
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
    st.subheader("2. Statistical Reversion Monitor")
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
    """)

# --- TAB 9: LIVE MARKET EXECUTION TERMINAL ---
# Module-level variables for monitoring
all_monitor_tickers = []
gap_df = None


def fetch_premarket_and_gap(tickers):
    """Fetch pre-market and gap analysis for given tickers."""
    if not tickers:
        return None

    data, hist = get_premarket_data(tickers)
    if data is None or hist is None:
        return None

    try:
        results = []
        for t in tickers:
            try:
                # 1. Previous Day Close
                prev_close = hist[t].iloc[-2]

                # 2. Today's First Price (Pre-market start or Open)
                today_data = data['Close'][t].dropna()

                # Pre-market price (last point)
                current_extended = today_data.iloc[-1]

                # 3. Calculate Gap
                gap_pct = ((today_data.iloc[0] / prev_close) - 1) * 100

                results.append({
                    "Ticker": t,
                    "Prev Close": round(prev_close, 2),
                    "Pre/Live Price": round(current_extended, 2),
                    "Overnight Gap (%)": round(gap_pct, 2),
                    "Session Performance (%)": round(((current_extended / today_data.iloc[0]) - 1) * 100, 2)
                })
            except Exception:
                continue

        return pd.DataFrame(results).sort_values("Overnight Gap (%)", ascending=False) if results else None
    except Exception:
        return None


with tab9:
    st.header("🎛️ Live Market Execution Terminal")
    st.markdown("""
    **Objective:** Real-time monitoring of Price vs. Benchmark.
    Use this to identify 'Slippage' and 'Relative Strength' during intraday surges.
    """)

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
            bench_current = live_data[selected_benchmark].iloc[-1]
            # Start of the 2d window
            bench_prev = live_data[selected_benchmark].iloc[0]
            benchmark_change = (bench_current / bench_prev - 1) * 100

            for t in monitor_list:
                if t == selected_benchmark:
                    continue  # Skip benchmark in the individual report
                current_p = live_data[t].iloc[-1]
                # Last known valid price from previous day
                prev_p = live_data[t].dropna().iloc[0]
                change_abs = current_p - prev_p
                change_pct = (change_abs / prev_p) * 100
                rel_perf = change_pct - benchmark_change  # Alpha check

                price_report.append({
                    "Ticker": t,
                    "Current Price": round(current_p, 2),
                    "Day Change (%)": round(change_pct, 2),
                    "Rel. to Bench (%)": round(rel_perf, 2),
                    "Status": "🔥 Outperforming" if rel_perf > 0 else "❄️ Lagging"
                })

            # 3. Visualization: Metrics Row
            st.subheader(f"System Pulse vs. {selected_benchmark}")
            col1, col2, col3 = st.columns(3)
            col1.metric(f"Benchmark: {selected_benchmark}",
                        f"{bench_current:.2f}", f"{benchmark_change:.2f}%")
            col2.metric("Portfolio Sentiment",
                        "BULLISH" if benchmark_change > 0.3 else "NEUTRAL" if benchmark_change >= -0.3 else "BEARISH")
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
    # --- Streamlit Display ---
    st.subheader("🏁 Pre-Market & Gap Analysis")
    st.info("The 'Gap' represents institutional overnight sentiment re-pricing.")

    if st.button("Refresh Pre-Market/Gap Audit"):
        monitor_list = globals().get('all_monitor_tickers', [])
        globals()['gap_df'] = fetch_premarket_and_gap(monitor_list)
        if globals()['gap_df'] is not None:
            st.success("Gap data refreshed.")
        else:
            st.error("Unable to fetch gap data.")

    # Display gap table if data is available
    gap_data = globals().get('gap_df')
    if gap_data is not None:
        # Highlight significant gaps (> 2%)
        def highlight_gaps(val):
            color = 'red' if val < -2 else 'green' if val > 2 else 'white'
            return f'color: {color}; font-weight: bold'

        st.table(gap_data.style.map(
            highlight_gaps, subset=['Overnight Gap (%)']))
    else:
        st.info("Click 'Refresh Pre-Market/Gap Audit' to load gap analysis.")

    # --- Risk Alert Portal ---
    st.divider()
    st.subheader("Risk Alert Portal")
    st.markdown(
        """**Regime Icon Legend:** 🟢 Bullish | 🔴 Bearish | 🟡 Neutral | ⚠️ Unstable | 🚨 Tail Risk | 🔎 Other  
        **CVD Trend Icons:** ⬆️ Up | ⬇️ Down | → Flat"""
    )
    st.write(f"**Monitor List:** {', '.join(sorted(monitor_list))}")

    alert_ticker = st.selectbox(
        "Select Ticker for Regime Scan",
        options=sorted(set(monitor_list)),
        index=sorted(set(monitor_list)).index(selected_benchmark)
        if selected_benchmark in monitor_list else 0,
    )
    if st.button("Run Risk Alert Scan"):
        result = scan_market(alert_ticker)

        if not isinstance(result, dict):
            st.error(f"Scan failed: {result}")
        else:
            st.write(f"**Price:** {result['Price']:.2f}")
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
    st.subheader("Scan All Tickers & List Signals")
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
                    "Ticker": ticker,
                    "Price": f"{result['Price']:.2f}",
                    "Hurst": f"{result['Hurst']:.3f}",
                    "Tail Index": f"{result['Tail Index']:.3f}",
                    "VPIN": f"{result.get('VPIN', 0.0):.3f}",
                    "CVD Trend": f"{cvd_icon} {result['CVD Trend']}",
                    "Signal/Regime": f"{result['Regime']} {get_regime_icon(result['Regime'])}",
                    "Verdict": result['Verdict'],
                    "Suggestion": result['Suggestion'],
                    "Reason": result['Reason'],
                    "Fragility": result.get('Fragility Alert', '')
                })

        # Display signals table
        df_signals = pd.DataFrame(all_signals).sort_values(
            by=["Signal/Regime", "Verdict", "Ticker"],
            ascending=[True, True, True]
        )
        st.subheader("📊 All Risk Alert Signals")
        st.dataframe(df_signals, hide_index=True, width='stretch')

        # Summary statistics
        st.subheader("🎯 Signal Summary")
        signal_counts = df_signals["Signal/Regime"].value_counts()
        st.bar_chart(signal_counts)

    # 5. Tactical Consultant Action
    st.divider()
    st.subheader("Consultant's Intraday Audit")

    if benchmark_change > 1.5:
        st.warning(
            "🚨 **Gamma Warning:** Market is rolling up >1.5%. Check for 'Melt-up' exhaustion in your INTC and MU satellites.")
    elif benchmark_change < -1.5:
        st.error(
            "📉 **Liquidation Alert:** Systemic sell-off detected. Monitor KILO.TO for safe-haven decoupling.")
    else:
        st.info("Regime: Normal Intraday Variance. No emergency rebalancing required.")

# --- FOOTER: DECISION LOG ---
st.divider()
st.subheader("Decision Log Entry")
note = st.text_area("Record today's rationale (GIPS Governance):",
                    placeholder="e.g., Retained GIL despite volatility due to Hanes synergy targets.")
if st.button("Save Entry"):
    with open("governance_ips/decision_log.txt", "a") as f:
        f.write(f"\n{pd.Timestamp.now()}: {note}")
    st.success("Entry saved to /governance_ips/")
