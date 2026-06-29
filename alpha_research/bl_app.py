import uuid
from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import inv
from ef_utils import market_cap
try:
    from ..data_pipeline.data_cache import get_data_persistent
except ImportError:
    try:
        from data_pipeline.data_cache import get_data_persistent
    except ImportError:
        package_root = Path(__file__).resolve().parent.parent
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from data_pipeline.data_cache import get_data_persistent

deployment = 1000

st.set_page_config(page_title="Professional BL Manager", layout="wide")

def black_litterman_individual(returns, views, confidences, w_mkt=None):
    """
    BL Model with Individualized Confidence
    """
    tickers = returns.columns
    n = len(tickers)
    cov = returns.cov() * 252
    
    # 1. Equilibrium Prior
    if w_mkt is None:
        w_mkt = np.array([1.0/n] * n) # Equal-weight prior
    
    delta = 2.5
    pi = delta * cov.dot(w_mkt)
    
    # 2. Individualized Omega (Uncertainty)
    # Omega is a diagonal matrix of the variance of the views
    tau = 0.05
    P = np.eye(n)
    Q = views.values
    
    # Uncertainty calculation: (1 - Confidence) / Confidence
    # We add a small epsilon to avoid division by zero
    uncertainties = []
    for c in confidences:
        c = np.clip(c, 0.01, 0.99)
        # Higher confidence = lower uncertainty (omega)
        uncertainties.append(((1 - c) / c) * tau)
    
    Omega = np.diag(uncertainties)
    
    # 3. BL Posterior Returns
    # term1 = [(tau*Sigma)^-1 + P' * Omega^-1 * P]^-1
    # term2 = (tau*Sigma)^-1 * Pi + P' * Omega^-1 * Q
    M_inv = inv(tau * cov)
    Omega_inv = inv(Omega)
    
    term1 = inv(M_inv + P.T.dot(Omega_inv).dot(P))
    term2 = M_inv.dot(pi) + P.T.dot(Omega_inv).dot(Q)
    
    posterior_returns = term1.dot(term2)
    
    # 4. Final Weights (Mean-Variance Optimization)
    weights = inv(delta * cov).dot(posterior_returns)
    weights = np.clip(weights, 0, 1) # No shorting
    return pd.Series(weights / weights.sum(), index=tickers)


# --- STATE MANAGEMENT ---
# Initialize session state for the confidence table and a unique key for the editor
if 'universe' not in st.session_state:
    st.session_state.universe = ['MSFT', 'NVDA', 'GOOG', 'AVGO', 'VRT', 'ENB.TO', 'XEQT.TO', 'AMD']

if 'conf_df' not in st.session_state:
    st.session_state.conf_df = pd.DataFrame({
        'Ticker': st.session_state.universe,
        'Confidence': [0.5] * len(st.session_state.universe)
    })

if 'editor_key' not in st.session_state:
    st.session_state.editor_key = str(uuid.uuid4())

# --- CALLBACKS FOR BULK UPDATE ---
def set_all_views(value):
    st.session_state.conf_df['Confidence'] = value
    # Changing the key forces the data_editor to re-render with new values
    st.session_state.editor_key = str(uuid.uuid4())
    
# --- UI INTERFACE ---
st.title("🛡️ Institutional BL-Momentum Dashboard")
st.subheader("Individualized Confidence Allocation Manager")

# Sidebar Configuration
st.sidebar.header("Individual Views")

# Bulk Update Buttons
col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Set All to 0 (Market Only)"):
        set_all_views(0.01)
with col_b:
    if st.button("Set All to 1 (Max Tilt)"):
        set_all_views(0.99)

# The Individualized Data Editor
# Using the session_state key to allow programmatic resets
edited_df = st.sidebar.data_editor(
    st.session_state.conf_df, 
    key=st.session_state.editor_key, 
    hide_index=True,
    width='stretch'
)
# Sync the edited data back to session state
st.session_state.conf_df = edited_df


if st.button("Run Individualized Optimization"):
    with st.spinner("Analyzing Variance-Covariance Matrix..."):
        # Fetching Data (2026 'Close' fix)
        for t in st.session_state.universe:
            _df = get_data_persistent(t)[['Close']].dropna()
            _df.columns = [t]
            # print(_df)
            df = pd.concat([df, _df], axis=1) if 'df' in locals() else _df
        
        returns = df.pct_change(fill_method=None).dropna()
        
        market_caps, w_mkt = market_cap(st.session_state.universe)
        
        st.text("Market Caps and Weights:")
        for t, mc, w in zip(st.session_state.universe, market_caps, w_mkt):
            st.text(f"{t}: {mc:.0f} -> {w:.2%}")
            
        # Calculate Momentum View (Risk Adjusted)
        df = df.dropna()
        mom_signal = (df.iloc[-1] / df.iloc[-252]) - 1
        st.text("Momentum Signals (Annualized):")
        for t, m in zip(st.session_state.universe, mom_signal):
            st.text(f"{t}: {m:.2%}")
        vol = returns.std() * np.sqrt(252)
        views = mom_signal / vol
        # print(returns.head())
        # Run BL
        bl_weights = black_litterman_individual(returns, views, edited_df['Confidence'].values, w_mkt=w_mkt)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Optimized Portfolio Weights")
            st.bar_chart(bl_weights)
            
        with col2:
            st.subheader(f"Deployment of ${deployment:,.2f} CAD")
            allocation = (bl_weights * deployment).round(2)
            # Create a nice table for the buy orders
            orders = pd.DataFrame(allocation[allocation > 0.01]).rename(columns={0: 'Buy Amount ($)'})
            st.table(orders)

        # Volatility Comparison
        st.divider()
        st.subheader("Volatility Projection")
        port_vol = np.sqrt(bl_weights.T.dot(returns.cov() * 252).dot(bl_weights))
        st.metric("Expected Annual Volatility", f"{port_vol:.2%}")
        
        if port_vol > 0.25:
            st.warning("⚠️ High Volatility: Consider increasing confidence in ENB.TO or XEQT.TO to stabilize.")