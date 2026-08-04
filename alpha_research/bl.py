import numpy as np
import pandas as pd
from pathlib import Path
from numpy.linalg import inv
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
from data_pipeline import get_price_history

REPO_ROOT = ensure_repo_root(REPO_ROOT)

# 1. Setup Data
tickers = ['MSFT', 'GOOG', 'NVDA', 'AVGO', 'VRT', 'XEQT.TO', 'ENB.TO']
data = get_price_history(tickers, period="2y", interval="1d")
returns = data.pct_change(fill_method=None).dropna()
cov = returns.cov() * 252
delta = 2.5  # Risk aversion coefficient (standard is 2.5)

# 2. Market Weights (use actual Market Caps in production)

# Get real market caps from yfinance and compute market weights
market_caps = []
for t in tickers:
    try:
        info = yf.Ticker(t).info
        mc = info.get('marketCap') or info.get('marketCap')
        if mc is None:
            raise KeyError
    except Exception:
        # fallback: use last price * sharesOutstanding if available, else np.nan
        try:
            print("Failed to get market cap for", t, "- trying fallback method")
            info = yf.Ticker(t).info
            price = info.get('previousClose') or info.get('regularMarketPrice')
            shares = info.get('sharesOutstanding')
            mc = price * shares if price and shares else np.nan
        except Exception:
            print("Failed to get fallback market cap for", t)
            mc = np.nan
    market_caps.append(mc)

market_caps = np.array(market_caps, dtype=float)
if np.isnan(market_caps).any():
    # replace NaNs with equal small weights to avoid breaking things
    nan_idx = np.isnan(market_caps)
    market_caps[nan_idx] = np.nanmean(market_caps)

w_mkt = market_caps / np.nansum(market_caps)


print('Market caps:')
for t, mc, w in zip(tickers, market_caps, w_mkt):
    print(f"{t}: {mc:.0f} -> {w:.2%}")

# 3. Step 1: Calculate Implied Equilibrium Returns (Market Wisdom)
"""What returns must exist to justify current market weights?"""

pi = delta * cov.dot(w_mkt)

# 4. Step 2: Define YOUR VIEWS (The PEAD Input)
# View: MSFT and NVDA will outperform the others by 5% (annualized)
# P is the 'Picking Matrix', Q is the 'Expected Return' of the view
P = np.zeros((1, len(tickers)))
P[0, 0] = 1 # MSFT
P[0, 2] = 1 # NVDA
Q = np.array([0.05]) # Your 5% outperformance view

# 5. Step 3: The Black-Litterman Formula (The Deduction)
tau = 0.05 # Scaler for uncertainty
omega = np.diag(P.dot(tau * cov).dot(P.T)) # Uncertainty of views

# The Master Formula
term1 = inv(inv(tau * cov) + P.T.dot(inv(np.diag(omega))).dot(P))
term2 = inv(tau * cov).dot(pi) + P.T.dot(inv(np.diag(omega))).dot(Q)
bl_returns = term1.dot(term2)

# 6. Result: New Balanced Weights
def optimize_weights(target_returns, covariance):
    # Simplified Mean-Variance using BL Returns
    res = inv(delta * covariance).dot(target_returns)
    return res / res.sum()

new_weights = optimize_weights(bl_returns, cov)
print("--- Black-Litterman Optimized Weights ---")
for t, w in zip(tickers, new_weights):
    print(f"{t}: {w:.2%}")