import numpy as np
import pandas as pd
import sys
from pathlib import Path
from scipy.optimize import minimize

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline import get_price_history

# 1. Tickers from your screenshot + stabilizing 'Fund 2' candidates
tickers = ['MSFT', 'GOOG', 'NVDA', 'AVGO', 'VRT', 'XEQT.TO', 'ENB.TO', 'XDIV.TO']
risk_free_rate = 0.038 # Current 2026 Risk-Free Rate

# 2. Download Data (Updated to 'Close')
# Note: Period set to 1y to capture the recent high-vol AI regime
data = get_price_history(tickers, period="1y", interval="1d")
returns = data.pct_change().dropna()

# 3. Covariance and Mean Returns
mu = returns.mean() * 252
sigma = returns.cov() * 252

# 4. Maximize Sharpe Ratio (The Tangency Portfolio)
def get_sharpe(weights):
    p_ret = np.sum(mu * weights)
    p_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
    return -(p_ret - risk_free_rate) / p_vol

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(tickers)))
optimized = minimize(get_sharpe, len(tickers)*[1./len(tickers)], 
                     method='SLSQP', bounds=bounds, constraints=constraints)

weights = pd.Series(optimized.x, index=tickers)
print("--- Optimal 'Risky Fund' Weights ---")
print(weights[weights > 0.01].sort_values(ascending=False)) # Showing only significant weights