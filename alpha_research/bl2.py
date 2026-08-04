import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from data_pipeline.data_cache import get_daily_returns

REPO_ROOT = ensure_repo_root(REPO_ROOT)

# -----------------------------
# 1. Download market data
# -----------------------------
assets = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
start = "2020-01-01"

data = get_daily_returns(assets, assets[0], start)

# Compute daily returns
returns = data

# Annualize covariance
Sigma = returns.cov() * 252

# -----------------------------
# 2. Market weights (equal-weight proxy)
# -----------------------------
n = len(assets)
w_mkt = np.ones(n) / n

# -----------------------------
# 3. Risk aversion parameter
# -----------------------------
# Approximate using market return
market_return = returns.mean().mean() * 252
market_variance = np.mean(np.diag(Sigma))

delta = market_return / market_variance

# -----------------------------
# 4. Implied equilibrium returns
# -----------------------------
Pi = delta * Sigma.values @ w_mkt

# -----------------------------
# 5. Investor Views
# -----------------------------
# Example views:
# 1) AAPL will outperform MSFT by 3%
# 2) TSLA will outperform AMZN by 5%

P = np.array([
    [1, -1, 0, 0, 0],
    [0, 0, 0, -1, 1]
])

Q = np.array([0.03, 0.05])

tau = 0.05

# Omega = uncertainty of views
Omega = np.diag(np.diag(P @ (tau * Sigma.values) @ P.T))

# -----------------------------
# 6. Black-Litterman formula
# -----------------------------
tauSigma = tau * Sigma.values

inv_tauSigma = np.linalg.inv(tauSigma)
inv_Omega = np.linalg.inv(Omega)

middle = np.linalg.inv(inv_tauSigma + P.T @ inv_Omega @ P)

mu_bl = middle @ (
    inv_tauSigma @ Pi + P.T @ inv_Omega @ Q
)

# -----------------------------
# 7. Optimal weights
# -----------------------------
weights = np.linalg.inv(Sigma.values) @ mu_bl / delta

# Normalize weights
weights = weights / weights.sum()

# -----------------------------
# 8. Results
# -----------------------------
bl_returns = pd.Series(mu_bl, index=assets)
bl_weights = pd.Series(weights, index=assets)

print("\nBlack-Litterman Expected Returns:")
print(bl_returns)

print("\nOptimal Portfolio Weights:")
print(bl_weights)

# -----------------------------
# 9. Plot weights
# -----------------------------
bl_weights.plot(kind='bar', title='Black-Litterman Portfolio')
plt.show()