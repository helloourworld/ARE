import pandas as pd
import numpy as np
from pathlib import Path
from requests import Session
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

class MomentumEngine:
    def __init__(self, tickers):
        # Do not set a custom requests Session for yfinance — let yfinance manage its session
        self.tickers = tickers

    def get_data(self):
        # Fetch through persistent cache
        data = get_price_history(self.tickers, period="2y", interval="1d")
        return data

    def calculate_momentum(self, df):
        """
        Calculates Risk-Adjusted Momentum: 
        (12-month return - 1-month return) / 12-month Volatility
        """
        # 1. Total Return (excluding the most recent month to avoid 'reversal' effect)
        df = df.dropna()
        df = df[df.columns.intersection(self.tickers)]  # Ensure we only use the specified tickers  

        
        # Calculate momentum signal
   
        returns_12m = df.pct_change(252)
        returns_1m = df.pct_change(21)

        momentum_signal = returns_12m - returns_1m

        # 2. Volatility (Standard Deviation of daily returns)
        volatility = df.pct_change().rolling(252).std() * np.sqrt(252)
        
        # 3. Risk-Adjusted Score
        score = momentum_signal / volatility
        return score.iloc[-1].sort_values(ascending=False)