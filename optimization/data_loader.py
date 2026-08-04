import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
from datetime import datetime, timedelta

from enable_repo_root import ensure_repo_root
from data_pipeline import get_price_history, get_ohlcv_history

REPO_ROOT = ensure_repo_root(Path(__file__).resolve().parents[1])

class AREDataLoader:
    def __init__(self):
        pass

    # --- 1. MACRO & EQUITY DATA (For HMM & ATR) ---
    def fetch_macro_data(self, tickers=["SPY", "MU", "^VIX", "^IRX", "SHY"]):
        """
        ^VIX: Volatility Index
        ^IRX: 13-Week Treasury Bill (Proxy for Warsh/Rates)
        SHY: 1-3 Year Treasury Bond ETF
        """
        print("Fetching Macro and Equity data...")
        data = get_price_history(tickers, period="1y", interval="1d")
        
        # Calculate Returns and Volatility for HMM
        features = pd.DataFrame()
        features['SPY_Returns'] = data['SPY'].pct_change()
        features['MU_Returns'] = data['MU'].pct_change()
        features['VIX_Level'] = data['^VIX']
        features['Rate_Proxy'] = data['^IRX']
        
        return features.dropna()

    # --- 2. OPTION CHAIN & GAMMA (For GEX) ---
    def fetch_gex_data(self, ticker="SPY"):
        """
        Fetches option chain and calculates Gamma manually.
        Note: Free data doesn't include Gamma; we must derive it.
        """
        print(f"Fetching Option Chain for {ticker}...")
        tk = yf.Ticker(ticker)
        expiry = tk.options[0] # Get the closest expiry (e.g., tomorrow/Tuesday)
        chain = tk.option_chain(expiry)
        
        spot = tk.history(period="1d")['Close'].iloc[-1]
        
        # Combine Calls and Puts
        calls = chain.calls[['strike', 'openInterest', 'lastPrice', 'impliedVolatility']]
        calls['type'] = 'call'
        puts = chain.puts[['strike', 'openInterest', 'lastPrice', 'impliedVolatility']]
        puts['type'] = 'put'
        
        df = pd.concat([calls, puts])
        
        # Calculate Manual Gamma (Black-Scholes Approximation)
        # T = time to expiry (1 day = 1/365)
        T = 1/365
        r = 0.045 # Current 2026 interest rate estimate
        
        def calculate_gamma(row):
            S = spot
            K = row['strike']
            iv = row['impliedVolatility']
            if iv == 0: return 0
            d1 = (np.log(S/K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
            gamma = norm.pdf(d1) / (S * iv * np.sqrt(T))
            return gamma

        df['gamma'] = df.apply(calculate_gamma, axis=1)
        return df, spot

    # --- 3. INTRADAY TICK DATA (For VPIN) ---
    def fetch_intraday_data(self, ticker="MU"):
        """
        Fetches 1-minute data for VPIN toxicity analysis.
        """
        print(f"Fetching Intraday 1-min data for {ticker}...")
        data = get_ohlcv_history(ticker, period="5d", interval="1m", force_refresh=False)
        
        # yfinance doesn't provide trade side (buy/sell), so we use the 
        # 'Tick Rule' (Price > Prev Price = Buy)
        df = data.copy()
        df['price_change'] = df['Close'].diff()
        df['side'] = np.where(df['price_change'] >= 0, 'buy', 'sell')
        df = df.rename(columns={'Volume': 'volume'})
        
        return df[['Close', 'volume', 'side']]