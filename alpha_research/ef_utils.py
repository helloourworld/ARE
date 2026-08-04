import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf
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
from data_pipeline.data_cache import normalize_timestamp_for_index

REPO_ROOT = ensure_repo_root(REPO_ROOT)

TRADING_DAYS = 252


def download_price_data(tickers, start_date="2020-01-01"):
    data = get_price_history(tickers, period="5y", interval="1d")
    if data.empty:
        raise ValueError(f"No price data returned for tickers: {tickers}")

    start_ts = normalize_timestamp_for_index(start_date, data.index)
    data = data[data.index >= start_ts]
    prices = data
    prices = prices.dropna(axis=0, how="any")
    if prices.empty:
        raise ValueError("Downloaded price data contains no complete rows after dropping NA values.")

    return prices


def annualized_return_covariance(prices, trading_days=TRADING_DAYS):
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * trading_days
    cov_matrix = returns.cov() * trading_days
    return mean_returns, cov_matrix, returns


def market_cap(tickers, anchor_ticker="XEQT.TO", anchor_weight=0.60):
    market_caps = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            mc = info.get("marketCap")
            if mc is None:
                raise KeyError
        except Exception:
            try:
                info = yf.Ticker(ticker).info
                price = info.get("previousClose") or info.get("regularMarketPrice")
                shares = info.get("sharesOutstanding")
                mc = price * shares if price and shares else np.nan
            except Exception:
                mc = np.nan
        market_caps.append(mc)

    market_caps = np.array(market_caps, dtype=float)
    if np.isnan(market_caps).all():
        raise ValueError("Unable to resolve any market caps for the provided tickers.")

    if np.isnan(market_caps).any():
        market_caps[np.isnan(market_caps)] = np.nanmean(market_caps)

    weights = market_caps / np.nansum(market_caps)
    weights = pd.Series(weights, index=tickers)

    if anchor_ticker in tickers:
        anchor_weight = float(anchor_weight)
        if not 0 <= anchor_weight <= 1:
            raise ValueError("Anchor weight must be between 0 and 1.")

        weights.loc[anchor_ticker] = anchor_weight
        other_tickers = [t for t in tickers if t != anchor_ticker]
        if other_tickers:
            weights.loc[other_tickers] = (
                (1.0 - anchor_weight)
                * weights.loc[other_tickers]
                / weights.loc[other_tickers].sum()
            )

    return market_caps, weights


def simulate_random_portfolios(mean_returns, cov_matrix, num_portfolios=100_000, rf=0.045, seed=42):
    tickers = mean_returns.index
    n = len(tickers)

    rng = np.random.default_rng(seed)
    raw_weights = rng.random((num_portfolios, n))
    normalized_weights = raw_weights / raw_weights.sum(axis=1, keepdims=True) # Normalize to sum to 1

    portfolio_returns = normalized_weights.dot(mean_returns.values)
    # np.einsum is used for efficient computation of the quadratic form w^T * Cov * w for each portfolio
    portfolio_volatilities = np.sqrt(
        np.einsum("ij,jk,ik->i", normalized_weights, cov_matrix.values, normalized_weights)
    )
    sharpe_ratios = (portfolio_returns - rf) / portfolio_volatilities

    results = pd.DataFrame(
        {
            "Return": portfolio_returns,
            "Volatility": portfolio_volatilities,
            "Sharpe": sharpe_ratios,
        }
    )

    return results, normalized_weights


def extract_efficient_frontier(portfolios):
    sorted_portfolios = portfolios.sort_values(by="Volatility").reset_index(drop=True)
    best_return = -np.inf
    frontier_rows = []

    for _, row in sorted_portfolios.iterrows():
        if row["Return"] > best_return:
            frontier_rows.append(row)
            best_return = row["Return"]

    return pd.DataFrame(frontier_rows)


def build_cml(opt_return, opt_vol, rf=0.045, num_points=100):
    x_values = np.linspace(0, opt_vol, num_points)
    y_values = rf + (opt_return - rf) / opt_vol * x_values
    return x_values, y_values


def compute_portfolio_metrics(weights, mean_returns, cov_matrix, rf=0.045):
    weights = np.asarray(weights, dtype=float)
    expected_return = np.dot(weights, mean_returns.values)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
    sharpe = (expected_return - rf) / volatility
    return expected_return, volatility, sharpe
