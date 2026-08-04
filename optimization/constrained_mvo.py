import pandas as pd
from pathlib import Path
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
import sys


def _find_repo_root(start_path: Path) -> Path:
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "enable_repo_root.py").exists():
            return candidate
    return start_path


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enable_repo_root import ensure_repo_root, load_config
from data_pipeline import get_price_history

REPO_ROOT = ensure_repo_root(REPO_ROOT)


def build_optimization_report(weights, targets, bounds):
    """Create a structured optimization report and a compact summary for downstream use."""
    tickers = list(targets.keys())
    if len(tickers) != len(weights):
        raise ValueError("Weight and target counts must match")

    report = pd.DataFrame({
        "Ticker": tickers,
        "Strategic Target": [targets[ticker] for ticker in tickers],
        "Optimized Weight": [weights.get(ticker, 0.0) for ticker in tickers],
    })
    report["Delta"] = report["Optimized Weight"] - report["Strategic Target"]

    bounds_lookup = {ticker: bounds[idx] for idx, ticker in enumerate(tickers)}
    within_bounds = all(
        lower <= report.loc[report["Ticker"] == ticker, "Optimized Weight"].iloc[0] <= upper
        for ticker, (lower, upper) in bounds_lookup.items()
    )

    summary = {
        "num_assets": len(tickers),
        "within_bounds": within_bounds,
        "max_abs_delta": float(abs(report["Delta"]).max()),
    }

    return report, summary


def run_constrained_optimization():
    # 1. Load Configuration
    cfg = load_config("config.yaml", REPO_ROOT)
    
    tickers = list(cfg['constraints'].keys())
    bounds = [(v['min'], v['max']) for v in cfg['constraints'].values()]
    targets = {k: v['target'] for k, v in cfg['constraints'].items()}

    # 2. Ingest Data
    data = get_price_history(tickers, period="2y", interval="1d")
    # returns = data.pct_change().dropna()

    # 3. Calculate Risk/Return Proxies
    mu = expected_returns.capm_return(data) # Forward-looking via CAPM
    S = risk_models.CovarianceShrinkage(data).ledoit_wolf()

    # 4. Initialize Efficient Frontier with Box Constraints
    ef = EfficientFrontier(mu, S, weight_bounds=bounds)
    
    # 5. Add L2 Regularization (Institutional "Anchor")
    # This penalizes weights that stray too far from your targets
    # gamma=0.1 is your "strength of conviction" in your Strategic Targets
    ef.add_objective(objective_functions.L2_reg, gamma=cfg['parameters']['l2_lambda'])
    
    # 6. Optimize for Max Sharpe
    # raw_weights = ef.max_sharpe()
    # cleaned_weights = ef.clean_weights()
    # 6. Use Quadratic Utility instead of Max Sharpe
    # risk_aversion=3 is standard for a "Balanced/Growth" investor
    raw_weights = ef.max_quadratic_utility(risk_aversion=cfg['parameters']['risk_aversion'])
    cleaned_weights = ef.clean_weights()
    
    # 7. Comparison Report
    report, summary = build_optimization_report(cleaned_weights, targets, bounds)
    
    print("--- Constrained Optimization Results ---")
    print(report.to_string(formatters={'Strategic Target': '{:,.2%}'.format, 
                                       'Optimized Weight': '{:,.2%}'.format,
                                       'Delta': '{:+.2%}'.format}))
    print(f"Summary: {summary}")
    return cleaned_weights, report, summary

if __name__ == "__main__":
    run_constrained_optimization()