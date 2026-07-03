import yaml
import pandas as pd
import sys
from pathlib import Path
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline import get_price_history

def run_constrained_optimization():
    # 1. Load Configuration
    config_path = REPO_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
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
    report = pd.DataFrame({
        "Strategic Target": targets,
        "Optimized Weight": cleaned_weights
    })
    report['Delta'] = report['Optimized Weight'] - report['Strategic Target']
    
    print("--- Constrained Optimization Results ---")
    print(report.to_string(formatters={'Strategic Target': '{:,.2%}'.format, 
                                       'Optimized Weight': '{:,.2%}'.format,
                                       'Delta': '{:+.2%}'.format}))
    return cleaned_weights

if __name__ == "__main__":
    run_constrained_optimization()