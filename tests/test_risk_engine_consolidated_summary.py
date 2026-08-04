import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from risk_modeling.risk_engine import AlphaRiskEngine


def test_get_risk_summary_returns_structured_metrics(monkeypatch):
    engine = AlphaRiskEngine(tickers=["AAPL"], benchmark="SPY")
    returns = pd.DataFrame(
        {
            "AAPL": [0.01, -0.02, 0.005, 0.03, 0.012, -0.015],
            "SPY": [0.004, -0.01, 0.002, 0.015, 0.008, -0.006],
        },
        index=pd.date_range("2026-07-01", periods=6, freq="D"),
    )
    monkeypatch.setattr(engine, "ingest_data", lambda: returns)

    summary = engine.get_risk_summary()

    assert isinstance(summary, dict)
    assert "summary" in summary
    assert "by_asset" in summary
    assert "consistency_checks" in summary
    assert "advanced_metrics" in summary
    assert "AAPL" in summary["by_asset"]

    asset_metrics = summary["by_asset"]["AAPL"]
    for field in ["daily_es", "theoretical_annualized_es", "empirical_annualized_es", "beta", "data_points"]:
        assert field in asset_metrics
