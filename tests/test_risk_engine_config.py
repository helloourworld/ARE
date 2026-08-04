from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from risk_modeling.risk_engine import AlphaRiskEngine


def test_risk_engine_uses_configured_defaults():
    engine = AlphaRiskEngine(["SPY"], benchmark="SPY", start_date="2024-01-01")

    assert engine.benchmark == "SPY"
    assert engine.rf == 0.00015873015873015872
