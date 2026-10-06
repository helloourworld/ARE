import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import fundamental.Economic as economic


def test_leading_indicators_use_valid_fred_ids():
    assert "ISM/MAN_PMI" not in economic.LEADING_INDICATORS.values()
    assert "ISM/SERV_PMI" not in economic.LEADING_INDICATORS.values()
    assert "UMCSENT" in economic.LEADING_INDICATORS.values()
    assert "RECPROUSM156N" in economic.LEADING_INDICATORS.values()
    for value in economic.LEADING_INDICATORS.values():
        assert "/" not in value


def test_economic_series_include_treasury_yield_curve():
    assert economic.ECONOMIC_SERIES["2Y Treasury Yield"] == "DGS2"
    assert economic.ECONOMIC_SERIES["10Y Treasury Yield"] == "DGS10"
    assert economic.ECONOMIC_SERIES["30Y Treasury Yield"] == "DGS30"


def test_macro_health_summary_generates_score_and_label():
    sample = {
        "GDP YoY": [2.0],
        "Unemployment Rate": [4.0],
        "Consumer Sentiment": [99.0],
        "Chicago Activity Index": [0.5],
        "Recession Probability": [0.2],
    }
    summary = economic.compute_macro_health_summary(sample)
    assert 0 <= summary["score"] <= 100
    assert summary["label"] in {"Expansionary", "Neutral", "Softening", "Cooling", "Risky"}
