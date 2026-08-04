from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimization.constrained_mvo import build_optimization_report


def test_build_optimization_report_exposes_summary_metrics():
    weights = {"XEQT.TO": 0.28, "MSFT": 0.17}
    targets = {"XEQT.TO": 0.25, "MSFT": 0.15}
    bounds = [(0.15, 0.30), (0.05, 0.20)]

    report, summary = build_optimization_report(weights, targets, bounds)

    assert list(report["Ticker"]) == ["XEQT.TO", "MSFT"]
    assert report.loc[report["Ticker"] == "XEQT.TO", "Optimized Weight"].iloc[0] == 0.28
    assert summary["num_assets"] == 2
    assert summary["within_bounds"] is True
    assert abs(summary["max_abs_delta"] - 0.03) < 1e-12
