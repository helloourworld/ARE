import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import data_pipeline.data_utils as data_utils


def test_get_10y_yield_falls_back_to_yfinance(monkeypatch):
    def boom(*args, **kwargs):
        raise TimeoutError("FRED timeout")

    monkeypatch.setattr(data_utils, "_fetch_10y_yield_from_fred", boom)
    monkeypatch.setattr(
        data_utils,
        "_fetch_10y_yield_from_yfinance",
        lambda: {"val": 4.5, "delta": 0.1, "delta_pct": 2.27},
    )

    result = data_utils.get_10y_yield()

    assert result["val"] == 4.5
    assert result["delta"] == 0.1
