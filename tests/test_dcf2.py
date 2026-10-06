import sys
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from alpha_research import DCF2


def test_run_ai_valuation_uses_balanced_quality_compounder_logic(monkeypatch, capsys):
    stock = type("Stock", (), {})()
    stock.info = {
        "currentPrice": 150.0,
        "sharesOutstanding": 2_500_000_000,
        "beta": 1.2,
        "totalCash": 12_000_000,
        "totalDebt": 8_000_000,
        "freeCashflow": 9_000_000_000,
    }
    stock.cashflow = pd.DataFrame(
        [[9_000_000_000], [-1_000_000_000]],
        index=["Operating Cash Flow", "Capital Expenditure"],
        columns=["2024"],
    )

    monkeypatch.setattr(DCF2.yf, "Ticker", lambda symbol: stock)
    monkeypatch.setattr(DCF2, "_peer_ev_fcf_benchmark", lambda *args, **kwargs: 120.0)

    DCF2.run_ai_valuation("TEST", short_term_growth=0.18, terminal_growth=0.03, risk_free_rate=0.04, equity_risk_premium=0.05)

    captured = capsys.readouterr().out
    assert "Intrinsic Fair Value:" in captured
    assert "Peer Median EV/FCF" in captured

    value_line = next(line for line in captured.splitlines() if "Intrinsic Fair Value:" in line)
    fair_value = float(value_line.split("$")[1])
    assert fair_value > 127.5
    assert fair_value < 172.5
