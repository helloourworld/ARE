"""Screen Canadian-listed utility stocks with Yahoo Finance fundamentals.

Yahoo Finance reports ``dividendYield`` and ``returnOnEquity`` as decimal
fractions, while ``debtToEquity`` is reported as a percentage.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import pandas as pd
import yfinance as yf

CANADIAN_UTILITY_TICKERS = (
    "FTS.TO",
    "EMA.TO",
    "CU.TO",
    "AQN.TO",
    "CPX.TO",
    "NPI.TO",
    "BEP-UN.TO",
    "BIP-UN.TO",
)

RESULT_COLUMNS = (
    "Ticker",
    "Name",
    "Sector",
    "Dividend Yield (%)",
    "ROE (%)",
    "Debt/Equity (%)",
    "Passes",
    "Reason",
    "Error",
)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def screen_canadian_utility_stocks(
    tickers: Iterable[str] = CANADIAN_UTILITY_TICKERS,
    *,
    min_dividend_yield: float = 0.04,
    min_roe: float = 0.15,
    max_debt_to_equity_pct: float = 100.0,
) -> pd.DataFrame:
    """Return one screening result per ticker, including reasons for exclusion.

    ``min_dividend_yield`` and ``min_roe`` are decimal fractions (0.04 means
    4%). Yahoo Finance's ``debtToEquity`` is treated as a percentage (100 means
    debt equals equity). All boundaries are strict: values equal to a threshold
    do not pass.
    """
    thresholds = {
        "min_dividend_yield": min_dividend_yield,
        "min_roe": min_roe,
        "max_debt_to_equity_pct": max_debt_to_equity_pct,
    }
    for name, value in thresholds.items():
        numeric_value = _finite_number(value)
        if numeric_value is None or numeric_value < 0:
            raise ValueError(f"{name} must be a finite, non-negative number.")
        thresholds[name] = numeric_value

    if isinstance(tickers, str):
        normalized_tickers = [tickers.strip()]
    else:
        normalized_tickers = [str(ticker).strip() for ticker in tickers]
    normalized_tickers = list(dict.fromkeys(ticker for ticker in normalized_tickers if ticker))

    rows = []
    for ticker in normalized_tickers:
        row = {
            "Ticker": ticker,
            "Name": ticker,
            "Sector": None,
            "Dividend Yield (%)": None,
            "ROE (%)": None,
            "Debt/Equity (%)": None,
            "Passes": False,
            "Reason": "",
            "Error": None,
        }
        try:
            info = yf.Ticker(ticker).info
        except Exception as exc:
            row["Reason"] = "无法获取基本面数据"
            row["Error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue

        row["Name"] = info.get("shortName") or info.get("longName") or ticker
        row["Sector"] = info.get("sector")
        dividend_yield = _finite_number(info.get("dividendYield"))
        roe = _finite_number(info.get("returnOnEquity"))
        debt_to_equity_pct = _finite_number(info.get("debtToEquity"))

        if dividend_yield is not None:
            row["Dividend Yield (%)"] = dividend_yield * 100
        if roe is not None:
            row["ROE (%)"] = roe * 100
        row["Debt/Equity (%)"] = debt_to_equity_pct

        reasons = []
        if row["Sector"] != "Utilities":
            reasons.append("行业不是 Utilities")
        if dividend_yield is None:
            reasons.append("缺少股息率数据")
        elif dividend_yield <= thresholds["min_dividend_yield"]:
            reasons.append(
                f"股息率不高于 {thresholds['min_dividend_yield'] * 100:.2f}%"
            )
        if roe is None:
            reasons.append("缺少 ROE 数据")
        elif roe <= thresholds["min_roe"]:
            reasons.append(f"ROE 不高于 {thresholds['min_roe'] * 100:.2f}%")
        if debt_to_equity_pct is None:
            reasons.append("缺少负债/权益数据")
        elif debt_to_equity_pct >= thresholds["max_debt_to_equity_pct"]:
            reasons.append(
                "负债/权益不低于 "
                f"{thresholds['max_debt_to_equity_pct']:.2f}%"
            )

        row["Passes"] = not reasons
        row["Reason"] = "符合条件" if not reasons else "；".join(reasons)
        rows.append(row)

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if not results.empty:
        results = results.sort_values(
            ["Passes", "Dividend Yield (%)"],
            ascending=[False, False],
            na_position="last",
            ignore_index=True,
        )
    return results


if __name__ == "__main__":
    screening = screen_canadian_utility_stocks()
    print(screening.to_string(index=False))