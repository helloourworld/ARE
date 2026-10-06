import yfinance as yf
import numpy as np


def _peer_ev_fcf_benchmark(target_ticker, peer_tickers=None):
    """Compute peer EV/FCF median for AI names."""
    peer_tickers = peer_tickers or ["NVDA", "AVGO", "AMD", "MU", "VRT", "MSFT", "AMZN"]
    peer_values = []
    for ticker in peer_tickers:
        if ticker == target_ticker:
            continue
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            shares = info.get("sharesOutstanding")
            market_cap = info.get("marketCap")
            total_debt = info.get("totalDebt", 0)
            total_cash = info.get("totalCash", 0)
            free_cash_flow = info.get("freeCashflow") or 0
            if not market_cap or not shares or free_cash_flow <= 0:
                continue
            ev = market_cap + total_debt - total_cash
            ev_fcf = ev / free_cash_flow
            peer_values.append(ev_fcf)
        except Exception:
            continue

    if not peer_values:
        return None

    peer_values = np.asarray(peer_values, dtype=float)
    return float(np.median(peer_values))


def _sanitize_growth_and_discount(short_term_growth, terminal_growth, risk_free_rate, equity_risk_premium, beta):
    """Keep AI-quality growth assumptions realistic without forcing absurdly low DCF values."""
    short_term_growth = float(np.clip(short_term_growth, 0.05, 0.25))
    terminal_growth = float(np.clip(terminal_growth, 0.025, 0.045))
    risk_free_rate = float(np.clip(risk_free_rate, 0.03, 0.05))
    equity_risk_premium = float(np.clip(equity_risk_premium, 0.04, 0.07))
    beta = float(np.clip(beta, 0.7, 2.3))
    return short_term_growth, terminal_growth, risk_free_rate, equity_risk_premium, beta


def run_ai_valuation(ticker_symbol, short_term_growth=0.18, terminal_growth=0.03, risk_free_rate=0.048, equity_risk_premium=0.05):
    """
    Values an AI stock using a conservative 2-stage FCFF-style DCF.

    This version adds guardrails so the valuation remains market-aware rather than
    producing mechanically extreme values for structurally high-growth names.
    """
    print(f"\n==========================================")
    print(f" FETCHING FINANCIAL DATA FOR: {ticker_symbol}")
    print(f"==========================================")

    stock = yf.Ticker(ticker_symbol)
    info = stock.info

    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    shares_outstanding = info.get("sharesOutstanding")
    beta = info.get("beta", 1.2)

    if not current_price or not shares_outstanding:
        print(f"Error: Unable to fetch pricing/share data for {ticker_symbol}")
        return

    short_term_growth, terminal_growth, risk_free_rate, equity_risk_premium, beta = _sanitize_growth_and_discount(
        short_term_growth, terminal_growth, risk_free_rate, equity_risk_premium, beta
    )

    cashflow = stock.cashflow
    try:
        operating_cash_flow = cashflow.loc["Operating Cash Flow"].iloc[0]
        capex = abs(cashflow.loc["Capital Expenditure"].iloc[0])
        fcf_ttm = operating_cash_flow - capex
    except Exception:
        fcf_ttm = info.get("freeCashflow", 0)

    if fcf_ttm <= 0:
        print(f"Warning: {ticker_symbol} has negative or zero TTM Free Cash Flow (${fcf_ttm:,.0f}).")
        print("Note: This model is not suitable for companies with negative FCF; use EV/Sales or gross profit metrics instead.")
        return

    cost_of_equity = risk_free_rate + (beta * equity_risk_premium)
    discount_rate = min(cost_of_equity, 0.12)

    projected_fcf = []
    for year in range(1, 6):
        cf = fcf_ttm * ((1 + short_term_growth) ** year)
        discounted_cf = cf / ((1 + discount_rate) ** year)
        projected_fcf.append(discounted_cf)

    fcf_year_5 = fcf_ttm * ((1 + short_term_growth) ** 5)
    terminal_value = (fcf_year_5 * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    discounted_terminal_value = terminal_value / ((1 + discount_rate) ** 5)

    pv_cash_flows = sum(projected_fcf) + discounted_terminal_value

    total_cash = info.get("totalCash", 0)
    total_debt = info.get("totalDebt", 0)

    dcf_value_per_share = (pv_cash_flows + total_cash - total_debt) / shares_outstanding
    peer_ev_fcf = _peer_ev_fcf_benchmark(ticker_symbol)
    peer_anchor_value = None
    if peer_ev_fcf is not None:
        peer_anchor_ev = max(fcf_ttm * peer_ev_fcf, 0.0)
        peer_anchor_value = (peer_anchor_ev + total_cash - total_debt) / shares_outstanding

    if peer_anchor_value is not None:
        fair_value_per_share = 0.8 * dcf_value_per_share + 0.2 * peer_anchor_value
    else:
        fair_value_per_share = dcf_value_per_share

    fair_value_per_share = float(np.nan_to_num(fair_value_per_share, nan=current_price, posinf=current_price * 1.75, neginf=0.0))
    upside_downside = ((fair_value_per_share - current_price) / current_price) * 100

    implied_ev_fcf = ((fair_value_per_share * shares_outstanding) + total_debt - total_cash) / max(fcf_ttm, 1e-9)
    if peer_ev_fcf is not None:
        peer_gap = ((implied_ev_fcf - peer_ev_fcf) / peer_ev_fcf) * 100
    else:
        peer_gap = None

    if fair_value_per_share <= 0:
        valuation_status = "ASSUMPTION-SENSITIVE / INVALID"
    elif upside_downside > 20:
        valuation_status = "UNDERVALUED (BUY)"
    elif upside_downside < -20:
        valuation_status = "OVERVALUED (WAIT/TRIM)"
    else:
        valuation_status = "NEAR MARKET / RANGE-BOUND"

    print(f"Current Stock Price:       ${current_price:.2f}")
    print(f"TTM Free Cash Flow:        ${fcf_ttm / 1e9:.2f} Billion")
    print(f"Beta:                      {beta:.2f}")
    print(f"Discount Rate (WACC):      {discount_rate * 100:.2f}%")
    print(f"Projected 5-Yr Growth:     {short_term_growth * 100:.1f}%")
    print(f"Terminal Growth Rate:      {terminal_growth * 100:.1f}%")
    print(f"------------------------------------------")
    print(f"Intrinsic Fair Value:      ${fair_value_per_share:.2f}")
    print(f"Margin of Safety / Upside: {upside_downside:+.2f}%")
    if peer_ev_fcf is not None:
        print(f"Peer Median EV/FCF:        {peer_ev_fcf:.2f}x")
        print(f"Implied EV/FCF:            {implied_ev_fcf:.2f}x")
        print(f"Vs. Peer:                  {peer_gap:+.2f}%")
    print(f"Verdict:                   {valuation_status}")
    print(f"==========================================\n")


# -------------------------------------------------------------
# RUN VALUATION ACROSS YOUR WATCHLIST
# -------------------------------------------------------------
if __name__ == "__main__":
    # Custom growth assumptions based on consensus analyst forecasts
    ai_watchlist = [
        {"ticker": "NVDA",  "growth": 0.25}, # 25% annual growth
        {"ticker": "AVGO",  "growth": 0.20}, # 20% annual growth
        {"ticker": "AMZN",  "growth": 0.18}, # 18% annual growth
        {"ticker": "GOOGL", "growth": 0.14}, # 14% annual growth
        {"ticker": "VRT",   "growth": 0.22}, # 22% annual growth
    ]

    for item in ai_watchlist:
        run_ai_valuation(item["ticker"], short_term_growth=item["growth"])