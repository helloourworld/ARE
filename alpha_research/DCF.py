import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import sys
import argparse

try:
    from alpha_research.DCF_recovery import calculate_recovery_dcf, parse_forecast
except ModuleNotFoundError:
    from DCF_recovery import calculate_recovery_dcf, parse_forecast


def _latest_statement_value(statement, labels):
    """Return the latest non-null value for the first matching Yahoo row."""
    for label in labels:
        if label in statement.index:
            values = statement.loc[label].dropna()
            if not values.empty:
                return float(values.iloc[0])
    available = ", ".join(str(label) for label in statement.index)
    raise KeyError(f"None of {labels!r} found; available rows: {available}")


def get_live_metrics(ticker_symbol):
    """
    Fetches required trailing financial data directly from Yahoo Finance.
    """
    print(f"🚀 Fetching live financial data for {ticker_symbol}...")
    tk = yf.Ticker(ticker_symbol)
    
    # Financial Statement Extraction
    cash_flow = tk.cashflow
    balance_sheet = tk.balance_sheet
    info = tk.info
    
    # Prefer Yahoo's reported FCF; its capex row changed from plural to singular.
    try:
        fcf_base = _latest_statement_value(cash_flow, ['Free Cash Flow']) / 1e6
    except KeyError:
        ocf = _latest_statement_value(
            cash_flow, ['Operating Cash Flow', 'Total Cash From Operating Activities']
        )
        capex = abs(_latest_statement_value(
            cash_flow, ['Capital Expenditures', 'Capital Expenditure', 'Capital Expenditures In PPE']
        ))
        fcf_base = (ocf - capex) / 1e6
    
    # Balance Sheet Data
    cash = _latest_statement_value(
        balance_sheet, ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments']
    ) / 1e6
    try:
        debt = _latest_statement_value(balance_sheet, ['Total Debt']) / 1e6
    except KeyError:
        st_debt = _latest_statement_value(balance_sheet, ['Current Debt']) if 'Current Debt' in balance_sheet.index else 0
        lt_debt = _latest_statement_value(balance_sheet, ['Long Term Debt']) if 'Long Term Debt' in balance_sheet.index else 0
        debt = (st_debt + lt_debt) / 1e6

    shares_out = info.get('sharesOutstanding')
    if not shares_out:
        shares_out = _latest_statement_value(
            balance_sheet, ['Ordinary Shares Number', 'Share Issued']
        )
    shares_out /= 1e6
    market_price = info.get('currentPrice') or info.get('regularMarketPrice')
    
    return {
        "fcf_base": fcf_base,
        "shares_out": shares_out,
        "debt": debt,
        "cash": cash,
        "market_price": market_price,
    }

def calculate_dcf_core(fcf_base, growth_high, growth_terminal, wacc, shares_out, debt, cash, years=5):
    """
    Core mathematical execution of a Multi-Period DCF. Returns price per share.
    """
    projected_fcf = []
    current_fcf = fcf_base
    for _ in range(years):
        current_fcf *= (1 + growth_high)
        projected_fcf.append(current_fcf)
        
    terminal_value = (projected_fcf[-1] * (1 + growth_terminal)) / (wacc - growth_terminal)
    pv_fcf = [projected_fcf[i] / ((1 + wacc) ** (i + 1)) for i in range(years)]
    pv_terminal_value = terminal_value / ((1 + wacc) ** years)
    
    enterprise_value = sum(pv_fcf) + pv_terminal_value
    equity_value = enterprise_value + cash - debt
    return equity_value / shares_out

def run_valuation_suite(ticker_symbol="AVGO", recovery_forecast=None):
    # 1. Fetch live metrics or use baseline defaults on failure
    try:
        inputs = get_live_metrics(ticker_symbol)
    except Exception as e:
        print(f"⚠️ API Fetch failed ({e}), using default baseline metrics.")
        inputs = {
            "fcf_base": 19000.0, "shares_out": 4650.0, "debt": 70000.0,
            "cash": 12000.0, "market_price": None,
        }

    # Append structural growth parameters
    inputs.update({"growth_high": 0.24, "years": 5})
    
    print("\n" + "="*60)
    print(f"📊 BASE DATA EXTRACTION FOR {ticker_symbol}")
    print(f"Trailing Free Cash Flow: ${inputs['fcf_base']:,.2f} M")
    print(f"Shares Outstanding:     {inputs['shares_out']:,.2f} M")
    print(f"Total Debt:             ${inputs['debt']:,.2f} M")
    print(f"Cash & Equivalents:     ${inputs['cash']:,.2f} M")
    if inputs.get("market_price"):
        print(f"Current Market Price:   ${inputs['market_price']:,.2f}")
    if inputs["fcf_base"] <= 0:
        print("Warning:                 Negative FCF; use a recovery-case forecast.")
    print("="*60)

    base_case_price = calculate_dcf_core(
        fcf_base=inputs["fcf_base"], growth_high=inputs["growth_high"],
        growth_terminal=0.035, wacc=0.080, shares_out=inputs["shares_out"],
        debt=inputs["debt"], cash=inputs["cash"], years=inputs["years"]
    )
    print(f"Base-Case DCF Value:    ${base_case_price:,.2f}")
    if inputs.get("market_price"):
        difference = base_case_price / inputs["market_price"] - 1
        if base_case_price <= 0:
            valuation = "NEGATIVE DCF - RECOVERY CASE NEEDED"
        else:
            valuation = "UNDERVALUED" if difference > 0 else "OVERVALUED" if difference < 0 else "FAIRLY VALUED"
        print(f"DCF vs Market:          {difference:+.1%} ({valuation})")
    if inputs["fcf_base"] <= 0 and recovery_forecast:
        recovery_price = calculate_recovery_dcf(
            fcf_forecast=recovery_forecast, growth_terminal=0.035, wacc=0.080,
            shares_out=inputs["shares_out"], debt=inputs["debt"], cash=inputs["cash"]
        )
        print(f"Recovery-Case DCF:      ${recovery_price:,.2f}")
        print(f"Recovery FCF Forecast:   {recovery_forecast} M")

    # 2. Generate Sensitivity Matrix
    wacc_range = [0.070, 0.075, 0.080, 0.085, 0.090]
    tg_range = [0.025, 0.030, 0.035, 0.040, 0.045]
    matrix_data = np.zeros((len(wacc_range), len(tg_range)))

    for i, wacc in enumerate(wacc_range):
        for j, tg in enumerate(tg_range):
            matrix_data[i, j] = calculate_dcf_core(
                fcf_base=inputs["fcf_base"], growth_high=inputs["growth_high"],
                growth_terminal=tg, wacc=wacc, shares_out=inputs["shares_out"],
                debt=inputs["debt"], cash=inputs["cash"], years=inputs["years"]
            )

    df_sensitivity = pd.DataFrame(
        matrix_data,
        index=[f"WACC {w*100:.1f}%" for w in wacc_range],
        columns=[f"Term Growth {g*100:.1f}%" for g in tg_range]
    )
    
    print("\n" + "="*70)
    print(f"   {ticker_symbol} SENSITIVITY MATRIX (INTRINSIC STOCK PRICE)   ")
    print(f"   Assumed High Growth Rate: {inputs['growth_high']*100:.1f}% for {inputs['years']} Years")
    print("="*70)
    print(df_sensitivity.round(2).to_string())
    print("="*70)

    # 3. Monte Carlo Simulation (10,000 Paths)
    print("\n🎲 Running 10,000 Monte Carlo Simulation Paths...")
    iterations = 10000
    wacc_sim = np.random.normal(loc=0.080, scale=0.005, size=iterations)
    growth_high_sim = np.random.normal(loc=0.24, scale=0.04, size=iterations)
    growth_term_sim = np.random.normal(loc=0.035, scale=0.003, size=iterations)
    
    simulated_prices = []
    for i in range(iterations):
        w, gh, gt = wacc_sim[i], growth_high_sim[i], growth_term_sim[i]
        if w <= gt or w <= 0 or gh <= 0:
            continue
            
        price = calculate_dcf_core(
            fcf_base=inputs["fcf_base"], growth_high=gh, growth_terminal=gt,
            wacc=w, shares_out=inputs["shares_out"], debt=inputs["debt"],
            cash=inputs["cash"], years=inputs["years"]
            )
        simulated_prices.append(price)
        
    prices = np.array(simulated_prices)
    p10, p50, p90 = np.percentile(prices, [10, 50, 90])
    
    print("=" * 50)
    print(f"      {ticker_symbol} MONTE CARLO SIMULATION SUMMARY        ")
    print("=" * 50)
    print(f"Mean Estimated Price:     ${np.mean(prices):.2f}")
    print(f"Median Price (50th%):     ${p50:.2f}")
    print(f"Bear Case Target (10th%): ${p10:.2f}")
    print(f"Bull Case Target (90th%): ${p90:.2f}")
    print("=" * 50)
    
    # 4. Plot Combined Visual Dashboard
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Heatmap Plot
    sns.heatmap(df_sensitivity, annot=True, fmt=".2f", cmap="YlGnBu", 
                cbar_kws={'label': 'Intrinsic Stock Price ($)'}, ax=axes[0])
    axes[0].set_title(f'{ticker_symbol} DCF Sensitivity Matrix', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Terminal Growth Rate (%)', labelpad=8)
    axes[0].set_ylabel('WACC / Discount Rate (%)', labelpad=8)
    
    # Monte Carlo Plot
    axes[1].hist(prices, bins=100, density=True, alpha=0.6, color='g', edgecolor='black')
    axes[1].axvline(p50, color='red', linestyle='dashed', linewidth=2, label=f'Median: ${p50:.2f}')
    axes[1].axvline(p10, color='orange', linestyle='dotted', linewidth=2, label=f'10th Pct: ${p10:.2f}')
    axes[1].axvline(p90, color='blue', linestyle='dotted', linewidth=2, label=f'90th Pct: ${p90:.2f}')
    axes[1].set_title(f'{ticker_symbol} Monte Carlo Price Distribution Density', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Simulated Intrinsic Share Price ($)')
    axes[1].set_ylabel('Probability Density')
    axes[1].set_xlim(50, p90 * 1.5) # Dynamic visual limits framing distribution curve
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a multi-period DCF valuation.")
    parser.add_argument("ticker", nargs="?", default="MSFT")
    parser.add_argument(
        "--recovery-fcf", type=parse_forecast,
        help="optional comma-separated recovery-case annual FCF forecast in $M"
    )
    arguments = parser.parse_args()
    run_valuation_suite(arguments.ticker.upper(), arguments.recovery_fcf)

"""
python alpha_research\DCF.py INTC --recovery-fcf=-2000,1000,4000,6000,8000"""
