Alpha Risk Engine (ARE)
===============

Investment Framework & Analytical Architecture
----------------

**Lead Analyst:** Davdi Yu, CFA  
**Strategy:** Multi-Factor Tactical Allocation (Growth/Value/Quality)  
**Universe:** North American Equities  & Global (EAFE & EM)

Executive Summary
------------------------

The **Alpha Risk Engine (ARE)** is a systematic framework designed to manage a concentrated **high-conviction** portfolio through the lens of institutional risk management.

By integrating **Fundamental Equity Research** (Alpha) with **Quantitative Factor Modeling** (Risk), ARE aims to achieve risk-adjusted outperformance while mitigating tail-risks associated with cyclical infrastructure buildouts (e.g., AI/NAND cycles) and macroeconomic shocks.

Directory Architecture
-------------------------

The engine is organized into six functional modules:

```Table
/ARE_Project_Root

├── alpha_research/            # Fundamental deep-dives (GIL Mergers, SNDK Cycle Analysis)
├── data_pipeline/             # Python scripts for IBKR TWS API & YFinance data ingestion
├── risk_modeling/             # Factor-based risk engines (Fama-French, Ledoit-Wolf, CVaR)
├── optimization/              # Black-Litterman & Mean-Variance Optimization models
├── governance_ips/            # Investment Policy Statement (IPS) & Decision Logs (Minutes)
└── reporting_attribution/     # GIPS-compliant performance & Brinson attribution reports
````

Quantitative Methodologies
--------------------------

ARE utilizes institutional-grade Python libraries to execute the following models:

* Robust Covariance: Utilizing Ledoit-Wolf Shrinkage to stabilize the covariance matrix and solve the "Suboptimal Benchmark" ($\rho$ > 1) problem.

* Factor Attribution: A global multi-factor model decompose returns into Market ($\beta$), Size (SMB), and Value (HML) exposures.

* Tail Risk Analysis: Calculating Conditional Value at Risk (CVaR) to account for non-normal distributions in tech-heavy positions (SNDK).

* Black-Litterman Allocation: Blending market equilibrium weights with analyst-specific views on GIL (Post-Merger Value) and SNDK (Infrastructure Peak).

* Market News Sentiment: Yahoo Finance headlines are scored with pretrained FinBERT. The headline score is positive-class probability minus negative-class probability; a 24-hour half-life weights recent headlines more heavily when aggregating sentiment. The dashboard also reports sentiment breadth, model confidence, and effective news count. Google News RSS is used only if Yahoo Finance returns no headlines and is restricted to personal, non-commercial feed-reader use. These are language indicators, not return forecasts or standalone trade signals.

Current Strategic Views (2026 Q2)
---------------------------------

* AI Infrastructure Cycle: Underweight hardware (SNDK) due to anticipated saturation of the CAPEX supercycle; pivot to software/services (GOOG).

* Apparel/Basics Integration: Overweight GIL based on vertical integration efficiencies and HanesBrands synergy capture.

* Defensive Core: Utilization of FBAL and XDIV to maintain a lower-volatility systematic floor during geopolitical volatility.

Operational Governance
-----------------------

* Execution Desk: All trades are executed via broker utilizing Single Login System (SLS) for Joint/Individual account consolidation.

* Decision Logging: Every rebalancing event requires a written entry in the `governance_ips` folder to mitigate behavioral bias.

* Tax Efficiency: Prioritization of "In-Kind" transfers to minimize capital gains drag on long-term "winners."

Tech Stack
----------

Language: Python 3.x
Data: yfinance, pandas_datareader (IBKR API for live execution)
Analytics: pandas, numpy, statsmodels, PyPortfolioOpt
Optimization: PyPortfolioOpt
Visualization: matplotlib, plotly

How to use this README
-----------------------------

1. **Identity:** As an analyst, this doc is your "Anchor." When the market is volatile (like **MSFT**'s current weakness), read Section 1 and Section 4 to remind yourself of the strategy.
2. **Governance:** If you decide to sell **MCD**, you must first record the reason in the `governance_ips` folder mentioned in the README.
3. **Onboarding:** If you ever decide to hire a junior analyst or a partner, this README serves as the "Instruction Manual" for how your fund operates.
