# 🧠 Mandelbrot Fractal Engine: Indicator Guide

This system uses **Fractal Geometry** and **Information Theory** to identify market regimes. It moves beyond traditional "indicators" by measuring the **statistical structure** of price action.

---

## 1. The Hurst Exponent (Trend Persistence)

**What it measures:** Market "Memory." It tells you if the current move is a self-reinforcing trend or just random noise.

* **Logic:**
  * **$H > 0.55$ (Persistence):** Returns show persistence over the calculation window. This supports a **Trending Regime**, but does not guarantee the next bar's direction.
  * **$H \approx 0.50$ (Random Walk):** The market is a coin flip. No memory.
  * **$H < 0.45$ (Anti-Persistence):** Returns are anti-persistent and may support mean-reversion analysis.
* **Example:**
  * **MU @ $1,228 (Hurst 0.52):** Even though it's up 16%, the memory is low. It's a "Random Walk" at a high level.
  * **NVDA @ $202 (Hurst 0.61):** High persistence. The recent return path is more trend-like, although the metric does not identify who is trading.

---

## 2. The Tail Index ($\alpha$) (Crash/Jump Risk)

**What it measures:** "Wild Randomness." It calculates the thickness of the "Fat Tails" in the return distribution.

* **Logic:**
  * **$\alpha > 1.7$ (Tail-Stable):** Fewer extreme negative-return observations are present in the estimator's view; this is not a guarantee of normal or safe returns.
  * **$\alpha < 1.55$ (Tail Risk):** Negative returns have heavier estimated tails. Reduce exposure and account for jump risk; the metric does not predict a specific move or size.
* **Example:**
  * **GOOG @ $343 (\alpha = 1.50):** Elevated tail concern. The estimate supports wider risk assumptions, but does not specify a target or jump size.
  * **SPY @ 734 (\alpha = 3.12):** Tail conditions appear more stable in this sample; diversification does not remove market risk.

---

## 3. Cumulative Volume Delta (CVD) Proxy

**What it measures:** A volume-weighted proxy for buying and selling pressure.

* **Logic:** Estimates directional volume using return/volatility normalization. It does not identify specific institutions, retail traders, or order-book participants.
* **Key Pattern: The Divergence.**
  * **Price UP + CVD DOWN:** **Bearish divergence.** Price is rising while the estimated directional volume weakens.
  * **Price DOWN + CVD UP:** **Bullish absorption candidate.** Selling pressure may be weakening, but this is not a confirmed bottom signal.
* **Example:**
  * **MU @ $1,235 (Price Up, CVD Trend Down):** Warning: price is rising while estimated directional volume weakens. Review for divergence rather than assuming a specific participant's action.

---

## 4. Shannon Entropy (Market Complexity)

**What it measures:** "Noise" vs. "Signal."

* **Logic:**
  * **Low Entropy (< 2.2):** Returns are more concentrated across the configured bins; Hurst may be easier to interpret.
  * **High Entropy (> 2.8):** Returns are more dispersed across the configured bins; treat Hurst and directional signals with more caution.
* **Example:**
  * **High-entropy periods:** Directional signals may be less reliable. Treat Hurst as context and confirm it with the other risk measures.

---

## 5. VPIN (Volume Imbalance Proxy)

**What it measures:** Estimated volume imbalance and potential liquidity stress.

* **Logic:**
  * **VPIN > 0.75:** High estimated volume imbalance and elevated liquidity risk. Confirm with volatility, CVD, and the risk scores; a jump is not guaranteed.
  * **VPIN < 0.40:** Lower estimated volume imbalance. It does not identify the market participants driving the flow.
* **Example:**
  * **Pre-Earnings MU:** A VPIN increase can indicate changing volume imbalance around an event, but the proxy cannot establish who is trading or predict the event outcome.

---

## 6. Fragility Score (Structural Market Weakness)

**What it measures:** Whether market internals are deteriorating beneath the current price action. Fragility is different from realized stress: a market can be structurally fragile before a large move appears in volatility or drawdown data.

The score is bounded from **0 to 100** and is built from rolling percentile ranks of five components:

* **Downside flow (35%):** Negative Chaikin Money Flow.
* **Breadth weakness (25%):** Deterioration in the asset's price relative to its market benchmark. Canadian listings use XIU; US listings use RSP.
* **Weak memory (15%):** Hurst values below the $0.55$ reference level.
* **Concentration (15%):** Concentration risk. The current Mandelbrot scanner supplies a neutral zero history because no concentration series is connected yet.
* **Drawdown acceleration (10%):** Faster deterioration in drawdown.

**Interpretation:**

| Score | Meaning | Icon |
| :--- | :--- | :--- |
| $<30$ | Healthy | ✅ |
| $30$–$<50$ | Normal | 🟢 |
| $50$–$<70$ | Watch | ⚠️ |
| $70$–$<85$ | Elevated | 🟠 |
| $\geq85$ | Critical | 🚨 |

The alert levels used by the scanner are **Watch at 60**, **Elevated at 75**, and **Critical at 85**. The display interpretation is slightly more granular than the alert level, so a score can display as "Watch" before it reaches the formal watch alert threshold.

**Key warning:** A high fragility score describes weak internal structure; it does not by itself predict the direction or timing of the next move. Use it with regime, tail index, CVD, and VPIN.

For broad Canadian ETFs such as **XDIV**, **XEQT**, and **XIU**, the scanner does not use ETF trading-volume CMF as a proxy for the underlying holdings' money flow. Their fragility calculation emphasizes benchmark-relative breadth and drawdown acceleration instead. This avoids treating secondary-market ETF trading activity as direct evidence of portfolio-level selling.

---

## 7. Stress Score (Realized Market Stress)

**What it measures:** How extreme current realized market conditions are relative to their own recent history. Higher values mean more observed stress, not a forecast of a crash.

The score is bounded from **0 to 100** and combines rolling percentile ranks using these default weights:

* **Realized volatility (20%):** 20-day annualized volatility.
* **Drawdown depth (20%):** Current distance below the asset's running high.
* **VIX (20%):** Yahoo Finance `^VIX` daily close.
* **High-yield credit spread (20%):** FRED series `BAMLH0A0HYM2`, the ICE BofA US High Yield Index Option-Adjusted Spread.
* **Investment-grade credit spread (10%):** FRED series `BAMLC0A0CM`, the ICE BofA US Corporate Index Option-Adjusted Spread.
* **Credit risk premium (10%):** High-yield OAS minus investment-grade OAS.

The model uses up to **504 observations** and requires at least **126 valid observations**. When three or more components are at or above their 80th percentile, the weighted raw score receives a 1.10 confirmation multiplier before conversion to the final score.

**Interpretation:**

| Score | Meaning | Icon |
| :--- | :--- | :--- |
| $<60$ | Normal | 🟢 |
| $60$–$<75$ | Watch | ⚠️ |
| $75$–$<85$ | Elevated | 🟠 |
| $\geq85$ | Critical | 🚨 |

The FRED and VIX histories are persisted under `data/` and refreshed when stale. Set the `FRED_API_KEY` environment variable before the first download. If no key or cached FRED series is available, the scanner marks stress as unavailable rather than substituting a proxy.

Stress and fragility answer different questions:

* **Fragility:** Are the market internals weakening or becoming structurally unstable?
* **Stress:** Are realized volatility, drawdown, volatility expectations, and credit conditions currently extreme?

When both scores are elevated, treat the reading as stronger confirmation of defensive conditions. When they disagree, inspect the component diagnostics rather than averaging them blindly.

---

## 🗒️ Decision Cheat Sheet (Bringing it together)

| Scenario | Hurst | Tail Index | CVD | Judgment |
| :--- | :--- | :--- | :--- | :--- |
| **Trend candidate** | $>0.55$ | $>1.7$ | **UP** | Trend conditions are supportive; confirm risk scores and sizing. |
| **Possible absorption** | $<0.45$ | $<1.55$ | **UP** | Potential buying support, but tail risk remains high. Do not treat as a standalone buy. |
| **Bull-trap warning** | $>0.55$ | Any | **DOWN** | Price persistence conflicts with flow; review fragility and stress. |
| **Flash-crash conditions** | Variable | **<1.55** | **DOWN** | Defensive posture; confirm with VPIN, stress, and fragility before acting. |

---

> Note: These thresholds are model-specific to this Mandelbrot-based engine and should be interpreted as system-guided risk signals, not universal market laws. Review the component diagnostics before making decisions.

## 🚀 2026 Real-World Example: The "Micron Shakeout"

1. **Wednesday 3:55 PM (Pre-Earnings):**
    * Price: $999 | Hurst: 0.64 | **Tail Index: 1.10** | **VPIN: 0.85**
    * **Interpretation:** Total **Regime 5** risk conditions. VPIN indicates elevated estimated volume imbalance and the Tail Index indicates heavier estimated downside tails. **Result:** Use defensive sizing and account for jump risk.
2. **Thursday 10:00 AM (Post-Earnings):**
    * Price: $1,235 | Hurst: 0.51 | **Tail Index: 3.10** | **CVD: DOWN**
    * **Interpretation:** The event move has settled into **Regime 3** (Random Walk). The higher Tail Index is not a safety guarantee, while **CVD Down** is a divergence warning. **Result:** Do not chase; wait for clearer confirmation.
