# 🧠 Mandelbrot Fractal Engine: Indicator Guide

This system uses **Fractal Geometry** and **Information Theory** to identify market regimes. It moves beyond traditional "indicators" by measuring the **statistical structure** of price action.

---

## 1. The Hurst Exponent (Trend Persistence)

**What it measures:** Market "Memory." It tells you if the current move is a self-reinforcing trend or just random noise.

* **Logic:**
  * **$H > 0.55$ (Persistence):** The market has "memory." If it went up last minute, it is likely to go up the next. This is a **Trending Regime**.
  * **$H \approx 0.50$ (Random Walk):** The market is a coin flip. No memory.
  * **$H < 0.45$ (Anti-Persistence):** Mean-reverting. The market "bounces" back and forth like a rubber band.
* **Example:**
  * **MU @ $1,228 (Hurst 0.52):** Even though it's up 16%, the memory is low. It's a "Random Walk" at a high level.
  * **NVDA @ $202 (Hurst 0.61):** High persistence. Institutional buyers are "stacking" orders, creating a clean, memorable trend.

---

## 2. The Tail Index ($\alpha$) (Crash/Jump Risk)

**What it measures:** "Wild Randomness." It calculates the thickness of the "Fat Tails" in the return distribution.

* **Logic:**
  * **$\alpha > 1.7$ (Gaussian):** The "Wall of Safety." The market is behaving "normally." Stop-losses work here.
  * **$\alpha < 1.55$ (Tail Risk):** The "Trapdoor." The probability of a sudden 5–10% "Jump" is high.
* **Example:**
  * **GOOG @ $343 (\alpha = 1.50):** Dangerous. The fractal structure is "unzipping." A "Mandelbrotian Jump" to $330 is statistically probable.
  * **SPY @ 734 (\alpha = 3.12):** Very safe. The broad market is diversified and stable, acting as a "Gaussian Shield."

---

## 3. Cumulative Volume Delta (CVD) Proxy

**What it measures:** Institutional Aggression vs. Retail Chasing.

* **Logic:** Tracks whether the "Aggressor" (the one hitting the Market Order button) is a buyer or a seller.
* **Key Pattern: The Divergence.**
  * **Price UP + CVD DOWN:** **Bearish Distribution.** Big Money is selling their shares into the hands of retail traders who are "chasing" the pop.
  * **Price DOWN + CVD UP:** **Bullish Absorption.** A "Whale" is sitting with a massive Buy Limit order, absorbing all the panic selling. This is a "Bottom" signal.
* **Example:**
  * **MU @ $1,235 (Price Up, CVD Trend Down):** Warning! Institutions are exiting their positions into the 16% rally. The "Smart Money" is leaving the party.

---

## 4. Shannon Entropy (Market Complexity)

**What it measures:** "Noise" vs. "Signal."

* **Logic:**
  * **Low Entropy (< 2.0):** The market is "Ordered." The Hurst trend is clean and reliable.
  * **High Entropy (> 3.0):** The market is "Noisy." The Hurst might look high, but it’s being faked by chaotic, low-liquidity wiggles.
* **Example:**
  * **Lunch Hour Trading:** Usually has high entropy. The "Signal" is weak, and algorithmic "noise" dominates. Don't trust Hurst signals during high entropy periods.

---

## 5. VPIN (Probability of Informed Trading)

**What it measures:** "Toxic Flow." It detects when "sharks" (informed traders) are in the water.

* **Logic:**
  * **VPIN > 0.75:** High Toxicity. Informed traders are aggressively imbalancing the order book. A **Regime 5 Jump** is imminent.
  * **VPIN < 0.40:** Low Toxicity. Trading is balanced and "retail-heavy."
* **Example:**
  * **Pre-Earnings MU:** VPIN often spikes 10 minutes before a news release as "informed" algorithms begin to position for the volatility.

---

## 🗒️ Decision Cheat Sheet (Bringing it together)

| Scenario | Hurst | Tail Index | CVD | Judgment |
| :--- | :--- | :--- | :--- | :--- |
| **The "Perfect" Long** | $>0.55$ | $>1.7$ | **UP** | High-conviction trend. Buy and hold. |
| **The "Value" Bottom** | $<0.45$ | $<1.55$ | **UP** | Institutions absorbing a crash. High-risk, high-reward Buy. |
| **The "Bull Trap"** | $>0.55$ | $>1.7$ | **DOWN** | Price is rising, but Big Money is exiting. Prepare to Sell. |
| **The "Flash Crash"** | Variable | **<1.55** | **DOWN** | **Regime 5.** Exit all positions. The trapdoor is open. |

---

> Note: These thresholds are model-specific to this Mandelbrot-based engine and should be interpreted as system-guided risk signals, not universal market laws.

## 🚀 2026 Real-World Example: The "Micron Shakeout"

1. **Wednesday 3:55 PM (Pre-Earnings):**
    * Price: $999 | Hurst: 0.64 | **Tail Index: 1.10** | **VPIN: 0.85**
    * **Interpretation:** Total **Regime 5** chaos. The VPIN says "sharks are in the water," and the Tail Index says "the floor is gone." **Result:** High probability of a massive jump.
2. **Thursday 10:00 AM (Post-Earnings):**
    * Price: $1,235 | Hurst: 0.51 | **Tail Index: 3.10** | **CVD: DOWN**
    * **Interpretation:** The "Jump" is over. We are in **Regime 3** (Random Walk). The high Tail Index says we are safe from crashing, but the **CVD Down** says Big Money is taking profits. **Result:** Do not chase. Wait for a new trend to form.
