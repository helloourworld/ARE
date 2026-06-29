import sys
import warnings
import numpy as np
from scipy.stats import linregress, norm
from pathlib import Path


try:
    from .liquidity import calculate_liquidity_signals
except ImportError:
    try:
        from risk_modeling.liquidity import calculate_liquidity_signals
    except ImportError:
        package_root = Path(__file__).resolve().parent.parent
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from liquidity import calculate_liquidity_signals
try:
    from ..data_pipeline.data_cache import get_data_persistent
except ImportError:
    try:
        from data_pipeline.data_cache import get_data_persistent
    except ImportError:
        package_root = Path(__file__).resolve().parent.parent
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from data_pipeline.data_cache import get_data_persistent
        
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)




# ============================================================================
# 1. PARAMETERS & THRESHOLDS
# ============================================================================
# Hurst: > 0.55 = Trend Memory, < 0.45 = Mean Reversion
HURST_WINDOW = 300            # 5 hours of 1m data
HURST_TREND_MIN = 0.55
HURST_CHOP_MAX = 0.45

# Tail Index: < 1.55 = Wild Randomness (Jump Risk)
TAIL_BLOCK_SIZE = 200         # Robust window for Hill Estimator
TAIL_PERCENTILE = 5
TAIL_MIN_K = 15
TAIL_RISKY = 1.55             # Regime 5 Trigger
TAIL_SAFE = 1.70              # Required for Regime 1/2 stability

# Volatility: For Active vs. Dormant Risk
VOL_LOOKBACK = 30             # 30-minute window for "Current Vol"
VOL_THRESHOLD = 0.0015        # Threshold to define "Active" movement
FRAGILITY_THRESHOLD = 2.5     # Fragility if negative money flow is large relative to low vol

# ============================================================================
# 2. FRACTAL MATHEMATICS
# ============================================================================


def calculate_hurst_vw(prices: np.ndarray, volumes: np.ndarray, window: int = 300):
    """
    Volume-Weighted Hurst that adapts to Zero-Volume (After-Hours).
    """
    if len(prices) < window:
        return 0.5

    p = prices[-window:]
    v = volumes[-window:]
    returns = np.log(p[1:] / p[:-1])

    # 1. CHECK FOR ZERO VOLUME (After-Hours Fix)
    # If the average volume is near zero, weighting is impossible/useless.
    avg_vol = np.mean(v)

    if avg_vol > 0.0001:
        # We have volume! Use Volume-Weighting logic
        # Add epsilon to v to prevent division by zero in any specific bar
        vol_weights = (v[1:] + 1) / (avg_vol + 1)
        data = returns * vol_weights
    else:
        # NO VOLUME (Pre/Post Market): Use standard log returns
        # This prevents the Hurst from crashing when Yahoo returns 0 volume
        data = returns

    N = len(data)
    lags = np.unique(np.geomspace(10, N//2, num=20).astype(int))

    RS = []
    for lag in lags:
        num_blocks = N // lag
        rs_sub = []
        for i in range(num_blocks):
            block = data[i*lag: (i+1)*lag]
            s = np.std(block)
            if s > 1e-10:
                r = np.max(np.cumsum(block - np.mean(block))) - \
                    np.min(np.cumsum(block - np.mean(block)))
                rs_sub.append(r / s)
        if rs_sub:
            RS.append(np.mean(rs_sub))

    if len(RS) < 5:
        return 0.5
    return linregress(np.log(lags), np.log(RS)).slope


def calculate_tail_index_robust(returns: np.ndarray, lookback: int = 500):
    """
    Asymmetric Hill Estimator (Left-Tail).
    Filters out 'zero-return' noise bars to avoid Gaussian bias.
    """
    # 1. Filter for 'Active' negative returns only
    active_neg = np.abs(returns[(returns < -1e-6)])

    if len(active_neg) < TAIL_MIN_K:
        return 3.0

    subset = active_neg[-lookback:]

    def hill_est(data):
        k = max(int(len(data) * (TAIL_PERCENTILE / 100)), TAIL_MIN_K)
        tails = np.sort(data)[-k:]
        threshold = tails[0]
        if threshold <= 1e-10:
            return 3.0
        return 1.0 / np.mean(np.log(tails / threshold))

    # Rolling window percentile for conservative risk view
    results = [hill_est(subset[i:i+TAIL_BLOCK_SIZE])
               for i in range(0, len(subset)-TAIL_BLOCK_SIZE, 15)]

    return float(np.percentile(results, 15)) if results else 2.5

# ============================================================================
# 3. PERSISTENT DATA CACHE
# ============================================================================

# ============================================================================
# NEW JUDGMENT INDICATORS
# ============================================================================


def calculate_shannon_entropy(returns, bins=10):
    """
    Shannon Entropy: Measures Market Complexity.
    Low Entropy (< 2.0) = Highly ordered, Hurst is reliable.
    High Entropy (> 3.0) = Chaotic Noise, Hurst is likely a 'fake'.
    """
    if len(returns) < 50:
        return 0
    # Create probability distribution of returns
    hist, _ = np.histogram(returns, bins=bins, density=True)
    hist = hist[hist > 0]  # Remove zeros for log calculation
    probs = hist / np.sum(hist)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def calculate_vpin_lite(returns, volumes, window=50):
    """
    Quiet Markets: VPIN will oscillate between 0.15 and 0.40.
    Trending Markets: VPIN will move toward 0.50 - 0.70.
    Toxic/Pre-Crash (The SPY scenario): VPIN will climb above 0.80.
    """
    
    r = np.array(returns[-window:])
    v = np.array(volumes[-window:])

    # 1. Calculate rolling volatility (std dev of returns)
    # We need this to know if a move is "large" or "small"
    sigma = np.std(r)
    if sigma < 1e-9: # Avoid division by zero in flat markets
        return 0.5

    # 2. Bulk Volume Classification (BVC) 
    # This splits volume: e.g., a small move up might be 55% buy, 45% sell.
    # A massive move up might be 99% buy, 1% sell.
    buy_fraction = norm.cdf(r / sigma)
    
    buy_v = v * buy_fraction
    sell_v = v * (1 - buy_fraction)

    # 3. Calculate Imbalance
    total_vol = np.sum(v)
    if total_vol < 1e-9:
        return 0.5

    oi_imbalance = np.abs(buy_v - sell_v)
    vpin = np.sum(oi_imbalance) / total_vol
    
    return vpin


def calculate_cvd_refined(data_df, window=100):
    df = data_df.tail(window).copy()
    if len(df) < window or df['Volume'].sum() < 1e-9:
        return 0.0, 0.0

    # 1. Use BVC for better Delta (aligns with your VPIN)
    returns = df['Close'].pct_change().fillna(0)
    sigma = returns.std() + 1e-10
    
    # Probability of buy volume (0 to 1)
    buy_prob = norm.cdf(returns / sigma)
    delta = (2 * buy_prob - 1) * df['Volume'] # Scales from -Volume to +Volume
    
    cvd = delta.cumsum()

    # 2. Normalize CVD to handle scaling (Z-score or Min-Max)
    # We want to know the "direction" regardless of total volume
    cvd_array = cvd.values
    x = np.arange(len(cvd_array))
    
    # Calculate Slope
    slope, intercept, r_value, p_value, std_err = linregress(x, cvd_array)
    
    # 3. Sensitivity Adjustment: The "CVD Intensity"
    # Instead of raw slope, use the R-squared or Correlation Coefficient
    # This tells you how "solid" the CVD trend is.
    # If r_value is -0.9, the CVD is crashing in a perfect line.
    
    # Normalize slope by the average volume to make it asset-agnostic
    avg_vol = df['Volume'].mean()
    normalized_slope = slope / avg_vol 

    return normalized_slope, r_value

# ============================================================================
# UPDATED SCANNER WITH JUDGMENT LOGIC
# ============================================================================


def compute_judgment(prices, volumes, data_1m, h_val, alpha, cmf=0.0, breadth_slope=0.0):
    """
    Refined Market Regime Judgment Engine
    """
    # 1. Fetch Core Metrics
    returns = np.diff(np.log(prices))
    cvd_slope, cvd_confidence = calculate_cvd_refined(data_1m) # Using the R-Value/Confidence
    entropy = calculate_shannon_entropy(returns[-200:])
    vpin = calculate_vpin_lite(returns, volumes[-len(returns):]) # Using BVC version
    
    # 2. Determine Price Direction (Last 10 bars)
    price_change = (prices[-1] - prices[-10]) / prices[-10]
    is_price_up = price_change > 0.001  # 0.1% threshold to avoid noise
    is_price_down = price_change < -0.001
    
    # 3. Define Thresholds
    CVD_THRESHOLD = 0.05       # Minimum slope to care about
    CONFIDENCE_THRESHOLD = 0.5 # Minimum R-Value to trust CVD
    VPIN_TOXIC = 0.75          # Level where liquidity becomes fragile
    weak_liquidity = (cmf < 0) and (breadth_slope < 0)
    
    # 4. Hierarchical Judgment Logic
    judgment = "NEUTRAL / STABLE"

    # --- REGIME A: THE TOXIC TRAP (Highest Priority) ---
    if vpin > VPIN_TOXIC:
        if is_price_up and cvd_slope < -CVD_THRESHOLD:
            judgment = "TOXIC DISTRIBUTION (Price up, but Smart Money is exiting aggressively)"
        elif is_price_down and cvd_slope > CVD_THRESHOLD:
            judgment = "TOXIC ACCUMULATION (Price down, but Smart Money is absorbing everything)"
        else:
            judgment = "TOXIC FLOW (High probability of a structural Jump soon)"

    # --- REGIME A1: LIQUIDITY FADING ---
    elif weak_liquidity and is_price_up and vpin > 0.6:
        judgment = "WEAK RALLY / LIQUIDITY FADING"
    elif weak_liquidity and is_price_up:
        judgment = "WEAK RALLY (Liquidity is not confirming price)"
    elif weak_liquidity and is_price_down:
        judgment = "LIQUIDITY DRAIN (Selling pressure and weak internals)"

    # --- REGIME B: DIVERGENCE (Medium Priority) ---
    elif abs(cvd_slope) > CVD_THRESHOLD and cvd_confidence > CONFIDENCE_THRESHOLD:
        if is_price_up and cvd_slope < -CVD_THRESHOLD:
            judgment = "BEARISH DIVERGENCE (Weak rally, lack of aggressive buyers)"
        elif is_price_down and cvd_slope > CVD_THRESHOLD:
            judgment = "BULLISH ABSORPTION (Institutional buying support detected)"
        elif is_price_up and cvd_slope > CVD_THRESHOLD:
            judgment = "HEALTHY MOMENTUM (Price and Volume are in sync)"
        elif is_price_down and cvd_slope < -CVD_THRESHOLD:
            judgment = "AGGRESSIVE SELLING (Trend is backed by real volume)"

    # --- REGIME C: TREND QUALITY (Hurst/Entropy) ---
    elif h_val > 0.60:
        if entropy > 2.8: # High Entropy = Chaos
            judgment = "NOISY TREND (Persistence is high but price action is erratic)"
        else:
            judgment = "BULLISH PERSISTENCE (Clean, high-confidence trend)"
            
    # --- REGIME D: EXHAUSTION ---
    elif h_val < 0.45 and entropy < 2.2:
        judgment = "MEAN REVERSION (Trend is dead, price likely to return to average)"

    return judgment, entropy, vpin, cvd_slope


def scan_with_judgment(ticker):
    return scan_market(ticker, show_judgment=True)

# ============================================================================
# 4. MAIN SCANNER
# ============================================================================

def generate_suggestion(regime, judgment, entropy, alpha, h_val, vpin, cvd_slope):
    """
    Synthesizes fractal metrics into actionable institutional-grade advice.
    """
    
    # --- 1. THE KILL SWITCH: TOXIC FLOW & STRUCTURAL JUMP RISK ---
    # Toxicity overrides all other "Bullish" regimes because liquidity is failing.
    if "TOXIC" in judgment or vpin > 0.80:
        if "DISTRIBUTION" in judgment:
            return "🚨 EXIT / AGGRESSIVE SELL", "Structural Jump Risk: Smart money is exiting into a hollow rally. Rug-pull imminent."
        if "ACCUMULATE" in judgment or "ABSORPTION" in judgment:
            return "🛡️ HEDGE / PROTECT", "Toxic flow detected during accumulation. High volatility ahead; use protective puts."
        return "🚫 AVOID / STAY CASH", "Order flow is predatory (VPIN > 0.8). Market makers are withdrawing liquidity."

    # --- 2. TAIL RISK & EXTREME INSTABILITY (Regime 5) ---
    if "5 - TAIL RISK" in regime:
        if "BULLISH ABSORPTION" in judgment:
            return "🔥 SPECULATIVE BUY", "Tail Risk is high but Institutions are catching the knife. High risk/reward."
        return "⚠️ CAUTION", "Non-Normal return distribution. Probability of an extreme 'fat-tail' move is high."

    # --- 3. TREND PERSISTENCE (Regime 1 & 2) ---
    if "1 - BULLISH PERSISTENCE" in regime:
        if "BEARISH DIVERGENCE" in judgment or "DISTRIBUTION" in judgment:
            return "📉 REDUCE / TAKE PROFIT", "Trend is real (Hurst > 0.6) but aggressive sellers are dominating the tape."
        if "HEALTHY MOMENTUM" in judgment:
            return "🚀 STRONG BUY / HOLD", "Price and Order Flow are in sync. High conviction uptrend."
        return "✅ HOLD / BUY DIPS", "Healthy persistent uptrend."

    if "2 - BEARISH PERSISTENCE" in regime:
        if "BULLISH ABSORPTION" in judgment:
            return "⏳ WATCH FOR REVERSAL", "Downtrend is persistent but smart money is building a floor. Look for Hurst to drop."
        if "AGGRESSIVE SELLING" in judgment:
            return "🛑 SELL / SHORT", "Downward momentum is backed by aggressive market orders. No bottom in sight."
        return "🛑 STAY SHORT", "Persistent bearish memory."

    # --- 4. QUALITY OF TREND (Entropy Check) ---
    if "NOISY" in judgment or entropy > 2.8:
        return "🔄 REDUCE POSITION SIZE", "Trend exists but is highly chaotic. Expect frequent stop-outs."

    # --- 5. EQUILIBRIUM & MEAN REVERSION (Regime 3 & 4) ---
    if "4 - UNSTABLE" in regime or "MEAN REVERSION" in judgment:
        return "↕️ SCALP RANGE", "Market is mean-reverting. Sell resistance, buy support. Avoid trend-following."

    if "3 - NEUTRAL" in regime:
        if "BULLISH ABSORPTION" in judgment:
            return "➕ ACCUMULATE", "Price is sideways but aggressive buyers are soaking up supply."
        return "💤 NEUTRAL", "Fractal noise. Wait for Hurst to break above 0.55 or below 0.45."

    return "🔎 MONITOR", "Metrics are inconclusive. Wait for fractal alignment."

def scan_market(ticker, show_judgment=True):
    print(f"--- Scanning {ticker} Mandelbrot Status ---")

    # Load 1m (Trend) and 1d (Structural Risk)
    data_1m = get_data_persistent(ticker, "1m")
    data_1d = get_data_persistent(ticker, "1d")

    if data_1m.empty or data_1d.empty:
        print("ERROR: Insufficient data returned from Yahoo Finance.")
        return "ERROR - No Data"

    required_1m = max(HURST_WINDOW, VOL_LOOKBACK, 60) + 1
    if len(data_1m) < required_1m:
        print(
            f"ERROR: Not enough intraday data ({len(data_1m)} bars, need {required_1m}).")
        return "ERROR - Insufficient 1m data"

    if len(data_1d) < 2:
        print("ERROR: Not enough daily history.")
        return "ERROR - Insufficient 1d data"

    prices = data_1m['Close'].values
    volumes = data_1m['Volume'].values
    returns_1m = np.diff(np.log(prices))

    # 1. Calculate Core Metrics
    h_val = calculate_hurst_vw(prices, volumes, window=HURST_WINDOW)

    # We calculate the 'Worst Case' Alpha between live intraday and structural daily
    alpha_live = calculate_tail_index_robust(returns_1m, lookback=500)
    alpha_daily = calculate_tail_index_robust(
        data_1d['Close'].pct_change().dropna().values, lookback=500)
    alpha_eff = min(alpha_live, alpha_daily)

    # 2. Intraday Volatility (Standard Deviation of log returns over 30 mins)
    recent_vol = np.std(returns_1m[-VOL_LOOKBACK:])

    # Fragility: negative money flow during low volatility
    benchmark_df = get_data_persistent("RSP", "1d")
    if benchmark_df.empty:
        liquidity_signals = {'cmf': 0.0, 'breadth_ratio': 1.0, 'breadth_slope': 0.0, 'rsi': 50.0}
    else:
        liquidity_signals = calculate_liquidity_signals(data_1d, benchmark_df)

    cash_cmf = liquidity_signals['cmf']
    fragility_score = abs(cash_cmf) / max(recent_vol, 1e-12)
    if h_val < 0.53 and cash_cmf < -0.15 and fragility_score > FRAGILITY_THRESHOLD:
        print("ALERT: CRITICAL FRAGILITY - High Risk of Phase Transition (Gap Down)")

    # 3. Directional Check (Slope of last hour)
    slope = linregress(np.arange(60), prices[-60:]).slope

    # 4. REGIME CLASSIFICATION
    prev_close = data_1d['Close'].values[-1]
    gap_pct = abs((prices[-1] - prev_close) / prev_close)
    tail_state = "ACTIVE" if recent_vol > VOL_THRESHOLD else "DORMANT"

    if alpha_eff < TAIL_RISKY:
        regime = f"5 - TAIL RISK ({tail_state} Danger of Sudden Move)"
    elif gap_pct > 0.05 and h_val < HURST_TREND_MIN:
        regime = "1 - BULLISH PERSISTENCE (Post-Gap Consolidation)"
    elif h_val > HURST_TREND_MIN:
        if alpha_eff >= TAIL_SAFE:
            regime = "1 - BULLISH PERSISTENCE (Trend is Real)"
        else:
            regime = "2 - BEARISH PERSISTENCE (Trend is Real)"
    elif h_val < HURST_CHOP_MAX:
        regime = "4 - UNSTABLE (Mean Reverting / Chop)"
    else:
        regime = "3 - NEUTRAL / RANDOM WALK"

    # 5. OUTPUT
    print(f"Price:         {prices[-1]:.2f}")
    print(f"Hurst (Trend): {h_val:.3f}")
    print(f"Tail Index:    {alpha_eff:.3f}")
    print(f"Intraday Vol:  {recent_vol:.5f}")
    print(f"RESULT REGIME: {regime}")

    # Compute liquidity signals for better tape/flow context.
    result = {
        "Price": float(prices[-1]),
        "Hurst": float(h_val),
        "Tail Index": float(alpha_eff),
        "Intraday Vol": float(recent_vol),
        "Regime": regime,
        "Fragility Score": float(fragility_score),
        "Fragility Alert": "",
        "Verdict": "N/A",
        "Suggestion": "N/A",
        "Reason": "N/A",
        "CVD Trend": "N/A"
    }

    if show_judgment:
        judgment, entropy, vpin, cvd_slope = compute_judgment(
            prices, volumes, data_1m, h_val, alpha_eff,
            cmf=liquidity_signals['cmf'],
            breadth_slope=liquidity_signals['breadth_slope']
        )
        cvd_arrow = "⬆️" if cvd_slope > 0 else "⬇️" if cvd_slope < 0 else "→"
        cvd_label = "UP" if cvd_slope > 0 else "DOWN" if cvd_slope < 0 else "FLAT"
        print(f"Entropy: {entropy:.2f} (Low is better) | VPIN: {vpin:.2f}")
        print(f"Liquidity CMF: {liquidity_signals['cmf']:.3f} | Breadth Slope: {liquidity_signals['breadth_slope']:.5f}")
        print(f"CVD Trend: {cvd_slope: .2f} {cvd_arrow} {cvd_label}")
        print(f"VERDICT: {judgment}")

        # 6. ADD SUGGESTION
        action, reason = generate_suggestion(
            regime, judgment, entropy, alpha_eff, h_val, vpin, cvd_slope)
        print(f"\nSUGGESTION: {action}")
        print(f"REASON:     {reason}")

        result.update({
            "Verdict": judgment,
            "Suggestion": action,
            "Reason": reason,
            "CVD Trend": f"{cvd_slope:.2f} {cvd_label}"
        })

    if h_val < 0.53 and cash_cmf < -0.15 and fragility_score > FRAGILITY_THRESHOLD:
        result["Fragility Alert"] = "CRITICAL FRAGILITY"

    print("-" * 45)
    return result


if __name__ == "__main__":
    # Test on your Core Universe
    import time
    tickers = ["MU", 'SPY']
    while True:
        for t in tickers:
            scan_market(t)
            # scan_with_judgment(t)
        time.sleep(60)  # Wait 1 minute before next scan
"""
Summary Table for Tail-Index Risk Regimes

Tail Index (α)      Risk Category            Market Behavior
---------------------------------------------------------------
> 3.0               Gaussian / Safe         Movements resemble a normal distribution; standard models are reliable.
2.0 - 3.0           Heavy Tails             Large moves occur more often; risk is elevated but still manageable.
1.55 - 2.0          Unstable Zone           Tail behavior dominates; standard risk metrics begin to fail.
< 1.55              Tail Risk (Regime 5)    High probability of jumps and gaps; use protective sizing.
< 1.0               Structural Collapse     Extreme instability; theoretical mean may not exist.
"""
