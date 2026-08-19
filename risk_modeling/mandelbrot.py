import sys
import warnings
import logging
import numpy as np
import pandas as pd
from scipy.stats import linregress, norm
from pathlib import Path
from datetime import time


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
    from .bolling_bands import get_hybrid_risk_signal
except ImportError:
    try:
        from risk_modeling.bolling_bands import get_hybrid_risk_signal
    except ImportError:
        package_root = Path(__file__).resolve().parent.parent
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from bolling_bands import get_hybrid_risk_signal
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

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_SCAN_LOG_PATH = LOG_DIR / "mandelbrot_scan.log"
FRAGILITY_CALIBRATION_PATH = DATA_DIR / "fragility_calibration.csv"
_LAST_SCAN_BAR_TS = {}
_LAST_SCAN_RESULT = {}


def configure_scan_file_logging(log_path: Path = DEFAULT_SCAN_LOG_PATH) -> Path:
    """Attach a file handler for mandelbrot scan logs if not already present."""
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            existing = Path(getattr(handler, "baseFilename", ""))
            if existing == target:
                return target

    file_handler = logging.FileHandler(target, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(file_handler)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return target




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
FRAGILITY_THRESHOLD = 35.0    # Tuned trigger: avoids low-signal criticals while preserving severe cases
FRAGILITY_CMF_MIN = 0.12      # Stronger downside flow gate to reduce fragility false positives
FRAGILITY_BREADTH_MIN = -0.002  # Require clearer negative breadth confirmation
FRAGILITY_VOL_FLOOR_RATIO = 0.35  # Adaptive floor vs VOL_THRESHOLD to avoid divide-by-micro-vol noise
FRAGILITY_UNIVERSE_Q = 0.80   # Cross-sectional score quantile for dynamic critical trigger
FRAGILITY_THRESHOLD_MIN = 25.0
FRAGILITY_THRESHOLD_MAX = 60.0
FRAGILITY_CAL_MIN_SAMPLES = 8

# Others
GAP_THRESHOLD = 0.05  # e.g., 0.05 = 5% or 0.005 = 0.5% depending on scale
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

    valid_lags = []
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
            valid_lags.append(lag)
            RS.append(np.mean(rs_sub))

    if len(RS) < 5 or len(valid_lags) != len(RS):
        return 0.5
    return linregress(np.log(valid_lags), np.log(RS)).slope


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
    Toxic/Pre-Crash: VPIN will climb above 0.80.
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
    
    # Normalize slope by the average volume to make it asset-agnostic
    avg_vol = df['Volume'].mean()
    
    delta = (2 * buy_prob - 1) * df['Volume'] / avg_vol if avg_vol else 0  # Scales from -Volume to +Volume
    
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
    

    normalized_slope = slope

    return normalized_slope, r_value


def calibrate_cvd_threshold(data_df, lookback=240, base_threshold=0.03):
    """Calibrate CVD slope threshold from recent flow intensity.

    The threshold adapts by symbol/session so divergence checks remain meaningful
    across different volume scales and tape regimes.
    """
    df = data_df.tail(lookback).copy()
    if len(df) < 120 or "Close" not in df.columns or "Volume" not in df.columns:
        return float(base_threshold)

    returns = df["Close"].pct_change().fillna(0.0)
    sigma = returns.rolling(20, min_periods=20).std().replace(0.0, np.nan)
    z = (returns / sigma).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    buy_prob = norm.cdf(z)
    delta = (2.0 * buy_prob - 1.0) * df["Volume"].astype(float)

    avg_vol = max(float(df["Volume"].astype(float).mean()), 1e-9)
    delta_norm = (delta / avg_vol).abs()

    q75 = float(delta_norm.quantile(0.75)) if len(delta_norm.dropna()) else 0.0
    adaptive = max(base_threshold, q75 * 0.25)
    return float(np.clip(adaptive, 0.02, 0.25))


def _get_universe_symbols_from_cache(max_symbols=80):
    """Infer active symbol universe from local 1m cache files."""
    symbols = []
    for path in sorted(DATA_DIR.glob("cache_*_1m.csv")):
        stem = path.stem
        if not stem.startswith("cache_") or not stem.endswith("_1m"):
            continue
        symbol = stem[len("cache_"):-len("_1m")]
        if symbol:
            symbols.append(symbol)
    return symbols[:max_symbols]


def _load_fragility_calibration(session_date):
    """Load a same-day universe calibration if present and valid."""
    if not FRAGILITY_CALIBRATION_PATH.exists():
        return None
    try:
        cal = pd.read_csv(FRAGILITY_CALIBRATION_PATH)
        if cal.empty or "date" not in cal.columns or "threshold" not in cal.columns:
            return None
        same_day = cal.loc[cal["date"].astype(str) == str(session_date)]
        if same_day.empty:
            return None
        sample_size = int(same_day.get("sample_size", pd.Series([0])).iloc[-1])
        if sample_size < FRAGILITY_CAL_MIN_SAMPLES:
            return None
        threshold = float(same_day["threshold"].iloc[-1])
        return float(np.clip(threshold, FRAGILITY_THRESHOLD_MIN, FRAGILITY_THRESHOLD_MAX))
    except Exception:
        return None


def _save_fragility_calibration(session_date, threshold, sample_size):
    """Persist daily calibration for reuse across scans."""
    row = pd.DataFrame([
        {
            "date": str(session_date),
            "threshold": float(threshold),
            "sample_size": int(sample_size),
        }
    ])
    try:
        if FRAGILITY_CALIBRATION_PATH.exists():
            old = pd.read_csv(FRAGILITY_CALIBRATION_PATH)
            old = old.loc[old.get("date", pd.Series(dtype=str)).astype(str) != str(session_date)]
            out = pd.concat([old, row], ignore_index=True)
        else:
            out = row
        out.to_csv(FRAGILITY_CALIBRATION_PATH, index=False)
    except Exception as exc:
        logger.debug("Failed to persist fragility calibration | error=%s", exc)


def calibrate_fragility_threshold_for_session(session_date, benchmark_df=None, base_threshold=FRAGILITY_THRESHOLD):
    """Calibrate fragility threshold from same-day cross-sectional universe scores.

    The goal is to keep CRITICAL alerts sparse and meaningful as market-wide
    volatility/liquidity regimes drift.
    """
    cached = _load_fragility_calibration(session_date)
    if cached is not None:
        return float(cached)

    symbols = _get_universe_symbols_from_cache(max_symbols=80)
    if not symbols:
        return float(base_threshold)

    if benchmark_df is None or benchmark_df.empty:
        benchmark_df = get_data_persistent("RSP", "1d")

    scores = []
    for symbol in symbols:
        try:
            data_1m = get_data_persistent(symbol, "1m")
            data_1d = get_data_persistent(symbol, "1d")
            if data_1m is None or data_1d is None or data_1m.empty or data_1d.empty:
                continue
            if len(data_1m) < (HURST_WINDOW + 1):
                continue

            prices = data_1m["Close"].values
            volumes = data_1m["Volume"].values
            returns_1m = np.diff(np.log(prices))
            if len(returns_1m) < VOL_LOOKBACK:
                continue

            h_val = calculate_hurst_vw(prices, volumes, window=HURST_WINDOW)
            recent_vol = float(np.std(returns_1m[-VOL_LOOKBACK:]))

            if benchmark_df is None or benchmark_df.empty:
                liq = {"cmf": 0.0, "breadth_slope": 0.0}
            else:
                liq = calculate_liquidity_signals(data_1d, benchmark_df)

            fragility = calculate_fragility_score(
                recent_vol=recent_vol,
                returns_1m=returns_1m,
                cmf=liq.get("cmf", 0.0),
                breadth_slope=liq.get("breadth_slope", 0.0),
                h_val=h_val,
                threshold=base_threshold,
            )

            # Use full cross-section (including low/zero scores) so the threshold
            # tracks universe-wide tape conditions instead of only stressed names.
            scores.append(float(fragility["score"]))
        except Exception:
            continue

    if len(scores) < FRAGILITY_CAL_MIN_SAMPLES:
        threshold = float(base_threshold)
    else:
        q_score = float(np.quantile(scores, FRAGILITY_UNIVERSE_Q))
        threshold = float(np.clip(q_score, FRAGILITY_THRESHOLD_MIN, FRAGILITY_THRESHOLD_MAX))

    _save_fragility_calibration(session_date, threshold, len(scores))
    return threshold


def calculate_fragility_score(recent_vol, returns_1m, cmf, breadth_slope, h_val, threshold=FRAGILITY_THRESHOLD):
    """Compute a robust fragility score with adaptive vol floor and flow/breadth confirmation.

    Why this is more stable:
    - Uses a percentile-based volatility floor to prevent low-volatility explosions.
    - Requires downside money flow to be present before scoring.
    - Penalizes negative breadth and weak memory regimes.
    """
    downside_cmf = max(-float(cmf), 0.0)

    ret = np.asarray(returns_1m[-180:], dtype=float)
    if ret.size > 0:
        abs_ret = np.abs(ret)
        vol_floor = float(np.percentile(abs_ret, 60))
    else:
        vol_floor = VOL_THRESHOLD

    adaptive_floor = max(VOL_THRESHOLD * FRAGILITY_VOL_FLOOR_RATIO, vol_floor, 1e-6)
    effective_vol = max(float(recent_vol), adaptive_floor)

    flow_component = downside_cmf / effective_vol
    breadth_multiplier = 1.0 + min(max(-float(breadth_slope), 0.0) * 40.0, 1.0)
    memory_multiplier = 1.15 if float(h_val) < 0.53 else 1.0
    score = float(flow_component * breadth_multiplier * memory_multiplier)

    is_critical = (
        downside_cmf > FRAGILITY_CMF_MIN
        and float(breadth_slope) < FRAGILITY_BREADTH_MIN
        and float(h_val) < 0.55
        and score > float(threshold)
    )
    is_watch = (
        not is_critical
        and downside_cmf > (FRAGILITY_CMF_MIN * 0.75)
        and float(h_val) < 0.56
        and score > (float(threshold) * 0.75)
    )

    return {
        "score": score,
        "effective_vol": float(effective_vol),
        "downside_cmf": float(downside_cmf),
        "is_critical": bool(is_critical),
        "is_watch": bool(is_watch),
        "threshold": float(threshold),
    }

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
    CVD_THRESHOLD = calibrate_cvd_threshold(data_1m, base_threshold=0.03)
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
            judgment = "PERSISTENCE Trend(Low Entropy) - Price likely to continue in the same direction"
            
    # --- REGIME D: EXHAUSTION ---
    elif h_val < 0.45 and entropy < 2.2:
        judgment = "MEAN REVERSION (Trend is dead, price likely to return to average)"

    return judgment, entropy, vpin, cvd_slope, CVD_THRESHOLD


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
        if "ACCUMULATE" in judgment or "ACCUMULATION" in judgment or "ABSORPTION" in judgment:
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


def _get_session_open(data_1m, data_1d, session_date):
    """Prefer the official daily open, with a regular-session intraday fallback."""
    daily_index = pd.DatetimeIndex(pd.to_datetime(data_1d.index, errors="coerce"))
    daily_rows = data_1d.loc[daily_index.date == session_date]
    if not daily_rows.empty and "Open" in daily_rows.columns:
        official_open = daily_rows["Open"].dropna()
        if not official_open.empty:
            return float(official_open.iloc[-1])

    intraday_index = pd.DatetimeIndex(pd.to_datetime(data_1m.index, errors="coerce"))
    if intraday_index.tz is not None:
        intraday_index = intraday_index.tz_convert("America/New_York")
    regular_session = (intraday_index.date == session_date) & (intraday_index.time >= time(9, 30))
    regular_rows = data_1m.iloc[np.flatnonzero(regular_session)]
    if regular_rows.empty:
        return None
    if "Open" in regular_rows.columns:
        return float(regular_rows["Open"].iloc[0])
    return float(regular_rows["Close"].iloc[0])


def _resolve_day_baseline(prev_close, session_open, market_ts_et):
    """Use the prior close before the market opens and the session open once live."""
    if market_ts_et is None:
        return float(prev_close)
    if market_ts_et.time() < time(9, 30):
        return float(prev_close)
    if session_open is not None:
        return float(session_open)
    return float(prev_close)


def scan_market(ticker, show_judgment=True, data_1m=None, data_1d=None, check_new_data=True):
    """
    Scan one ticker and return regime metrics + actionable fields.

    Notes on intraday baseline handling:
    - `data_1m` may span multiple days (for example 7d windows).
    - Session open is taken from the first bar of the current ET trading day,
        not from the first bar in the full 1m window.
    - Day % uses previous close in premarket, and switches to current day open
        after regular session starts (09:30 ET).
    """
    logger.info("Scanning Mandelbrot status | ticker=%s", ticker)
    logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)
    # Load 1m (trend/tape) and 1d (structural) views if not provided.
    if data_1m is None:
            data_1m = get_data_persistent(ticker, "1m")
    if data_1d is None:
            data_1d = get_data_persistent(ticker, "1d")

    if data_1m is None or data_1d is None or data_1m.empty or data_1d.empty:
        logger.error("Insufficient data returned from data source")
        return "ERROR - No Data"

    ticker_key = str(ticker).upper()
    market_ts = data_1m.index[-1] if len(data_1m.index) > 0 else "N/A"
    market_ts_pd = pd.Timestamp(market_ts)
    market_ts_key = market_ts_pd.tz_convert("UTC") if market_ts_pd.tzinfo is not None else market_ts_pd

    if check_new_data:
        last_seen_ts = _LAST_SCAN_BAR_TS.get(ticker_key)
        if last_seen_ts is not None and market_ts_key <= last_seen_ts:
            logger.info("Waiting for new 1m data | ticker=%s | latest_bar=%s", ticker, market_ts_pd)
            stale_result = dict(_LAST_SCAN_RESULT.get(ticker_key, {}))
            stale_result.update({
                "Scan Status": "WAIT",
                "Wait Reason": "No new 1m bar",
                "Bar Time": market_ts_pd.strftime("%H:%M:%S"),
            })
            return stale_result if stale_result else "WAIT - No new 1m data"

    _LAST_SCAN_BAR_TS[ticker_key] = market_ts_key

    required_1m = max(HURST_WINDOW, VOL_LOOKBACK, 60) + 1
    if len(data_1m) < required_1m and not ticker.endswith(".TO"):
        logger.error(f"{ticker} Not enough intraday data | bars={len(data_1m)} | required={required_1m}")
        return "ERROR - Insufficient 1m data"

    if len(data_1d) < 2:
        logger.error("Not enough daily history")
        return "ERROR - Insufficient 1d data"

    hybrid_signal_result = {
        "Hybrid Signal": "N/A",
        "Hybrid Reason": "Unable to compute hybrid signal.",
        "Hybrid VPIN": 0.0,
        "Hybrid CVD Trend": "N/A",
        "Hybrid Upper Band": None,
        "Hybrid Lower Band": None,
    }
    try:
        hybrid = get_hybrid_risk_signal(data_1m)
        hybrid_signal_result.update({
            "Hybrid Signal": hybrid.get("signal", "N/A"),
            "Hybrid Reason": hybrid.get("signal_reason", "N/A"),
            "Hybrid VPIN": float(hybrid.get("vpin", 0.0)),
            "Hybrid CVD Trend": hybrid.get("cvd_slope", "N/A"),
            "Hybrid Upper Band": float(hybrid.get("upper_band", np.nan)),
            "Hybrid Lower Band": float(hybrid.get("lower_band", np.nan)),
        })
    except Exception as exc:
        logger.warning(f"⚠️ {ticker} Hybrid signal computation failed | error=%s", exc)

    prices = data_1m['Close'].values
    volumes = data_1m['Volume'].values
    returns_1m = np.diff(np.log(prices))
    # Normalize latest timestamp to ET for session-state consistency.
    if market_ts_pd.tzinfo is not None:
        market_ts_et = market_ts_pd.tz_convert("America/New_York")
    else:
        market_ts_et = market_ts_pd
    bar_time_hms = market_ts_et.strftime("%H:%M:%S")
    current_session_date = market_ts_et.date()

    # 1. Calculate Core Metrics
    h_val = calculate_hurst_vw(prices, volumes, window=HURST_WINDOW)

    # We calculate the 'Worst Case' Alpha between live intraday and structural daily
    alpha_live = calculate_tail_index_robust(returns_1m, lookback=500)
    daily_prices = data_1d['Close'].astype(float).values
    daily_log_returns = np.diff(np.log(daily_prices))
    alpha_daily = calculate_tail_index_robust(daily_log_returns, lookback=500)
    alpha_eff = min(alpha_live, alpha_daily)

    # 2. Intraday Volatility (Standard Deviation of log returns over 30 mins)
    recent_vol = np.std(returns_1m[-VOL_LOOKBACK:])

    # **Fragility: negative money flow during low volatility**
    benchmark_df = get_data_persistent("RSP", "1d")
    if benchmark_df.empty:
        liquidity_signals = {'cmf': 0.0, 'breadth_ratio': 1.0, 'breadth_slope': 0.0, 'rsi': 50.0}
    else:
        liquidity_signals = calculate_liquidity_signals(data_1d, benchmark_df)

    fragility_threshold = calibrate_fragility_threshold_for_session(
        session_date=current_session_date,
        benchmark_df=benchmark_df,
        base_threshold=FRAGILITY_THRESHOLD,
    )

    cash_cmf = liquidity_signals['cmf']
    fragility = calculate_fragility_score(
        recent_vol=recent_vol,
        returns_1m=returns_1m,
        cmf=cash_cmf,
        breadth_slope=liquidity_signals['breadth_slope'],
        h_val=h_val,
        threshold=fragility_threshold,
    )
    fragility_score = fragility["score"]
    if fragility["is_critical"]:
        logger.warning(
            "⚠️ %s Critical fragility detected | score=%.2f | thr=%.2f | cmf=%.3f | eff_vol=%.6f",
            ticker,
            fragility_score,
            fragility_threshold,
            fragility["downside_cmf"],
            fragility["effective_vol"],
        )
    elif fragility["is_watch"]:
        logger.info(
            "⚠️ %s Fragility watch | score=%.2f | thr=%.2f | cmf=%.3f | eff_vol=%.6f",
            ticker,
            fragility_score,
            fragility_threshold,
            fragility["downside_cmf"],
            fragility["effective_vol"],
        )

    # 3. Directional Check (Slope of last hour)
    slope = linregress(np.arange(60), prices[-60:]).slope

    # 4. REGIME CLASSIFICATION + SESSION BASELINES
    daily_index = pd.to_datetime(data_1d.index, errors="coerce")
    latest_daily_date = daily_index[-1].date() if len(daily_index) else None

    # Determine the previous close for premarket vs. regular session calculations.
    if latest_daily_date == current_session_date:
        prev_close = float(data_1d['Close'].iloc[-2])
    else:
        prev_close = float(data_1d['Close'].iloc[-1])
    current_price = float(prices[-1])

    # data_1m can include multiple days; isolate bars from the current ET date.
    idx_1m = pd.DatetimeIndex(pd.to_datetime(data_1m.index, errors="coerce"))
    if idx_1m.tz is not None:
        idx_1m_et = idx_1m.tz_convert("America/New_York")
    else:
        idx_1m_et = idx_1m

    same_day_mask = idx_1m_et.date == market_ts_et.date()
    same_day_pos = np.flatnonzero(same_day_mask)

    if same_day_pos.size > 0:
        day_slice = data_1m.iloc[same_day_pos]
    else:
        logger.warning(f"⚠️ {ticker} Unable to isolate current session bars; falling back to first bar in window")
        day_slice = data_1m

    session_open = _get_session_open(data_1m, data_1d, current_session_date)
    if session_open is None:
        if 'Open' in day_slice.columns and not day_slice['Open'].empty:
            session_open = float(day_slice['Open'].iloc[0])
        else:
            session_open = float(day_slice['Close'].iloc[0])

    # Baseline switch:
    # - Premarket uses previous close.
    # - Regular session uses today's session open.
    day_baseline = _resolve_day_baseline(prev_close, session_open, market_ts_et)

    gap_pct = (session_open - prev_close) / prev_close if session_open is not None else 0.0
    # print("prev_close:", prev_close, "session_open:", session_open, "current_price:", current_price, "gap_pct:", gap_pct
    #       )

    # ---------------------------------------------------------------------
    # 1. Price Action Diagnostics
    # ---------------------------------------------------------------------
    # Performance relative to the active intraday baseline.
    day_pct = ((current_price - day_baseline) / day_baseline) * 100.0

    # Overall performance relative to previous close (Net market state)
    net_pct = ((current_price - prev_close) / prev_close) * 100.0

    # Post-gap intraday behavior flags
    is_gap_up = gap_pct > GAP_THRESHOLD
    is_gap_down = gap_pct < -GAP_THRESHOLD

    holding_gap_up = is_gap_up and day_pct >= 0      # Price holding or expanding above Open
    fading_gap_up  = is_gap_up and day_pct < 0       # Price failing & filling gap back to Prev Close

    holding_gap_dn = is_gap_down and day_pct <= 0    # Price holding or expanding below Open
    bouncing_gap_dn = is_gap_down and day_pct > 0    # Price recovering back towards Prev Close

    tail_state = "ACTIVE" if recent_vol > VOL_THRESHOLD else "DORMANT"

    # Optional: Add VWAP alignment if available (e.g., current_price > vwap)
    # vwap_bullish = current_price > vwap if vwap else True

    # ---------------------------------------------------------------------
    # 2. Regime Classification
    # ---------------------------------------------------------------------

    # A. Tail Risk Override (Highest Priority)
    if alpha_eff < TAIL_RISKY:
        regime = f"5 - TAIL RISK ({tail_state} Danger of Sudden Move)"
        tail_quality = "Tail-Risk"

    # B. Strong Trend Regimes (Hurst indicates strong persistence)
    elif h_val > HURST_TREND_MIN:
        # Intraday AND net direction must agree for true trend persistence
        is_bullish = net_pct > 0 and day_pct >= -0.2  # Net positive without severe intraday collapse
        
        tail_quality = "Tail-Stable" if alpha_eff >= TAIL_SAFE else "Tail-Caution"
        
        if is_bullish:
            regime = f"1 - BULLISH PERSISTENCE (Trend is Real | {tail_quality})"
        else:
            regime = f"2 - BEARISH PERSISTENCE (Trend is Real | {tail_quality})"

    # C. Post-Gap Regimes (Hurst is low/neutral, so intraday hold/fade dictates outcome)
    elif holding_gap_up:
        regime = "1 - BULLISH PERSISTENCE (Post-Gap Acceptance & Hold)"
        tail_quality = "N/A"

    elif fading_gap_up:
        regime = "4 - UNSTABLE (Post-Gap Fade / Supply Overhead)"
        tail_quality = "N/A"

    elif holding_gap_dn:
        regime = "2 - BEARISH PERSISTENCE (Post-Gap Breakdown & Hold)"
        tail_quality = "N/A"

    elif bouncing_gap_dn:
        regime = "4 - UNSTABLE (Post-Gap Short Cover / Counter-Bounce)"
        tail_quality = "N/A"

    # D. Low Hurst (Mean Reverting / Range Bound)
    elif h_val < HURST_CHOP_MAX:
        regime = "4 - UNSTABLE (Mean Reverting / Chop)"
        tail_quality = "N/A"

    # E. Random Walk
    else:
        regime = "3 - NEUTRAL / RANDOM WALK"
        tail_quality = "N/A"

    # 5. OUTPUT (Organized in logical sequence)
    logger.info("StockTs=%s | Price=%.2f | Hurst=%.3f | TailIndex=%.3f | IntradayVol=%.5f | Regime=%s", market_ts, prices[-1], h_val, alpha_eff, recent_vol, regime)

    # Build result in logical sequence: Metadata → Price → Metrics → Regime → Risk → Flow → Action
    result = {
        # --- METADATA & TIME ---
        "Bar Time": bar_time_hms,
        "Scan Status": "RUN",
        
        # --- PRICE ACTION ---
        "Price": float(prices[-1]),
        "Open": float(session_open) if latest_daily_date == current_session_date else float(prev_close),
        "Day %": float(day_pct),
        "Day % Net": float(net_pct),
        
        # --- FRACTAL METRICS (Quality of Price Movement) ---
        "Hurst": float(h_val),
        "Tail Index": float(alpha_eff),
        "Intraday Vol": float(recent_vol),
        
        # --- REGIME CLASSIFICATION ---
        "Regime": regime,
        "Tail Quality": tail_quality,
        
        # --- RISK ASSESSMENT (Fragility / Stability) ---
        "Fragility Score": float(fragility_score),
        "Fragility Threshold": float(fragility_threshold),
        "Fragility Alert": "",
        
        # --- FLOW & LIQUIDITY ANALYSIS ---
        "VPIN": float(0.0),
        "CVD Trend": "N/A",
        "CVD Threshold": float(np.nan),
        
        # --- VERDICT & ACTION (What to Do) ---
        "Verdict": "N/A",
        "Suggestion": "N/A",
        "Reason": "N/A",
    }
    # Merge hybrid signal results into flow section
    result.update(hybrid_signal_result)

    if show_judgment:
        judgment, entropy, vpin, cvd_slope, cvd_threshold = compute_judgment(
            prices, volumes, data_1m, h_val, alpha_eff,
            cmf=liquidity_signals['cmf'],
            breadth_slope=liquidity_signals['breadth_slope']
        )
        cvd_arrow = "⬆️" if cvd_slope > 0 else "⬇️" if cvd_slope < 0 else "→"
        cvd_label = "UP" if cvd_slope > 0 else "DOWN" if cvd_slope < 0 else "FLAT"
        logger.info("StockTs=%s | Entropy=%.2f | VPIN=%.2f | LiquidityCMF=%.3f | BreadthSlope=%.5f | CVDThr=%.3f | CVDTrend=%.2f %s %s | Verdict=%s", market_ts, entropy, vpin, liquidity_signals['cmf'], liquidity_signals['breadth_slope'], cvd_threshold, cvd_slope, cvd_arrow, cvd_label, judgment)

        # 6. ADD SUGGESTION
        action, reason = generate_suggestion(
            regime, judgment, entropy, alpha_eff, h_val, vpin, cvd_slope)
        logger.info("StockTs=%s | Suggestion=%s | Reason=%s", market_ts, action, reason)

        # Update flow and action sections with judgment-derived data
        result.update({
            # Flow section updates
            "VPIN": float(vpin),
            "CVD Trend": f"{cvd_slope:.2f}",
            "CVD Threshold": float(cvd_threshold),
            # Action section updates
            "Verdict": judgment,
            "Suggestion": action,
            "Reason": reason,
        })

    if fragility["is_critical"]:
        result["Fragility Alert"] = "CRITICAL FRAGILITY"
    elif fragility["is_watch"]:
        result["Fragility Alert"] = "FRAGILITY WATCH"

    _LAST_SCAN_RESULT[ticker_key] = dict(result)

    logger.debug("Scan complete | ticker=%s", ticker)
    return result


if __name__ == "__main__":
    # Test on your Core Universe
    log_path = configure_scan_file_logging()
    logger.info("Mandelbrot scan logging enabled | file=%s", log_path)
    tickers = ["MSFT"]
    while True:
        for t in tickers:
            scan_market(t)     
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
