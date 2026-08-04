import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
from pathlib import Path
import sys


def _find_repo_root(start_path: Path) -> Path:
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "enable_repo_root.py").exists():
            return candidate
    return start_path


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enable_repo_root import ensure_repo_root

REPO_ROOT = ensure_repo_root(REPO_ROOT)

try:
    from data_pipeline.data_cache import get_data_persistent, normalize_timestamp_for_index
except ImportError:
    from data_pipeline import get_data_persistent

# Suppress noisy Future/Deprecation warnings during test runs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Your custom imports (ensure these files are in your directory)
from utils import compute_features, classify_regime, generate_signal, compute_position_size
from ml_filter import MLTradeFilter, TradeDataCollector

# Setup logging to replace self.Debug
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class MandelbrotSwingStandalone:
    def __init__(self):
        # --- Configuration (from initialize) ---
        self.tickers = ["GOOGL", "TSLA", "NVDA",
                        "AAPL", "AMZN", "META", "MSFT"]
        self.cash_parking_ticker = "QQQ"
        self.initial_cash = 1_000_000
        self.current_cash = self.initial_cash

        self.lookback = 260
        self.min_data_for_trading = 52
        self.min_data_for_tail = 100

        self.max_positions = 5
        self.position_size_pct = 0.2
        self.holding_period = 20
        self.option_stop_loss = 0.4
        self.option_take_profit = 5.0
        self.min_signal_strength = 0.3

        self.signal_cfg = {
            'rsi_oversold': 40, 'rsi_overbought': 60,
            'rsi_mr_oversold': 35, 'rsi_mr_overbought': 65,
        }

        # --- State Management ---
        self.price_history = {t: pd.DataFrame()
                              for t in self.tickers + [self.cash_parking_ticker]}
        self.active_positions = {}  # ticker: {entry_price, entry_date, signal, contract_info}
        self.trade_log = []
        self.iv_cache = {}

        # --- ML Components ---
        self.use_ml = False  # Set to True if you have your ML environment ready
        self.ml_filter = MLTradeFilter(
            retrain_interval=20, min_samples=30, threshold=0.45)
        self.trade_collector = TradeDataCollector(max_samples=2000)
        self.ml_stats = {'blocked': 0, 'passed': 0}

    def update_data(self, ticker, bar_data):
        """
        Mimics on_data: Call this daily with new OHLCV data.
        bar_data should be a dict: {'open', 'high', 'low', 'close', 'volume', 'date'}
        """
        # Append data to local storage and maintain lookback
        new_row = pd.DataFrame([bar_data]).set_index('date')
        self.price_history[ticker] = pd.concat(
            [self.price_history[ticker], new_row]).tail(self.lookback + 10)

    def _find_option_contract_sim(self, ticker, direction, current_price, date):
        """
        Placeholder for Option Selection Logic.
        In a real environment, you'd query an Options API (Alpaca/Polygon/ThetaData).
        """
        # In standalone, we simulate the 'contract' as a tracking object
        return {
            'symbol': f"{ticker}_OPT_{date.strftime('%Y%m%d')}",
            'strike': current_price * (1.02 if direction > 0 else 0.98),
            'expiry': date + timedelta(days=30),
            'type': 'call' if direction > 0 else 'put'
        }

    def run_daily_logic(self, current_date):
        """Replacement for _daily_analysis and the scheduled event"""
        signals = {}

        # 1. Feature Engineering & Regime Classification
        for ticker in self.tickers:
            # Convert internal storage to the format expected by your utils.py
            # Note: utils.py needs to be compatible with standard Dict/DataFrame
            features = compute_features(
                ticker,
                self.price_history,  # Ensure this matches your utils.py signature
                self.min_data_for_trading,
                self.min_data_for_tail,
                self.iv_cache
            )

            if features is None:
                continue

            regime, danger = classify_regime(features)
            signals[ticker] = generate_signal(
                features, regime, danger, self.signal_cfg)

        # 2. Check Exits
        self._check_exits(signals, current_date)

        # 3. Check Entries
        if len(self.active_positions) < self.max_positions:
            self._check_entries(signals, current_date)

    def _check_entries(self, signals, current_date):
        candidates = [
            (t, s) for t, s in signals.items()
            if s['direction'] != 0 and t not in self.active_positions and s['strength'] >= self.min_signal_strength
        ]
        candidates.sort(key=lambda x: x[1]['strength'], reverse=True)

        for ticker, signal in candidates[:2]:
            if len(self.active_positions) >= self.max_positions:
                break

            # ML Filtering Logic
            if self.use_ml and self.ml_filter.is_trained:
                if not self.ml_filter.should_take_trade(signal):
                    self.ml_stats['blocked'] += 1
                    continue
                self.ml_stats['passed'] += 1

            # Execute Entry (Simulated)
            current_price = self.price_history[ticker]['close'].iloc[-1]
            contract = self._find_option_contract_sim(
                ticker, signal['direction'], current_price, current_date)

            # Position Sizing
            size = compute_position_size(signal, self.position_size_pct)
            notional = (self.current_cash * size)

            # Recording the position
            self.active_positions[ticker] = {
                'entry_price': current_price * 0.05,  # Simulating option premium as 5% of stock
                'entry_date': current_date,
                'signal': signal,
                'contract': contract
            }
            logger.info(
                f"ENTRY: {ticker} at {current_date} @ {current_price:.2f} | Signal Strength: {signal['strength']:.2f} | ML Filter: {'Passed' if self.use_ml and self.ml_filter.is_trained else 'N/A'}")

    def _check_exits(self, signals, current_date):
        for ticker in list(self.active_positions.keys()):
            pos = self.active_positions[ticker]
            current_price_stock = self.price_history[ticker]['close'].iloc[-1]

            # Simple option price simulation (In reality, use Black-Scholes or API price)
            entry_price = pos['entry_price']
            current_option_price = entry_price * \
                (current_price_stock /
                 self.price_history[ticker]['close'].asof(pos['entry_date']))

            pnl_pct = (current_option_price - entry_price) / entry_price
            days_held = (current_date - pos['entry_date']).days

            exit_reason = None
            # Standard Exits
            if pnl_pct < -self.option_stop_loss:
                exit_reason = "STOP_LOSS"
                logger.info(f"STOP LOSS hit for {ticker}: PnL {pnl_pct:.2%}")
            elif pnl_pct > self.option_take_profit:
                exit_reason = "TAKE_PROFIT"
                logger.info(f"TAKE PROFIT hit for {ticker}: PnL {pnl_pct:.2%}")
            elif days_held >= self.holding_period:
                exit_reason = "TIME_EXIT"
                logger.info(f"HOLDING PERIOD exceeded for {ticker}: Days Held {days_held}")

            # Signal/Regime Exits
            if ticker in signals:
                curr_s = signals[ticker]
                if curr_s['regime'] >= 4 and pnl_pct < 0:
                    exit_reason = "CRISIS_EXIT"
                    logger.info(f"CRISIS EXIT for {ticker}: Regime {curr_s['regime']} | PnL {pnl_pct:.2%}")
                elif curr_s['regime'] <= 1 and pnl_pct > 0:
                    exit_reason = "BULL_EXIT"
                    logger.info(f"BULL EXIT for {ticker}: Regime {curr_s['regime']} | PnL {pnl_pct:.2%}")

            if exit_reason:
                self._close_position(ticker, pnl_pct, days_held, exit_reason)

    def _close_position(self, ticker, pnl_pct, days_held, reason):
        pos = self.active_positions.pop(ticker)
        self.trade_log.append(f"EXIT: {ticker} | Reason: {reason} | PnL: {pnl_pct:.2%}")
        logger.info(f"EXIT: {ticker} | Reason: {reason} | PnL: {pnl_pct:.2%}")

        if self.use_ml:
            self.trade_collector.record_exit(ticker, pnl_pct)
            self.ml_filter.maybe_retrain(self.trade_collector)


def load_yf_spy_history(ticker="SPY", start_date="2024-01-01"):
    """Load ticker OHLCV history from persistent cache for testing."""
    history = get_data_persistent(ticker, interval="1d", period="2y", force_refresh=False)
    if history.empty:
        return pd.DataFrame()
    # Filter by start_date
    start_ts = normalize_timestamp_for_index(start_date, history.index)
    history = history[history.index >= start_ts]
    history = history[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return history


# --- Main Execution Block ---
if __name__ == "__main__":
    algo = MandelbrotSwingStandalone()
    ticker = "QQQ"  # Using QQQ as the cash parking ticker for testing
    algo.tickers = [ticker]  # Override tickers to just QQQ for focused testing
    algo.cash_parking_ticker = ticker
    algo.price_history = {ticker: pd.DataFrame()}
    algo.use_ml = True  # Enable ML filtering for testing

    spy_history = load_yf_spy_history(ticker=ticker, start_date="2024-01-01")
    for current_date, row in spy_history.iterrows():
        bar_data = {
            'open': float(row['Open']),
            'high': float(row['High']),
            'low': float(row['Low']),
            'close': float(row['Close']),
            'volume': float(row['Volume']),
            'date': current_date,
        }
        algo.update_data(ticker, bar_data)
        algo.run_daily_logic(current_date)

    logger.info(f"====Completed {ticker} test run with {len(algo.trade_log)} trades====")
    for trade in algo.trade_log[:]:
        logger.info(trade)
