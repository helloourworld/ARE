"""Automated IB trading bot that consumes Streamlit-backed signal recommendations.

This bot uses `risk_modeling.mandelbrot.scan_market()` as its signal engine, which
mirrors the Streamlit dashboard's hybrid risk signal logic. It fetches 1-minute historical
bars from IB, then uses the scanned suggestion to decide buy/exit actions.
"""

import datetime
import io
import itertools
import json
import logging
import os
import uuid
from logging.handlers import RotatingFileHandler
from contextlib import redirect_stdout
import sys
import time
from pathlib import Path
import pytz
import numpy as np
import pandas as pd

# Ensure the project root is on sys.path so imports work when running bot/BotMandelbrot.py directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ib_insync import IB, MarketOrder, Stock
from data_pipeline.data_cache import get_data_persistent
from risk_modeling.mandelbrot import scan_market


class RunContextFilter(logging.Filter):
    """Attach per-run context fields to each log record."""

    def __init__(self, run_id):
        super().__init__()
        self.run_id = run_id

    def filter(self, record):
        record.run_id = self.run_id
        return True


class JsonFormatter(logging.Formatter):
    """Render logs as JSON for ingestion in external logging systems."""

    def format(self, record):
        payload = {
            "timestamp": datetime.datetime.utcfromtimestamp(record.created).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "run_id": getattr(record, "run_id", "n/a"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _setup_logger():
    """Configure console and rotating file logger for trading and backtest runs."""
    logger = logging.getLogger("mandelbrot_bot")
    if logger.handlers:
        return logger

    log_level = os.getenv("BOT_LOG_LEVEL", "INFO").upper()
    log_json = os.getenv("BOT_LOG_JSON", "0").lower() in ("1", "true", "yes", "json")
    run_id = os.getenv("BOT_RUN_ID", uuid.uuid4().hex[:12])
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "mandelbrot_bot.log"

    if log_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | run=%(run_id)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    context_filter = RunContextFilter(run_id)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.addFilter(context_filter)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    file_handler.addFilter(context_filter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    # Keep scanner diagnostics in the same structured logging stream.
    scanner_logger = logging.getLogger("risk_modeling.mandelbrot")
    scanner_logger.setLevel(getattr(logging, log_level, logging.INFO))
    scanner_logger.addHandler(console_handler)
    scanner_logger.addHandler(file_handler)
    scanner_logger.propagate = False

    return logger


logger = _setup_logger()


ENTRY_SIGNALS = {"🚀 STRONG BUY / HOLD", "✅ HOLD / BUY DIPS", "➕ ACCUMULATE"}
EXIT_SIGNALS = {"📉 REDUCE / TAKE PROFIT", "🛑 SELL / SHORT", "🛑 STAY SHORT", "↕️ SCALP RANGE", "🚫 AVOID / STAY CASH", "🔄 REDUCE POSITION SIZE", "🚨 EXIT / AGGRESSIVE SELL"}
ENTRY_CONFIRMATION_BARS = 3
SOFT_EXIT_CONFIRMATION_BARS = 2
CONSOLIDATED_ENTRY_MIN_SCORE = 3
NO_NEW_ENTRY_AFTER = "16:30"
TRADE_SESSION_START = "09:35"
TRADE_SESSION_END = "15:55"
TRAILING_ACTIVATION_PCT = 0.010
DEFAULT_STOP_LOSS_PCT = 0.018
DEFAULT_TAKE_PROFIT_PCT = 0.030
DEFAULT_TRAILING_STOP_PCT = 0.012


def _get_consolidated_signal_score(signal_result):
    """Combine multiple signal dimensions into one score to reduce false entries."""
    signal = signal_result.get("Suggestion", "")
    verdict = str(signal_result.get("Verdict", "")).upper()
    fragility = signal_result.get("Fragility Alert", "")
    vpin = float(signal_result.get("VPIN", 0.0) or 0.0)
    hybrid_signal = str(signal_result.get("Hybrid Signal", "")).upper()
    hybrid_vpin = float(signal_result.get("Hybrid VPIN", 0.0) or 0.0)
    regime = str(signal_result.get("Regime", "")).upper()

    score = 0

    if signal in ENTRY_SIGNALS:
        score += 1
    if any(token in verdict for token in ("HEALTHY MOMENTUM", "BULLISH ABSORPTION", "BULLISH PERSISTENCE")):
        score += 1
    if "1 - BULLISH PERSISTENCE" in regime:
        score += 1
    if any(token in hybrid_signal for token in ("BUY", "ACCUMULATE", "BULL")):
        score += 1

    if vpin > 0.72:
        score -= 2
    elif vpin < 0.60:
        score += 1

    if hybrid_vpin > 0.72:
        score -= 1

    if fragility == "CRITICAL FRAGILITY":
        score -= 3

    blocked_verdict_words = ("TOXIC", "DISTRIBUTION", "AGGRESSIVE SELLING", "WEAK RALLY", "NOISY")
    if any(word in verdict for word in blocked_verdict_words):
        score -= 2

    return score


def _is_entry_candidate(signal_result, min_score=CONSOLIDATED_ENTRY_MIN_SCORE):
    """Allow entries only when consolidated score and quality checks are favorable."""
    signal = signal_result.get("Suggestion", "")
    verdict = str(signal_result.get("Verdict", "")).upper()
    regime = str(signal_result.get("Regime", "")).upper()
    fragility = signal_result.get("Fragility Alert", "")
    vpin = float(signal_result.get("VPIN", 0.0) or 0.0)

    if signal not in ENTRY_SIGNALS:
        return False
    if fragility == "CRITICAL FRAGILITY":
        return False
    if vpin > 0.72:
        return False
    if "1 - BULLISH PERSISTENCE" not in regime:
        return False
    if not any(token in verdict for token in ("HEALTHY MOMENTUM", "BULLISH PERSISTENCE")):
        return False
    return _get_consolidated_signal_score(signal_result) >= min_score


def _get_exit_reason(
    signal_result,
    price,
    entry_price,
    peak_price,
    stop_loss_pct=DEFAULT_STOP_LOSS_PCT,
    take_profit_pct=DEFAULT_TAKE_PROFIT_PCT,
    trailing_stop_pct=DEFAULT_TRAILING_STOP_PCT,
    trailing_activation_pct=TRAILING_ACTIVATION_PCT,
):
    """Return an exit reason when risk or signal quality deteriorates."""
    signal = signal_result.get("Suggestion", "")
    verdict = str(signal_result.get("Verdict", "")).upper()
    fragility = signal_result.get("Fragility Alert", "")
    vpin = float(signal_result.get("VPIN", 0.0) or 0.0)

    if fragility == "CRITICAL FRAGILITY":
        return "CRITICAL_FRAGILITY"
    if signal in EXIT_SIGNALS:
        return f"SIGNAL:{signal}"
    if "TOXIC" in verdict or "AGGRESSIVE SELLING" in verdict:
        return "VERDICT_RISK_OFF"
    if vpin > 0.80:
        return "VPIN_TOXIC"

    if entry_price is not None and entry_price > 0:
        pnl_pct = (price - entry_price) / entry_price
        if pnl_pct <= -abs(stop_loss_pct):
            return "STOP_LOSS"
        if pnl_pct >= abs(take_profit_pct):
            return "TAKE_PROFIT"

    if peak_price is not None and peak_price > 0 and entry_price is not None and entry_price > 0:
        pnl_pct = (price - entry_price) / entry_price
        drawdown_from_peak = (peak_price - price) / peak_price
        if pnl_pct >= abs(trailing_activation_pct) and drawdown_from_peak >= abs(trailing_stop_pct):
            return "TRAILING_STOP"

    return None


def _is_hard_exit_reason(reason):
    """Hard exits bypass confirmation to reduce tail risk."""
    if reason is None:
        return False
    if reason in {"CRITICAL_FRAGILITY", "VPIN_TOXIC", "STOP_LOSS", "VERDICT_RISK_OFF"}:
        return True
    if isinstance(reason, str) and reason.startswith("SIGNAL:🚨 EXIT"):
        return True
    return False


def _is_in_trade_window(hhmm):
    """Only open new positions during regular trading session."""
    return TRADE_SESSION_START <= hhmm <= TRADE_SESSION_END


def _build_trade_audit(trades_df):
    """Build a compliance-style audit table with realized PnL on exits."""
    if trades_df.empty:
        return pd.DataFrame(columns=[
            "timestamp",
            "action",
            "qty",
            "price",
            "signal",
            "reason",
            "notional",
            "entry_price",
            "realized_pnl",
            "realized_return_pct",
        ])

    rows = []
    open_qty = 0.0
    open_price = None

    for _, trade in trades_df.iterrows():
        action = str(trade.get("action", ""))
        qty = float(trade.get("qty", 0.0) or 0.0)
        price = float(trade.get("price", 0.0) or 0.0)
        signal = trade.get("signal", "")
        reason = trade.get("reason", "")
        timestamp = trade.get("timestamp")
        notional = qty * price

        realized_pnl = np.nan
        realized_return_pct = np.nan
        entry_price = np.nan

        if action == "BUY" and qty > 0:
            open_qty = qty
            open_price = price
        elif action.startswith("SELL") and qty > 0 and open_qty > 0 and open_price is not None:
            entry_price = open_price
            realized_pnl = (price - open_price) * qty
            if open_price > 0:
                realized_return_pct = (price / open_price - 1.0) * 100.0
            open_qty = 0.0
            open_price = None

        rows.append({
            "timestamp": timestamp,
            "action": action,
            "qty": qty,
            "price": price,
            "signal": signal,
            "reason": reason,
            "notional": notional,
            "entry_price": entry_price,
            "realized_pnl": realized_pnl,
            "realized_return_pct": realized_return_pct,
        })

    return pd.DataFrame(rows)


def _compute_backtest_metrics(equity_df, trades_df, initial_cash):
    """Compute compact KPIs for strategy quality and risk review."""
    if equity_df.empty:
        return {
            "final_equity": float(initial_cash),
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "closed_trades": 0,
            "win_rate_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
        }

    final_equity = float(equity_df["equity"].iloc[-1])
    total_return_pct = (final_equity / float(initial_cash) - 1.0) * 100.0

    run_max = equity_df["equity"].cummax()
    max_drawdown_pct = float(((equity_df["equity"] / run_max) - 1.0).min() * 100.0)

    audit_df = _build_trade_audit(trades_df)
    closed_df = audit_df[audit_df["realized_return_pct"].notna()].copy()
    closed_trades = int(len(closed_df))

    if closed_trades == 0:
        return {
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "closed_trades": 0,
            "win_rate_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
        }

    wins = closed_df[closed_df["realized_pnl"] > 0]
    losses = closed_df[closed_df["realized_pnl"] < 0]

    win_rate_pct = float(len(wins) / closed_trades * 100.0)
    avg_win_pct = float(wins["realized_return_pct"].mean()) if not wins.empty else 0.0
    avg_loss_pct = float(losses["realized_return_pct"].mean()) if not losses.empty else 0.0

    gross_profit = float(wins["realized_pnl"].sum()) if not wins.empty else 0.0
    gross_loss_abs = abs(float(losses["realized_pnl"].sum())) if not losses.empty else 0.0
    if gross_loss_abs > 0:
        profit_factor = gross_profit / gross_loss_abs
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return {
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "closed_trades": closed_trades,
        "win_rate_pct": win_rate_pct,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "profit_factor": profit_factor,
    }


class MandelbrotBot:
    """Automated trading bot driven by scan_market signal outputs."""
    def __init__(self):
        self.ib = IB()
        self.ib.connect('127.0.0.1', PORT, clientId=1)
        self.ib.reqMarketDataType(4)  # Switch to delayed-frozen data if live not available
        self.contract = Stock(TICKER, 'SMART', 'USD', primaryExchange='NASDAQ')
        self.ib.qualifyContracts(self.contract)
        self.initial_equity = self.get_total_equity()
        logger.info("Bot initialized | ticker=%s | start_balance=%.2f", TICKER, self.initial_equity)

    def get_total_equity(self):
        account = self.ib.accountSummary()
        return float([v.value for v in account if v.tag == 'NetLiquidation'][0])

    def check_kill_switch(self):
        """If account drops 15%, close everything and stop."""
        current_equity = self.get_total_equity()
        drawdown = (self.initial_equity - current_equity) / self.initial_equity
        if drawdown >= MAX_ACCOUNT_DRAWDOWN:
            logger.critical("Kill switch triggered | drawdown=%.2f%%", drawdown * 100.0)
            self.ib.reqGlobalCancel()
            self.ib.placeOrder(self.contract, MarketOrder(
                'SELL', self.get_position_size()))
            self.ib.disconnect()
            raise SystemExit(1)

    def get_position_size(self):
        pos = [p for p in self.ib.positions() if p.contract.symbol == TICKER]
        return pos[0].position if pos else 0

    def _bars_to_live_df(self, bars):
        """Convert IB historical bars into a timezone-aware 1-minute OHLCV DataFrame."""
        rows = []
        for b in bars:
            rows.append({
                "datetime": pd.to_datetime(b.date, utc=True),
                "Open": b.open,
                "High": b.high,
                "Low": b.low,
                "Close": b.close,
                "Volume": b.volume,
            })
        df = pd.DataFrame(rows).set_index("datetime")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df

    def run_strategy(self):
        """Main bot loop: exit, risk checks, data fetch, and signal-driven execution."""
        at_tz = pytz.timezone('Canada/Atlantic')
        last_entry_signal = None
        entry_signal_streak = 0
        last_exit_reason = None
        exit_signal_streak = 0
        entry_price = None
        peak_price = None

        while True:
            self.ib.waitOnUpdate(timeout=30)
            now_at = datetime.datetime.now(at_tz)

            # 1. TIME EXIT CHECK (close any remaining position at the target time)
            if now_at.strftime("%H:%M") >= TIME_EXIT_AT:
                qty = self.get_position_size()
                if qty != 0:
                    logger.info("Time exit reached | action=liquidate | qty=%s", qty)
                    self.ib.placeOrder(self.contract, MarketOrder(
                        'SELL' if qty > 0 else 'BUY', abs(qty)))
                break

            # 2. EMERGENCY KILL SWITCH CHECK (account-level drawdown protection)
            self.check_kill_switch()

            # 3. DATA ACQUISITION (1-minute bars for the current session)
            bars = self.ib.reqHistoricalData(
                self.contract, '', durationStr='1 D', barSizeSetting='1 min', whatToShow='TRADES', useRTH=False)
            if not bars:
                logger.warning("No live bars received | time=%s | reconnect_check=true", now_at.strftime('%H:%M:%S'))
                if not self.ib.isConnected():
                    self.ib.reconnect()
                self.ib.sleep(5)  # Wait before retry
                continue
            prices = np.array([b.close for b in bars])
            logger.debug("Fetched bars | count=%s", len(prices))
            volumes = np.array([b.volume for b in bars])

            # 4. SIGNAL-DRIVEN TRADE LOGIC
            # Convert live IB bars into a DataFrame and pass it to scan_market.
            live_1m = self._bars_to_live_df(bars)
            previous_daily = get_data_persistent(TICKER, interval="1d", period="2y")
            result = scan_market(
                TICKER,
                show_judgment=True,
                data_1m=live_1m,
                data_1d=previous_daily,
            )
            if not isinstance(result, dict):
                logger.error("Signal scan failed | ticker=%s | error=%s", TICKER, result)
                continue

            qty = self.get_position_size()
            current_price = prices[-1]
            market_ts = live_1m.index[-1] if not live_1m.empty else "N/A"
            signal = result.get("Suggestion", "N/A")
            verdict = result.get("Verdict", "N/A")
            fragility = result.get("Fragility Alert", "")
            logger.info(
                "Market state | stock_ts=%s | local_time=%s | price=%.2f | volume=%s | signal=%s | verdict=%s | fragility=%s",
                market_ts,
                now_at.strftime('%H:%M:%S'),
                current_price,
                volumes[-1],
                signal,
                verdict,
                fragility,
            )

            if qty > 0:
                if entry_price is None:
                    entry_price = current_price
                peak_price = current_price if peak_price is None else max(peak_price, current_price)
            else:
                entry_price = None
                peak_price = None

            # Generate orders only from stronger confirmed signal conditions.
            now_hhmm = now_at.strftime("%H:%M")
            can_open_new_position = _is_in_trade_window(now_hhmm) and now_hhmm < NO_NEW_ENTRY_AFTER

            if qty == 0 and can_open_new_position and _is_entry_candidate(result):
                if signal == last_entry_signal:
                    entry_signal_streak += 1
                else:
                    entry_signal_streak = 1
                    last_entry_signal = signal

                if entry_signal_streak < ENTRY_CONFIRMATION_BARS:
                    continue

                cash_to_use = min(Cash, self.initial_equity * POSITION_SIZE_PCT)
                share_qty = cash_to_use / current_price
                order = MarketOrder('BUY', round(share_qty, 6))
                trade = self.ib.placeOrder(self.contract, order)
                logger.info(
                    "Order placed | stock_ts=%s | side=BUY | status=%s | qty=%s | price=%.2f",
                    market_ts,
                    trade.orderStatus.status,
                    order.totalQuantity,
                    current_price,
                )
                entry_price = current_price
                peak_price = current_price
                entry_signal_streak = 0
                last_entry_signal = None
                time.sleep(10)
            elif qty == 0:
                entry_signal_streak = 0
                last_entry_signal = None
                exit_signal_streak = 0
                last_exit_reason = None

            elif qty > 0:
                exit_reason = _get_exit_reason(result, current_price, entry_price, peak_price)
                if exit_reason is None:
                    exit_signal_streak = 0
                    last_exit_reason = None
                    continue

                hard_exit = _is_hard_exit_reason(exit_reason)
                if not hard_exit:
                    if exit_reason == last_exit_reason:
                        exit_signal_streak += 1
                    else:
                        last_exit_reason = exit_reason
                        exit_signal_streak = 1
                    if exit_signal_streak < SOFT_EXIT_CONFIRMATION_BARS:
                        continue
                else:
                    exit_signal_streak = 0
                    last_exit_reason = None

                logger.info(
                    "Exit signal | stock_ts=%s | reason=%s | signal=%s | fragility=%s | qty=%s | price=%.2f",
                    market_ts,
                    exit_reason,
                    signal,
                    fragility,
                    qty,
                    current_price,
                )
                self.ib.placeOrder(self.contract, MarketOrder('SELL', qty))
                entry_price = None
                peak_price = None
                exit_signal_streak = 0
                last_exit_reason = None
                time.sleep(10)


def backtest_mandelbrot_strategy(
    ticker,
    start_date=None,
    end_date=None,
    initial_cash=100000.0,
    position_size_pct=0.05,
    time_exit_at="17:00",
    output_dir=None,
    step_minutes=1,
    verbose=False,
    min_hold_steps=2,
    stop_loss_pct=DEFAULT_STOP_LOSS_PCT,
    take_profit_pct=DEFAULT_TAKE_PROFIT_PCT,
    trailing_stop_pct=DEFAULT_TRAILING_STOP_PCT,
):
    """Backtest the Streamlit-driven bot using historical 1m and 1d data.

    If output_dir is provided, equity and trade results are saved as CSV files.
    Use step_minutes to sample the backtest at a coarser resolution for faster execution.
    """
    df_1m = get_data_persistent(ticker, interval="1m", period="7d")
    df_1d = get_data_persistent(ticker, interval="1d", period="2y")
    if df_1m.empty or df_1d.empty:
        raise ValueError("Insufficient historical data for backtest.")

    if start_date is not None:
        start_ts = pd.to_datetime(start_date, utc=True)
    else:
        start_ts = df_1m.index[0] + pd.Timedelta(minutes=300)

    if end_date is not None:
        end_ts = pd.to_datetime(end_date, utc=True)
    else:
        end_ts = df_1m.index[-1]

    df_1m = df_1m[(df_1m.index >= start_ts) & (df_1m.index <= end_ts)]
    if df_1m.empty:
        raise ValueError("No 1m data in requested backtest range.")

    cash_balance = float(initial_cash)
    position_size = 0.0
    trade_log = []
    equity_history = []
    tz_atl = pytz.timezone("Canada/Atlantic")
    trade_lines = []
    last_entry_signal = None
    entry_signal_streak = 0
    last_exit_reason = None
    exit_signal_streak = 0
    entry_price = None
    peak_price = None
    bars_in_position = 0

    if step_minutes < 1:
        raise ValueError("step_minutes must be >= 1")

    scanner_logger = logging.getLogger("risk_modeling.mandelbrot")
    original_scanner_level = scanner_logger.level
    if not verbose:
        scanner_logger.setLevel(logging.WARNING)

    try:
        indices = range(0, len(df_1m), step_minutes)
        if indices[-1] != len(df_1m) - 1:
            indices = list(indices) + [len(df_1m) - 1]

        for idx in indices:
            timestamp = df_1m.index[idx]
            row = df_1m.iloc[idx]
            intraday = df_1m.iloc[: idx + 1]
            daily = df_1d.loc[: timestamp.normalize()]

            if len(intraday) < 300 or len(daily) < 2:
                equity_history.append({"timestamp": timestamp, "cash": cash_balance, "position": position_size, "equity": cash_balance + position_size * row["Close"]})
                continue

            if verbose:
                signal_result = scan_market(
                    ticker,
                    show_judgment=True,
                    data_1m=intraday,
                    data_1d=daily,
                )
            else:
                with redirect_stdout(io.StringIO()):
                    signal_result = scan_market(
                        ticker,
                        show_judgment=True,
                        data_1m=intraday,
                        data_1d=daily,
                    )
            if not isinstance(signal_result, dict):
                if verbose:
                    logger.warning("Backtest signal scan failed | timestamp=%s | error=%s", timestamp, signal_result)
                equity_history.append({"timestamp": timestamp, "cash": cash_balance, "position": position_size, "equity": cash_balance + position_size * row["Close"]})
                continue

            signal = signal_result.get("Suggestion", "N/A")
            fragility = signal_result.get("Fragility Alert", "")
            price = float(row["Close"])
            if position_size > 0:
                peak_price = price if peak_price is None else max(peak_price, price)
                bars_in_position += 1
            else:
                bars_in_position = 0

            hhmm_atl = timestamp.tz_convert(tz_atl).strftime("%H:%M")
            is_after_entry_cutoff = hhmm_atl >= NO_NEW_ENTRY_AFTER
            can_open_new_position = _is_in_trade_window(hhmm_atl) and (not is_after_entry_cutoff)

            if position_size == 0 and can_open_new_position and _is_entry_candidate(signal_result):
                if signal == last_entry_signal:
                    entry_signal_streak += 1
                else:
                    entry_signal_streak = 1
                    last_entry_signal = signal

                if entry_signal_streak >= ENTRY_CONFIRMATION_BARS:
                    amount = min(cash_balance, cash_balance * position_size_pct)
                    qty = round(amount / price, 6)
                    if qty > 0:
                        position_size = qty
                        cash_balance -= qty * price
                        entry_price = price
                        peak_price = price
                        bars_in_position = 0
                        entry_signal_streak = 0
                        last_entry_signal = None
                        trade_log.append({"timestamp": timestamp, "action": "BUY", "qty": qty, "price": price, "signal": signal, "reason": "CONFIRMED_ENTRY"})
                        trade_lines.append(f"{timestamp} BUY  qty={qty} price={price:.2f} signal={signal} reason=CONFIRMED_ENTRY")
                        if verbose:
                            logger.info(trade_lines[-1])
            elif position_size == 0:
                entry_signal_streak = 0
                last_entry_signal = None
                exit_signal_streak = 0
                last_exit_reason = None

            elif position_size > 0:
                exit_reason = _get_exit_reason(
                    signal_result,
                    price,
                    entry_price,
                    peak_price,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    trailing_stop_pct=trailing_stop_pct,
                )
                hard_exit = _is_hard_exit_reason(exit_reason)
                if exit_reason is None:
                    exit_signal_streak = 0
                    last_exit_reason = None
                elif not hard_exit:
                    if exit_reason == last_exit_reason:
                        exit_signal_streak += 1
                    else:
                        last_exit_reason = exit_reason
                        exit_signal_streak = 1
                else:
                    exit_signal_streak = 0
                    last_exit_reason = None

                has_soft_exit_confirmation = exit_signal_streak >= SOFT_EXIT_CONFIRMATION_BARS
                can_exit_now = hard_exit or (bars_in_position >= min_hold_steps and has_soft_exit_confirmation)

                if exit_reason is not None and can_exit_now:
                    cash_balance += position_size * price
                    trade_log.append({"timestamp": timestamp, "action": "SELL", "qty": position_size, "price": price, "signal": signal, "reason": exit_reason})
                    trade_lines.append(f"{timestamp} SELL qty={position_size} price={price:.2f} signal={signal} reason={exit_reason}")
                    if verbose:
                        logger.info(trade_lines[-1])
                    position_size = 0.0
                    entry_price = None
                    peak_price = None
                    bars_in_position = 0
                    exit_signal_streak = 0
                    last_exit_reason = None

            now_atl = timestamp.tz_convert(tz_atl)
            if position_size > 0 and now_atl.strftime("%H:%M") >= time_exit_at:
                cash_balance += position_size * price
                trade_log.append({"timestamp": timestamp, "action": "SELL_EOD", "qty": position_size, "price": price, "signal": "TIME_EXIT", "reason": "TIME_EXIT"})
                trade_lines.append(f"{timestamp} SELL_EOD qty={position_size} price={price:.2f} signal=TIME_EXIT reason=TIME_EXIT")
                if verbose:
                    logger.info(trade_lines[-1])
                position_size = 0.0
                entry_price = None
                peak_price = None
                bars_in_position = 0

            equity_history.append({"timestamp": timestamp, "cash": cash_balance, "position": position_size, "equity": cash_balance + position_size * price})
    finally:
        scanner_logger.setLevel(original_scanner_level)

    if position_size > 0:
        final_price = float(df_1m["Close"].iloc[-1])
        cash_balance += position_size * final_price
        trade_log.append({"timestamp": df_1m.index[-1], "action": "SELL_FINAL", "qty": position_size, "price": final_price, "signal": "FINAL_EXIT", "reason": "FINAL_EXIT"})
        trade_lines.append(f"{df_1m.index[-1]} SELL_FINAL qty={position_size} price={final_price:.2f} signal=FINAL_EXIT reason=FINAL_EXIT")
        if verbose:
            logger.info(trade_lines[-1])
        position_size = 0.0
        equity_history[-1]["cash"] = cash_balance
        equity_history[-1]["position"] = 0.0
        equity_history[-1]["equity"] = cash_balance

    equity_df = pd.DataFrame(equity_history).set_index("timestamp")
    trades_df = pd.DataFrame(trade_log)

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        equity_df.to_csv(output_path / f"{ticker}_backtest_equity.csv")
        trades_df.to_csv(output_path / f"{ticker}_backtest_trades.csv")
        audit_df = _build_trade_audit(trades_df)
        audit_df.to_csv(output_path / f"{ticker}_trade_audit.csv", index=False)
        with open(output_path / f"{ticker}_backtest_trades.txt", "w", encoding="utf-8") as trade_file:
            trade_file.write("\n".join(trade_lines))

    return equity_df, trades_df


def run_backtest_sweep(
    ticker,
    output_dir="backtest_results",
    initial_cash=1000.0,
    position_size_pct=0.05,
    step_minutes_options=None,
    min_hold_steps_options=None,
    stop_loss_options=None,
    take_profit_options=None,
    trailing_stop_options=None,
    quick=False,
):
    """Run parameter sweep and rank strategies by return, then drawdown."""
    if quick:
        if step_minutes_options is None:
            step_minutes_options = [5, 10]
        if min_hold_steps_options is None:
            min_hold_steps_options = [1, 2]
        if stop_loss_options is None:
            stop_loss_options = [0.010, 0.014]
        if take_profit_options is None:
            take_profit_options = [0.020, 0.028]
        if trailing_stop_options is None:
            trailing_stop_options = [0.008]
    else:
        if step_minutes_options is None:
            step_minutes_options = [1, 3, 5]
        if min_hold_steps_options is None:
            min_hold_steps_options = [1, 2, 3]
        if stop_loss_options is None:
            stop_loss_options = [0.008, 0.012, 0.016]
        if take_profit_options is None:
            take_profit_options = [0.018, 0.024, 0.032]
        if trailing_stop_options is None:
            trailing_stop_options = [0.006, 0.009, 0.012]

    results = []
    combinations = list(itertools.product(
        step_minutes_options,
        min_hold_steps_options,
        stop_loss_options,
        take_profit_options,
        trailing_stop_options,
    ))

    for idx, (step_minutes, min_hold_steps, stop_loss_pct, take_profit_pct, trailing_stop_pct) in enumerate(combinations, start=1):
        logger.info(
            "Sweep progress | run=%s/%s | step=%s | hold=%s | sl=%.3f | tp=%.3f | trail=%.3f",
            idx,
            len(combinations),
            step_minutes,
            min_hold_steps,
            stop_loss_pct,
            take_profit_pct,
            trailing_stop_pct,
        )
        try:
            equity_df, trades_df = backtest_mandelbrot_strategy(
                ticker=ticker,
                initial_cash=initial_cash,
                position_size_pct=position_size_pct,
                output_dir=None,
                step_minutes=step_minutes,
                verbose=False,
                min_hold_steps=min_hold_steps,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                trailing_stop_pct=trailing_stop_pct,
            )
            final_equity = float(equity_df["equity"].iloc[-1])
            total_return_pct = (final_equity / initial_cash - 1.0) * 100.0
            run_max = equity_df["equity"].cummax()
            drawdown = (equity_df["equity"] / run_max - 1.0).min() * 100.0
            results.append({
                "step_minutes": step_minutes,
                "min_hold_steps": min_hold_steps,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "trailing_stop_pct": trailing_stop_pct,
                "final_equity": final_equity,
                "total_return_pct": total_return_pct,
                "max_drawdown_pct": float(drawdown),
                "trade_rows": int(len(trades_df)),
                "status": "ok",
            })
        except Exception as exc:
            results.append({
                "step_minutes": step_minutes,
                "min_hold_steps": min_hold_steps,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "trailing_stop_pct": trailing_stop_pct,
                "final_equity": np.nan,
                "total_return_pct": np.nan,
                "max_drawdown_pct": np.nan,
                "trade_rows": 0,
                "status": f"error: {exc}",
            })

    results_df = pd.DataFrame(results)
    ok_df = results_df[results_df["status"] == "ok"].copy()
    if not ok_df.empty:
        ok_df = ok_df.sort_values(["total_return_pct", "max_drawdown_pct"], ascending=[False, False])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path / f"{ticker}_backtest_sweep_results.csv", index=False)
    if not ok_df.empty:
        ok_df.head(20).to_csv(output_path / f"{ticker}_backtest_sweep_top20.csv", index=False)

    return results_df, ok_df


# --- EXECUTION ---
if __name__ == "__main__":
    PORT = 4002
    TICKER = 'MRVL'
    Cash = 1000  # Your current balance
    MAX_ACCOUNT_DRAWDOWN = 0.15  # 15% Total Stop Loss
    POSITION_SIZE_PCT = 0.05  # Use up to 5% of equity per signal-driven entry
    TIME_EXIT_AT = "17:00"  # 5:00 PM Atlantic Time

    if len(sys.argv) > 1 and sys.argv[1].lower() == "backtest":
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "backtest_results"
        step_minutes = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        verbose = sys.argv[4].lower() in ("1", "true", "yes", "verbose") if len(sys.argv) > 4 else False
        min_hold_steps = int(sys.argv[5]) if len(sys.argv) > 5 else 2
        stop_loss_pct = float(sys.argv[6]) if len(sys.argv) > 6 else DEFAULT_STOP_LOSS_PCT
        take_profit_pct = float(sys.argv[7]) if len(sys.argv) > 7 else DEFAULT_TAKE_PROFIT_PCT
        trailing_stop_pct = float(sys.argv[8]) if len(sys.argv) > 8 else DEFAULT_TRAILING_STOP_PCT
        equity_df, trades_df = backtest_mandelbrot_strategy(
            TICKER,
            initial_cash=Cash,
            position_size_pct=POSITION_SIZE_PCT,
            output_dir=output_dir,
            step_minutes=step_minutes,
            verbose=verbose,
            min_hold_steps=min_hold_steps,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            trailing_stop_pct=trailing_stop_pct,
        )
        logger.info("Backtest complete")
        logger.info("Results saved | path=%s", Path(output_dir).resolve())
        if not trades_df.empty:
            logger.info("Last trades:\n%s", trades_df.tail(10).to_string(index=False))
        else:
            logger.info("No trades were executed in this run")
        metrics = _compute_backtest_metrics(equity_df, trades_df, Cash)
        logger.info(
            "Backtest summary | final_equity=%.2f | total_return_pct=%.2f | max_drawdown_pct=%.2f | closed_trades=%s | win_rate_pct=%.2f | avg_win_pct=%.2f | avg_loss_pct=%.2f | profit_factor=%s",
            metrics["final_equity"],
            metrics["total_return_pct"],
            metrics["max_drawdown_pct"],
            metrics["closed_trades"],
            metrics["win_rate_pct"],
            metrics["avg_win_pct"],
            metrics["avg_loss_pct"],
            "inf" if np.isinf(metrics["profit_factor"]) else f"{metrics['profit_factor']:.3f}",
        )
        if verbose:
            logger.info("Verbose logging enabled. Check the trade details in the output folder")
    elif len(sys.argv) > 1 and sys.argv[1].lower() == "sweep":
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "backtest_results"
        quick = sys.argv[3].lower() in ("quick", "1", "true", "yes") if len(sys.argv) > 3 else False
        results_df, top_df = run_backtest_sweep(
            ticker=TICKER,
            output_dir=output_dir,
            initial_cash=Cash,
            position_size_pct=POSITION_SIZE_PCT,
            quick=quick,
        )
        logger.info("Parameter sweep complete")
        logger.info("Results saved | path=%s", Path(output_dir).resolve())
        if top_df.empty:
            logger.warning("No successful sweep runs found. Check sweep results CSV for errors")
        else:
            logger.info("Top 10 configurations:\n%s", top_df.head(10).to_string(index=False))
    else:
        bot = MandelbrotBot()
        try:
            bot.run_strategy()
        except KeyboardInterrupt:
            bot.ib.disconnect()

# Notes:
# - This bot uses the same signal engine as the Streamlit dashboard via scan_market().
# - Configure PORT, TICKER, Cash, and POSITION_SIZE_PCT before running.
# - Run this script with a live IB Gateway/TWS connection, and test in paper trading first.
