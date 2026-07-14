import math
import os
import time
import logging
from pathlib import Path
from ib_insync import *

# --- CONFIGURATION ---
ACCOUNT_ID = os.getenv('IB_PAPER_ID')
TOTAL_BUDGET_CAD = 4700.0
USD_ALLOCATION_PCT = 0.25 # 25% of portfolio is USD (MSFT, GOOG, AMZN)
SLIPPAGE_BUFFER = 0.001 
# Defensive Weights
TARGETS = [
    {'symbol': 'XEQT.TO', 'exch': 'TSX',   'curr': 'CAD', 'weight': 0.35, 'limit': 45.1},
    {'symbol': 'XDIV.TO', 'exch': 'TSX',   'curr': 'CAD', 'weight': 0.20, 'limit': 45.1},
    {'symbol': 'CLML.TO', 'exch': 'TSX',   'curr': 'CAD', 'weight': 0.20, 'limit': 53.5},
    {'symbol': 'MSFT', 'exch': 'SMART', 'curr': 'USD', 'weight': 0.083},
    {'symbol': 'GOOGL','exch': 'SMART', 'curr': 'USD', 'weight': 0.083}, # GOOGL usually has more liquidity
    {'symbol': 'AMZN', 'exch': 'SMART', 'curr': 'USD', 'weight': 0.084},
]


def _setup_logger():
    logger = logging.getLogger('bot_initial')
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    log_dir = Path(__file__).resolve().parents[1] / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / 'bot_initial_trades.log', encoding='utf-8')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


logger = _setup_logger()

class FHSADefensiveTrader:
    def __init__(self):
        self.ib = IB()
        self.ib.connect('127.0.0.1', 4002, clientId=1)
        self.account_id = self._resolve_account_id()

    def _resolve_account_id(self):
        managed_accounts = self.ib.managedAccounts()

        if ACCOUNT_ID:
            if ACCOUNT_ID in managed_accounts:
                return ACCOUNT_ID
            raise RuntimeError(f"Configured account '{ACCOUNT_ID}' is not available in IB managed accounts: {managed_accounts}")

        if len(managed_accounts) == 1:
            selected = managed_accounts[0]
            logger.info("No account ID configured; using single available managed account: %s", selected)
            return selected

        raise RuntimeError(
            "Account ID not configured. Set IB_PAPER_ID environment variable "
            f"(available managed accounts: {managed_accounts})"
        )

    def _wait_and_log_trade(self, trade, label, timeout_sec=20):
        start = time.time()
        last_status = ''
        while time.time() - start < timeout_sec:
            self.ib.sleep(0.5)
            status = trade.orderStatus.status or ''
            if status != last_status and status:
                logger.info("%s | status=%s | symbol=%s | qty=%s", label, status, trade.contract.symbol, trade.order.totalQuantity)
                last_status = status

            if status in {'Filled', 'Cancelled', 'ApiCancelled', 'Inactive'}:
                break

        final_status = trade.orderStatus.status or 'Unknown'
        logger.info(
            "%s | final_status=%s | symbol=%s | qty=%s | avg_fill_price=%s",
            label,
            final_status,
            trade.contract.symbol,
            trade.order.totalQuantity,
            trade.orderStatus.avgFillPrice,
        )
        return final_status

    def _validate_targets(self):
        required = {'symbol', 'exch', 'curr', 'weight'}
        allowed_currencies = {'CAD', 'USD'}
        seen_symbols = set()
        total_weight = 0.0
        usd_weight = 0.0

        for idx, item in enumerate(TARGETS):
            missing = required - set(item.keys())
            if missing:
                raise ValueError(f"TARGETS[{idx}] is missing required fields: {sorted(missing)}")

            symbol = str(item['symbol']).strip()
            curr = str(item['curr']).strip().upper()
            weight = float(item['weight'])

            if not symbol:
                raise ValueError(f"TARGETS[{idx}] has an empty symbol")
            if symbol in seen_symbols:
                raise ValueError(f"TARGETS contains duplicate symbol: {symbol}")
            if curr not in allowed_currencies:
                raise ValueError(f"TARGETS[{idx}] has unsupported currency: {curr}")
            if weight <= 0:
                raise ValueError(f"TARGETS[{idx}] has non-positive weight: {weight}")
            if curr == 'CAD' and 'limit' in item:
                try:
                    limit = float(item['limit'])
                except (TypeError, ValueError):
                    raise ValueError(f"TARGETS[{idx}] has invalid limit price: {item['limit']}")
                if limit <= 0:
                    raise ValueError(f"TARGETS[{idx}] has non-positive limit price: {limit}")

            seen_symbols.add(symbol)
            total_weight += weight
            if curr == 'USD':
                usd_weight += weight

        if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
            raise ValueError(f"TARGETS weights must sum to 1.0, got {total_weight:.6f}")

        if not math.isclose(usd_weight, USD_ALLOCATION_PCT, abs_tol=0.02):
            logger.warning(
                "USD_ALLOCATION_PCT (%.3f) differs from USD target weight sum (%.3f)",
                USD_ALLOCATION_PCT,
                usd_weight,
            )

    def get_market_price(self, contract, use_snapshot=False):
        self.ib.qualifyContracts(contract)

        if use_snapshot:
            ticker = self.ib.reqTickers(contract)[0]
        else:
            self.ib.reqMktData(contract, '', False, False)
            time.sleep(2)
            ticker = self.ib.reqTickers(contract)[0]

        for candidate in (ticker.ask, ticker.last, ticker.close):
            if candidate is not None and candidate > 0:
                return candidate

        raise RuntimeError(
            f"No valid market price for {contract.symbol} "
            f"(ask={ticker.ask}, last={ticker.last}, close={ticker.close})"
        )

    def _build_stock_contract(self, item):
        """Build an IB stock contract with exchange/symbol normalization."""
        symbol = item['symbol']
        exch = item['exch']
        curr = item['curr']

        # IB expects Canadian TSX symbols without the .TO suffix.
        if curr == 'CAD' and symbol.endswith('.TO'):
            symbol = symbol[:-3]

        # Route Canadian equities via SMART and pin the primary listing exchange.
        if exch == 'TSX':
            return Stock(symbol, 'SMART', curr, primaryExchange='TSE')

        return Stock(symbol, exch, curr)

    def _get_cash_balance(self, currency):
        """Return account cash balance for a currency using IB account summary."""
        summary = self.ib.accountSummary(account=self.account_id)
        preferred_tags = ('TotalCashValue', 'CashBalance')

        for tag in preferred_tags:
            for row in summary:
                if row.tag == tag and row.currency == currency and row.account == self.account_id:
                    try:
                        return float(row.value)
                    except (TypeError, ValueError):
                        continue

        return 0.0

    def _get_position_size(self, contract):
        """Return current position size for contract in selected account."""
        self.ib.qualifyContracts(contract)
        target_con_id = contract.conId

        for pos in self.ib.positions(account=self.account_id):
            if pos.contract.conId == target_con_id:
                return int(math.floor(pos.position))

        return 0

    def convert_cad_to_usd(self, amount_cad, usd_cad_rate):
        """Converts a lump sum of CAD to USD to save on per-trade fees."""
        if usd_cad_rate <= 0:
            raise RuntimeError(f"Invalid USDCAD rate for conversion: {usd_cad_rate}")

        usd_qty = max(1, int(round(amount_cad / usd_cad_rate)))
        logger.info(
            "ACTION REQUIRED: Converting %.2f CAD to about %s USD at USDCAD %.6f",
            amount_cad,
            usd_qty,
            usd_cad_rate,
        )
        pair = Forex('USDCAD')
        self.ib.qualifyContracts(pair)
        # In IBKR, buying USDCAD means buying USD with CAD
        order = MarketOrder('BUY', usd_qty, account=self.account_id)
        trade = self.ib.placeOrder(pair, order)
        logger.info("FX trade submitted for account=%s", self.account_id)
        self._wait_and_log_trade(trade, 'FX_CONVERSION')

    def run(self):
        logger.info("Starting bot run | account=%s", self.account_id)
        try:
            self._validate_targets()

            # 1. Fetch FX rate for calculations
            fx_ticker = self.ib.reqTickers(Forex('USDCAD'))[0]
            usd_cad_rate = fx_ticker.midpoint()
            if usd_cad_rate is None or usd_cad_rate <= 0:
                raise RuntimeError(f"Invalid USDCAD midpoint: {usd_cad_rate}")
            logger.info("USDCAD midpoint fetched: %.6f", usd_cad_rate)

            # 2. Convert only the missing USD cash for this allocation target.
            usd_target = (TOTAL_BUDGET_CAD * USD_ALLOCATION_PCT) / usd_cad_rate
            usd_cash_now = self._get_cash_balance('USD')
            usd_shortfall = max(0.0, usd_target - usd_cash_now)

            if usd_shortfall >= 1.0:
                cad_needed = usd_shortfall * usd_cad_rate
                do_fx = input(
                    f"Need about ${cad_needed:.2f} CAD to top up {usd_shortfall:.2f} USD. Convert now? (y/n): "
                )
                if do_fx.lower() == 'y':
                    self.convert_cad_to_usd(cad_needed, usd_cad_rate)
                else:
                    logger.info("FX conversion skipped by user")
            else:
                logger.info(
                    "USD cash already sufficient (target=%.2f USD, current=%.2f USD); skipping FX conversion",
                    usd_target,
                    usd_cash_now,
                )

            # 3. Process Stocks
            orders = []
            for item in TARGETS:
                contract = self._build_stock_contract(item)
                logger.info(
                    "CONTRACT: requested=%s | symbol=%s | exchange=%s | primary=%s | currency=%s",
                    item['symbol'],
                    contract.symbol,
                    contract.exchange,
                    getattr(contract, 'primaryExchange', ''),
                    contract.currency,
                )
                manual_limit = item.get('limit') if item.get('curr') == 'CAD' else None
                if manual_limit is not None:
                    price = float(manual_limit)
                    lmt_price = round(float(manual_limit), 2)
                    logger.info("Using manual limit price for %s: %.2f CAD", item['symbol'], lmt_price)
                else:
                    use_snapshot = item.get('exch') == 'TSX'
                    if use_snapshot:
                        logger.info("Using snapshot market data for %s", item['symbol'])
                    price = self.get_market_price(contract, use_snapshot=use_snapshot)
                    lmt_price = round(price * (1 + SLIPPAGE_BUFFER), 2)

                # Use real-time rate to decide how many shares to buy
                price_in_cad = price * usd_cad_rate if item['curr'] == 'USD' else price
                target_qty = math.floor((TOTAL_BUDGET_CAD * item['weight']) / price_in_cad)
                held_qty = self._get_position_size(contract)
                qty = max(0, target_qty - held_qty)

                if qty > 0:
                    orders.append((contract, qty, lmt_price))
                    logger.info(
                        "PLAN: Buy %s %s at %.2f %s (target=%s, held=%s)",
                        qty,
                        item['symbol'],
                        lmt_price,
                        item['curr'],
                        target_qty,
                        held_qty,
                    )
                else:
                    logger.info(
                        "No buy needed for %s (target=%s, held=%s)",
                        item['symbol'],
                        target_qty,
                        held_qty,
                    )

            # 4. Final confirmation
            if input("\nExecute all stock orders? (y/n): ").lower() == 'y':
                for contract, qty, lmt in orders:
                    order = LimitOrder('BUY', qty, lmt, account=self.account_id)
                    trade = self.ib.placeOrder(contract, order)
                    logger.info("Order submitted | symbol=%s | qty=%s | limit=%.2f", contract.symbol, qty, lmt)
                    self._wait_and_log_trade(trade, 'EQUITY_ORDER')
            else:
                logger.info("Order execution cancelled by user at final confirmation")
        finally:
            if self.ib.isConnected():
                self.ib.disconnect()
            logger.info("Bot run finished and disconnected")

if __name__ == "__main__":
    try:
        FHSADefensiveTrader().run()
    except Exception as exc:
        logger.exception("Bot terminated with error: %s", exc)