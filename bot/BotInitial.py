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
    {'symbol': 'XEQT.TO', 'exch': 'TSX',   'curr': 'CAD', 'weight': 0.35},
    {'symbol': 'XDIV.TO', 'exch': 'TSX',   'curr': 'CAD', 'weight': 0.20},
    {'symbol': 'CLML.TO', 'exch': 'TSX',   'curr': 'CAD', 'weight': 0.20},
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

    def get_market_price(self, contract):
        self.ib.qualifyContracts(contract)
        self.ib.reqMktData(contract, '', False, False)
        time.sleep(2)
        ticker = self.ib.reqTickers(contract)[0]
        return ticker.ask if ticker.ask > 0 else ticker.close

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

    def convert_cad_to_usd(self, amount_cad):
        """Converts a lump sum of CAD to USD to save on per-trade fees."""
        logger.info("ACTION REQUIRED: Converting %.2f CAD to USD", amount_cad)
        pair = Forex('USDCAD')
        self.ib.qualifyContracts(pair)
        # In IBKR, buying USDCAD means buying USD with CAD
        order = MarketOrder('BUY', amount_cad, account=self.account_id)
        trade = self.ib.placeOrder(pair, order)
        logger.info("FX trade submitted for account=%s", self.account_id)
        self._wait_and_log_trade(trade, 'FX_CONVERSION')

    def run(self):
        logger.info("Starting bot run | account=%s", self.account_id)

        # 1. Fetch FX Rate for calculations
        fx_ticker = self.ib.reqTickers(Forex('USDCAD'))[0]
        usd_cad_rate = fx_ticker.midpoint()
        logger.info("USDCAD midpoint fetched: %.6f", usd_cad_rate)

        # 2. Check if we should do a lump-sum FX conversion first
        usd_needed_cad = TOTAL_BUDGET_CAD * USD_ALLOCATION_PCT
        do_fx = input(f"Do you want to convert ${usd_needed_cad:.2f} CAD to USD now? (y/n): ")
        if do_fx.lower() == 'y':
            self.convert_cad_to_usd(usd_needed_cad)
        else:
            logger.info("FX conversion skipped by user")

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
            price = self.get_market_price(contract)
            
            # Use real-time rate to decide how many shares to buy
            price_in_cad = price * usd_cad_rate if item['curr'] == 'USD' else price
            qty = math.floor((TOTAL_BUDGET_CAD * item['weight']) / price_in_cad)
            
            if qty > 0:
                lmt_price = round(price * (1 + SLIPPAGE_BUFFER), 2)
                orders.append((contract, qty, lmt_price))
                logger.info("PLAN: Buy %s %s at %.2f %s", qty, item['symbol'], lmt_price, item['curr'])
            else:
                logger.warning("Skipped %s because computed quantity is %s", item['symbol'], qty)

        # 4. Final Confirmation
        if input("\nExecute all stock orders? (y/n): ").lower() == 'y':
            for contract, qty, lmt in orders:
                order = LimitOrder('BUY', qty, lmt, account=self.account_id)
                trade = self.ib.placeOrder(contract, order)
                logger.info("Order submitted | symbol=%s | qty=%s | limit=%.2f", contract.symbol, qty, lmt)
                self._wait_and_log_trade(trade, 'EQUITY_ORDER')
        else:
            logger.info("Order execution cancelled by user at final confirmation")
        
        self.ib.disconnect()
        logger.info("Bot run finished and disconnected")

if __name__ == "__main__":
    try:
        FHSADefensiveTrader().run()
    except Exception as exc:
        logger.exception("Bot terminated with error: %s", exc)