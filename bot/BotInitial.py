import math
import time
from ib_insync import *

# --- CONFIGURATION ---
ACCOUNT_ID = "XXX" # <--- YOUR FHSA ACCOUNT
TOTAL_BUDGET_CAD = 4700.0
USD_ALLOCATION_PCT = 0.25 # 25% of portfolio is USD (MSFT, GOOG, AMZN)
SLIPPAGE_BUFFER = 0.001 

# Defensive Weights
TARGETS = [
    {'symbol': 'XEQT', 'exch': 'TSX',   'curr': 'CAD', 'weight': 0.35},
    {'symbol': 'XDIV', 'exch': 'TSX',   'curr': 'CAD', 'weight': 0.20},
    {'symbol': 'CLML', 'exch': 'TSX',   'curr': 'CAD', 'weight': 0.20},
    {'symbol': 'MSFT', 'exch': 'SMART', 'curr': 'USD', 'weight': 0.083},
    {'symbol': 'GOOGL','exch': 'SMART', 'curr': 'USD', 'weight': 0.083}, # GOOGL usually has more liquidity
    {'symbol': 'AMZN', 'exch': 'SMART', 'curr': 'USD', 'weight': 0.084},
]

class FHSADefensiveTrader:
    def __init__(self):
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=1)

    def get_market_price(self, contract):
        self.ib.qualifyContracts(contract)
        self.ib.reqMktData(contract, '', False, False)
        time.sleep(2)
        ticker = self.ib.reqTickers(contract)[0]
        return ticker.ask if ticker.ask > 0 else ticker.close

    def convert_cad_to_usd(self, amount_cad):
        """Converts a lump sum of CAD to USD to save on per-trade fees."""
        print(f"ACTION REQUIRED: Converting ${amount_cad} CAD to USD...")
        pair = Forex('USDCAD')
        self.ib.qualifyContracts(pair)
        # In IBKR, buying USDCAD means buying USD with CAD
        order = MarketOrder('BUY', amount_cad, account=ACCOUNT_ID)
        trade = self.ib.placeOrder(pair, order)
        print("FX Trade submitted. Waiting for settlement...")
        time.sleep(5)

    def run(self):
        if ACCOUNT_ID not in self.ib.managedAccounts():
            print(f"Error: Account {ACCOUNT_ID} not found.")
            return

        # 1. Fetch FX Rate for calculations
        fx_ticker = self.ib.reqTickers(Forex('USDCAD'))[0]
        usd_cad_rate = fx_ticker.midpoint()

        # 2. Check if we should do a lump-sum FX conversion first
        usd_needed_cad = TOTAL_BUDGET_CAD * USD_ALLOCATION_PCT
        do_fx = input(f"Do you want to convert ${usd_needed_cad:.2f} CAD to USD now? (y/n): ")
        if do_fx.lower() == 'y':
            self.convert_cad_to_usd(usd_needed_cad)

        # 3. Process Stocks
        orders = []
        for item in TARGETS:
            contract = Stock(item['symbol'], item['exch'], item['curr'])
            price = self.get_market_price(contract)
            
            # Use real-time rate to decide how many shares to buy
            price_in_cad = price * usd_cad_rate if item['curr'] == 'USD' else price
            qty = math.floor((TOTAL_BUDGET_CAD * item['weight']) / price_in_cad)
            
            if qty > 0:
                lmt_price = round(price * (1 + SLIPPAGE_BUFFER), 2)
                orders.append((contract, qty, lmt_price))
                print(f"PLAN: Buy {qty} {item['symbol']} at {lmt_price} {item['curr']}")

        # 4. Final Confirmation
        if input("\nExecute all stock orders? (y/n): ").lower() == 'y':
            for contract, qty, lmt in orders:
                order = LimitOrder('BUY', qty, lmt, account=ACCOUNT_ID)
                self.ib.placeOrder(contract, order)
                print(f"Order sent: {contract.symbol}")
        
        self.ib.disconnect()

if __name__ == "__main__":
    FHSADefensiveTrader().run()