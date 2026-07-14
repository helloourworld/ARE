from ib_insync import *
import pandas as pd
from datetime import datetime

ib = IB()
try:
    ib.connect('127.0.0.1', 4002, clientId=2)
except Exception as e:
    print(f"Connection failed: {e}")
    exit()


def get_contract_name(contract):
    """Helper to format name for Stocks, Options, and Spreads (Bags)."""
    if contract.secType == 'OPT':
        return contract.localSymbol
    if contract.secType == 'BAG':
        # For spreads, show symbol + legs info
        legs = "/".join([str(leg.conId) for leg in contract.comboLegs])
        return f"{contract.symbol} [SPREAD]"
    return contract.symbol


def show_my_account():
    # Sync all orders (from TWS/API/Mobile)
    ib.reqAllOpenOrders()
    ib.waitOnUpdate(timeout=0.5)

    # --- 1. HOLDINGS ---
    print("\n" + "="*95)
    print(f"{'SYMBOL/CONTRACT':<20} | {'SEC':<4} | {'QTY':<6} | {'AVG COST':<10}")
    print("-" * 95)
    pos = ib.positions()
    if pos:
        for p in pos:
            name = get_contract_name(p.contract)
            print(
                f"{name:<20} | {p.contract.secType:<4} | {p.position:<6} | {p.avgCost:<10.2f}")
    else:
        print("No open positions.")

    # --- 2. OPEN ORDERS + SPREAD PRICES ---
    print("\n" + "="*95)
    print(f"{'ID (PRNT)':<12} | {'CONTRACT':<20} | {'ACTION':<6} | {'ORD PX':<10} | {'MKT PRICE':<10} | {'STATUS'}")
    print("-" * 95)

    trades = ib.openTrades()
    if trades:
        # Request tickers for all unique contracts (including BAGs)
        unique_contracts = []
        seen = set()
        for t in trades:
            contract_id = t.contract.conId
            if contract_id not in seen:
                unique_contracts.append(t.contract)
                seen.add(contract_id)
        tickers = ib.reqTickers(*unique_contracts)
        price_map = {ticker.contract.conId if ticker.contract.secType !=
                     'BAG' else 0: ticker for ticker in tickers}

        # Note: BAG contracts often have conId=0 in the ticker,
        # so we match BAGs manually if needed.

        for t in trades:
            name = get_contract_name(t.contract)
            order_px = t.order.lmtPrice if t.order.orderType == 'LMT' else t.order.auxPrice
            parent = f"{t.order.orderId}({t.order.parentId})"

            # Find the correct ticker for this contract
            ticker = next(
                (tick for tick in tickers if tick.contract == t.contract), None)

            if ticker:
                bid = ticker.bid
                ask = ticker.ask

                # --- NETTING LOGIC ---
                # If both bid/ask exist, mid is the average.
                # This is the "Net Price" of the spread.
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2
                else:
                    # Fallback to marketPrice() or 0.0 if spread is wide/missing
                    mid = ticker.marketPrice() if not pd.isna(ticker.marketPrice()) else 0.0

                bid_str = f"{bid:8.2f}" if bid > 0 else "  N/A"
                ask_str = f"{ask:8.2f}" if ask > 0 else "  N/A"
                mid_str = f"{mid:8.2f}" if mid > 0 else " Loading.."
            else:
                bid_str = ask_str = mid_str = " Loading.."

            print(f"{parent:<12} | {name:<20} | {t.order.action:<6} | "
                  f"{order_px:<10.2f} | {mid_str:<10} | {t.orderStatus.status}")
    else:
        print("No active orders found.")

    # --- 3. RECENT FILLS (Timezone Fixed) ---
    print("\n" + "="*95)
    print(f"{'CONTRACT':<20} | {'SIDE':<6} | {'FILL PX':<10} | {'LOCAL TIME'}")
    print("-" * 95)
    fills = ib.fills()
    if fills:
        for f in fills[-5:]:
            name = get_contract_name(f.contract)
            local_time = f.execution.time.astimezone().strftime('%H:%M:%S')
            print(
                f"{name:<20} | {f.execution.side:<6} | {f.execution.avgPrice:<10.2f} | {local_time}")
    else:
        print("No trades filled this session.")
    print("="*95 + "\n")


show_my_account()
ib.disconnect()
