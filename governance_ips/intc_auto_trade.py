import asyncio
import os
import pandas as pd
import pandas_ta as ta
from ib_async import *
from cozepy import Coze, TokenAuth, COZE_CN_BASE_URL
from dotenv import load_dotenv

# 1. Environment & Config
load_dotenv()
COZE_TOKEN = os.getenv("COZE_TOKEN")
COZE_BOT_ID = os.getenv("COZE_BOT_ID")
IB_PORT = 4002  # TWS Paper Trading
TICKER = 'INTC'
LEVEL_LOOKBACK = 20
MAX_RISK_DOLLARS = float(os.getenv("INTC_MAX_RISK_DOLLARS", "500"))
MAX_SHARES = int(os.getenv("INTC_MAX_SHARES", "100"))
PAPER_TRADING = os.getenv("IB_TRADING_MODE", "paper").lower() == "paper"

if not PAPER_TRADING:
    raise RuntimeError("Live trading is disabled. Set IB_TRADING_MODE=paper to run this bot.")

# Initialize Coze Client
coze = Coze(auth=TokenAuth(COZE_TOKEN), base_url=COZE_CN_BASE_URL)

class INTCBot:
    def __init__(self):
        self.ib = IB()
        self.contract = Stock(TICKER, 'SMART', 'USD')
        self.is_position_open = False
        self.entry_price = None
        self.initial_stop = None
        self.profit_target = None
        self.pending_order = None

    async def get_ai_sentiment(self):
        """Calls Coze AI Skill for real-time news filter"""
        print(f"--- Calling AI Scout for {TICKER} ---")
        try:
            # Using non-streaming chat for structured decision
            chat = coze.chat.create(
                bot_id=COZE_BOT_ID,
                user_id="trader_01",
                additional_messages=[{"role": "user", "content": f"Analyze {TICKER} for today."}]
            )
            # Polling for completion (Coze 2026 pattern)
            while chat.status == "in_progress":
                await asyncio.sleep(1)
                chat = coze.chat.retrieve(chat_id=chat.id, conversation_id=chat.conversation_id)
            
            # Simple logic: Bot should return 'BULL' or 'BEAR' in response
            # In production, use the JSON prompt we discussed earlier
            messages = coze.conversations.messages.list(conversation_id=chat.conversation_id)
            ai_text = messages[0].content
            return 1.0 if "BULL" in ai_text.upper() else 0.0
        except Exception as e:
            print(f"AI Skill Error: {e}")
            return 0.5 # Neutral fallback

    async def run_strategy(self):
        """Main Loop: Check Technicals + AI Sentiment"""
        await self.ib.connectAsync('127.0.0.1', IB_PORT, clientId=1)
        print(f"Connected to IBKR. Monitoring {TICKER}...")
        positions = self.ib.positions()
        open_quantity = sum(
            float(position.position)
            for position in positions
            if position.contract.symbol == TICKER
        )
        self.is_position_open = open_quantity > 0
        if self.is_position_open:
            matching = [
                position for position in positions
                if position.contract.symbol == TICKER and position.position > 0
            ]
            self.entry_price = (
                float(matching[0].averageCost)
                if matching else None
            )
            print(f"Reconciled existing {TICKER} position: {open_quantity} shares")

        while True:
            try:
                # 1. Get Market Data (1-minute bars)
                bars = await self.ib.reqHistoricalDataAsync(
                    self.contract, endDateTime='', durationStr='1 D',
                    barSizeSetting='1 min', whatToShow='TRADES', useRTH=True)
                df = util.df(bars)

                # 2. Dynamic levels from recent 20-minute highs and lows
                if df.empty or len(df) < 2:
                    await asyncio.sleep(30)
                    continue
                prior_bars = df.iloc[:-1]
                current_price = float(prior_bars['close'].iloc[-1])
                if len(prior_bars) < LEVEL_LOOKBACK:
                    print(
                        f"Waiting for {LEVEL_LOOKBACK} bars to calculate levels "
                        f"(currently {len(prior_bars)})."
                    )
                    await asyncio.sleep(30)
                    continue

                breakout_level = prior_bars['high'].rolling(
                    LEVEL_LOOKBACK, min_periods=LEVEL_LOOKBACK
                ).max().iloc[-1]
                support_level = prior_bars['low'].rolling(
                    LEVEL_LOOKBACK, min_periods=LEVEL_LOOKBACK
                ).min().iloc[-1]

                reconciled_quantity = sum(
                    float(position.position)
                    for position in self.ib.positions()
                    if position.contract.symbol == TICKER
                )
                if reconciled_quantity <= 0 and self.is_position_open:
                    self.is_position_open = False
                    self.entry_price = None
                    self.initial_stop = None
                    self.profit_target = None
                elif reconciled_quantity > 0 and not self.is_position_open:
                    self.is_position_open = True
                
                # 3. AI Filter (Only check sentiment if close to breakout)
                sentiment_score = 0
                if current_price >= support_level:
                    sentiment_score = await self.get_ai_sentiment()

                # 4. Execution Logic (Long Only)
                print(
                    f"Price: {current_price:.2f} | Support: {support_level:.2f} | "
                    f"Breakout: {breakout_level:.2f} | AI Score: {sentiment_score}"
                )
                
                if not self.is_position_open:
                    if self.pending_order is not None:
                        pending_status = getattr(self.pending_order.orderStatus, "status", "Unknown")
                        if pending_status in {"Filled", "Cancelled", "ApiCancelled", "Inactive"}:
                            self.pending_order = None
                    if self.pending_order is None and current_price >= breakout_level and sentiment_score > 0.7:
                        print("🚀 SIGNAL: Technical Breakout + AI Confirmation!")
                        stop_distance = max(self.entry_price - support_level, 0.01) if self.entry_price else max(current_price - support_level, 0.01)
                        quantity = min(
                            MAX_SHARES,
                            max(1, int(MAX_RISK_DOLLARS / stop_distance)),
                        )
                        order = MarketOrder('BUY', quantity)
                        trade = self.ib.placeOrder(self.contract, order)
                        self.pending_order = trade
                        print(f"Order Placed: {trade}")
                        await asyncio.sleep(2)
                        fill_status = getattr(trade.orderStatus, "status", "Unknown")
                        filled_quantity = float(getattr(trade.orderStatus, "filled", 0) or 0)
                        if fill_status == "Filled" and filled_quantity > 0:
                            average_fill = float(getattr(trade.orderStatus, "avgFillPrice", 0) or current_price)
                            self.is_position_open = True
                            self.entry_price = average_fill
                            initial_risk = max(self.entry_price - support_level, 0.01)
                            self.initial_stop = max(support_level, self.entry_price - initial_risk)
                            self.profit_target = self.entry_price + (2 * initial_risk)
                            self.pending_order = None
                        else:
                            print(f"BUY not confirmed filled | status={fill_status} filled={filled_quantity}")
                
                # 5. Profit Taking / Trailing Stop Logic
                elif self.is_position_open:
                    if self.pending_order is not None:
                        pending_status = getattr(self.pending_order.orderStatus, "status", "Unknown")
                        if pending_status in {"Filled", "Cancelled", "ApiCancelled", "Inactive"}:
                            self.pending_order = None
                    if self.entry_price is None:
                        positions = self.ib.positions()
                        matching = [
                            position for position in positions
                            if position.contract.symbol == TICKER and position.position > 0
                        ]
                        self.entry_price = float(matching[0].averageCost) if matching else current_price
                    if self.initial_stop is None:
                        initial_risk = max(self.entry_price - support_level, 0.01)
                        self.initial_stop = max(support_level, self.entry_price - initial_risk)
                        self.profit_target = self.entry_price + (2 * initial_risk)
                    trailing_stop = max(self.initial_stop, support_level)

                    if current_price >= self.profit_target:
                        print("💰 Target Hit. Closing Position.")
                        matching_quantity = abs(sum(
                            float(position.position)
                            for position in self.ib.positions()
                            if position.contract.symbol == TICKER
                        ))
                        if self.pending_order is not None or matching_quantity <= 0:
                            await asyncio.sleep(30)
                            continue
                        trade = self.ib.placeOrder(self.contract, MarketOrder('SELL', int(matching_quantity)))
                        self.pending_order = trade
                        await asyncio.sleep(2)
                        if getattr(trade.orderStatus, "status", "") == "Filled":
                            self.is_position_open = False
                            self.entry_price = None
                            self.initial_stop = None
                            self.profit_target = None
                            self.pending_order = None
                    elif current_price < trailing_stop:
                        print("🚨 Stop Hit. Protecting Capital.")
                        matching_quantity = abs(sum(
                            float(position.position)
                            for position in self.ib.positions()
                            if position.contract.symbol == TICKER
                        ))
                        if self.pending_order is not None or matching_quantity <= 0:
                            await asyncio.sleep(30)
                            continue
                        trade = self.ib.placeOrder(self.contract, MarketOrder('SELL', int(matching_quantity)))
                        self.pending_order = trade
                        await asyncio.sleep(2)
                        if getattr(trade.orderStatus, "status", "") == "Filled":
                            self.is_position_open = False
                            self.entry_price = None
                            self.initial_stop = None
                            self.profit_target = None
                            self.pending_order = None

                await asyncio.sleep(30) # Refresh every 30 seconds

            except Exception as e:
                print(f"Loop Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = INTCBot()
    asyncio.run(bot.run_strategy())