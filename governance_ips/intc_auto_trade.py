import asyncio
import os
import pandas as pd
import pandas_ta as ta
from ib_async import *
from cozepy import Coze, TokenAuth, COZE_CN_BASE_URL
from dotenv import load_dotenv

# 1. Environment & Config
load_dotenv()
COZE_TOKEN = os.getenv("pat_7wpwlmMcgmuKlJUyIzJ66nQuzQc2ITgLA8qEJxC7gAfF03OCSM8MwgDOOYejQMtH") # Add to your .env file
COZE_BOT_ID = os.getenv("COZE_BOT_ID")
IB_PORT = 4002  # TWS Paper Trading
TICKER = 'INTC'

# Initialize Coze Client
coze = Coze(auth=TokenAuth(COZE_TOKEN), base_url=COZE_CN_BASE_URL)

class INTCBot:
    def __init__(self):
        self.ib = IB()
        self.contract = Stock(TICKER, 'SMART', 'USD')
        self.is_position_open = False

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

        while True:
            try:
                # 1. Get Market Data (1-minute bars)
                bars = await self.ib.reqHistoricalDataAsync(
                    self.contract, endDateTime='', durationStr='1 D',
                    barSizeSetting='1 min', whatToShow='TRADES', useRTH=True)
                df = util.df(bars)

                # 2. Technical Breakout Logic ($135 Trigger)
                current_price = df['close'].iloc[-1]
                breakout_level = 135.20
                
                # 3. AI Filter (Only check sentiment if close to breakout)
                sentiment_score = 0
                if current_price > 134:
                    sentiment_score = await self.get_ai_sentiment()

                # 4. Execution Logic (Long Only)
                print(f"Price: {current_price} | AI Score: {sentiment_score}")
                
                if not self.is_position_open:
                    if current_price >= breakout_level and sentiment_score > 0.7:
                        print("🚀 SIGNAL: Technical Breakout + AI Confirmation!")
                        order = MarketOrder('BUY', 100) # Use LimitOrder for better fills
                        trade = self.ib.placeOrder(self.contract, order)
                        self.is_position_open = True
                        print(f"Order Placed: {trade}")
                
                # 5. Profit Taking / Trailing Stop Logic
                elif self.is_position_open:
                    if current_price >= 141.50:
                        print("💰 Target Hit. Closing Position.")
                        self.ib.placeOrder(self.contract, MarketOrder('SELL', 100))
                        self.is_position_open = False
                    elif current_price < 133.50: # Trailing Stop
                        print("🚨 Stop Hit. Protecting Capital.")
                        self.ib.placeOrder(self.contract, MarketOrder('SELL', 100))
                        self.is_position_open = False

                await asyncio.sleep(30) # Refresh every 30 seconds

            except Exception as e:
                print(f"Loop Error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    bot = INTCBot()
    asyncio.run(bot.run_strategy())