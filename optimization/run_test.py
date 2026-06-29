import numpy as np
from scipy.stats import norm
import pandas as pd

# --- BLACK-SCHOLES PRICING ENGINE ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    S: Current Stock Price
    K: Strike Price
    T: Time to Expiry (in years)
    r: Risk-free interest rate (e.g., 0.0525 for 5.25%)
    sigma: Implied Volatility (e.g., 0.30 for 30%)
    """
    if T <= 0: return max(0, S - K) if option_type == 'call' else max(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# --- 1. INITIAL SETUP (JUNE 29, 2026) ---
spot_price = 130.66
strike_call = 138.00
strike_put = 124.00
expiry_days = 18 / 365  # July 17 Expiry (roughly 18 days away)
risk_free_rate = 0.0525 # 5.25% (Warsh-era hawkish rates)
initial_iv = 0.32       # 32% IV (Dormant/Low Vol)

# Calculate entry cost
call_entry = black_scholes(spot_price, strike_call, expiry_days, risk_free_rate, initial_iv, 'call')
put_entry = black_scholes(spot_price, strike_put, expiry_days, risk_free_rate, initial_iv, 'put')
total_entry_cost = call_entry + put_entry

print(f"--- ENTRY (June 29) ---")
print(f"INTC Price: ${spot_price}")
print(f"Strangle Cost: ${total_entry_cost:.2f} (Call: ${call_entry:.2f}, Put: ${put_entry:.2f})")

# --- 2. SCENARIO A SIMULATION (THE POP) ---
# Assume 10 days pass, price hits $142, and IV spikes to 48% due to the "Tail Risk" triggering.
days_passed = 10
new_spot = 142.00
new_expiry_days = (18 - days_passed) / 365
new_iv = 0.48  # The "Vega Spike" mentioned in the analysis

call_exit = black_scholes(new_spot, strike_call, new_expiry_days, risk_free_rate, new_iv, 'call')
put_exit = black_scholes(new_spot, strike_put, new_expiry_days, risk_free_rate, new_iv, 'put')
total_exit_value = call_exit + put_exit

# --- 3. RESULTS ---
pnl_dollar = total_exit_value - total_entry_cost
pnl_percent = (pnl_dollar / total_entry_cost) * 100

print(f"\n--- EXIT SCENARIO A (10 Days Later) ---")
print(f"New INTC Price: ${new_spot}")
print(f"New IV: {new_iv*100}% (Volatility Crush/Spike)")
print(f"Strangle Value: ${total_exit_value:.2f} (Call: ${call_exit:.2f}, Put: ${put_exit:.2f})")
print(f"Net P&L: ${pnl_dollar:.2f}")
print(f"Percent Return: {pnl_percent:.2f}%")

# --- 4. ALTERNATE DOWNSIDE POP ---
new_spot_down = 118.00
call_exit_down = black_scholes(new_spot_down, strike_call, new_expiry_days, risk_free_rate, new_iv, 'call')
put_exit_down = black_scholes(new_spot_down, strike_put, new_expiry_days, risk_free_rate, new_iv, 'put')
total_exit_down = call_exit_down + put_exit_down
pnl_down = ((total_exit_down - total_entry_cost) / total_entry_cost) * 100

print(f"\n--- EXIT SCENARIO A (Downside Pop to $118) ---")
print(f"Percent Return: {pnl_down:.2f}%")