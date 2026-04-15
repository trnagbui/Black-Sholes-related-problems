import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# 1. Define the known parameters at Year 1
C_target = 2.0
K = 40.0
T = 1.0  # 1 year remaining
r = 0.08
delta = 0.0
sigma = 0.3

# 2. Define the Black-Scholes Call Formula
def bs_call(S):
    d1 = (np.log(S / K) + (r - delta + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-delta * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# 3. Create an "Objective Function"
# We want this function to equal 0. So we take the BS Formula output and subtract our target (2.00).
def objective(S):
    return bs_call(S) - C_target

# 4. Run the Root-Finding Algorithm
# This tells the computer to test stock prices between 0.00001 and 100 until the objective function equals 0.
S_star = brentq(objective, 1e-5, 100)

print(f"Calculated Stock Price: {S_star}")
