# Import libraries
import numpy as np
import math

# Parameters
K = 10 # Strike price of option
r = 0.02 # Constant risk-free interest rate
sigma = 0.25 # Constant volatility of the stock price
T = 0.25 # Time to maturity
S0 = 10 # Current stock price

# Function to calculate u and d for the binomial tree
def calc_u_d(sigma, T, N):
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt)) # Compute u
    d = 1 / u # Compute d
    return u, d