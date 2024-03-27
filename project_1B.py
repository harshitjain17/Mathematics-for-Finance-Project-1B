# Import libraries
import numpy as np
import math

# Parameters
K = 10                                          # Strike price of option
r = 0.02                                        # Constant risk-free interest rate
sigma = 0.25                                    # Constant volatility of the stock price
T = 0.25                                        # Time to maturity
S0 = 10                                         # Current stock price

# Function to calculate u and d for the binomial tree
def calculated_u_d(sigma, T, N):
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))         # Compute u
    d = 1 / u                                   # Compute d
    return u, d

# Function to calculate option price using binomial tree method
def calculate_price_binomial_tree(N):
    dt = T / N
    u, d = calculated_u_d(sigma, T, N)
    p = (math.exp(r * dt) - d) / (u - d)                # Probability of up movement
    stock_price = [0] * (N + 1)                         # Initialize the stock price at maturity
    option_value = [0] * (N + 1)                        # Initialize the option values at maturity
    
    # Loop through each time step
    for i in range(N + 1):
        stock_price[i] = S0 * (u ** (N - i)) * (d ** i) # Calculate stock price at maturity
        option_value[i] = max(stock_price[i] - K, 0)    # Calculate option value at maturity
    
    # Calculate option values at earlier time steps using backward recursion
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            option_value[i] = math.exp(-r * dt) * (p * option_value[i] + (1 - p) * option_value[i + 1]) # Calculate option value using the binomial tree method
    
    return option_value[0]

