from numba import njit
import numpy as np




"""

Functions compiled using the Numba package to accelerate the methods used during optimizations. 
Numba is a just-in-time compiler that translates a subset of Python and NumPy code into optimized machine code at runtime, significantly boosting the performance of numerical computations. 
This acceleration is especially beneficial for iterative and computationally intensive optimization routines.

"""


@njit
def tracking_error_numba(weights, universe_returns, benchmark_returns):
    """
    Calculates the tracking error (standard deviation of the difference between the portfolio's return and the benchmark) using explicit loops to benefit from Numba's JIT optimization.
    """
    
    weights = np.asarray(weights, dtype=np.float64)
    universe_returns = np.asarray(universe_returns, dtype=np.float64)
    benchmark_returns = np.asarray(benchmark_returns, dtype=np.float64)

    n = universe_returns.shape[0]  # Number of days
    m = universe_returns.shape[1]  # Number of assets
    portfolio_returns = np.empty(n) 

    # daily portfolio returns
    for i in range(n):
        s = 0.0
        for j in range(m):
            s += universe_returns[i, j] * weights[j]  # Dot product of the weights and the returns
        portfolio_returns[i] = s

    # compute the difference between the portfolio and benchmark returns
    diff = np.empty(n)
    for i in range(n):
        diff[i] = portfolio_returns[i] - benchmark_returns[i]

    # compute the mean of the differences
    mean_diff = 0.0
    for i in range(n):
        mean_diff += diff[i]
    mean_diff /= n

    # Compute the variance
    var = 0.0
    for i in range(n):
        var += (diff[i] - mean_diff) ** 2
    var /= n

    # return the square root of the variance
    return np.sqrt(var)

@njit
def tracking_error_series_numba(portfolio_returns, benchmark_returns):
    """
    Calculates the tracking error on 1D series.
    """

    portfolio_returns = np.asarray(portfolio_returns, dtype=np.float64)
    benchmark_returns = np.asarray(benchmark_returns, dtype=np.float64)

    n = portfolio_returns.shape[0]
    diff = np.empty(n)

    #Calculation of the day-to-day return difference.
    for i in range(n):
        diff[i] = portfolio_returns[i] - benchmark_returns[i]
    
    #Calculates the average of these differences.
    mean_diff = 0.0
    for i in range(n):
        mean_diff += diff[i]
    mean_diff /= n

    # Compute the variance
    var = 0.0
    for i in range(n):
        var += (diff[i] - mean_diff) ** 2
    var /= n

    return np.sqrt(var) #Return the square root of the variance


@njit
def transaction_cost_numba(weights, previous_weights, fee_rate, core_portfolio_weight):
    """
    Calculates the transaction fees using NumPy arrays.
    """

    weights = np.asarray(weights, dtype=np.float64)
    previous_weights = np.asarray(previous_weights, dtype=np.float64)
    fee_rate = np.asarray(fee_rate, dtype=np.float64)
    
    m = weights.shape[0]
    cost = 0.0
    
    #Calculates the transaction cost for each
    for i in range(m):
        cost += core_portfolio_weight * abs(weights[i] - previous_weights[i]) * fee_rate[i]
    return cost