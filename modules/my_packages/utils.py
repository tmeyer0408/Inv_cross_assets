import pandas as pd
import numpy as np
from .metrics_numba import tracking_error_numba, tracking_error_series_numba, transaction_cost_numba



class Utilities:
    """
    Class that holds utility methods for the various strategies used in the project
    """

    def __init__(self):
        pass

    @staticmethod
    def compute_returns(data:pd.DataFrame, fill_method:str=None):
        returns = data.pct_change(fill_method=fill_method)
        returns = returns.iloc[1:, :]
        return returns
    
    @staticmethod
    def normalize_weights(weights):
        """
        Normalize a weight vector so that their sum equals 1.
        """
        total = np.sum(weights)
        if total == 0:
            return weights
        return weights / total


    @staticmethod
    def information_ratio_ex_ante(weights, universe_returns, benchmark_returns, previous_weights, fee_rate, core_portfolio_weight):
        """
        Calculate the ex-ante information ratio adjusted for fees.
        """
        portfolio_returns = np.sum(universe_returns * weights, axis=1) # Compute the portfolio returns with the given weights
        excess_return = np.mean(portfolio_returns) - np.mean(benchmark_returns) # Compute the excess return

        # Compute the tracking error with numba method
        te = tracking_error_numba(np.asarray(weights, dtype=np.float64),
                                           np.asarray(universe_returns, dtype=np.float64),
                                             np.asarray(benchmark_returns, dtype=np.float64))
        
        return -(excess_return - Utilities.transaction_cost(weights, previous_weights, fee_rate, core_portfolio_weight)) / te # Return the information ratio adjusted for fees

    @staticmethod
    def information_ratio_ex_post(portfolio_returns, benchmark_returns):
        """
        Calculate the ex-post information ratio from historical returns.
        """
        # Convert the lists to numpy arrays for numba method
        portfolio_returns = np.array(portfolio_returns)
        benchmark_returns = np.array(benchmark_returns)
        if portfolio_returns.shape[0] != benchmark_returns.shape[0]:
            raise ValueError("# The series must have the same length.")
        
        excess_returns = portfolio_returns - benchmark_returns # Compute the excess returns
        std_excess = np.std(excess_returns) # Compute the standard deviation of the excess returns
        if std_excess == 0:
            return np.nan
        return np.mean(excess_returns) / std_excess # Return the information ratio

    @staticmethod
    def align_dataframes(benchmark_df, universe_df):
        """
        Align the DataFrames of the benchmark and the universe on their common dates.
        """
        common_dates = benchmark_df.index.intersection(universe_df.index) # Get the common dates
        benchmark_aligned = benchmark_df.loc[common_dates] # Align the benchmark
        universe_aligned = universe_df.loc[common_dates] # Align the universe
        return benchmark_aligned, universe_aligned

    @staticmethod
    def tracking_error_ex_ante(weights, universe_returns, benchmark_returns):
        """
        Wrapper around the Numba function to calculate tracking error ex ante.
        """
        return tracking_error_numba(np.asarray(weights, dtype=np.float64),
                                    np.asarray(universe_returns, dtype=np.float64),
                                    np.asarray(benchmark_returns, dtype=np.float64))

    @staticmethod
    def tracking_error_ex_post(portfolio_returns, benchmark_returns):
        """
        Wrapper around the Numba function to calculate tracking error ex post.
        """
        return tracking_error_series_numba(np.asarray(portfolio_returns, dtype=np.float64),
                                           np.asarray(benchmark_returns, dtype=np.float64))

    @staticmethod
    def transaction_cost(weights, previous_weights, fee_rate, core_portfolio_weight):
        """
        Wrapper around the Numba function to calculate transaction fees.
        """
        
        return transaction_cost_numba(np.asarray(weights, dtype=np.float64),
                                      np.asarray(previous_weights, dtype=np.float64),
                                      np.asarray(fee_rate, dtype=np.float64),
                                      core_portfolio_weight)
    
    @staticmethod
    def round_and_renormalize(weights, threshold=1e-3):
        """
        Round the weights to zero if they are below a certain threshold and renormalize them."
        """
        w_rounded = np.where(weights < threshold, 0, weights)
        # Calcule la somme après mise à 0
        s = w_rounded.sum()
        if s > 0:
            w_rounded = w_rounded / s
        else:
            pass
        return w_rounded
    
    @staticmethod
    def calculate_management_fees(weights_df: pd.DataFrame, fees_df, use_compounding: bool = True) -> pd.Series:
        """
        Calculates the management fees incurred by the portfolio.
        """

        # Create a DataFrame to store the daily fee rates for each asset
        daily_fee_rates = pd.DataFrame(index=weights_df.index, columns=weights_df.columns, dtype=float)
        for asset in weights_df.columns:
            if asset in fees_df.index:
                annual_fee_rate = fees_df.loc[asset].iloc[0]
                if use_compounding:
                    # Convert the annual fee rate to a daily rate using the formula for compounding
                    daily_fee_rate = (1 + annual_fee_rate)**(1/252) - 1
                else:
                    # Convert the annual fee rate to a daily rate using simple division
                    daily_fee_rate = annual_fee_rate / 252
            else:
                daily_fee_rate = 0.0  
            daily_fee_rates[asset] = daily_fee_rate  

        # Calculate the daily fees incurred by the portfolio, multiply the daily fee rates by the weights for each asset and sum across assets
        daily_fees = (weights_df * daily_fee_rates).sum(axis=1)

        #Peut à a revoir ça
        annual_average_fee = daily_fees.mean() * 252
        print("Frais annuels moyens supportés : {:.4%}".format(annual_average_fee))

        return daily_fees

    @staticmethod
    def calculate_transaction_fees(weights_df: pd.DataFrame, transaction_fee_rate = 0.010) -> pd.Series:
        """
        Calculates the transaction fees incurred by the portfolio based on the daily weights matrix.
        """
        
   
        fee_rate = pd.Series(transaction_fee_rate, index=weights_df.columns)
        weight_changes = weights_df.diff()
        weight_changes.iloc[0] = weights_df.iloc[0].abs()
        fee_matrix = weight_changes.multiply(fee_rate, axis=1)
        daily_transaction_fees = fee_matrix.sum(axis=1)
        annual_average_fee = daily_transaction_fees.mean() * 252
        print("Frais annuels moyens de transaction supportés : {:.4%}".format(annual_average_fee))

        return daily_transaction_fees


