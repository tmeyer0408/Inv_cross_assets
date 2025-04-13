import pandas as pd
import numpy as np

class PerformanceAnalyser:
    """Class to analyse the performance of a strategy"""
    def __init__(self, portfolio_returns:pd.DataFrame):
        self.portfolio_returns = portfolio_returns
        self.cumulative_performance = None
        self.equity_curve = None

    def compute_cumulative_performance(self, compound_type:str="geometric"):
        """Compute the cumulative performance of the strategy"""
        if compound_type == "geometric":
            self.cumulative_performance = (1 + self.portfolio_returns).cumprod() - 1
        elif compound_type == "arithmetic":
            self.cumulative_performance = self.portfolio_returns.cumsum()
        else:
            raise ValueError("Compound type not supported")

        return self.cumulative_performance

    def compute_equity_curve(self):
        """Compute the equity curve of the strategy"""
        self.equity_curve = self.compute_cumulative_performance(compound_type="arithmetic")
        return self.equity_curve

    def compute_metrics(self):
        """Compute the performance metrics of the strategy"""
        if self.cumulative_performance is None:
            self.compute_cumulative_performance()

        # Compute basic performance metrics
        total_return = self.cumulative_performance.iloc[-1, 0]
        annualized_return = (1 + total_return) ** (252 / len(self.portfolio_returns)) - 1
        volatility = self.portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility

        # Maximum drawdown
        rolling_max = self.cumulative_performance.cummax()
        drawdown = (self.cumulative_performance / rolling_max) - 1
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

