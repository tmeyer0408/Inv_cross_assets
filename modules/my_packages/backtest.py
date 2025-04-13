from typing import Union
import pandas as pd
import numpy as np

class Backtest:
    """Class to backtest a strategy"""
    def __init__(self, returns:pd.DataFrame, weights:pd.DataFrame, rebal_periods:Union[int, None]=None, transaction_cost_bps:int=0,
                 first_rebal_date_idx:int = 3): # 3 because the first 3 rows of the weights as 0 as it requires 3 periods to compute the weights (rolling vol)
        self.returns = returns
        self.weights = weights
        self.portfolio_returns = None
        self.rebal_periods = rebal_periods
        self.transaction_cost_bps = transaction_cost_bps
        self.transaction_cost_vect = pd.DataFrame(data=np.nan, index=self.weights.index, columns=['transaction_costs'])
        self.first_rebal_date_idx = first_rebal_date_idx
        self.rebal_dates = self.weights.index[self.first_rebal_date_idx::self.rebal_periods]

    def run_backtest(self):
        """Run the backtest"""
        self.portfolio_returns = (pd.DataFrame(
            (self.returns * self.weights).sum(axis=1),
        columns=['portfolio_returns'])
        )

        if (self.transaction_cost_bps != 0) and (self.rebal_periods is not None):
            for t in self.weights.index[self.first_rebal_date_idx:]:
                if t in self.rebal_dates:
                    # rebalancing date
                    delta_weights = self.weights.loc[t, :] # no previous weights, initial buy
                else:
                    t_idx = self.weights.index.get_loc(t)
                    delta_weights = self.weights.loc[t, :] - self.weights.iloc[t_idx-1, :]
                gross_returns = self.portfolio_returns.loc[t,:]
                volume_exchanged = np.sum(np.abs(delta_weights))
                total_cost = self.transaction_cost_bps * volume_exchanged
                self.portfolio_returns.loc[t, :] = self.portfolio_returns.loc[t, :] - total_cost/10000

        return self.portfolio_returns

    def get_results(self):
        """Get the backtest results"""
        if self.portfolio_returns is None:
            self.run_backtest()
        return self.portfolio_returns

    def compute_management_fees(self, file_path:str=r".\data\data_ETFs_funds_final.xlsx", sheet_name:str="management_fees"):
        fees = pd.read_excel(file_path, sheet_name=sheet_name).values
        weights = self.weights.values
        managemement_fees = (fees/252) * weights
        managemement_fees = pd.DataFrame(data=managemement_fees.sum(axis=1), index=self.weights.index, columns=['management_fees'])
        return managemement_fees
