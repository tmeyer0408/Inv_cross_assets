from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Union

class WeightingScheme(ABC):
    """Abstract class to define the interface for the weighting scheme"""
    def __init__(self, returns:pd.DataFrame, signals:pd.DataFrame, rebal_periods:Union[int, None], portfolio_type:str="long_only"):
        self.returns = returns
        self.signals = signals
        if not isinstance(rebal_periods, (int,type(None))):
            raise ValueError("rebal_periods must be an int or None.")
        self.rebal_periods = rebal_periods
        self.portfolio_type = portfolio_type
        self.weights = None
        self.rebalanced_weights = None

    @abstractmethod
    def compute_weights(self):
        """Compute the weights for the strategy"""
        pass

    def get_weights(self):
        """Get the computed weights"""
        if self.weights is None:
            self.compute_weights()
        return self.weights

class EqualWeightingScheme(WeightingScheme):
    """Class to implement the equal weighting scheme"""
    def __init__(self, returns:pd.DataFrame, signals:pd.DataFrame, rebal_periods:int=0, portfolio_type:str="long_only",
                 min_periods_rolling_vol:int=2):
        super().__init__(returns, signals, rebal_periods, portfolio_type)
        self.min_periods_rolling_vol = min_periods_rolling_vol
        self.first_date_weights = None
        self.first_date_idx_weights = None
        self.rebal_dates = None


    def compute_weights(self):
        """Compute the weights for the equal weighting scheme"""
        number_of_positive_signals = (self.signals == 1).sum(axis=1)
        number_of_negative_signals = (self.signals == -1).sum(axis=1)

        weights = pd.DataFrame(data=0.0, index=self.signals.index, columns=self.signals.columns)

        if self.portfolio_type == "long_only":
            # Check that there is at least one positive signal at each date
            if (number_of_positive_signals > 0).sum() == number_of_positive_signals.shape[0]:
                weights[self.signals == 1] = self.signals[self.signals == 1].div(number_of_positive_signals, axis=0)
            else:
                # raise ValueError("Error division by 0. There is no positive signals for some dates.")
                # Calculer les poids là où il y a au moins un signal positif sur la ligne
                mask_valid = number_of_positive_signals > 0
                weights[self.signals == 1] = self.signals[self.signals == 1].div(
                    number_of_positive_signals.where(mask_valid), axis=0)

                # Mettre les poids à 0 pour les lignes où tous les signaux sont manquants (aucun 1)
                weights.loc[~mask_valid, :] = 0

                print("For some dates, there are no signals. Weights set to 0 by default.")

        elif self.portfolio_type == "short_only":
            # Check that there is at least one negative signal at each date
            if (number_of_negative_signals > 0).sum() == number_of_negative_signals.shape[0]:
                weights[self.signals == -1] = self.signals[self.signals == -1].div(number_of_negative_signals, axis=0)
            else:
                raise ValueError("Error division by 0. There is no negative signals for some dates.")

        elif self.portfolio_type == "long_short":
            # Check that there is at least one positive and one negative signal at each date
            if ((number_of_positive_signals > 0).sum() == number_of_positive_signals.shape[0] and
                (number_of_negative_signals > 0).sum() == number_of_negative_signals.shape[0]):
                weights[self.signals == 1] = self.signals[self.signals == 1].div(number_of_positive_signals, axis=0)
                weights[self.signals == -1] = self.signals[self.signals == -1].div(number_of_negative_signals, axis=0)
            else:
                raise ValueError("Error division by 0. There is no positive or negative signal for some dates.")

        else:
            raise ValueError("portfolio_type not supported")

        first_date_weights = weights.first_valid_index()
        self.first_date_weights = first_date_weights
        self.first_date_idx_weights = weights.index.get_loc(first_date_weights)
        self.weights = weights.fillna(0)  # Ne pas utiliser inplace=True
        return self.weights

    def rebalance_portfolio(self):
        if self.weights is None:
            self.weights = self.compute_weights()

        if (self.rebalanced_weights is None) and (self.rebal_periods is not None):
            self.rebalanced_weights = pd.DataFrame(data=self.weights, index=self.weights.index,
                                                   columns=self.weights.columns)
            self.rebal_dates = self.weights.index[self.first_date_idx_weights::self.rebal_periods]
            for t in self.weights.index[self.first_date_idx_weights:]:
                if t in self.rebal_dates:
                    self.rebalanced_weights.loc[t, :] = self.weights.loc[t, :]
                else:
                    # non rebalancing date, weights must derive
                    g = (1 + self.returns.loc[t, :])
                    t_idx = self.rebalanced_weights.index.get_loc(t)
                    w = self.rebalanced_weights.iloc[t_idx - 1] * g
                    w = w / w.sum()
                    self.rebalanced_weights.loc[t, :] = w

            self.rebalanced_weights = self.rebalanced_weights.fillna(0.0)
            return self.rebalanced_weights


class NaiveRiskParity(WeightingScheme):
    """Class to implement the naive risk parity weighting scheme"""

    def __init__(self, returns: pd.DataFrame, signals: pd.DataFrame, rebal_periods: int = 0, portfolio_type: str = "long_only",
                 vol_lookback: int = 252, min_periods_rolling_vol: int = 2):
        """
        Initialize the naive risk parity model.

        Parameters:
        - returns (pd.DataFrame): DataFrame of asset returns.
        - signals (pd.DataFrame): DataFrame of trading signals (1 for long, -1 for short, 0 for neutral).
        - rebal_periods (int): Rebalancing frequency (default: 0 for daily).
        - portfolio_type (str): "long_only", "short_only", or "long_short".
        - vol_lookback (int): Lookback period for volatility calculation (default: 252 days ~ 1 year).
        """
        super().__init__(returns, signals, rebal_periods, portfolio_type)
        self.vol_lookback = vol_lookback
        self.min_periods_rolling_vol = min_periods_rolling_vol
        self.first_date_weights = None
        self.first_date_idx_weights = None
        self.rebal_dates = None

    def compute_weights(self):
        """Compute weights using the inverse volatility method."""
        rolling_vol = self.returns.rolling(window=self.vol_lookback, min_periods=self.min_periods_rolling_vol).std()
        inv_vol = 1 / rolling_vol  # Inverse of volatility
        inv_vol.replace([np.inf, -np.inf], np.nan, inplace=True)  # Handle division by zero
        inv_vol = inv_vol.fillna(0)  # Replace NaN values

        # Mask to apply signals
        number_of_positive_signals = (self.signals == 1).sum(axis=1)
        number_of_negative_signals = (self.signals == -1).sum(axis=1)

        weights = pd.DataFrame(data=0.0, index=self.signals.index, columns=self.signals.columns)

        if self.portfolio_type == "long_only":
            valid_mask = number_of_positive_signals > 0
            masked_inv_vol = inv_vol * (self.signals == 1)
            weights = masked_inv_vol.div(masked_inv_vol.sum(axis=1), axis=0)
            weights.loc[~valid_mask, :] = 0  # Set weights to 0 if no valid signals

        elif self.portfolio_type == "short_only":
            valid_mask = number_of_negative_signals > 0
            masked_inv_vol = inv_vol * (self.signals == -1)
            weights = masked_inv_vol.div(masked_inv_vol.sum(axis=1), axis=0)
            weights.loc[~valid_mask, :] = 0

        elif self.portfolio_type == "long_short":
            valid_mask = (number_of_positive_signals > 0) & (number_of_negative_signals > 0)
            long_weights = inv_vol * (self.signals == 1)
            short_weights = inv_vol * (self.signals == -1)

            long_weights = long_weights.div(long_weights.sum(axis=1), axis=0)
            short_weights = short_weights.div(short_weights.sum(axis=1), axis=0)

            weights = long_weights - short_weights
            weights.loc[~valid_mask, :] = 0

        else:
            raise ValueError("portfolio_type not supported")

        first_date_weights = weights.first_valid_index()
        self.first_date_weights = first_date_weights
        self.first_date_idx_weights = weights.index.get_loc(first_date_weights)
        self.weights = weights.fillna(0)  # Ne pas utiliser inplace=True
        return self.weights

    def rebalance_portfolio(self):
        if self.weights is None:
            self.weights = self.compute_weights()

        if (self.rebalanced_weights is None) and (self.rebal_periods is not None):
            self.rebalanced_weights = pd.DataFrame(data=self.weights, index=self.weights.index, columns=self.weights.columns)
            self.rebal_dates = self.weights.index[self.first_date_idx_weights::self.rebal_periods]
            for t in self.weights.index[self.first_date_idx_weights:]:
                if t in self.rebal_dates:
                    self.rebalanced_weights.loc[t, :] = self.weights.loc[t, :]
                else:
                    # non rebalancing date, weights must derive
                    g = (1+self.returns.loc[t,:])
                    t_idx = self.rebalanced_weights.index.get_loc(t)
                    w = self.rebalanced_weights.iloc[t_idx-1] * g
                    w = w / w.sum()
                    self.rebalanced_weights.loc[t,:] = w

            self.rebalanced_weights = self.rebalanced_weights.fillna(0.0)
            return self.rebalanced_weights
