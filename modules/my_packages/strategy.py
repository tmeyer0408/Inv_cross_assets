from abc import ABC, abstractmethod
from typing import Union, Tuple
import pandas as pd
import numpy as np
from .strategy_utilities import MomentumStrategyUtilities


class Strategy(ABC):
    """Abstract class to define the interface for the strategy"""
    def __init__(self, returns:pd.DataFrame):
        self.returns = returns
        self.signals = None

    @abstractmethod
    def compute_signals(self):
        """Compute the signals for the strategy"""
        pass

    def get_signals(self):
        """Get the computed signals"""
        if self.signals is None:
            self.compute_signals()
        return self.signals

class BuyAndHold(Strategy):
    """Class to implement the buy and hold strategy"""
    def compute_signals(self):
        """Compute the signals for the buy and hold strategy"""
        self.signals = (~np.isnan(self.returns)).astype(int)
        self.signals.iloc[0,:] = 1  # First signal is 1, to avoid 0 in the weights in the first line because it'll be
        # this weight up to the first rebalancing date
        return self.signals

class Momentum(Strategy):
    def __init__(self, returns: pd.DataFrame, lookback: int = 252, skip_periods: int = 21):
        """
        Initializes the Momentum strategy.

        Parameters:
        - returns: pd.DataFrame, DataFrame of asset returns.
        - lookback: int, the lookback period to compute momentum (e.g., 252 for 12 months).
        - skip_periods: int, the number of periods to exclude at the end (e.g., 21 for 1 month).
        """
        super().__init__(returns)
        self.lookback = lookback
        self.skip_periods = skip_periods

    def compute_signals(self):
        """
        Compute the momentum signals for each asset in the DataFrame.

        A signal of 1 indicates a positive momentum, 0 indicates a negative or neutral momentum.
        """
        signals = pd.DataFrame(index=self.returns.index, columns=self.returns.columns)

        for col in self.returns.columns:  # Boucle sur chaque actif
            asset_returns = self.returns[col]

            cumulative_returns = []
            for i in range(len(asset_returns)):
                # On prend les `lookback` dernières valeurs, en excluant les `skip_periods` dernières
                if i >= self.lookback:
                    window = asset_returns.iloc[i - self.lookback:i - self.skip_periods]
                    window = window.dropna()  # Supprime les NaN avant calcul

                    if len(window) > 0:
                        cum_return = np.prod(1 + window) - 1
                    else:
                        cum_return = 0  # Si toutes les valeurs sont NaN, on met 0 par défaut
                else:
                    cum_return = 0  # Pas assez de données

                cumulative_returns.append(cum_return)

            signals[col] = (np.array(cumulative_returns) > 0).astype(int)  # 1 si momentum positif, sinon 0

        self.signals = signals
        return self.signals

class CrossSectionalMomentum(Strategy):
    def __init__(self, returns: pd.DataFrame, df_prices:pd.DataFrame, window_mom:int, nb_period_to_exclude:Union[None, int],
                 exclude_last_period:bool, percentiles:Tuple[Union[int, float], Union[int, float]], rolling_window: int=2):
        # Calling parent constructor
        super().__init__(returns=returns)

        # Check new attributes
        if not isinstance(df_prices, pd.DataFrame):
            raise ValueError("df must be a pandas dataframe.")
        if not isinstance(rolling_window, int):
            raise ValueError("window must be an int.")
        if rolling_window > df_prices.shape[0]:
            raise ValueError("rolling_window must be < df_prices.shape[0]")
        if not isinstance(window_mom, int):
            raise ValueError("window_mom must be an int.")
        if not isinstance(percentiles, tuple) and len(percentiles) == 2 and all(isinstance(p, (int, float)) for p in percentiles):
            raise ValueError("percentiles must be a tuple of length 2 and of int or float.")


        # Assigning new attributes
        self.df_prices = df_prices
        self.rolling_window = rolling_window
        self.window_mom = window_mom
        self.percentiles = percentiles
        self.nb_period_to_exclude = nb_period_to_exclude
        self.exclude_last_period = exclude_last_period


    def compute_signals(self):
        """
        This functions returns a dataframe of signals (-1 or 1) based on percentiles of rolling momentum
        """
        rolling_mom = MomentumStrategyUtilities.rolling_momentum(df=self.df_prices, nb_period=self.window_mom, nb_period_to_exclude=self.nb_period_to_exclude,
                                                exclude_last_period=self.exclude_last_period)

        # To store
        signal = pd.DataFrame(np.nan, index=rolling_mom.index, columns=rolling_mom.columns)
        for i_end in range(self.rolling_window, rolling_mom.shape[0]+1):
            print(f"computing CSMOM signal: loop {i_end} out of {rolling_mom.shape[0]}")
            # Current window (1 row)
            local_rolling_mom = rolling_mom.iloc[i_end-self.rolling_window:i_end,]

            # Percentiles
            lower_pct = np.nanpercentile(rolling_mom.values.flatten(), self.percentiles[0], axis=0)
            upper_pct = np.nanpercentile(rolling_mom.values.flatten(), self.percentiles[1], axis=0)

            # Formatting for comparison
            lower_pct_df = pd.DataFrame(np.tile(lower_pct, reps=(1, local_rolling_mom.shape[1])), index=local_rolling_mom.index,
                                        columns=local_rolling_mom.columns)
            upper_pct_df = pd.DataFrame(np.tile(upper_pct, reps=(1, local_rolling_mom.shape[1])), index=local_rolling_mom.index,
                                        columns=local_rolling_mom.columns)

            # Signal
            mask_lower = local_rolling_mom <= lower_pct_df
            mask_upper = local_rolling_mom >= upper_pct_df

            col_lower = mask_lower.columns[mask_lower.values[0]]
            col_upper = mask_upper.columns[mask_upper.values[0]]

            row_date = local_rolling_mom.index

            signal.loc[row_date, col_lower] = -1.0
            signal.loc[row_date, col_upper] = 1.0

        return signal


class FactorStrategy:
    def __init__(self, df, signal_cols, window=6, net_expo=0.5, max_lvg_factor=2):
        self.df = df
        self.signal_cols = signal_cols  # colonnes avec les prix des indices factorielles calculé en relatif par rapport au Msci World
        self.window = window  # taille de la fenêtre de moyenne mobile
        self.net_expo = net_expo  # exposition nette cible vis-à-vis du benchmark (MSCI World)
        self.max_lvg_factor = max_lvg_factor  # poids maximal autorisé par facteur en cas de momentum positif
        self.df_signal = None  # dataframe contenant les signaux mensuels
        self.wgt = None  # dataframe où sont stockés les poids à chaque fin de mois
        self.df_strat = None  # dataframe final contenant tous les détails de la stratégie (signaux, poids, perfs)

    def generate_signals(self):
        # on calcule les signaux à partir des prix relatifs factoriels (mensuels)
        df_signal = self.df[self.signal_cols].resample('M').last()  # on prend les derniers prix chaque mois
        signal_cols = []

        for col in self.signal_cols:  # pour chaque facteur
            rolling_mean = df_signal[col].rolling(window=self.window).mean()  # moyenne mobile
            factor_letter = col.split()[-1][0]  # extrait M, V ou Q
            signal_col = f"Signal {factor_letter}"
            df_signal[signal_col] = np.where(df_signal[col] > rolling_mean, 'Y', 'N')  # momentum : oui/non
            signal_cols.append(signal_col)

        df_signal['nb_pos_momentum'] = (df_signal[signal_cols] == 'Y').sum(axis=1)  # nombre de signaux positifs
        self.df_signal = df_signal.drop(columns=self.signal_cols)  # on garde uniquement les signaux (pas les prix relatifs)
        return self.df_signal

    def generate_weights(self):
        # génération des poids à partir des signaux
        if self.df_signal is None:
            raise ValueError("Les signaux doivent être générés avant les poids.")

        wgt = pd.DataFrame(index=self.df_signal.index)
        pos_mom = self.df_signal['nb_pos_momentum']  # nb de facteurs en momentum positif
        inv_pos_mom = 3 - pos_mom  # nb de facteurs en momentum négatif

        for factor in ['M', 'V', 'Q']:
            signal_col = f'Signal {factor}'
            weight_col = f'Poids {factor}'

            wgt[weight_col] = np.where(
                self.df_signal[signal_col] == 'Y',
                self.max_lvg_factor / pos_mom,  # on achete les facteurs avec un momentum positif. Les poids sont équipondérés si pos_mom > 1
                -((self.max_lvg_factor / inv_pos_mom))  # on short les autres (qui ont un momentum négatif par rapport au msci world)
            )

        # poids du MSCI World calculé pour maintenir l'exposition nette à la valeur cible
        wgt['Poids World'] = self.net_expo - wgt[['Poids M', 'Poids V', 'Poids Q']].sum(axis=1)
        self.wgt = wgt
        return self.wgt

    def build_daily_weights(self):
        # propagation des poids jour par jour jusqu'au prochain rebalancement
        if self.df_signal is None or self.wgt is None:
            raise ValueError("Génère d'abord les signaux et les poids mensuels.")

        rdt_facteurs = self.df[['ETF MOM', 'ETF VALUE', 'ETF QUALITY', 'ETF WORLD']].pct_change().dropna()
        self.df_signal = self.df_signal.reindex(rdt_facteurs.index)
        poids_monthly = self.wgt.reindex(rdt_facteurs.index)

        df_strat = pd.concat([self.df_signal, rdt_facteurs, poids_monthly], axis=1)
        df_strat = df_strat[df_strat.index >= '2015-12-31']

        poids_cols = ['Poids M', 'Poids V', 'Poids Q', 'Poids World']
        rdt_cols = ['ETF MOM', 'ETF VALUE', 'ETF QUALITY', 'ETF WORLD']

        for l in range(1, len(df_strat)):
            if df_strat.index[l].is_month_end:
                continue
            w_prev = df_strat.iloc[l - 1][poids_cols]
            r_now = df_strat.iloc[l][rdt_cols]
            return_strat = (w_prev * r_now).sum()

            for p_col, r_col in zip(poids_cols, rdt_cols):
                df_strat.loc[df_strat.index[l], p_col] = (
                    df_strat.iloc[l - 1][p_col] * (1 + df_strat.iloc[l][r_col])
                ) / (1 + return_strat)

        self.df_strat = df_strat
        return self.df_strat

    def compute_performance(self):
        # éxécution de la stratégie
        if self.df_strat is None:
            raise ValueError("Les poids journaliers doivent être construits avant de calculer la performance.")

        poids_cols = ['Poids M', 'Poids V', 'Poids Q', 'Poids World']
        rdt_cols = ['ETF MOM', 'ETF VALUE', 'ETF QUALITY', 'ETF WORLD']

        self.df_strat['return strat'] = np.nan

        for l in range(1, len(self.df_strat)):
            w_prev = self.df_strat.iloc[l - 1][poids_cols]
            r_now = self.df_strat.iloc[l][rdt_cols]
            return_strat = np.dot(w_prev.values, r_now.values)
            self.df_strat.at[self.df_strat.index[l], 'return strat'] = return_strat

        self.df_strat['Track strat'] = 100 * (1 + self.df_strat['return strat']).cumprod()
        self.df_strat['Track ETF World'] = 100 * (1 + self.df_strat['ETF WORLD']).cumprod()

        return {
            'DataFrame Stratégie': self.df_strat,  # dataframe final avec les signaux, poids, performances
            'return strat': self.df_strat['return strat'],  # rendements journaliers de la stratégie
            'Poids facteurs': self.df_strat[poids_cols]  # poids alloués chaque jour à chaque facteur
        }


class DynamicAllocation:
    """
    Class for Dynamically Calculating the Allocations Between the Core and Satellite Portfolios
    """

    def __init__(self, core_returns, satellite_returns ,benchmark_returns = None):
        self.core_returns = core_returns
        self.satellite_returns = satellite_returns
        self.benchmark_returns = benchmark_returns

    def _risk_parity_weight(self, window_core , window_sat,risk_budget_core=0.70, risk_budget_sat=0.30):
        """
        Calculate the optimal weight of the core portfolio according to the risk parity (ERC) method for two assets.
        """

        if isinstance(window_core, pd.DataFrame):
            window_core = window_core.iloc[:, 0]
        if isinstance(window_sat, pd.DataFrame):
            window_sat = window_sat.iloc[:, 0]

        # Calculation of the volatility (standard deviation) over the analysis window
        sigma_core = window_core.std(ddof=1)
        sigma_sat = window_sat.std(ddof=1)
        
        # To avoid division by zero if both volatilities are zero.
        if sigma_core + sigma_sat == 0:
            return 0.5  # We then allocate equally.
        
        # Direct weight calculation for the core
        core_weight = (sigma_sat / risk_budget_sat) / ((sigma_core / risk_budget_core) + (sigma_sat / risk_budget_sat))
        return core_weight
    

    def _semi_variance_weight(self, window_sat, min_core, k):
        """
        Calculate the core's weight based on the negative semi-variance of the satellite.
        """

        # Make sure that window_sat is a Series.
        if isinstance(window_sat, pd.DataFrame):
            window_sat = window_sat.iloc[:, 0]
        
        negatives = window_sat[window_sat < 0]
        semi_var = negatives.var(ddof=1) if not negatives.empty else 0
        semi_vol_annualized = np.sqrt(252 * semi_var)
        
        #Maximum allowed weight for the satellite (to ensure the min_core)
        max_sat_weight = 1 - min_core

        # Linear adjustment of the satellite weight based on the semi-variance
        target_sat_weight = max_sat_weight - k * (1/(1+(semi_vol_annualized*100)))
        if target_sat_weight < 0:
            target_sat_weight = 0

        core_weight = 1 - target_sat_weight
        
        # Ensure that the core weight is at least equal to min_core
        if core_weight < min_core:
            core_weight = min_core
        return core_weight,semi_vol_annualized
    

    def _semi_variance_weight_zscore(self, t, min_core, k, window) :
        """
        Calculate the core's weight based on the z-score of the benchmark's semi-variance.
        """

        # retrieve the current date
        current_date = self.core_returns.index[t]
    
        # Semi-variance calculation on the benchmark
        benchmark_window = self.benchmark_returns.loc[:current_date].tail(window)
        negatives_bench = benchmark_window[benchmark_window < 0]
        bench_semi_var = negatives_bench.var(ddof=1) if not negatives_bench.empty else 0
        current_semi_vol = np.sqrt(252 * bench_semi_var)
        
        # Calculation of the z-score
        benchmark_hist = self.benchmark_returns.loc[:current_date].tail(window*2)
        if len(benchmark_hist) < window:
            mean_bench = current_semi_vol
            std_bench = 1 
        else:
            # Calculation of the rolling semi-variance
            def calc_semi_vol(x):
                neg = x[x < 0]
                var_val = neg.var(ddof=1) if not neg.empty else 0
                return np.sqrt(252 * var_val)
            
            rolling_semi_vols = benchmark_hist.rolling(window=window).apply(calc_semi_vol, raw=False)
            valid_vals = rolling_semi_vols.dropna()
            if valid_vals.empty:
                mean_bench = current_semi_vol
                std_bench = 1
            else:
                mean_bench = valid_vals.mean()
                std_bench = valid_vals.std(ddof=1)
                if std_bench == 0:
                    std_bench = 1
        z_score_series = (valid_vals - mean_bench) / std_bench
        
        z_score__actu = z_score_series.iloc[-5].mean()
        max_sat_weight = 1 - min_core

        # linear adjustment of the satellite weight based on the z-score
        target_sat_weight = max_sat_weight - k * (1/(z_score__actu))
        target_sat_weight = np.clip(target_sat_weight, 0, max_sat_weight)
        core_weight = 1 - target_sat_weight

        # Ensure that the core weight is at least equal to min_core
        if core_weight < min_core:
            core_weight = min_core
        return core_weight, current_semi_vol, z_score__actu
    
    def _target_vol_weight(self, t, window, portfolio_returns, min_core, target_vol, tol, start_idx):

        """
        Calculate the optimal core weight to achieve an ex-ante volatility of target_vol, based on the portfolio's ex-post volatility over the 'window'.
        """

        # Calculation of the ex-post volatility over the 'window'.
        recent_returns = portfolio_returns.iloc[t-window+1:t+1]
        realized_vol = recent_returns.std(ddof=1) * np.sqrt(252)
        print(f"Ex-post volatility (annualized): {realized_vol:.4f}")

        window_core = self.core_returns.iloc[start_idx:t+1]
        if isinstance(window_core, pd.DataFrame):
            window_core = window_core.iloc[:, 0]
        window_core = np.asarray(window_core).flatten()

        window_sat = self.satellite_returns.iloc[start_idx:t+1]
        if isinstance(window_sat, pd.DataFrame):
            window_sat = window_sat.iloc[:, 0]
        window_sat = np.asarray(window_sat).flatten()

        sigma_core = window_core.std(ddof=1)
        sigma_sat = window_sat.std(ddof=1)
        
        # Check if the ex-post volatility is outside the range [target_vol - tol, target_vol + tol].
        if realized_vol < (target_vol - tol) or realized_vol > (target_vol + tol):
            print(f"Rebalancing triggered: realized volatility {realized_vol:.4f} is outside the target bounds [{target_vol-tol:.4f}, {target_vol+tol:.4f}]")

            cov_matrix = np.cov(window_core, window_sat)
            cov = cov_matrix[0, 1]
            
            # Function calculating the portfolio volatility based on the core weight.
            def portfolio_vol(w):
                return np.sqrt(252 * (w**2 * sigma_core**2 +
                                    (1 - w)**2 * sigma_sat**2 +
                                    2 * w * (1 - w) * cov))
            
            # Grid search for the core weight that minimizes |vol_ex_ante - target_vol|.
            weights = np.linspace(min_core, 1, 1000)
            vols = np.array([portfolio_vol(w) for w in weights])
            idx = np.argmin(np.abs(vols - target_vol))
            optimal_core_weight = weights[idx]
            optimal_ex_ante_vol = portfolio_vol(optimal_core_weight)
            return optimal_core_weight,realized_vol, optimal_ex_ante_vol,sigma_core, sigma_sat
        else:
            # If the ex-post volatility is within the tolerated range, the allocation remains unchanged.
            return None,realized_vol, None, sigma_core, sigma_sat


    
    def calculate_allocation(self, min_core, window, ndays_rebal = 1, 
                             method = "erc", k = 10,risk_budget_core=0.70, risk_budget_sat=0.30, target_vol=0.10, tol=0.01):
        """
        Dynamically calculate, for each date, the allocation weights between the core portfolio and the satellite portfolio, 
        using either the ERC method or methods based on semi-variance.
        """

        dates = self.core_returns.index
        core_weights = pd.Series(index=dates, dtype=float)
        portfolio_returns = pd.Series(index=dates, dtype=float)
        semi_vol_series  = pd.Series(index=dates, dtype=float)
        vol_ex_post_series  = pd.Series(index=dates, dtype=float)
        vol_ex_ante_series  = pd.Series(index=dates, dtype=float)
        sigma_core_series = pd.Series(index=dates, dtype=float)
        sigma_sat_series = pd.Series(index=dates, dtype=float)
        z_score_series = pd.Series(index=dates, dtype=float)
        current_core_weight = None

        # loop over the dates
        for t in range(0, len(dates) - 1):
            # Rebalancing every ndays_rebal days
            if t % ndays_rebal == 0:
                start_idx = 0 if t < window else t - window
                if method == "erc":
                    window_core = self.core_returns.iloc[start_idx:t+1]
                    window_sat = self.satellite_returns.iloc[start_idx:t+1]
                    computed_weight = self._risk_parity_weight(window_core, window_sat, risk_budget_core,risk_budget_sat)
                    current_core_weight = computed_weight if computed_weight >= min_core else min_core
                elif method == "semi variance Z_score":
                    # if a benchmark is provided, we use the semi-variance based on the benchmark
                    computed_weight, current_semi_vol, z_score = self._semi_variance_weight_zscore(t, min_core, k, window)
                    current_core_weight = computed_weight
                    semi_vol_series.iloc[t] = current_semi_vol  
                    z_score_series.iloc[t] = z_score
                elif method == "semi variance classique":
                    current_date = self.core_returns.index[t]
                    window_sat = self.benchmark_returns.loc[:current_date].tail(window)
                    computed_weight, current_semi_vol = self._semi_variance_weight(window_sat, min_core, k)
                    current_core_weight = computed_weight
                    semi_vol_series.iloc[t] = current_semi_vol
                elif method == "vol cible":
                    # If there are insufficient observations, we revert to a fallback method (for example, ERC).
                    if t < window:
                        window_core = self.core_returns.iloc[start_idx:t+1]
                        window_sat = self.satellite_returns.iloc[start_idx:t+1]
                        computed_weight = self._risk_parity_weight(window_core, window_sat, risk_budget_core, risk_budget_sat)
                        current_core_weight = computed_weight if computed_weight >= min_core else min_core
                    else:
                        optimal_weight,realized_vol, optimal_ex_ante_vol,sigma_core,sigma_sat = self._target_vol_weight(t, window, portfolio_returns, min_core, target_vol, tol, start_idx)
                        vol_ex_post_series.iloc[t] = realized_vol
                        vol_ex_ante_series.iloc[t] = optimal_ex_ante_vol
                        sigma_core_series.iloc[t] = sigma_core * np.sqrt(252)
                        sigma_sat_series.iloc[t] = sigma_sat * np.sqrt(252)
                        if optimal_weight is not None:
                            current_core_weight = optimal_weight
                else:
                    raise ValueError("The method must be either 'erc', 'vol cible', 'semi_variance Z_score' or 'semi variance classique'.")
            
            # Apply the computed weight
            core_weights.iloc[t + 1] = current_core_weight
            portfolio_returns.iloc[t + 1] = (
                current_core_weight * self.core_returns.iloc[t + 1] + 
                (1 - current_core_weight) * self.satellite_returns.iloc[t + 1]
            )
        
        core_weights.fillna(method='ffill', inplace=True)
        portfolio_returns.fillna(0, inplace=True)
        semi_vol_series.fillna(method='ffill', inplace=True)
        z_score_series.fillna(method='ffill', inplace=True)
        vol_ex_post_series.fillna(method='ffill', inplace=True)
        vol_ex_ante_series.fillna(method='ffill', inplace=True)
        sigma_core_series.fillna(method='ffill', inplace=True)
        sigma_sat_series.fillna(method='ffill', inplace=True)
        satellite_weights = 1 - core_weights
        
        return core_weights, satellite_weights, portfolio_returns,semi_vol_series ,z_score_series,vol_ex_post_series,vol_ex_ante_series,sigma_core_series, sigma_sat_series