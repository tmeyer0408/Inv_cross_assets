import numpy as np
import pandas as pd
from scipy.optimize import minimize


from .utils import Utilities



class Backtester:
    """
    Backtester for the Core strategy
    """
    def __init__(self, universe_returns_core, benchmark_returns,
                 objective_function_core, objective_threshold_core,te_tolerance, fee_rate_core=None):
        
        """
        Initialize the backtester with the input data and the optimization parameters.
        """
        self.universe_returns_core = universe_returns_core
        self.benchmark_returns = benchmark_returns
        self.objective_function_core = objective_function_core
        self.objective_threshold_core = objective_threshold_core
        self.fee_rate_core = fee_rate_core
        self.te_tolerance = te_tolerance
        self.core_optimizer = CoreOptimizer(universe_returns_core, fee_rate_core, 
                                            objective_function_core, objective_threshold_core,te_tolerance) # Creates an instance of CoreOptimizer that contains the methods for the core portfolio.


    
    def get_current_universe(self, universe_returns, i, rolling_window):
        """
        Function that determines which assets exist on date i, to handle the addition of new ETFs to the universe.
        """
        valid_assets = universe_returns.columns[universe_returns.iloc[:i].notna().sum() >= rolling_window] # Selects assets that have at least rolling_window non-NaN values
        current_universe = universe_returns[valid_assets] # Filters the universe to keep only the valid assets
        existing_mask = current_universe.iloc[i-1].notna()  # Mask to select existing assets
        existing_assets = current_universe.columns[existing_mask] # Selects the assets that existed on the previous date
        return current_universe, valid_assets, existing_assets


    @staticmethod
    def get_current_fee_rate(fee_rate, valid_assets):
        """
        Method to retrieve the transaction fees for each fee.s
        """
        if fee_rate is not None:
            return fee_rate[valid_assets]
        else:
            return pd.Series(0, index=valid_assets)

    def update_daily_weights(self, previous_weights, current_universe, i, existing_assets):
        """
        Function to update the daily weights by drift.
        """
        daily_returns = current_universe.loc[current_universe.index[i], existing_assets] # Selects the daily returns for the existing assets
        prev_weights = previous_weights.reindex(existing_assets, fill_value=0) # Reindexes the previous weights to keep only the existing assets
        updated_weights = Utilities.normalize_weights(prev_weights * (1 + daily_returns)) # Updates the weights by drift
        new_weights = prev_weights.copy() 
        new_weights.loc[existing_assets] = updated_weights # Updates the weights for the existing assets
        return new_weights


    def initialize_portfolio(self,i, current_universe_core,benchmark_returns, existing_assets_core,
                         objective_threshold_core, core_portfolio_weight, objective_function_core, current_fee_rate_core,
                         verbose,rolling_window):
        """
        Initialize the CORE portfolio by minimizing the ex ante information ratio, subject to a TE constraint.
        """

        # Retrieve the last 252 returns to initialize the portfolio.
        initial_universe_core = current_universe_core.iloc[i-rolling_window:i][existing_assets_core]
        initial_benchmark = benchmark_returns.iloc[i-rolling_window:i]

        previous_weights_core = pd.Series(0.0, index=current_universe_core.columns, dtype=np.float64)

        # Initialization of the optimization constraint
        bounds_core = [(0, 1)] * len(existing_assets_core)
        constraints_core = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: objective_threshold_core/np.sqrt(252) - 
              Utilities.tracking_error_ex_ante(w, initial_universe_core.values, initial_benchmark.values.flatten())} #Minimum TE constraint
        ]
        # Optimization function
        def obj_func_core(w, u, b):
            u_arr = u.values
            b_arr = b.values.flatten()
            return objective_function_core(w, u_arr, b_arr, 
                                        previous_weights_core.loc[existing_assets_core].values,
                                        current_fee_rate_core.loc[existing_assets_core].values,
                                        core_portfolio_weight)
        # Initial weights
        init_guess_core = np.ones(len(existing_assets_core)) / len(existing_assets_core)

        # Optimization
        res_core = minimize(obj_func_core, init_guess_core, args=(initial_universe_core, initial_benchmark),
                            method='SLSQP', bounds=bounds_core, constraints=constraints_core)
        
        #optimal weights
        weights_opt = res_core.x
        weights_opt = Utilities.round_and_renormalize(weights_opt, threshold=1e-4)
        previous_weights_core.loc[existing_assets_core] = weights_opt

        # Use the ex ante tracking error function to calculate TE (wrapper for the numba function).
        first_core_TE = Utilities.tracking_error_ex_ante(previous_weights_core.loc[existing_assets_core].values.astype(np.float64),
                                        initial_universe_core.values.astype(np.float64),
                                        initial_benchmark.values.flatten().astype(np.float64)) * np.sqrt(252)
 
        first_core_IR = -res_core.fun
        
        if verbose:
            print(f"Initialisation CORE : IR = {first_core_IR:.4f}, TE ex ante = {first_core_TE:.4f}")
        allocations_core_init = previous_weights_core.copy()
        rebalancing_date = benchmark_returns.index[i-1]
        return (previous_weights_core,first_core_TE, first_core_IR,allocations_core_init,rebalancing_date)
                
                
                


    def optimize_allocation_portfolio(self, rolling_window=252, rebalance_freq_core=5, rebalance_freq_sat=21,
                                      rebalance_freq_core_sat=5, core_portfolio_weight=0.70, te_tolerance=0.25,
                                      verbose=True):
        """
        Optimize the Core  portfolio.
        """
        self.core_optimizer.rolling_window = rolling_window  # We pass the window to CoreOptimizer."
        nb_iterations = len(self.universe_returns_core) - rolling_window

        # Initialization of variables
        core_returns_series = pd.Series(index=self.benchmark_returns.index[rolling_window:], dtype=float)
        allocations_core = []
        rebalancing_dates = []
        IR_core_ex_ante = []
        IR_core_ex_post = []
        TE_core_ex_ante = []
        TE_core_ex_post = []
        transaction_costs_core = []
        
        previous_weights_core = pd.Series(0.0, index=self.universe_returns_core.columns, dtype=np.float64)
        port_return_idx = 0

        # Iteration over each day (after the rolling window)
        for i in range(rolling_window, len(self.universe_returns_core)):

            # --- Retrieval of the current universe and valid assets. ---
            current_universe_core, valid_assets_core, existing_assets_core = self.get_current_universe(self.universe_returns_core, i, rolling_window)
            current_fee_rate_core = self.get_current_fee_rate(self.fee_rate_core, valid_assets_core)
            if current_universe_core.iloc[i-1].isna().all():
                continue

            # --- Initialization of the portfolio on the first date (first iteration) ---
            if i == rolling_window:

                # Portfolio initialization
                (previous_weights_core,
                 first_core_TE, first_core_IR,
                 alloc_core_init,
                 rebal_date) = self.initialize_portfolio(i, current_universe_core,self.benchmark_returns, existing_assets_core,
                                                         self.objective_threshold_core,core_portfolio_weight,self.objective_function_core,
                                                           current_fee_rate_core,verbose, rolling_window)
                
                # Calculation of the first transaction cost
                transaction_cost = Utilities.transaction_cost(
                    previous_weights_core.loc[existing_assets_core].values.astype(np.float64),
                    np.zeros_like(previous_weights_core.loc[existing_assets_core].values.astype(np.float64)), #Series of zeros for the first date, to model the initialization of the portfolio.
                    current_fee_rate_core.loc[existing_assets_core].values.astype(np.float64),
                    core_portfolio_weight
                 )

       
                

                current_transaction_cost = transaction_cost
                # Update the variables
                allocations_core.append(alloc_core_init)
                rebalancing_dates.append(rebal_date)
                IR_core_ex_ante.append(first_core_IR)
                TE_core_ex_ante.append(first_core_TE)
                transaction_costs_core.append({"Date": rebal_date, "Transaction Cost": transaction_cost})
                continue

            # --- Daily update by drift. ---
            new_weights_core = self.update_daily_weights(previous_weights_core, current_universe_core, i, existing_assets_core)

            # --- Calculation of candidate returns. ---
            current_date = current_universe_core.index[i]
            core_return = current_universe_core.loc[current_date, previous_weights_core.index].dot(previous_weights_core) -current_transaction_cost
            core_returns_series.loc[current_date] = core_return

            # --- CORE rebalancing decision ---
            # Calculation of the ex-post TE and IR on the rolling window.
            current_transaction_cost = 0.0

            TE_post, IR_post = self.core_optimizer.compute_core_te_ir(current_date, core_returns_series, self.benchmark_returns, rolling_window)


            TE_core_ex_post.append(TE_post)
            IR_core_ex_post.append(IR_post)
            if verbose:
                print(f"Date: {self.benchmark_returns.index[i]} | Combined TE ex post = {TE_post:.4f}")

            # --- CORE rebalancing (according to rebalance_freq_core and if the ex-post TE exceeds the tolerance; during the first 252 days, we rebalance by default because we don't have sufficient visibility on our ex-post TE). ---
            if ((TE_post is np.nan) or (abs(TE_post - self.objective_threshold_core) > te_tolerance)) and (i % rebalance_freq_core == 0):
                optimal_weights_core, TE_ante_val, IR_val, rebal_date,transaction_cost = self.core_optimizer.rebalance_core_portfolio(
                    i, current_universe_core, self.benchmark_returns, existing_assets_core,
                    previous_weights_core, core_portfolio_weight, verbose
                )
                print(f"Date: {self.benchmark_returns.index[i]} | TE ex ante after optimization = {TE_ante_val:.4f}")
                TE_core_ex_ante.append(TE_ante_val)
                allocations_core.append(optimal_weights_core.copy())
                rebalancing_dates.append(rebal_date)
                IR_core_ex_ante.append(IR_val)
                previous_weights_core = optimal_weights_core.copy()
                transaction_costs_core.append({"Date": rebal_date, "Transaction Cost": transaction_cost})
                current_transaction_cost = transaction_cost
            else:
                previous_weights_core = new_weights_core.copy()
                allocations_core.append(new_weights_core.copy())
                current_transaction_cost=0.0
            
            port_return_idx += 1



        # --- Construction of the final DataFrames. ---
        allocations_core_df = pd.DataFrame(allocations_core, index=self.universe_returns_core.index[rolling_window:], columns=self.universe_returns_core.columns)
        core_returns_series = pd.Series(core_returns_series[:port_return_idx], index=self.benchmark_returns.index[rolling_window:rolling_window+port_return_idx])
        IR_core_ex_ante_df = pd.DataFrame(IR_core_ex_ante, index=rebalancing_dates, columns=["Information ratio ex ante"])
        IR_core_ex_post_df = pd.DataFrame(IR_core_ex_post, index=self.universe_returns_core.index[rolling_window:rolling_window+len(IR_core_ex_post)], columns=["Information ratio ex post"])
        TE_core_ex_ante_df = pd.DataFrame(TE_core_ex_ante, index=rebalancing_dates, columns=['TE_ex_ante'])
        TE_core_ex_post_df = pd.DataFrame(TE_core_ex_post, index=self.universe_returns_core.index[rolling_window:rolling_window+len(TE_core_ex_post)], columns=['TE_ex_post'])
        transaction_costs_core_df = pd.DataFrame(transaction_costs_core, columns=["Date", "Transaction Cost"]).set_index("Date")

        
        return (core_returns_series,allocations_core_df,rebalancing_dates,
                IR_core_ex_ante_df, IR_core_ex_post_df, TE_core_ex_ante_df, TE_core_ex_post_df, 
                transaction_costs_core_df)





class CoreOptimizer:
    """
    Class that contains the methods used to optimize the CORE portfolio.
    """

    def __init__(self, universe_returns_core, fee_rate_core, objective_function_core, objective_threshold_core,te_tolerance):
        """
        Initialize the CoreOptimizer with the core data.
        """
        self.universe_returns_core = universe_returns_core
        self.fee_rate_core = fee_rate_core
        self.objective_function_core = objective_function_core
        self.objective_threshold_core = objective_threshold_core
        self.rolling_window = None
        self.te_tolerance = te_tolerance

    def optimize_core_portfolio(self, initial_universe, initial_benchmark, existing_assets, previous_weights, core_portfolio_weight):
        """
        Optimize the CORE portfolio by minimizing the ex ante information ratio, subject to a TE constraint, starting from the current weights
        """
        # Update of the optimization constraint.
        bounds = [(0, 1)] * len(existing_assets)
        constraints = [
             {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            #maximum TE constraint
             {'type': 'ineq', 'fun': lambda w: (self.objective_threshold_core + self.te_tolerance)/np.sqrt(252) -
             Utilities.tracking_error_ex_ante(w, initial_universe.values, initial_benchmark.values.flatten())},
            #minimum TE constraint
             {'type': 'ineq', 'fun': lambda w: -(self.objective_threshold_core - self.te_tolerance)/np.sqrt(252) +
             Utilities.tracking_error_ex_ante(w, initial_universe.values, initial_benchmark.values.flatten())}
             ]


        # Optimization function
        def obj_func(w, u, b):
            u_arr = u.values
            b_arr = b.values.flatten()
            return self.objective_function_core(
                w, u_arr, b_arr,
                previous_weights.reindex(existing_assets, fill_value=0).values,
                self.fee_rate_core.loc[existing_assets].values,
                core_portfolio_weight
            )
        
        # Initial weights equal to the previous weights
        init_guess = previous_weights.reindex(existing_assets, fill_value=0).values

        # Optimization
        res = minimize(obj_func, init_guess, args=(initial_universe, initial_benchmark),
                       method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Optimal weights
        weights_opt = res.x
        weights_opt = Utilities.round_and_renormalize(weights_opt, threshold=1e-4)
        optimal_weights = previous_weights.copy()
        optimal_weights = optimal_weights.reindex(existing_assets, fill_value=0)
        optimal_weights.loc[existing_assets] = weights_opt

        # Calculation of the ex ante TE and IR with the optimal weights
        TE_ante = Utilities.tracking_error_ex_ante(optimal_weights.loc[existing_assets].values,
                                       initial_universe.values,
                                       initial_benchmark.values.flatten()) * np.sqrt(252)
        
        ## Calculation of the transaction costs
        frais_transaction = Utilities.transaction_cost(
            optimal_weights.reindex(existing_assets, fill_value=0).values.astype(np.float64),
            previous_weights.reindex(existing_assets, fill_value=0).values.astype(np.float64),
            self.fee_rate_core.reindex(existing_assets, fill_value=0).values.astype(np.float64),
            core_portfolio_weight
        )

        IR_val = -res.fun * np.sqrt(252)

        return optimal_weights, TE_ante, IR_val,frais_transaction
    

    def compute_core_te_ir(self, current_date, core_returns_series, benchmark_returns, rolling_window):
        """
        Calculates the ex-post TE and IR for the core over a rolling window ending on current_date.
        """
        # We retrieve the returns for the rolling window
        window_returns = core_returns_series.loc[:current_date].iloc[-rolling_window:]
    
        # We check if we have enough data to calculate the TE and IR
        #if len(window_returns) < rolling_window:
            #return np.nan, np.nan

        # We retrieve the benchmark returns for the same period
        benchmark_window = benchmark_returns.loc[window_returns.index].values.flatten()

        # We calculate the TE and IR
        TE_post = Utilities.tracking_error_ex_post(window_returns.values, benchmark_window) * np.sqrt(252)
        IR_post = (Utilities.information_ratio_ex_post(window_returns.values, benchmark_window) * np.sqrt(252)
                if len(window_returns) == rolling_window else np.nan)
        return TE_post, IR_post

    def rebalance_core_portfolio(self, i, current_universe, benchmark_returns, existing_assets, previous_weights, core_portfolio_weight, verbose):
        """
        Performs the optimization of the CORE portfolio in the event of rebalancing.
        Returns the new weights, the ex ante TE, the IR, and the rebalancing date.
        """
        # We retrieve the returns for the rolling
        rolling_uc = current_universe.iloc[i - self.rolling_window:i][existing_assets]
        rolling_bench = benchmark_returns.iloc[i - self.rolling_window:i]

        # We optimize the portfolio
        optimal_weights, TE_ante, IR_val,transaction_cost = self.optimize_core_portfolio(
            rolling_uc, rolling_bench, existing_assets, previous_weights, core_portfolio_weight
        )
        rebal_date = benchmark_returns.index[i]
        if verbose:
            print(f"Rebalancement du portefeuille core à la date : {current_universe.index[i]}")
        return optimal_weights, TE_ante, IR_val, rebal_date,transaction_cost
    


