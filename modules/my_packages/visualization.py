import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import colorsys




class Visualizer:
    """Visualize results of the backtest"""

    def __init__(self, performance):
        self.performance = performance

    def plot_cumulative_performance(self, title="Cumulative Performance", figsize=(10, 6)):
        """Display the cumulative performance"""
        plt.figure(figsize=figsize)
        plt.plot(self.performance.cumulative_performance, label="Strategy Returns")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show(block=True)

    def plot_equity_curve(self, title="Equity Curve", figsize=(10, 6)):
        """Display the equity curve"""
        plt.figure(figsize=figsize)
        plt.plot(self.performance.equity_curve, label="Equity Curve")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show(block=True)

    def plot_drawdowns(self, title="Drawdowns", figsize=(10, 6)):
        """Display the drawdowns"""
        if self.performance.cumulative_performance is None:
            self.performance.compute_cumulative_performance()

        rolling_max = self.performance.cumulative_performance.cummax()
        drawdown = (self.performance.cumulative_performance / rolling_max) - 1

        plt.figure(figsize=figsize)
        plt.plot(drawdown, label="Drawdowns")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.fill_between(drawdown.index, drawdown.iloc[:, 0], 0, color='red', alpha=0.3)
        plt.show(block=True)

    def print_tracking_error_stats(self,final_te):
        """
        Displays the final annualized, semiannual, monthly, and weekly tracking error.
        """

        final_te_annualized = final_te * np.sqrt(252)
        final_te_semi_annual = final_te * np.sqrt(126)
        final_te_monthly = final_te * np.sqrt(21)
        final_te_weekly = final_te * np.sqrt(5)
        print(f"Tracking Error Hebdomadaire : {final_te_weekly:.4f}")
        print(f"Tracking Error Mensuelle : {final_te_monthly:.4f}")
        print(f"Tracking Error Semestrielle : {final_te_semi_annual:.4f}")
        print(f"Tracking Error Annualisée : {final_te_annualized:.4f}")
        print(f"Tracking Error finale du portefeuille optimisé : {final_te:.4f}")


    def plot_cumulative_performance_core(self,cum_portfolio, cum_benchmark, title="Cumulative Performance"):
        """
        Displays the core portfolio's cumulative performance compared to a benchmark.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(cum_portfolio, label="Portefeuille", linestyle="-", color="blue")
        plt.plot(cum_benchmark, label="Benchmark", linestyle="--", color="red")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Performance")
        plt.legend()
        plt.grid()
        plt.show()
    
    def plot_cumulative_returns_zoom(self, cumulative_core_returns, cumulative_benchmark_returns, zoom_start=None, zoom_end=None):
        """
        Display the chart of cumulative returns for the portfolio and benchmark, with the ability to zoom in on a specific period.
        """

        if zoom_start is not None:
            zoom_start = pd.to_datetime(zoom_start)
        if zoom_end is not None:
            zoom_end = pd.to_datetime(zoom_end)

        plt.figure(figsize=(12, 6))

        plt.plot(cumulative_core_returns.index, cumulative_core_returns, 
                    label="Portefeuille", color="blue", linewidth=2)

        plt.plot(cumulative_benchmark_returns.index, cumulative_benchmark_returns, 
                    label="Benchmark", color="orange", linewidth=2)

        plt.xlabel("Date")
        plt.ylabel("Rendement Cumulé")
        plt.title("Performance Cumulative du Portefeuille vs Benchmark")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        if zoom_start is not None or zoom_end is not None:
            plt.xlim(zoom_start, zoom_end)

        plt.show()



    def plot_cumulative_performance_multi(self, series_dict, custom_title=None):
        """
        Displays the cumulative performance of multiple series on the same plot.
        """

        plt.figure(figsize=(12, 6))
        for label, series in series_dict.items():
            if label in ["Core-Sat", "Benchmark"]:
                plt.plot(series.index, series, label=label, linestyle='-', marker=None)
            elif label in ["Core", "Satellite"]:
                plt.plot(series.index, series, label=label, linestyle='--', marker=None)
            else:
                plt.plot(series.index, series, label=label)
        title = custom_title if custom_title is not None else "Multi-series cumulative performance"
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Cumulative Performance")
        plt.legend()
        plt.grid(True)
        plt.show()



    def plot_tracking_errors(self,TE_ex_ante_df, TE_ex_post_df, te_target, te_tolerance, start_day=0):
        """
        Displays the comparison between the ex-ante TE and the ex-post TE from a certain number of days onward, and draws horizontal lines for the upper bound (te_target + te_tolerance) and the lower bound (te_target - te_tolerance).
        """
        
        TE_ex_ante_df = TE_ex_ante_df.reindex(TE_ex_post_df.index, method='ffill')

        # We only plot starting from start_day.
        TE_ex_ante_df = TE_ex_ante_df.iloc[start_day:]
        TE_ex_post_df = TE_ex_post_df.iloc[start_day:]
        
        plt.figure(figsize=(12, 6))
        
        # Plot the ex-ante TE as a red dashed line."
        plt.plot(TE_ex_ante_df.index, TE_ex_ante_df["TE_ex_ante"], 
                label="TE Ex Ante", linestyle='--', color='red', linewidth=1)
        
        # Plot the ex-post TE as a blue continuous line.
        plt.plot(TE_ex_post_df.index, TE_ex_post_df["TE_ex_post"], 
                label="TE Ex Post", linestyle='-', color='blue', linewidth=1)
        
        # Calculation of the upper and lower bounds.
        upper_bound = te_target + te_tolerance
        lower_bound = te_target - te_tolerance
        
        # Plot horizontal lines as grey dashed lines.
        plt.axhline(upper_bound, linestyle='--', color='grey', label='Borne Supérieure')
        plt.axhline(lower_bound, linestyle='--', color='grey', label='Borne Inférieure')
        
        plt.title("Comparaison TE Ex Ante vs TE Ex Post")
        plt.xlabel("Date")
        plt.ylabel("Tracking Error")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()


    def plot_tracking_errors_evol(self, TE_ex_ante_df, TE_ex_post_df, te_target, te_tolerance, rebalancing_dates, benchmark_returns_aligned,vol_window = 252, start_day=0):
        """
        Display in two subplots:
            On the top: the comparison between the ex-ante tracking error and the ex-post tracking error from start_day, with the upper and lower bounds (te_target ± te_tolerance) and the rebalancing dates indicated by vertical lines.
            On the bottom: the evolution of the benchmark's annualized historical volatility (calculated over a 21-day window).
        """

        TE_ex_ante_df = TE_ex_ante_df.reindex(TE_ex_post_df.index, method='ffill')
        TE_ex_ante_df = TE_ex_ante_df.iloc[start_day:]
        TE_ex_post_df = TE_ex_post_df.iloc[start_day:]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))

        ax1.plot(TE_ex_ante_df.index, TE_ex_ante_df["TE_ex_ante"], 
                    label="TE Ex Ante", linestyle='--', color='red', linewidth=1)
        ax1.plot(TE_ex_post_df.index, TE_ex_post_df["TE_ex_post"], 
                    label="TE Ex Post", linestyle='-', color='blue', linewidth=1)

        upper_bound = te_target + te_tolerance
        lower_bound = te_target - te_tolerance
        ax1.axhline(upper_bound, linestyle='--', color='grey', label='Borne Supérieure')
        ax1.axhline(lower_bound, linestyle='--', color='grey', label='Borne Inférieure')

        for date in rebalancing_dates:
            if date >= TE_ex_ante_df.index[0] and date <= TE_ex_ante_df.index[-1]:
                ax1.axvline(date, linestyle='--', color='orange', alpha=0.5)

        ax1.set_title("Comparaison TE Ex Ante vs TE Ex Post")
        ax1.set_ylabel("Tracking Error")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        if isinstance(benchmark_returns_aligned, pd.DataFrame):
            benchmark_series = benchmark_returns_aligned.iloc[:, 0]
        else:
            benchmark_series = benchmark_returns_aligned
            
        benchmark_vol = benchmark_series.rolling(window=vol_window).std() * np.sqrt(252)

        ax2.plot(benchmark_vol.index, benchmark_vol, label="Volatilité Annualisée", color='green', linewidth=1)
        ax2.set_title("Évolution de la volatilité historique annualisée du benchmark")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volatilité")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()


    def plot_IR(self,IR_ex_ante_df, IR_ex_post_df):
        """
        Function to display the Ex Ante vs Ex Post Information Ratio.
        """
        plt.figure(figsize=(12, 6))

  
        plt.plot(IR_ex_post_df.index, IR_ex_post_df["Information ratio ex post"], label="IR Ex Post", linestyle='-', color='blue', linewidth=1, alpha=0.8)

        plt.plot(IR_ex_ante_df.index, IR_ex_ante_df["Information ratio ex ante"], label="IR Ex Ante", linestyle='--', color='red', linewidth=1)

        plt.title("Comparaison IR Ex Ante vs IR Ex Post")
        plt.xlabel("Date")
        plt.ylabel("Information Ratio")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)  
        plt.show()



    def plot_objective_evolution(self,objective_values_df, optimization_type="minte"):
        """
        Display the evolution of the objective function (information ratio).
        """
    
        plt.figure(figsize=(12, 6))
        if objective_values_df.shape[1] == 1:
            x = objective_values_df.index
            y = objective_values_df["Objective Value"]
            plt.plot(x, y, marker='o', label="Objective Value")
            plt.fill_between(x, y, alpha=0.6)
        else:
            objective_values_df.plot.area(stacked=False, alpha=0.6)
            
        if optimization_type == "minte":
            title = "Evolution of the Tracking Error of the Optimized Portfolio."
            ylabel = "Tracking Error"
        else:
            title = "Evolution of the Information Ratio of the Optimized Portfolio."
            ylabel = "Ratio d'Information"
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.legend(title="Fonds", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.show()

    def plot_allocations_evolution_3(
    self, 
    allocations_df, 
    optimization_type="minte", 
    custom_title=None, 
    stacked=True
):
        # Filtrer les actifs avec des poids négligeables
        present_allocations = allocations_df.loc[:, (allocations_df > 0.0001).any()]
        num_assets = present_allocations.shape[1]
        
        # Générer des indices de 0 à 1 répartis uniformément pour chaque actif
        indices = np.linspace(0, 1, num_assets)
        
        # Récupérer la colormap HSV
        cmap = plt.cm.get_cmap("hsv")
        
        # Fonction pour désaturer les couleurs (pasteliser)
        def desaturate_color(color, factor=0.7):
            """
            Reduces the saturation (factor < 1) of an RGBA color.
            'color' is a tuple (r, g, b, a) within [0,1].
            """
            r, g, b, a = color
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            s *= factor
            r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
            return (r2, g2, b2, a)

        # Construire une liste de couleurs désaturées
        colors = [desaturate_color(cmap(i), factor=0.7) for i in indices]

        # === 1) PLOT DE L'ÉVOLUTION DES ALLOCATIONS (AREA CHART) ===
        plt.figure(figsize=(12, 6))
        present_allocations.plot.area(alpha=0.6, color=colors, stacked=stacked)

        # Gérer le titre
        if custom_title is not None:
            title = custom_title
        else:
            if optimization_type == "minte":
                title = "Evolution of the portfolio allocations (TE minimization)"
            else:
                title = "Evolution of the portfolio allocations (Information Ratio Maximization)"
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Weights")
        plt.legend(title="Assets", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.show()

        # === 2) PLOT DU CAMEMBERT (PIE CHART) AVEC LES MÊMES COULEURS ===
        # Récupérer la dernière allocation disponible
        final_allocations = present_allocations.iloc[-1]

        plt.figure(figsize=(8, 8))

        # Fonction de formatage : n'affiche un pourcentage que si la tranche est >= 2 %
        def autopct_function(pct):
            return ('%1.1f%%' % pct) if pct >= 2 else ''
        
        # Création du pie chart
        # -> labels=None pour ne pas afficher de légende, 
        #    autopct=autopct_function pour afficher uniquement les % > 2
        plt.pie(
            final_allocations,
            labels=None,
            colors=colors,
            autopct=autopct_function,
            startangle=90
        )
        plt.title("Final Portfolio Allocation")
        plt.show()

        # Retourne la liste des actifs (colonnes) pour éventuellement un usage ultérieur
        return present_allocations.columns.tolist()




    def plot_allocations_evolution_2(self,allocations_df, optimization_type="minte", custom_title=None,stacked=True):
        """
        Plot the evolution of the weights.
        """
        present_allocations = allocations_df.loc[:, (allocations_df > 0.0001).any()]
        num_assets = present_allocations.shape[1]
        cmap = plt.cm.get_cmap("tab20", num_assets)
        colors = [cmap(i) for i in range(num_assets)]
        
        plt.figure(figsize=(12, 6))
        present_allocations.plot.area(alpha=0.6, color=colors,stacked=stacked)
        
        if custom_title is not None:
            title = custom_title
        else:
            if optimization_type == "minte":
                title = "Evolution of the portfolio allocations (TE minimization)"
            else:
                title = "Evolution of the portfolio allocations (Information Ratio Maximization)"
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Weights")
        plt.legend(title="Fonds", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.show()

    def plot_allocations_evolution(self, allocations_df, optimization_type="minte", custom_title=None, stacked=True):
        present_allocations = allocations_df.loc[:, (allocations_df > 0.0001).any()]
        num_assets = present_allocations.shape[1]
        
        # Générer des indices de 0 à 1 répartis uniformément pour chaque actif
        indices = np.linspace(0, 1, num_assets)
        
        # Récupérer la colormap HSV
        cmap = plt.cm.get_cmap("hsv")
        
        # Fonction pour désaturer les couleurs
        def desaturate_color(color, factor=0.7):
            """
            Réduit la saturation (factor < 1) d'une couleur RGBA.
            color est un tuple (r, g, b, a) dans [0,1].
            """
            r, g, b, a = color
            # Conversion RGB -> HLS
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            # On réduit la saturation
            s *= factor  
            # HLS -> RGB
            r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
            return (r2, g2, b2, a)

        # Construire une liste de couleurs désaturées
        colors = [desaturate_color(cmap(i), factor=0.7) for i in indices]

        # Plot
        plt.figure(figsize=(12, 6))
        present_allocations.plot.area(alpha=0.6, color=colors, stacked=stacked)

        if custom_title is not None:
            title = custom_title
        else:
            if optimization_type == "minte":
                title = "Evolution of the portfolio allocations (TE minimization)"
            else:
                title = "Evolution of the portfolio allocations (Information Ratio Maximization)"
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Weights")
        plt.legend(title="Fonds", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.show()

        return present_allocations.columns.tolist()


    def plot_weights_pie(self,allocations_df, title="Répartition des poids"):
        """
        Displays the portfolio weight distribution as a pie chart, with the legend placed outside to preserve a large disc
        """

        weights = allocations_df.iloc[-1] * 100  
        weights = weights[weights > 0.0001]  # Keep only the significant weights.
        cmap = plt.cm.get_cmap("tab20", len(weights))
        colors = [cmap(i) for i in range(len(weights))]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the pie chart
        patches, texts, autotexts = ax.pie(
            weights,
            labels=None,        
            autopct='%1.1f%%',  
            startangle=140,
            colors=colors
        )
        ax.axis('equal')  
        ax.set_title(title, fontsize=14)

        ax.legend(
            patches,
            weights.index,
            title="Fonds",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            ncol=2  
        )

        fig.tight_layout(rect=[0, 0, 0.8, 1])

        plt.show()

    def plot_universe_cumulative_performance_by_asset_type(
    self,
    universe_returns_aligned: pd.DataFrame,
    benchmark_returns_aligned: pd.DataFrame,
    custom_title_bonds: str = None,
    custom_title_equity: str = None
    ):
        """
        Displays the cumulative performance of the asset universe in two distinct charts.
        """
        import matplotlib.lines as mlines

        universe_returns_aligned = universe_returns_aligned[universe_returns_aligned.index >= pd.Timestamp("2016-01-01")]
        cumulative_returns = (1 + universe_returns_aligned).cumprod()

        if isinstance(benchmark_returns_aligned, pd.DataFrame) and benchmark_returns_aligned.shape[1] >= 2:
            bonds_benchmark_series = benchmark_returns_aligned.iloc[:, 0]
            equity_benchmark_series = benchmark_returns_aligned.iloc[:, 1]
        else:
            bonds_benchmark_series = equity_benchmark_series = benchmark_returns_aligned

        bonds_benchmark_series = bonds_benchmark_series[bonds_benchmark_series.index >= pd.Timestamp("2016-01-01")]
        equity_benchmark_series = equity_benchmark_series[equity_benchmark_series.index >= pd.Timestamp("2016-01-01")]
        bonds_benchmark_cumulative = (1 + bonds_benchmark_series).cumprod()
        equity_benchmark_cumulative = (1 + equity_benchmark_series).cumprod()

        bonds_assets = [asset for asset in universe_returns_aligned.columns if "bond" in asset.lower()]
        equity_assets = [asset for asset in universe_returns_aligned.columns if "bond" not in asset.lower()]

        if bonds_assets:
            n_bonds = len(bonds_assets)
            cmap_bonds = plt.cm.get_cmap("nipy_spectral", n_bonds)
            asset_to_color_bonds = {asset: cmap_bonds(i) for i, asset in enumerate(bonds_assets)}

            plt.figure(figsize=(10, 6))
            for asset in bonds_assets:
                plt.plot(
                    cumulative_returns.index,
                    cumulative_returns[asset],
                    label=asset,
                    color=asset_to_color_bonds[asset]
                )

            plt.plot(
                bonds_benchmark_cumulative.index,
                bonds_benchmark_cumulative,
                label="Benchmark Bonds",
                color="black"
            )
            title_bonds = custom_title_bonds if custom_title_bonds is not None else "Performance Cumulée des ETFs Bonds"
            plt.title(title_bonds)
            plt.xlabel("Date")
            plt.ylabel("Performance Cumulée")
            plt.grid(True)
            plt.show()

            fig_legend_bonds = plt.figure(figsize=(3, 3))
            handles_bonds = []
            for asset in bonds_assets:
                handles_bonds.append(mlines.Line2D([], [], color=asset_to_color_bonds[asset], label=asset))
            handles_bonds.append(mlines.Line2D([], [], color="black", label="Benchmark Bonds"))
            fig_legend_bonds.legend(handles=handles_bonds, loc="center", frameon=False)
            plt.axis("off")
            plt.show()


        if equity_assets:
            n_equity = len(equity_assets)
            cmap_equity = plt.cm.get_cmap("nipy_spectral", n_equity)
            asset_to_color_equity = {asset: cmap_equity(i) for i, asset in enumerate(equity_assets)}

            plt.figure(figsize=(10, 6))
            for asset in equity_assets:
                plt.plot(
                    cumulative_returns.index,
                    cumulative_returns[asset],
                    label=asset,
                    color=asset_to_color_equity[asset]
                )
            plt.plot(
                equity_benchmark_cumulative.index,
                equity_benchmark_cumulative,
                label="Benchmark Actions",
                color="black"
            )
            title_equity = custom_title_equity if custom_title_equity is not None else "Performance Cumulée des ETFs Actions"
            plt.title(title_equity)
            plt.xlabel("Date")
            plt.ylabel("Performance Cumulée")
            plt.grid(True)
            plt.show()

            fig_legend_equity = plt.figure(figsize=(3, 3))
            handles_equity = []
            for asset in equity_assets:
                handles_equity.append(mlines.Line2D([], [], color=asset_to_color_equity[asset], label=asset))
            handles_equity.append(mlines.Line2D([], [], color="black", label="Benchmark Actions"))
            fig_legend_equity.legend(handles=handles_equity, loc="center", frameon=False)
            plt.axis("off")
            plt.show()





    def plot_costs(self, costs, title="Évolution des frais", xlabel="Date", ylabel="Frais", linestyle='-'):
        """
        Displays the evolution of a series or DataFrame of incurred fees (for example, transaction or management fees).
        """


        plt.figure(figsize=(10, 5))

    
        if isinstance(costs, pd.Series):
            label = costs.name if costs.name is not None else "Frais"
            plt.plot(costs.index, costs.values, linestyle=linestyle, label=label)
            plt.legend()
        elif isinstance(costs, pd.DataFrame):
            if costs.shape[1] == 1:
                col = costs.columns[0]
                plt.plot(costs.index, costs[col].values, linestyle=linestyle, label=col)
            else:
                for col in costs.columns:
                    plt.plot(costs.index, costs[col].values, linestyle=linestyle, label=col)
            plt.legend()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()

    def plot_costs_in_bp(self, costs, title="Évolution des frais", xlabel="Date", ylabel="Frais (bp)", linestyle='-',annualized=False):
        """
        Displays the evolution of a series or DataFrame of incurred fees in basis points.
        """
        plt.figure(figsize=(10, 5))

        if annualized: factor = 252
        else: factor = 1

        # Convertir en basis points
        costs_in_bp = costs * 1e4 * factor

        # Plot
        if isinstance(costs_in_bp, pd.Series):
            label = costs_in_bp.name if costs_in_bp.name is not None else "Frais"
            plt.plot(costs_in_bp.index, costs_in_bp.values, linestyle=linestyle, label=label)
            plt.legend()
        elif isinstance(costs_in_bp, pd.DataFrame):
            for col in costs_in_bp.columns:
                plt.plot(costs_in_bp.index, costs_in_bp[col].values, linestyle=linestyle, label=col)
            plt.legend()

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()


 
    def print_tracking_error_stats(self,final_te):
        """
        Displays the final annualized, semiannual, monthly, and weekly tracking error.
        """

        final_te_annualized = final_te * np.sqrt(252)
        final_te_semi_annual = final_te * np.sqrt(126)
        final_te_monthly = final_te * np.sqrt(21)
        final_te_weekly = final_te * np.sqrt(5)
        print(f"Tracking Error Hebdomadaire : {final_te_weekly:.4f}")
        print(f"Tracking Error Mensuelle : {final_te_monthly:.4f}")
        print(f"Tracking Error Semestrielle : {final_te_semi_annual:.4f}")
        print(f"Tracking Error Annualisée : {final_te_annualized:.4f}")
        print(f"Tracking Error finale du portefeuille optimisé : {final_te:.4f}")

    def plot_core_sat_weights_area(self,core_weights_series, sat_weights_series, custom_title=None,core_label="Core", sat_label="Satellite"):
        """
        Displays the evolution of the core and satellite weights as an area plot.
        """

        df_weights = pd.DataFrame({
            core_label: core_weights_series,
            sat_label: sat_weights_series
        })
        
        plt.figure(figsize=(12, 6))
        df_weights.plot.area(alpha=0.7)
        title = custom_title if custom_title is not None else "Évolution des poids (Area Plot) : Core vs Satellite"
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Poids")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
    
    def plot_volatilities(
    self,
    vol_ex_ante_series,
    vol_ex_post_series,
    target_vol=None,
    tol=None,
    core_vol_series=None,
    sat_vol_series=None,
    start_day=0,
    title="Volatility Comparison",
    figsize=(12, 6),
    label_ex_ante="Vol Ex Ante",
    label_ex_post="Vol Ex Post",
    label_core_vol="Core Volatility",
    label_sat_vol="Satellite Volatility",
    label_target="Target Vol",
    label_upper="Upper Bound",
    label_lower="Lower Bound"
):
        """
        Displays the evolution of ex-ante and ex-post volatilities on the same chart.
        """

        # Extract first column if input is a DataFrame
        if isinstance(vol_ex_ante_series, pd.DataFrame):
            vol_ex_ante_series = vol_ex_ante_series.iloc[:, 0]
        if isinstance(vol_ex_post_series, pd.DataFrame):
            vol_ex_post_series = vol_ex_post_series.iloc[:, 0]
        if core_vol_series is not None and isinstance(core_vol_series, pd.DataFrame):
            core_vol_series = core_vol_series.iloc[:, 0]
        if sat_vol_series is not None and isinstance(sat_vol_series, pd.DataFrame):
            sat_vol_series = sat_vol_series.iloc[:, 0]

        # Filter from start_day
        vol_ex_ante_series = vol_ex_ante_series.iloc[start_day:]
        vol_ex_post_series = vol_ex_post_series.iloc[start_day:]
        if core_vol_series is not None:
            core_vol_series = core_vol_series.iloc[start_day:]
        if sat_vol_series is not None:
            sat_vol_series = sat_vol_series.iloc[start_day:]

        plt.figure(figsize=figsize)

        # Plot ex-ante and ex-post
        plt.plot(vol_ex_ante_series.index, vol_ex_ante_series, label=label_ex_ante,
                linestyle='--', color='red', linewidth=1)
        plt.plot(vol_ex_post_series.index, vol_ex_post_series, label=label_ex_post,
                linestyle='-', color='blue', linewidth=1)

        # Target and bounds
        if target_vol is not None:
            if tol is not None:
                upper_bound = target_vol + tol
                lower_bound = target_vol - tol
                plt.axhline(upper_bound, linestyle='--', color='grey', label=label_upper)
                plt.axhline(lower_bound, linestyle='--', color='grey', label=label_lower)
            else:
                plt.axhline(target_vol, linestyle='--', color='grey', label=label_target)

        # Core and Satellite volatilities
        if core_vol_series is not None:
            plt.plot(core_vol_series.index, core_vol_series, label=label_core_vol,
                    linestyle='-.', color='green', linewidth=1)
        if sat_vol_series is not None:
            plt.plot(sat_vol_series.index, sat_vol_series, label=label_sat_vol,
                    linestyle='-.', color='orange', linewidth=1)

        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Volatility (annualized)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    def calculate_stats(self, portfolio_returns, benchmark_returns, window=252, core_returns=None, satellite_returns=None,
                        label_portfolio="Portfolio",label_benchmark="Benchmark",label_core="Core",label_satellite="Satellite"):
        """
        Calculates descriptive statistics and rolling metrics for the portfolio relative to the benchmark,
        and computes correlations among the portfolio, benchmark, core, and satellite returns.
        """

        if core_returns is not None and isinstance(core_returns, pd.DataFrame):
            core_returns = core_returns.iloc[:, 0]
        if satellite_returns is not None and isinstance(satellite_returns, pd.DataFrame):
            satellite_returns = satellite_returns.iloc[:, 0]
        if portfolio_returns is not None and isinstance(portfolio_returns, pd.DataFrame):
            portfolio_returns = portfolio_returns.iloc[:, 0]
        if benchmark_returns is not None and isinstance(benchmark_returns, pd.DataFrame):
            benchmark_returns = benchmark_returns.iloc[:, 0]

        # Global annualized volatility of the portfolio
        vol_annualized = portfolio_returns.std(ddof=1) * np.sqrt(252)
        
        # Cumulative returns for drawdown calculation
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / rolling_max) - 1
        max_drawdown = drawdowns.min()  # Negative value
        
        # Global annualized tracking error: difference between portfolio and benchmark
        diff_returns = portfolio_returns - benchmark_returns
        te_annualized = diff_returns.std(ddof=1) * np.sqrt(252)
        
        # Global correlations
        corr_portfolio_bench = portfolio_returns.corr(benchmark_returns)

        if core_returns is not None: corr_core_bench = core_returns.corr(benchmark_returns)
        if satellite_returns is not None:corr_sat_bench = satellite_returns.corr(benchmark_returns)
        corr_core_bench = core_returns.corr(benchmark_returns) if core_returns is not None else np.nan
        corr_sat_bench = satellite_returns.corr(benchmark_returns) if satellite_returns is not None else np.nan
        
        # Build a correlation matrix for the available series
        data = {
        label_portfolio: portfolio_returns,
        label_benchmark: benchmark_returns
        }
        if core_returns is not None:
            data[label_core] = core_returns
        if satellite_returns is not None:
            data[label_satellite] = satellite_returns


        df_corr = pd.DataFrame(data)
        corr_matrix = df_corr.corr()
        
        # Rolling annualized tracking error and volatility
        rolling_te = diff_returns.rolling(window=window).std(ddof=1) * np.sqrt(252)
        rolling_vol = portfolio_returns.rolling(window=window).std(ddof=1) * np.sqrt(252)
        
        # Rolling correlations with benchmark
        rolling_corr_portfolio = portfolio_returns.rolling(window=window).corr(benchmark_returns)
        if core_returns is not None: rolling_corr_core = core_returns.rolling(window=window).corr(benchmark_returns)
        if satellite_returns is not None: rolling_corr_sat = satellite_returns.rolling(window=window).corr(benchmark_returns)
        rolling_corr_core = core_returns.rolling(window=window).corr(benchmark_returns) if core_returns is not None else None
        rolling_corr_sat = satellite_returns.rolling(window=window).corr(benchmark_returns) if satellite_returns is not None else None

        # Compute global Sharpe Ratio
        global_sharpe = portfolio_returns.mean() / portfolio_returns.std(ddof=1) * np.sqrt(252)

        #Calulate global alpha and beta by regression
        p = np.polyfit(benchmark_returns.astype(float), portfolio_returns.astype(float), 1)
        beta_global = p[0]
        alpha_global = p[1] * 252 # annualized alpha

        # Calculate rolling Sharpe Ratio
        rolling_sharpe = portfolio_returns.rolling(window=window).apply(
        lambda x: np.nan if x.std(ddof=1)==0 else (x.mean() / x.std(ddof=1)) * np.sqrt(252),
        raw=False
    )
    
        # Rolling alpha calculation: for each window, a regression is performed between portfolio_returns and benchmark_returns, and the intercept is annualized.
        def calc_rolling_alpha(x):
            # x is a rolling window of portfolio returns
            indices = x.index
            y = benchmark_returns.loc[indices].astype(float)
            x = x.astype(float)
            if x.std(ddof=1) == 0 or y.std(ddof=1) == 0:
                return np.nan
            p = np.polyfit(y, x, 1)
            # p[1] is the intercept (alpha), annualized by multiplying by 252
            return p[1] * 252

        rolling_alpha_series = portfolio_returns.rolling(window=window).apply(calc_rolling_alpha, raw=False)
        
        # Plot rolling tracking error
        plt.figure(figsize=(12,6))
        plt.plot(rolling_te.index, rolling_te, label="Rolling Tracking Error", color="red")
        plt.title("Rolling Annualized Tracking Error")
        plt.xlabel("Date")
        plt.ylabel("Tracking Error (%)")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot rolling volatility
        plt.figure(figsize=(12,6))
        plt.plot(rolling_vol.index, rolling_vol, label="Rolling Volatility", color="blue")
        plt.title("Rolling Annualized Volatility")
        plt.xlabel("Date")
        plt.ylabel("Volatility (%)")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot rolling Sharpe Ratio
        plt.figure(figsize=(12,6))
        plt.plot(rolling_sharpe.index, rolling_sharpe, label="Rolling Sharpe Ratio", color="purple")
        plt.title("Rolling Annualized Sharpe Ratio")
        plt.xlabel("Date")
        plt.ylabel("Sharpe Ratio")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot rolling Alpha
        plt.figure(figsize=(12,6))
        plt.plot(rolling_alpha_series.index, rolling_alpha_series, label="Rolling Alpha (annualized)", color="brown")
        plt.title("Rolling Annualized Alpha")
        plt.xlabel("Date")
        plt.ylabel("Alpha")
        plt.legend()
        plt.grid(True)
        plt.show()


        # Plot rolling correlations
        plt.figure(figsize=(12, 6))
        plt.plot(rolling_corr_portfolio.index, rolling_corr_portfolio, label=f"{label_portfolio} vs {label_benchmark}", color="black", linestyle='-')
        if rolling_corr_core is not None:
            plt.plot(rolling_corr_core.index, rolling_corr_core, label=f"{label_core} vs {label_benchmark}", linestyle='--', color="green")
        if rolling_corr_sat is not None:
            plt.plot(rolling_corr_sat.index, rolling_corr_sat, label=f"{label_satellite} vs {label_benchmark}", linestyle='--', color="orange")
        plt.title("Rolling Correlations with Benchmark")
        plt.xlabel("Date")
        plt.ylabel("Correlation")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Print global metrics
        metrics = [
            "Annualized Volatility", 
            "Maximum Drawdown", 
            "Annualized Tracking Error",
            f"Corr ({label_portfolio} vs {label_benchmark})", 
            f"Corr ({label_core} vs {label_benchmark})", 
            f"Corr ({label_satellite} vs {label_benchmark})",
            "Global Sharpe Ratio",
            "Global Alpha (annualized)"
        ]
        values = [
            vol_annualized, 
            max_drawdown, 
            te_annualized, 
            corr_portfolio_bench, 
            corr_core_bench, 
            corr_sat_bench,
            global_sharpe,
            alpha_global
        ]

        stats_df = pd.DataFrame({"Metric": metrics, "Value": values})
        print(stats_df.to_string(index=False))

        print("Correlation Matrix:")
        print(corr_matrix.to_string())


        return



