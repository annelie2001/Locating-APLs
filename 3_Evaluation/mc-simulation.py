

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from collections import defaultdict
    import pysd
    return defaultdict, gpd, np, pd, pysd


@app.cell
def _(gpd, pd):
    # Optimization results
    results_df = pd.read_csv("./Data/combined_results_with_setup_costs.csv", sep=";")

    #Results from heuristic approach
    heuristic_results_df = pd.read_csv("./Data/apl_customer_mapping_heuristic.csv", sep=";")

    # 300m-grid population data
    wuerzburg_gdf_300m = gpd.read_file("./Data/wuerzburg_bevoelkerung_300m.geojson")
    return heuristic_results_df, results_df, wuerzburg_gdf_300m


@app.cell
def _(pysd):
    sd_model = pysd.read_vensim("./Vensim-Model/APL-SFD-Würzburg-V3.mdl")
    simulation_results = sd_model.run().reset_index()
    return (simulation_results,)


@app.cell
def _(simulation_results, wuerzburg_gdf_300m):
    total_demand_per_period = simulation_results["Number of deliveries"].values
    total_population = wuerzburg_gdf_300m["Einwohner"].sum()
    wuerzburg_gdf_300m["demand_share"] = wuerzburg_gdf_300m["Einwohner"] / total_population
    return (total_demand_per_period,)


@app.cell
def _(heuristic_results_df, pd, total_demand_per_period, wuerzburg_gdf_300m):
    # Demand per cell and period 

    periods = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    demand_list = []

    for t in periods:
        total_demand_t = total_demand_per_period[t]

        for _, row in wuerzburg_gdf_300m.iterrows():
            j = row["Gitter_ID_100m"]
            share = row["demand_share"]
            demand = total_demand_t * share

            demand_list.append({"j": j, "t": t, "demand": demand})

    demand_jt_df = pd.DataFrame(demand_list)
    demand_jt_df["mean"] = demand_jt_df["demand"]
    demand_jt_df["std"] = demand_jt_df["demand"] * 0.2

    # Expand Heuristic df for all periods
    dfs = []
    for period in periods:
        df = heuristic_results_df.copy()
        df["Period"] = period
        dfs.append(df)

    # Verbinde alle DataFrames in der Liste zu einem einzigen DataFrame
    heuristic_results_df_expanded = pd.concat(dfs, ignore_index=True)
    return demand_jt_df, periods


@app.cell
def _(defaultdict, np, periods):
    def run_monte_carlo_simulation(
        demand_jt_df,
        results_df,
        apl_capacity,
        num_runs=1000,
        random_seed=None
    ):
        """
        demand_jt_df: DataFrame with columns ["j", "t", "mean", "std"]
        results_df: Optimization results with columns ["Period", "APL_ID", "Customer_ID"]
        apl_capacity: scalar or dict with capacity per APL per period
        num_runs: number of Monte Carlo samples to simulate
        """

        if random_seed:
            np.random.seed(random_seed)

        overloaded_counts = []
        overloaded_demands = []

        # 1. Ermittle die zugeordneten Kunden
        assigned_customers = set(results_df["Customer_ID"].unique())

        # 2. Ermittle die nicht zugeordneten Zellen
        all_cells = set(demand_jt_df["j"].unique())
        unassigned_cells = all_cells - assigned_customers

        print(f"Anzahl nicht zugeordneter Zellen: {len(unassigned_cells)}")

        # Build a structure: {(t, apl): [assigned_customers]}
        assignment_map = defaultdict(list)
        for _, row in results_df.iterrows():
            t = row["Period"]
            apl = row["APL_ID"]
            customer = row["Customer_ID"]
            assignment_map[(t, apl)].append(customer)

        for run in range(num_runs):
            # 1. Draw demand ~ N(mean, std) per cell and period
            print(f"🔄 Simulation {run + 1}/{num_runs} gestartet...")
            demand_samples = demand_jt_df.copy()
            demand_samples["sample"] = np.random.normal(
                loc=demand_samples["mean"],
                scale=demand_samples["std"]
            ).clip(min=0)  # Negative demand not allowed

            # 2. Sum assigned customer demand per APL and period
            apl_demand = defaultdict(float)

            for (t, apl), customers in assignment_map.items():
                for j in customers:
                    demand_val = demand_samples[
                        (demand_samples["j"] == j) & (demand_samples["t"] == t)
                    ]["sample"].sum()
                    apl_demand[(t, apl)] += demand_val

            # 3. Berechne die nicht erfüllte Nachfrage pro Periode
            unassigned_demand = defaultdict(float)
            for t in periods:
                for j in unassigned_cells:
                    demand_val = demand_samples[
                        (demand_samples["j"] == j) & (demand_samples["t"] == t)
                    ]["sample"].sum()
                    unassigned_demand[t] += demand_val

            # 4. Check overloads
            total = 0
            overloads = 0
            total_demand = 0
            overloads_demand = 0
            for (t, apl), total_apl_demand in apl_demand.items():
                cap = apl_capacity.get(apl, apl_capacity) if isinstance(apl_capacity, dict) else apl_capacity
                total += 1
                total_demand += total_apl_demand
                if total_apl_demand > cap:
                    overloads += 1
                    overloads_demand += total_apl_demand-cap
                    # print(f"⚠️ Demand not satisfied: {total_apl_demand-cap}")

            # Berücksichtige die nicht erfüllte Nachfrage als zusätzliche "Überlast"
            for t, demand in unassigned_demand.items():
                total += 1
                overloads += 1
                total_demand += total_apl_demand
                overloads_demand += total_apl_demand            
                # print(f"⚠️ Demand not satisfied: {total_apl_demand}")

            reliability_counts = 1 - (overloads / total)
            reliability_demands = 1- (overloads_demand / total_demand)
            overloaded_counts.append(reliability_counts)
            overloaded_demands.append(reliability_demands)
            print(f"✅ Simulation {run + 1} abgeschlossen.\n")

        # Return result summary
        reliability_counts_mean = np.mean(overloaded_counts)
        reliability_demands_mean= np.mean(overloaded_demands)

        result_summary = {
            "mean_reliability": reliability_counts_mean,
            "mean_demand_satisfaction": reliability_demands_mean,
            "runs": num_runs,
        }

        print("🎉 Alle Simulationen abgeschlossen.")
        return result_summary, overloaded_counts

    return (run_monte_carlo_simulation,)


@app.cell
def _(demand_jt_df, results_df, run_monte_carlo_simulation):
    result_summary, reliability_runs = run_monte_carlo_simulation(
        demand_jt_df=demand_jt_df,
        results_df=results_df,
        apl_capacity=48000,
        num_runs=100,
        random_seed=42
    )
    return (result_summary,)


@app.cell
def _():
    # result_summary_heuristic, reliability_runs_heuristic = run_monte_carlo_simulation(
    #     demand_jt_df=demand_jt_df,
    #     results_df=heuristic_results_df_expanded,
    #     apl_capacity=48000,
    #     num_runs=100,
    #     random_seed=42
    # )
    return


@app.cell
def _(result_summary):
    def print_summary(summary):
        print("Simulation results:")
        print(f"  Service Reliability (Proportion of APLs without overload): {summary['mean_reliability']:.4f}")
        print(f"  Demand Satisfaction Rate (Proportion of demand satisfied): {summary['mean_demand_satisfaction']:.4f}")
        print(f"  Number of simulations: {summary['runs']}")

    print("Optimization model:")
    print_summary(result_summary)
    # print("\nHeuristic:")
    # print_summary(result_summary_heuristic)
    return


if __name__ == "__main__":
    app.run()
