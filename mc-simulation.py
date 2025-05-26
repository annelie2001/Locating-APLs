

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from collections import defaultdict
    return defaultdict, gpd, np, pd


@app.cell
def _(gpd, pd):
    # Optimization results
    results_df = pd.read_csv("./Data/combined_results_with_setup_costs.csv", sep=";")
    #Results from heuristic approach
    heuristic_results_df = pd.read_csv("./Data/heuristic_results.csv", sep=";")
    # Demand from Vensim simulation
    demand_df = pd.read_csv("./Data/results-model2-sim1.csv", sep=";")
    demand_df_filtered = demand_df.iloc[[1, 13, 25, 37, 49]]
    # 300m-grid population data
    wuerzburg_gdf_300m = gpd.read_file("./Data/wuerzburg_bevoelkerung_300m.geojson")
    return (
        demand_df_filtered,
        heuristic_results_df,
        results_df,
        wuerzburg_gdf_300m,
    )


@app.cell
def _(demand_df_filtered, wuerzburg_gdf_300m):
    # Demand share per cell
    total_demand_per_period = demand_df_filtered["Number of deliveries : Model-V2-S1"].values
    total_population = wuerzburg_gdf_300m["Einwohner"].sum()
    wuerzburg_gdf_300m["demand_share"] = wuerzburg_gdf_300m["Einwohner"] / total_population
    return (total_demand_per_period,)


@app.cell
def _(pd, total_demand_per_period, wuerzburg_gdf_300m):
    # Demand per cell and period 

    periods = [0, 1, 2, 3, 4]
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
    return (demand_jt_df,)


@app.cell
def _(defaultdict, np):
    def run_monte_carlo_simulation(
        demand_jt_df,
        results_df,
        apl_capacity,
        num_runs=1000,
        reliability_threshold=0.95,
        random_seed=None
    ):
        """
        demand_jt_df: DataFrame with columns ["j", "t", "mean", "std"]
        results_df: Optimization results with columns ["Period", "APL_ID", "Customer_ID"]
        apl_capacity: scalar or dict with capacity per APL per period
        num_runs: number of Monte Carlo samples to simulate
        reliability_threshold: max. % of overloaded APLs tolerated
        """

        if random_seed:
            np.random.seed(random_seed)

        overloaded_counts = []

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
                # print(f"  ⏳ Bearbeite Periode {t}/{len(periods)}...")
                for j in customers:
                    demand_val = demand_samples[
                        (demand_samples["j"] == j) & (demand_samples["t"] == t)
                    ]["sample"].sum()
                    apl_demand[(t, apl)] += demand_val

            # 3. Check overloads
            overloads = 0
            total = 0
            for (t, apl), total_demand in apl_demand.items():
                cap = apl_capacity.get(apl, apl_capacity) if isinstance(apl_capacity, dict) else apl_capacity
                total += 1
                if total_demand > cap:
                    overloads += 1
                    print(f"⚠️ APL {apl} überlastet in Periode {t} (Last: {total_demand}, Kapazität: {cap})")

            reliability = 1 - (overloads / total)
            overloaded_counts.append(reliability)
            print(f"✅ Simulation {run + 1} abgeschlossen.\n")

        # Return result summary
        reliability_mean = np.mean(overloaded_counts)
        success_rate = np.mean(np.array(overloaded_counts) >= reliability_threshold)

        result_summary = {
            "mean_reliability": reliability_mean,
            "success_rate": success_rate,
            "threshold": reliability_threshold,
            "runs": num_runs,
        }

        print("🎉 Alle Simulationen abgeschlossen.")
        return result_summary, overloaded_counts

    return (run_monte_carlo_simulation,)


@app.cell
def _(result_summary, result_summary_heuristic):
    def print_summary(summary):
        print("Simulation results:")
        print(f"  Mean reliability: {summary['mean_reliability']:.4f}")
        print(f"  Success rate (>{summary['threshold']:.2f} reliability): {summary['success_rate']:.4f}")
        print(f"  Number of simulations: {summary['runs']}")

    print("Optimization model:")
    print_summary(result_summary)
    print("\nHeuristic:")
    print_summary(result_summary_heuristic)
    return


@app.cell
def _(demand_jt_df, results_df, run_monte_carlo_simulation):
    result_summary, reliability_runs = run_monte_carlo_simulation(
        demand_jt_df=demand_jt_df,
        results_df=results_df,
        apl_capacity=4000,
        num_runs=50,
        reliability_threshold=0.95,
        random_seed=42
    )

    #print(result_summary)
    return (result_summary,)


@app.cell
def _(demand_jt_df, heuristic_results_df, run_monte_carlo_simulation):
    result_summary_heuristic, reliability_runs_heuristic = run_monte_carlo_simulation(
        demand_jt_df=demand_jt_df,
        results_df=heuristic_results_df,
        apl_capacity=4000,
        num_runs=50,
        reliability_threshold=0.95,
        random_seed=42
    )

    print(result_summary_heuristic)
    return (result_summary_heuristic,)


if __name__ == "__main__":
    app.run()
