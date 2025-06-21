import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import marimo as mo
    import altair as alt
    import pysd
    return np, pd, pysd


@app.cell
def _(pysd):
    sd_model = pysd.read_vensim("./Vensim-Model/APL-SFD-Würzburg-V3.mdl")
    simulation_results = sd_model.run().reset_index()
    simulation_results
    return (simulation_results,)


@app.cell
def _(simulation_results):
    demand_series = simulation_results["Number of deliveries"].values
    return (demand_series,)


@app.cell
def _(demand_series, np, pd):
    # Parameter
    num_scenarios = 20
    num_periods = 10
    delta = 0.01

    # Factors for scaling demand per scenario (linear from pessimistic to optimistic)
    scenario_factors = np.linspace(0.9, 1.1, num_scenarios) 

    # Prepare matrix
    scenarios = np.zeros((num_periods, num_scenarios))

    # Generate scenarios
    for s_idx, factor in enumerate(scenario_factors):
        for t in range(1, num_periods + 1):
            mu = factor * demand_series[t - 1]
            sigma = delta * t * mu  # Higher uncertainty in later periods
            scenarios[t - 1, s_idx] = int(np.random.normal(loc=mu, scale=sigma))

    # Scenario DataFrame
    scenario_df = pd.DataFrame(scenarios,
                               columns=[f"Scenario{i+1}" for i in range(num_scenarios)])

    # 8. Save
    scenario_df.to_csv("./Data/generated_scenarios.csv", sep=";", decimal='.', index=False)
    print("Generated scenarios and saved to 'generated_scenarios.csv'.")
    return


if __name__ == "__main__":
    app.run()
