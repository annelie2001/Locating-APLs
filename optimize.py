import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium")


@app.cell
def _():
    import pyomo.environ as pyo
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    from math import ceil
    return ceil, gpd, np, pd, pyo


@app.cell
def _(gpd, np, pd):
    scenarios_df = pd.read_csv("./Data/generated_scenarios.csv", sep=";")

    wuerzburg_gdf = gpd.read_file("./Data/wuerzburg_bevoelkerung_100m.geojson")
    grid_cells = wuerzburg_gdf.iloc[:, 0]
    total_population = wuerzburg_gdf["Einwohner"].sum()

    time_periods = list(range(1, 121))
    distances = np.load("./Data/distances.npy")
    return (
        distances,
        grid_cells,
        scenarios_df,
        time_periods,
        total_population,
        wuerzburg_gdf,
    )


@app.cell
def _(distances, grid_cells):
    distribution_costs = {}
    for i in grid_cells:
        for j in grid_cells:
            distribution_costs[i, j] = distances[grid_cells[grid_cells == i].index[0], grid_cells[grid_cells == j].index[0]]*2
    return distribution_costs, i, j


@app.cell
def _(scenarios_df, time_periods, total_population, wuerzburg_gdf):
    def calculate_demand_per_cell(scenario):
        demand = {}
        for t in time_periods:
            for j in wuerzburg_gdf.index:
                cell_population = wuerzburg_gdf["Einwohner"][j]
                demand[j, t] = (cell_population / total_population) * scenarios_df.iloc[t-1, scenario]
        return demand
    return (calculate_demand_per_cell,)


@app.cell
def _(calculate_demand_per_cell):
    demand_scenario_1 = calculate_demand_per_cell(0)
    return (demand_scenario_1,)


@app.cell
def _(demand_scenario_1, distribution_costs, grid_cells, pyo, time_periods):
    model = pyo.ConcreteModel()
    solver = pyo.SolverFactory('glpk')

    # Sets
    model.I = pyo.Set(initialize=grid_cells)   # APL-Standorte (IDs oder Indizes)
    model.J = pyo.Set(initialize=grid_cells)  # Kundenstandorte
    model.T = pyo.Set(initialize=time_periods)  # z. B. 10 Jahre à 12 Monate

    # Parameter
    model.c = pyo.Param(model.I, model.J, initialize=distribution_costs, within=pyo.NonNegativeReals)
    model.f = pyo.Param(model.I, model.T, initialize=lambda m, i, t: 5500 if t == 1 else 0) # Inflation noch einberechnen
    model.d = pyo.Param(model.J, model.T, initialize=demand_scenario_1, within=pyo.NonNegativeReals) # Nur Szenario 1
    model.a = pyo.Param(model.I, initialize=lambda model, i: 6000)  # 6000 Pakete/Monat
    m = 0.4

    # Decision Variables
    model.x = pyo.Var(model.I, model.J, model.T, domain=pyo.Binary)
    model.y = pyo.Var(model.I, model.T, domain=pyo.NonNegativeIntegers)
    return m, model, solver


@app.cell
def _(distribution_costs, grid_cells):
    print(type(grid_cells))
    print(grid_cells)
    print(distribution_costs.shape)
    return


if __name__ == "__main__":
    app.run()
