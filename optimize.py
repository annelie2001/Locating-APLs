

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pyomo.environ as pyo
    from pyomo.environ import value
    import pandas as pd
    import geopandas as gpd
    from rtree import index
    from shapely.geometry import Point
    return gpd, index, mo, pd, pyo, value


@app.cell
def _(gpd, pd):
    # Load data

    scenarios_df = pd.read_csv("./Data/generated_scenarios.csv", sep=";")
    scenarios_df_filtered = scenarios_df.iloc[[1, 13, 25, 37, 49, 61, 73, 85, 97, 109]] #Erster Monat mit einem Jahr Abstand
    time_periods = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 

    wuerzburg_gdf = gpd.read_file("./Data/wuerzburg_bevoelkerung_100m.geojson")
    wuerzburg_gdf_projected = wuerzburg_gdf.to_crs(epsg=25832)
    # grid_cells = wuerzburg_gdf.loc[:, 'Gitter_ID_100m']

    wuerzburg_gdf_200m = gpd.read_file("./Data/wuerzburg_bevoelkerung_200m.geojson")
    wuerzburg_gdf_200m_projected = wuerzburg_gdf_200m.to_crs(epsg=25832)
    grid_cells_200m = wuerzburg_gdf_200m.loc[:, 'Gitter_ID_100m']

    wuerzburg_gdf_300m = gpd.read_file("./Data/wuerzburg_bevoelkerung_300m.geojson")
    wuerzburg_gdf_300m_projected = wuerzburg_gdf_300m.to_crs(epsg=25832)
    grid_cells_300m = wuerzburg_gdf_300m.loc[:, 'Gitter_ID_100m']

    total_population = wuerzburg_gdf["Einwohner"].sum()

    apl_clusters_gdf = gpd.read_file("./Data/apl_candidates_clusters.geojson")
    apl_clusters_gdf_projected = apl_clusters_gdf.to_crs(epsg=25832)
    apl_cells = apl_clusters_gdf_projected.loc[:, 'Gitter_ID_100m']

    # Global parameters

    max_service_distance = 2000  # Max distance in meters
    max_neighbors = 15  # Max distance in terms of grid cells
    return (
        apl_cells,
        apl_clusters_gdf_projected,
        grid_cells_300m,
        max_service_distance,
        scenarios_df_filtered,
        time_periods,
        total_population,
        wuerzburg_gdf_300m_projected,
        wuerzburg_gdf_projected,
    )


@app.cell
def _(scenarios_df_filtered):
    print(scenarios_df_filtered.loc[:, "Scenario10"])
    return


@app.cell
def _(mo):
    mo.md("""# Calculate demand, valid pairs and distribution costs""")
    return


@app.cell
def _(
    index,
    scenarios_df_filtered,
    time_periods,
    total_population,
    wuerzburg_gdf_300m_projected,
):
    # Build the spacial index
    def build_spatial_index(gdf):
        # Erstelle räumlichen Index
        idx = index.Index()
        for i, row in gdf.iterrows():
            bounds = row['geometry'].bounds  # (minx, miny, maxx, maxy)
            idx.insert(i, bounds)
        return idx, {i: row['Gitter_ID_100m'] for i, row in gdf.iterrows()}

    spatial_idx, idx_to_id = build_spatial_index(wuerzburg_gdf_300m_projected)

    # Determine the nearest neighbors of each cell
    def get_nearest_neighbors(gdf, spatial_idx, idx_to_id, cell_idx, max_distance=2000, k=20):
        """Finde nahe Nachbarn für eine Zelle innerhalb max_distance"""
        cell_geom = gdf.loc[cell_idx, 'geometry']
        bounds = cell_geom.bounds
        centroid = cell_geom.centroid

        # Spacial index
        candidates_idx = list(spatial_idx.nearest(bounds, 2*k))

        # Exact distance
        neighbors = []
        for idx in candidates_idx:
            if idx == cell_idx:  # Skip comparison with itself
                continue
            neighbor_geom = gdf.loc[idx, 'geometry']
            distance = centroid.distance(neighbor_geom.centroid)
            if distance <= max_distance:
                neighbors.append(idx_to_id[idx])
                if len(neighbors) >= k:  # Stop if enough neighbours have been found
                    break

        return neighbors

    def get_valid_connections(gdf_projected, spatial_idx, idx_to_id, max_service_distance, max_neighbors):
        valid_connections = {}
        for i, row in gdf_projected.iterrows():
            cell_id = row['Gitter_ID_100m']
            neighbors = get_nearest_neighbors(gdf_projected, spatial_idx, idx_to_id, 
                                              i, max_service_distance, max_neighbors)
            valid_connections[cell_id] = neighbors
        return valid_connections

    # Calculate distance between cells
    def get_distance(gdf, id1, id2):
        row1 = gdf.loc[gdf['Gitter_ID_100m'] == id1]
        row2 = gdf.loc[gdf['Gitter_ID_100m'] == id2]

        if row1.empty or row2.empty:
            raise ValueError(f"IDs not found in GeoDataFrame: {id1}, {id2}")

        point1 = row1.iloc[0].geometry.centroid
        point2 = row2.iloc[0].geometry.centroid

        return point1.distance(point2)

    def calculate_distribution_cost(gdf_projected, valid_pairs):
        distribution_cost = {}
        for i, j in valid_pairs:
            # Kostenfaktor pro Meter und Paket
            cost_factor = 0.001  # Passe diesen Wert an
            distribution_cost[i, j] = get_distance(gdf_projected, i, j) * cost_factor
        return distribution_cost

    def calculate_demand_per_cell(gdf, scenario):
        demand = {}
        for t in time_periods:
            for _, row in gdf.iterrows():
                j = row['Gitter_ID_100m']
                cell_population = row["Einwohner"]
                demand[j, t] = (cell_population / total_population) * scenarios_df_filtered.iloc[t-1, scenario]
        return demand
    return calculate_demand_per_cell, calculate_distribution_cost


@app.cell
def _(
    apl_clusters_gdf_projected,
    calculate_demand_per_cell,
    calculate_distribution_cost,
    max_service_distance,
    wuerzburg_gdf_300m_projected,
    wuerzburg_gdf_projected,
):
    demand_scenario_neutral = calculate_demand_per_cell(wuerzburg_gdf_300m_projected, 10)

    valid_pairs = []

    for _, customer_row in wuerzburg_gdf_300m_projected.iterrows():
        customer_id = customer_row["Gitter_ID_100m"]
        customer_polygon = customer_row["geometry"]
        customer_centroid = customer_polygon.centroid

        for _, apl_row in apl_clusters_gdf_projected.iterrows():
            apl_id = apl_row["Gitter_ID_100m"]
            apl_polygon = apl_row["geometry"]
            apl_centroid = apl_polygon.centroid

            distance = customer_centroid.distance(apl_centroid)

            if distance <= max_service_distance:
                valid_pairs.append((apl_id, customer_id))

    print(f"Number of valid pairs: {len(valid_pairs)}")

    distribution_cost = calculate_distribution_cost(wuerzburg_gdf_projected, valid_pairs) 
    #100m-gdf, weil cluster-cells evtl nicht in 200m/300m-gdf sind
    return demand_scenario_neutral, distribution_cost, valid_pairs


@app.cell
def _(mo):
    mo.md(r"""# Define model and constraints""")
    return


@app.cell
def _(
    apl_cells,
    demand_scenario_neutral,
    distribution_cost,
    grid_cells_300m,
    pyo,
    time_periods,
    valid_pairs,
):
    model = pyo.ConcreteModel()

    model.ValidConnections = pyo.Set(initialize=valid_pairs)

    # Sets
    model.I = pyo.Set(initialize=apl_cells)   # APL-Standorte (IDs oder Indizes)
    model.J = pyo.Set(initialize=grid_cells_300m)  # Kundenstandorte
    model.T = pyo.Set(initialize=time_periods)  # Zeithorizont

    # Parameter
    model.c = pyo.Param(model.I, model.J, initialize=distribution_cost, within=pyo.NonNegativeReals)
    model.f = pyo.Param(
        model.I,
        model.T,
        initialize=lambda m, i, t: 5500 * (1.01) ** (t - 1),
        within=pyo.NonNegativeReals
    )
    model.d = pyo.Param(model.J, model.T, initialize=demand_scenario_neutral, within=pyo.NonNegativeReals) # Nur Szenario 10
    model.a = pyo.Param(initialize=4000)  # 4000 Pakete/Monat
    model.m = pyo.Param(initialize=0)   # minimum usage

    # Decision Variables
    model.x = pyo.Var(model.ValidConnections, model.T, domain=pyo.Binary)
    model.y = pyo.Var(model.I, model.T, domain=pyo.NonNegativeIntegers)
    return (model,)


@app.cell
def _(model, pyo):
    # Objective Function
    def objective_rule(m):
        # Verteilungskosten
        distribution_costs = sum(m.c[i, j] * m.d[j, t] * m.x[i, j, t] 
                                for (i, j) in m.ValidConnections for t in m.T)

        # Setup-Kosten (nur beim ersten Eröffnen)
        setup_costs = sum(m.f[i, t] * (m.y[i, t] - (0 if t == 1 else m.y[i, t-1])) 
                          for i in m.I for t in m.T)

        return distribution_costs + setup_costs

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraints
    # Jeder Kunde muss genau einem APL zugewiesen werden
    def assignment_rule(m, j, t):
        # Nur gültige Verbindungen berücksichtigen
        valid_apls = [i for i in m.I if (i, j) in m.ValidConnections]
        if not valid_apls:  # Falls keine gültigen APLs für diesen Kunden existieren
            print(f"Gitterzelle {j} hat keine gültige Verbindung zu einem APL")
            return pyo.Constraint.Skip
        return sum(m.x[i, j, t] for i in valid_apls) == 1

    model.assignment = pyo.Constraint(model.J, model.T, rule=assignment_rule)

    # APLs können nur einmal eröffnet werden
    def non_decreasing_rule(m, i, t):
        if t == 1:
            return pyo.Constraint.Skip
        return m.y[i, t] >= m.y[i, t-1]

    model.non_decreasing = pyo.Constraint(model.I, model.T, rule=non_decreasing_rule)

    # Kapazitätsbeschränkung für jeden APL
    def capacity_rule(m, i, t):
        # Nur gültige Verbindungen berücksichtigen
        valid_customers = [j for j in m.J if (i, j) in m.ValidConnections]
        return sum(m.d[j, t] * m.x[i, j, t] for j in valid_customers) <= m.a * m.y[i, t]

    model.capacity = pyo.Constraint(model.I, model.T, rule=capacity_rule)

    # Minimale Auslastung für alle APLs zusammen
    def min_utilization_rule(m, i, t):
        assigned_demand = sum(m.x[i, j, t] * m.d[j, t] for j in m.J if (i, j) in m.ValidConnections)
        min_demand_required = m.m * m.a * m.y[i, t]
        return assigned_demand >= min_demand_required

    model.min_utilization = pyo.Constraint(model.I, model.T, rule=min_utilization_rule)
    return


@app.cell
def _(mo):
    mo.md(r"""# Solve model and save results""")
    return


@app.cell
def _(model, pyo):
    #solver = pyo.SolverFactory('glpk')
    solver = pyo.SolverFactory('cplex')

    results = solver.solve(model, tee=True)
    results.write()

    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        print("Optimal solution found.")
    else:
        print("Solver did not find an optimal solution. Termination condition:", results.solver.termination_condition)
    return


@app.cell
def _(model, pd, pyo, value):
    # 1. Extrahiere x-Variablen (Zuweisungen)
    x_data = []
    for (i, j, t) in model.x:
        x_val = value(model.x[i, j, t])
        if x_val > 1e-6:  # Nur wenn APL i dem Kunden j in Periode t zugewiesen ist
            x_data.append({"APL_ID": i, "Customer_ID": j, "Period": t, "x": x_val})
    df_x = pd.DataFrame(x_data)

    # 2. Extrahiere y-Variablen (APL geöffnet)
    y_data = []
    for (i, t) in model.y:
        y_val = value(model.y[i, t])
        # Setup-Kosten nur, wenn APL i in t neu geöffnet wird
        y_prev = value(model.y[i, t - 1]) if t > 1 else 0
        was_opened = y_val - y_prev
        setup_cost = value(model.f[i, t]) * was_opened if was_opened > 0.5 else 0
        y_data.append({"APL_ID": i, "Period": t, "y": y_val, "SetupCost": setup_cost})
    df_y = pd.DataFrame(y_data)

    # 3. Merge x und y
    df_combined = df_x.merge(df_y, on=["APL_ID", "Period"], how="left")

    # 4. Zusätzliche Spalten: Demand, Kosten
    df_combined["Demand"] = df_combined.apply(lambda row: value(model.d[row["Customer_ID"], row["Period"]]), axis=1)
    df_combined["Cost_per_unit"] = df_combined.apply(lambda row: value(model.c[row["APL_ID"], row["Customer_ID"]]), axis=1)
    df_combined["Total_Cost"] = df_combined["x"] * df_combined["Cost_per_unit"]
    df_combined["Num_APLs"] = df_combined["y"]


    # 5. Sortieren nach Zeit, dann APL
    df_combined = df_combined.sort_values(by=["Period", "APL_ID"]).reset_index(drop=True)

    # 6. Export als CSV
    df_combined.to_csv("./Data/combined_results_with_setup_costs.csv", index=False, sep=";")
    print("Datei 'combined_results_with_setup_costs.csv' erfolgreich gespeichert.")
    print(df_combined)
    print("Objective value:", value(model.objective))

    for t in model.T:
        apls_in_t = sum(pyo.value(model.y[i, t]) for i in model.I)
        print(f"Period {t}: {apls_in_t} APLs deployed")
    return df_combined, df_y


@app.cell
def _(df_combined):
    demand_per_apl = df_combined.groupby(["APL_ID", "Period"])["Demand"].sum().reset_index()
    demand_per_apl_filtered = demand_per_apl[demand_per_apl["Demand"] < 800]

    print(demand_per_apl_filtered)
    print(demand_per_apl_filtered["APL_ID"].value_counts())
    return


@app.cell
def _(df_y):
    import matplotlib.pyplot as plt

    apl_over_time = df_y.groupby("Period")["y"].sum().reset_index()

    plt.figure(figsize=(8, 4))
    plt.plot(apl_over_time["Period"], apl_over_time["y"], marker="o", color="steelblue")
    plt.title("Total Number of APLs per Period")
    plt.xlabel("Period")
    plt.ylabel("Number of APLs")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
