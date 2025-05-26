

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
    from collections import defaultdict
    return gpd, mo, pd, pyo, value


@app.cell
def _(gpd, pd):
    # Load data

    # Demand scenarios
    scenarios_df = pd.read_csv("./Data/generated_scenarios.csv", sep=";")
    scenarios_df_filtered = scenarios_df.iloc[[1, 13, 25, 37, 49, 61, 73, 85, 97, 109]] #First month of each year, 10 years
    time_periods = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 

    # Population data Wuerzburg
    wuerzburg_gdf = gpd.read_file("./Data/wuerzburg_bevoelkerung_100m.geojson")
    wuerzburg_gdf_projected = wuerzburg_gdf.to_crs(epsg=25832)
    # grid_cells = wuerzburg_gdf.loc[:, 'Gitter_ID_100m']

    wuerzburg_gdf_300m = gpd.read_file("./Data/wuerzburg_bevoelkerung_300m.geojson")
    wuerzburg_gdf_300m_projected = wuerzburg_gdf_300m.to_crs(epsg=25832)
    grid_cells_300m = wuerzburg_gdf_300m.loc[:, 'Gitter_ID_100m']

    total_population = wuerzburg_gdf["Einwohner"].sum()

    #Potential APL-Locations
    apl_clusters_gdf = gpd.read_file("./Data/apl_candidates_clusters.geojson")
    apl_clusters_gdf_projected = apl_clusters_gdf.to_crs(epsg=25832)
    apl_cells = apl_clusters_gdf_projected.loc[:, 'Gitter_ID_100m']

    # Distance matrix
    df_walking = pd.read_csv("./Data/walking_distance_matrix.csv", sep=";")

    # Global parameters
    max_service_distance = 1700  # Max walking distance in meters
    return (
        apl_cells,
        df_walking,
        grid_cells_300m,
        max_service_distance,
        scenarios_df_filtered,
        time_periods,
        total_population,
        wuerzburg_gdf_300m_projected,
    )


@app.cell
def _():
    # # Distanz-Matrix CVRP
    # df_cvrp = pd.read_csv("./Data/cvrp_distance_matrix.csv", sep=";", index_col=0)

    # # Entferne "to_apl_" aus den Spaltennamen, um IDs zu extrahieren
    # df_cvrp.columns = [col.replace("to_apl_", "") for col in df_cvrp.columns]

    # # Liste aller IDs (inkl. Hub)
    # apl_ids_with_hub = df_cvrp.index.tolist()

    # # Dictionary mit allen Kombinationen für Pyomo-Parameter
    # cvrp_distance = {
    #     (i, j): df_cvrp.loc[i, j]
    #     for i in apl_ids_with_hub
    #     for j in apl_ids_with_hub
    # }
    return


@app.cell
def _(mo):
    mo.md("""# Calculate demand, valid pairs and distribution costs""")
    return


@app.cell
def _(scenarios_df_filtered, time_periods, total_population):
    # def get_distance(gdf, id1, id2):
    #     row1 = gdf.loc[gdf['Gitter_ID_100m'] == id1]
    #     row2 = gdf.loc[gdf['Gitter_ID_100m'] == id2]

    #     if row1.empty or row2.empty:
    #         raise ValueError(f"IDs not found in GeoDataFrame: {id1}, {id2}")

    #     point1 = row1.iloc[0].geometry.centroid
    #     point2 = row2.iloc[0].geometry.centroid

    #     return point1.distance(point2)

    # def calculate_distribution_cost(gdf_projected, valid_pairs):
    #     distribution_cost = {}
    #     for i, j in valid_pairs:
    #         # Kostenfaktor pro Meter und Paket
    #         cost_factor = 0.001  # Passe diesen Wert an
    #         distribution_cost[i, j] = get_distance(gdf_projected, i, j) * cost_factor
    #     return distribution_cost

    def calculate_demand_per_cell(gdf, scenario):
        demand = {}
        for t in time_periods:
            for _, row in gdf.iterrows():
                j = row['Gitter_ID_100m']
                cell_population = row["Einwohner"]
                demand[j, t] = (cell_population / total_population) * scenarios_df_filtered.iloc[t-1, scenario]
        return demand
    return (calculate_demand_per_cell,)


@app.cell
def _(
    calculate_demand_per_cell,
    df_walking,
    max_service_distance,
    wuerzburg_gdf_300m_projected,
):
    demand_scenario_neutral = calculate_demand_per_cell(wuerzburg_gdf_300m_projected, 10)

    # valid_pairs = []

    # for _, customer_row in wuerzburg_gdf_300m_projected.iterrows():
    #     customer_id = customer_row["Gitter_ID_100m"]
    #     customer_polygon = customer_row["geometry"]
    #     customer_centroid = customer_polygon.centroid

    #     for _, apl_row in apl_clusters_gdf_projected.iterrows():
    #         apl_id = apl_row["Gitter_ID_100m"]
    #         apl_polygon = apl_row["geometry"]
    #         apl_centroid = apl_polygon.centroid

    #         distance = customer_centroid.distance(apl_centroid)

    #         if distance <= max_service_distance:
    #             valid_pairs.append((apl_id, customer_id))

    # print(f"Number of valid pairs: {len(valid_pairs)}")

    valid_pairs = []
    no_connection = []

    for _, row in df_walking.iterrows():
        customer_id = row["from_customer"]
        has_connection = False 
        for col in row.index:
            if col.startswith("to_apl_"):
                apl_id = col.replace("to_apl_", "")
                distance = row[col]
                if distance <= max_service_distance:
                    valid_pairs.append((apl_id, customer_id))
                    has_connection = True
        if not has_connection:
            no_connection.append(customer_id)

    print(f"Anzahl gültiger Paare: {len(valid_pairs)}")
    print(f"Anzahl Gitterzellen ohne Verbindung: {len(no_connection)}")
    print(f"Gitterzellen ohne Verbindung: {no_connection}")

    # distribution_cost = calculate_distribution_cost(wuerzburg_gdf_projected, valid_pairs) 
    #100m-gdf, weil cluster-cells evtl nicht in 200m/300m-gdf sind
    return demand_scenario_neutral, valid_pairs


@app.cell
def _(mo):
    mo.md(r"""# Define model and constraints""")
    return


@app.cell
def _(
    apl_cells,
    demand_scenario_neutral,
    grid_cells_300m,
    pyo,
    time_periods,
    valid_pairs,
):
    model = pyo.ConcreteModel()

    model.ValidConnections = pyo.Set(initialize=valid_pairs, dimen=2)

    # Sets
    model.I = pyo.Set(initialize=apl_cells)   # APL-Standorte (IDs oder Indizes)
    model.J = pyo.Set(initialize=grid_cells_300m)  # Kundenstandorte
    model.T = pyo.Set(initialize=time_periods)  # Zeithorizont

    # Parameter
    # model.c = pyo.Param(model.I, model.J, initialize=distribution_cost, within=pyo.NonNegativeReals)
    model.f = pyo.Param(
        model.I,
        model.T,
        initialize=lambda m, i, t: 5500 * (1.02) ** (t - 1),
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
        # distribution_costs = sum(m.c[i, j] * m.d[j, t] * m.x[i, j, t] 
        #                         for (i, j) in m.ValidConnections for t in m.T)

        # Setup-Kosten (nur beim ersten Eröffnen)
        setup_costs = sum(m.f[i, t] * (m.y[i, t] - (0 if t == 1 else m.y[i, t-1])) 
                          for i in m.I for t in m.T)

        return setup_costs

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraints
    # Jeder Kunde muss genau einem APL zugewiesen werden
    def assignment_rule(m, j, t):
        # Nur gültige Verbindungen berücksichtigen
        valid_apls = [i for i in m.I if (i, j) in m.ValidConnections]
        if not valid_apls:  
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
            assigned_demand = sum(m.d[j, t] * m.x[i, j, t]
                                  for j in m.J if (i, j) in m.ValidConnections)
            return assigned_demand <= m.a * m.y[i, t]

    model.Capacity = pyo.Constraint(model.I, model.T, rule=capacity_rule)

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
def _(model, pd, value):
    # 1. Extrahiere x-Variablen (Zuweisungen)
    x_data = []
    for (i, j, t) in model.x:
        x_val = value(model.x[i, j, t])
        if x_val > 1e-6:  # Nur wenn APL i dem Kunden j in Periode t zugewiesen ist
            x_data.append({"APL_ID": i, "Customer_ID": j, "Period": t})
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

    # 4. Demand-Spalte
    df_combined["Demand"] = df_combined.apply(lambda row: value(model.d[row["Customer_ID"], row["Period"]]), axis=1)


    # 5. Sortieren nach Zeit, dann APL
    df_combined = df_combined.sort_values(by=["Period", "APL_ID"]).reset_index(drop=True)

    # 6. Export als CSV
    df_combined.to_csv("./Data/combined_results_with_setup_costs.csv", index=False, sep=";")
    print("Datei 'combined_results_with_setup_costs.csv' erfolgreich gespeichert.")
    df_combined
    print("Objective value:", value(model.objective))
    print(df_combined["Demand"].mean())
    return


@app.cell
def _(pd, pyo):
    def extract_cflp_results(model, output_path="./Data/cflp_apl_deployment_summary.csv"):
        results = []

        for i in model.I:
            opened_periods = []
            total_setup_costs = 0
            supplied_customers = set()
            prev_y = 0  # Anfangswert (kein APL aktiv)

            for t in model.T:
                current_y = int(round(pyo.value(model.y[i, t])))
                if current_y > prev_y:
                    newly_opened = current_y - prev_y
                    opened_periods.append((t, newly_opened))
                    total_setup_costs += newly_opened * pyo.value(model.f[i, t])
                prev_y = current_y  # merken für nächste Periode

            if prev_y > 0:
                for j in model.J:
                    if (i, j) in model.ValidConnections:
                        if any(pyo.value(model.x[i, j, t]) > 0.5 for t in model.T):
                            supplied_customers.add(j)
            
                #Summe der Nachfrage über alle Kunden und Perioden
                total_demand_per_apl = sum(
                    sum(pyo.value(model.d[j, t]) * pyo.value(model.x[i, j, t]) for j in supplied_customers) for t in model.T
                )
                # Auslastung pro Periode berechnen
                underutilized_count = 0
                for t in model.T:
                    # Summe der Nachfrage über alle Kunden für die jeweilige Periode
                    total_demand = sum(
                        pyo.value(model.d[j, t]) for j in supplied_customers
                    )
                    if total_demand < 1000:
                        underutilized_count += 1

                underutilized_flag = underutilized_count > len(model.T) / 2
                average_demand = total_demand_per_apl / len(model.T) if len(model.T) > 0 else 0

                results.append({
                    "APL_ID": i,
                    "Opened_Periods": str(opened_periods),  # z.B. [(1, 1), (3, 1)]
                    "Total_APLs_Opened": prev_y,
                    "Total_Setup_Costs": total_setup_costs,
                    "Supplied_Customers": ",".join(sorted(supplied_customers)),
                    "Underutilized_Most_Periods": underutilized_flag,
                    "Average_Demand": average_demand
                })

        df = pd.DataFrame(results)
        df.to_csv(output_path, sep=";", index=False)
        print(f"✅ Ergebnisse gespeichert unter: {output_path}")
        return df

    return (extract_cflp_results,)


@app.cell
def _(extract_cflp_results, model):
    df_apl_summary = extract_cflp_results(model)
    df_apl_summary
    print(f"Sum of opened APLs: {df_apl_summary['Total_APLs_Opened'].sum()}")
    print(df_apl_summary["Average_Demand"])
    return


if __name__ == "__main__":
    app.run()
