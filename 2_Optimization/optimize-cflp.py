

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
    scenario=10
    return (
        apl_cells,
        df_walking,
        grid_cells_300m,
        max_service_distance,
        scenario,
        scenarios_df,
        time_periods,
        total_population,
        wuerzburg_gdf_300m_projected,
    )


@app.cell
def _(mo):
    mo.md("""# Calculate demand and distances""")
    return


@app.cell
def _(
    scenario,
    scenarios_df,
    time_periods,
    total_population,
    wuerzburg_gdf_300m_projected,
):
    # Demand per cell in specific scenario

    def calculate_demand_per_cell(gdf, scenario):
        demand = {}
        for t in time_periods:
            for _, row in gdf.iterrows():
                j = row['Gitter_ID_100m']
                cell_population = row["Einwohner"]
                demand[j, t] = (cell_population / total_population) * scenarios_df.iloc[t-1, scenario]
        return demand

    demand_scenario = calculate_demand_per_cell(wuerzburg_gdf_300m_projected, scenario=scenario)
    return (demand_scenario,)


@app.cell
def _(df_walking, max_service_distance):
    # Calculate distances and determine valid pairs within max walking distance

    distances = {}
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
                    distances[apl_id, customer_id] = distance
                    has_connection = True
        if not has_connection:
            no_connection.append(customer_id)

    print(f"Number of valid pairs: {len(valid_pairs)}")
    print(f"Number of grid cells without valid connection: {len(no_connection)}")
    return distances, valid_pairs


@app.cell
def _(mo):
    mo.md(r"""# Define model and constraints""")
    return


@app.cell
def _(
    apl_cells,
    demand_scenario,
    distances,
    grid_cells_300m,
    pyo,
    time_periods,
    valid_pairs,
):
    model = pyo.ConcreteModel()

    model.ValidConnections = pyo.Set(initialize=valid_pairs, dimen=2)

    # Sets
    model.I = pyo.Set(initialize=apl_cells)   # APL nodes
    model.J = pyo.Set(initialize=grid_cells_300m)  # Customer nodes
    model.T = pyo.Set(initialize=time_periods)  # Time horizon

    # Parameter
    model.c = pyo.Param(model.I, model.J, initialize=distances, within=pyo.NonNegativeReals)
    model.f = pyo.Param(
        model.I,
        model.T,
        initialize=lambda m, i, t: 5500 * (1.02) ** (t - 1),
        within=pyo.NonNegativeReals
    )
    model.d = pyo.Param(model.J, model.T, initialize=demand_scenario, within=pyo.NonNegativeReals)
    model.a = pyo.Param(initialize=48000)  # 48000 deliveries/year, 4000 deliveries/month
    model.m = pyo.Param(initialize=0)   # minimum utilization

    # Decision Variables
    model.x = pyo.Var(model.ValidConnections, model.T, domain=pyo.Binary)
    model.y = pyo.Var(model.I, model.T, domain=pyo.NonNegativeIntegers)
    return (model,)


@app.cell
def _(model, pyo):
    # Objective Function
    def objective_rule(m):
        # Verteilungskosten
        customer_distances = sum(m.c[i, j] * m.d[j, t] * m.x[i, j, t] 
                                 for (i, j) in m.ValidConnections for t in m.T)

        # Setup-Kosten (nur beim ersten Eröffnen)
        setup_costs = sum(m.f[i, t] * (m.y[i, t] - (0 if t == 1 else m.y[i, t-1])) 
                          for i in m.I for t in m.T)

        return (setup_costs + 0*customer_distances)

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Constraints
    # Jeder Kunde muss genau einem APL zugewiesen werden
    def assignment_rule(m, j, t):
        # Nur gültige Verbindungen berücksichtigen
        valid_apls = [i for i in m.I if (i, j) in m.ValidConnections]
        if not valid_apls:  
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
def _(model, pd, pyo, value):
    solver = pyo.SolverFactory('cplex')
    # Setze Abbruchkriterien
    solver.options['timelimit'] = 600

    results = solver.solve(model, tee=True)

    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        print("Optimal solution found.")
    else:
        print("Solver did not find an optimal solution. Termination condition:", results.solver.termination_condition)

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
    print("Results saved to 'combined_results_with_setup_costs.csv'.")
    df_combined
    print("Objective value:", value(model.objective))
    print(f'Average demand: {df_combined["Demand"].mean()}')
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
                    if total_demand < (0.25 * model.a):
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
        print(f"✅ Results saved to: {output_path}")
        return df

    return (extract_cflp_results,)


@app.cell
def _(extract_cflp_results, model):
    df_apl_summary = extract_cflp_results(model)
    df_apl_summary
    print(f"Sum of opened APLs: {df_apl_summary['Total_APLs_Opened'].sum()}")
    print(f"Total setup costs: {df_apl_summary['Total_Setup_Costs'].sum()}")
    print(f"Number of underutilized APLs: {df_apl_summary['Underutilized_Most_Periods'].sum()}")
    return


if __name__ == "__main__":
    app.run()
