

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pyomo.environ as pyo
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    from math import ceil
    from rtree import index
    import statistics
    return gpd, index, pd, pyo, statistics


@app.cell
def _(gpd, pd):
    # Load data

    scenarios_df = pd.read_csv("./Data/generated_scenarios.csv", sep=";")

    wuerzburg_gdf = gpd.read_file("./Data/wuerzburg_bevoelkerung_100m.geojson")
    wuerzburg_gdf_projected = wuerzburg_gdf.to_crs(epsg=25832)
    grid_cells = wuerzburg_gdf.loc[:, 'Gitter_ID_100m']

    wuerzburg_gdf_200m = gpd.read_file("./Data/wuerzburg_bevoelkerung_200m.geojson")
    wuerzburg_gdf_projected_200m = wuerzburg_gdf_200m.to_crs(epsg=25832)
    grid_cells_200m = wuerzburg_gdf_200m.loc[:, 'Gitter_ID_100m']

    total_population = wuerzburg_gdf["Einwohner"].sum()
    time_periods = [1, 2]
    #time_periods = list(range(1, 37)) # 3 years

    # Global parameters

    max_service_distance = 2000  # Max distance in meters
    max_neighbors = 10  # Max distance in terms of grid cells
    return (
        grid_cells,
        grid_cells_200m,
        max_neighbors,
        max_service_distance,
        scenarios_df,
        time_periods,
        total_population,
        wuerzburg_gdf,
        wuerzburg_gdf_200m,
        wuerzburg_gdf_projected,
        wuerzburg_gdf_projected_200m,
    )


@app.cell
def _(wuerzburg_gdf, wuerzburg_gdf_200m):
    print(wuerzburg_gdf.shape)
    print(wuerzburg_gdf_200m.shape)
    return


@app.cell
def _(
    index,
    scenarios_df,
    time_periods,
    total_population,
    wuerzburg_gdf_projected_200m,
):
    # Build the spacial index
    def build_spatial_index(gdf):
        # Erstelle räumlichen Index
        idx = index.Index()
        for i, row in gdf.iterrows():
            bounds = row['geometry'].bounds  # (minx, miny, maxx, maxy)
            idx.insert(i, bounds)
        return idx, {i: row['Gitter_ID_100m'] for i, row in gdf.iterrows()}

    spatial_idx, idx_to_id = build_spatial_index(wuerzburg_gdf_projected_200m)

    # Determine the nearest neighbors of each cell
    def get_nearest_neighbors(gdf, spatial_idx, idx_to_id, cell_idx, max_distance=1500, k=20):
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
        """Berechnet die Distanz zwischen zwei Zellen anhand der Centroids"""
        point1 = gdf.loc[gdf['Gitter_ID_100m'] == id1, 'geometry'].centroid.iloc[0]
        point2 = gdf.loc[gdf['Gitter_ID_100m'] == id2, 'geometry'].centroid.iloc[0]
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
                demand[j, t] = (cell_population / total_population) * scenarios_df.iloc[t-1, scenario]
        return demand
    return (
        calculate_demand_per_cell,
        calculate_distribution_cost,
        get_distance,
        get_valid_connections,
        idx_to_id,
        spatial_idx,
    )


@app.cell
def _(
    calculate_demand_per_cell,
    calculate_distribution_cost,
    get_valid_connections,
    grid_cells,
    idx_to_id,
    max_neighbors,
    max_service_distance,
    spatial_idx,
    statistics,
    wuerzburg_gdf_200m,
    wuerzburg_gdf_projected_200m,
):
    demand_scenario_10 = calculate_demand_per_cell(wuerzburg_gdf_200m, 10)
    print("Calculated demand per cell done")

    valid_connections = get_valid_connections(wuerzburg_gdf_projected_200m, spatial_idx, idx_to_id, max_service_distance, max_neighbors)
    print("Get valid connections done")
    neighbor_counts = [len(neighbors) for neighbors in valid_connections.values()]
    print(f"Durchschnittliche Anzahl Nachbarn pro Zelle: {statistics.mean(neighbor_counts)}")
    print(f"Maximale Anzahl Nachbarn pro Zelle: {max(neighbor_counts)}")

    valid_pairs = [(i, j) for j in grid_cells for i in valid_connections.get(j, [])]
    print("Valid pairs done")
    print(f"Anzahl der valid_pairs: {len(valid_pairs)}")

    distribution_cost = calculate_distribution_cost(wuerzburg_gdf_projected_200m, valid_pairs)
    print("Calculate distribution cost done")
    return demand_scenario_10, distribution_cost, valid_pairs


@app.cell
def _(
    demand_scenario_10,
    distribution_cost,
    grid_cells_200m,
    pyo,
    time_periods,
    valid_pairs,
):
    model = pyo.ConcreteModel()

    model.ValidConnections = pyo.Set(initialize=valid_pairs)

    # Sets
    model.I = pyo.Set(initialize=grid_cells_200m)   # APL-Standorte (IDs oder Indizes)
    model.J = pyo.Set(initialize=grid_cells_200m)  # Kundenstandorte
    model.T = pyo.Set(initialize=time_periods)  # z. B. 10 Jahre à 12 Monate

    # Parameter
    model.c = pyo.Param(model.I, model.J, initialize=distribution_cost, within=pyo.NonNegativeReals)
    model.f = pyo.Param(model.I, model.T, initialize=lambda m, i, t: 5500 if t == 1 else 0) # Inflation noch einberechnen
    model.d = pyo.Param(model.J, model.T, initialize=demand_scenario_10, within=pyo.NonNegativeReals) # Nur Szenario 10
    model.a = pyo.Param(model.I, initialize=lambda model, i: 6000)  # 6000 Pakete/Monat
    model.m = pyo.Param(initialize=0.4)

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
        return sum(m.d[j, t] * m.x[i, j, t] for j in valid_customers) <= m.a[i] * m.y[i, t]

    model.capacity = pyo.Constraint(model.I, model.T, rule=capacity_rule)

    # Minimale Auslastung für alle APLs zusammen
    def min_utilization_rule(m, t):
        active_capacity = sum(m.a[i] * m.y[i, t] for i in m.I)
        # if active_capacity == 0:
        #     return pyo.Constraint.Skip
        total_demand = sum(m.d[j, t] for j in m.J)
        return total_demand >= m.m * active_capacity

    model.min_utilization = pyo.Constraint(model.T, rule=min_utilization_rule)
    return


@app.cell
def _(model, pyo):
    #solver = pyo.SolverFactory('glpk')
    solver = pyo.SolverFactory('cplex')

    results = solver.solve(model, tee=True)
    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        print("Optimal solution found.")
    else:
        print("Solver did not find an optimal solution. Termination condition:", results.solver.termination_condition)
    return (results,)


@app.cell
def _(
    get_distance,
    model,
    pd,
    pyo,
    results,
    wuerzburg_gdf,
    wuerzburg_gdf_projected,
):
    # Result analysis

    if results.solver.status == pyo.SolverStatus.ok:
        # Erstelle DataFrames für die Ergebnisse
        # APL-Standorte
        apl_results = []
        for t in model.T:
            for i in model.I:
                if model.y[i, t].value > 0.5:  # APL ist aktiv
                    valid_customers = [j for j in model.J if (i, j) in model.ValidConnections]
                    total_demand = sum(model.d[j, t].value for j in valid_customers if model.x[i, j, t].value > 0.5)
                    utilization = total_demand / model.a[i].value
                    apl_results.append({
                        'time_period': t,
                        'apl_id': i,
                        'active': 1,
                        'utilization': utilization,
                        'demand_served': total_demand
                    })
                else:
                    apl_results.append({
                        'time_period': t,
                        'apl_id': i,
                        'active': 0,
                        'utilization': 0,
                        'demand_served': 0
                    })

        # Zuweisungen
        assignment_results = []
        for t in model.T:
            for (i, j) in model.ValidConnections:
                if model.x[i, j, t].value > 0.5:  # Zuweisung aktiv
                    assignment_results.append({
                        'time_period': t,
                        'apl_id': i,
                        'customer_id': j,
                        'demand': model.d[j, t].value,
                        'distance': get_distance(wuerzburg_gdf_projected, i, j)
                    })

        # Erstelle DataFrames
        apl_df = pd.DataFrame(apl_results)
        assignment_df = pd.DataFrame(assignment_results)

        # Speichere als CSV
        apl_df.to_csv('apl_locations_results.csv', index=False)
        assignment_df.to_csv('customer_assignments_results.csv', index=False)

        # Erstelle ein GeoDataFrame für die aktiven APLs in der letzten Periode
        last_period = max(model.T)
        active_apl_ids = [i for i in model.I if model.y[i, last_period].value > 0.5]
        active_apls_gdf = wuerzburg_gdf[wuerzburg_gdf['Gitter_ID_100m'].isin(active_apl_ids)].copy()

        # Füge Informationen über Nutzung hinzu
        for idx, row in active_apls_gdf.iterrows():
            apl_id = row['Gitter_ID_100m']
            utilization = next((item['utilization'] for item in apl_results 
                              if item['apl_id'] == apl_id and item['time_period'] == last_period), 0)
            active_apls_gdf.loc[idx, 'utilization'] = utilization
            active_apls_gdf.loc[idx, 'demand_served'] = next((item['demand_served'] for item in apl_results 
                                                           if item['apl_id'] == apl_id and item['time_period'] == last_period), 0)

        # Speichere als GeoJSON
        active_apls_gdf.to_file('active_apl_locations.geojson', driver='GeoJSON')

        # Zusammenfassung ausgeben
        print("\nOptimierungsergebnisse wurden gespeichert:")
        print("1. apl_locations_results.csv - alle APL-Standorte mit Nutzungsstatistiken")
        print("2. customer_assignments_results.csv - Zuweisungen von Kunden zu APLs")
        print("3. active_apl_locations.geojson - GeoJSON der aktiven APLs in der letzten Periode")

        # Zusätzlich noch Zusammenfassung ausgeben
        print("\nZusammenfassung der Ergebnisse:")
        for t in model.T:
            active_count = sum(1 for i in model.I if model.y[i, t].value > 0.5)
            total_demand = sum(model.d[j, t].value for j in model.J)
            print(f"Periode {t}: {active_count} aktive APLs, Gesamtnachfrage: {total_demand:.2f}")
    else:
        print("Problemlösung fehlgeschlagen.")
    return


if __name__ == "__main__":
    app.run()
