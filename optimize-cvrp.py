

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    return pd, pywrapcp, routing_enums_pb2


@app.cell
def _(pd):
    # === Daten einlesen ===

    df_dist = pd.read_csv("./Data/cvrp_distance_matrix.csv", index_col=0, sep=";")
    df_dist.columns = df_dist.columns.str.replace(r"^to_apl_", "", regex=True)
    df_dist.index = df_dist.index.str.replace(r"^to_apl_", "", regex=True)
    hub_id = df_dist.index[-1]

    # Bedarfe laden und nur aktive APLs behalten
    df_apl = pd.read_csv("./Data/cflp_apl_deployment_summary.csv", sep=";")
    return df_apl, df_dist, hub_id


@app.cell
def _(
    df_apl,
    df_dist,
    expand_distance_matrix,
    process_apl_data_with_splitting,
):
    # === Parameter ===

    #hub_coords = (9.999129, 49.772268) 
    # noch durch gesnappte Koordinaten austauschen, bisher nicht gespeichert

    vehicle_capacity = 250
    num_vehicles = 15 
    depot_index = 0

    df_expanded, df_original = process_apl_data_with_splitting(df_apl=df_apl)
    apl_ids = df_expanded['APL_ID'].tolist()
    demands = [0] + df_expanded['Average_Demand'].astype(int).tolist()  # Hub + APLs
    daily_demand = [int(d/30) for d in demands]

    new_distance_matrix, apl_mapping = expand_distance_matrix(
        df_expanded=df_expanded, df_dist=df_dist
    )
    distance_matrix = new_distance_matrix.values.astype(int)
    return (
        apl_ids,
        daily_demand,
        depot_index,
        distance_matrix,
        num_vehicles,
        vehicle_capacity,
    )


@app.cell
def _(daily_demand, num_vehicles, vehicle_capacity):
    # === DEBUGGING: Kapazitäts-Analyse ===

    print("\n=== KAPAZITÄTS-ANALYSE ===")
    print(f"Demands: {daily_demand}")
    print(f"Anzahl Knoten (inkl. Hub): {len(daily_demand)}")
    print(f"Gesamtbedarf: {sum(daily_demand)}")
    print(f"Fahrzeugkapazität: {vehicle_capacity}")
    print(f"Anzahl Fahrzeuge: {num_vehicles}")
    print(f"Gesamtkapazität: {vehicle_capacity * num_vehicles}")
    print(f"Kapazität ausreichend: {sum(daily_demand) <= vehicle_capacity * num_vehicles}")

    # Überprüfe einzelne Bedarfe vs. Kapazität
    oversized_demands = [i for i, d in enumerate(daily_demand) if d > vehicle_capacity]
    if oversized_demands:
        print(f"WARNUNG: Bedarfe größer als Fahrzeugkapazität bei Knoten: {oversized_demands}")
        for idx in oversized_demands:
            print(f"  Knoten {idx}: Bedarf {daily_demand[idx]} > Kapazität {vehicle_capacity}")
    return


@app.cell
def _(pd):
    # Split nodes with multiple APLs

    def process_apl_data_with_splitting(df_apl):

            print(f"Original APLs: {len(df_apl)}")

            # Finde APLs mit mehrfachen Öffnungen
            multi_apls = df_apl[df_apl['Total_APLs_Opened'] > 1]
            if len(multi_apls) > 0:
                print(f"APLs mit mehrfachen Öffnungen: {len(multi_apls)}")
                for _, row in multi_apls.iterrows():
                    print(f"  {row['APL_ID']}: {row['Total_APLs_Opened']} Öffnungen, Bedarf: {row['Average_Demand']}")
            else:
                print("Keine APLs mit mehrfachen Öffnungen gefunden")

            # Erweiterte APL-Liste erstellen
            expanded_apls = []

            for _, row in df_apl.iterrows():
                apl_id = row['APL_ID']
                total_opened = int(row['Total_APLs_Opened'])
                avg_demand = row['Average_Demand']

                if total_opened == 1:
                    # Standard APL - keine Änderung
                    expanded_apls.append({
                        'APL_ID': apl_id,
                        'Original_APL_ID': apl_id,
                        'Part_Number': 1,
                        'Total_Parts': 1,
                        'Average_Demand': avg_demand,
                        'Original_Demand': avg_demand
                    })
                else:
                    # APL aufteilen
                    demand_per_part = avg_demand / total_opened

                    for part in range(total_opened):
                        expanded_apls.append({
                            'APL_ID': f"{apl_id}_part{part+1}",
                            'Original_APL_ID': apl_id,
                            'Part_Number': part + 1,
                            'Total_Parts': total_opened,
                            'Average_Demand': demand_per_part,
                            'Original_Demand': avg_demand
                        })

            df_expanded = pd.DataFrame(expanded_apls)

            print(f"Erweiterte APLs: {len(df_expanded)}")

            return df_expanded, df_apl
    return (process_apl_data_with_splitting,)


@app.cell
def _(hub_id, pd):
    def expand_distance_matrix(df_expanded, df_dist):
        """
        Erweitert die Distance Matrix um die aufgeteilten APLs
        """    
        # Mapping von Original APL IDs zu neuen APL IDs
        original_to_new = {}
        for _, row in df_expanded.iterrows():
            orig_id = row['Original_APL_ID']
            new_id = row['APL_ID']

            if orig_id not in original_to_new:
                original_to_new[orig_id] = []
            original_to_new[orig_id].append(new_id)

        # Neue Distance Matrix erstellen
        new_apl_ids = df_expanded['APL_ID'].tolist()
        all_ids = [hub_id] + new_apl_ids

        new_distance_matrix = pd.DataFrame(
            index=all_ids, 
            columns=all_ids, 
            dtype=float
        )

        # Distance Matrix erweitern
        for i, id1 in enumerate(all_ids):
            for j, id2 in enumerate(all_ids):

                if i == j:
                    new_distance_matrix.loc[id1, id2] = 0
                    continue

                # Original IDs bestimmen
                orig_id1 = id1 if id1 == hub_id else df_expanded[df_expanded['APL_ID'] == id1]['Original_APL_ID'].iloc[0]
                orig_id2 = id2 if id2 == hub_id else df_expanded[df_expanded['APL_ID'] == id2]['Original_APL_ID'].iloc[0]

                # Distanz zwischen Teilen derselben APL = 0
                if orig_id1 == orig_id2 and id1 != id2:
                    new_distance_matrix.loc[id1, id2] = 0
                else:
                    # Original Distanz verwenden
                    if orig_id1 in df_dist.index and orig_id2 in df_dist.columns:
                        new_distance_matrix.loc[id1, id2] = df_dist.loc[orig_id1, orig_id2]
                    else:
                        print(f"Warnung: Distanz für {orig_id1} -> {orig_id2} nicht gefunden")
                        new_distance_matrix.loc[id1, id2] = 999999  # Sehr hohe Distanz als Fallback

        print(f"Neue Distance Matrix: {new_distance_matrix.shape}")
        print(f"APLs mit Distanz 0 zu sich selbst (aufgeteilte Teile): {(new_distance_matrix == 0).sum().sum() - len(all_ids)}")

        return new_distance_matrix, original_to_new

    return (expand_distance_matrix,)


@app.cell
def _(
    daily_demand,
    depot_index,
    distance_matrix,
    num_vehicles,
    pywrapcp,
    vehicle_capacity,
):
    # === Routing-Modell aufbauen ===

    def create_data_model():
        return {
            'distance_matrix': distance_matrix.tolist(),
            'demands': daily_demand,
            'vehicle_capacities': [vehicle_capacity] * num_vehicles,
            'num_vehicles': num_vehicles,
            'depot': depot_index
        }

    data = create_data_model()

    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Kapazitätsbeschränkungen
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # Null Kapazitäts-Spielraum
        data['vehicle_capacities'],
        True,
        'Capacity'
    )

    return data, manager, routing


@app.cell
def _(pywrapcp, routing, routing_enums_pb2):
    # Lösungsstrategie festlegen
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.log_search = True

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)

    search_parameters.time_limit.seconds = 10  
    search_parameters.solution_limit = 10

    # Lösung berechnen
    solution = routing.SolveWithParameters(search_parameters)
    return (solution,)


@app.cell
def _(apl_ids):
    # === Ergebnisse auslesen ===
    def get_routes(data, manager, routing, solution):
        routes = []
        total_distance = 0
        total_load = 0
        for vehicle_id in range(data['num_vehicles']):
            route = []
            index = routing.Start(vehicle_id)
            route_distance = 0
            route_load = 0
            plan_output = f'Route for vehicle {vehicle_id}:'
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += data['demands'][node_index]
                # route.append(node_index)
                if node_index == 0:
                    route.append("Hub")  # Knoten 0 ist der Hub
                    plan_output += " Hub ->"
                else:
                    route.append(apl_ids[node_index - 1])  # APL_ID für andere Knoten
                    plan_output += f' {apl_ids[node_index - 1]} (load: {route_load}) ->'
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            node_index = manager.IndexToNode(index)
            # route.append(node_index)
            if node_index == 0:
                route.append("Hub")  # Knoten 0 ist der Hub
            else:
                route.append(apl_ids[node_index - 1])  # APL_ID für andere Knoten
            routes.append(route)
            plan_output += " Hub \n"
            plan_output += f'Distance: {route_distance}m, Load: {route_load}\n'
            print(plan_output)
            total_distance += route_distance
            total_load += route_load
        print(f'Total distance of all routes: {total_distance}m')
        print(f'Total load delivered: {total_load}')
        return routes
    return (get_routes,)


@app.cell
def _(data, get_routes, manager, pd, routing, solution):
    if solution:
        routes = get_routes(data, manager, routing, solution)
    else:
        print("Keine Lösung gefunden.")

    # Save routes

    max_len = max(len(route) for route in routes)
    padded_routes = [route + [''] * (max_len - len(route)) for route in routes]
    df_routes = pd.DataFrame(padded_routes)
    df_routes = df_routes.T
    df_routes.to_csv('./Data/routes.csv', index=False, header=False, sep=";")
    return


if __name__ == "__main__":
    app.run()
