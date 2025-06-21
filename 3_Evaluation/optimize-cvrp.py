import marimo

__generated_with = "0.13.15"
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
    # Read data

    df_dist = pd.read_csv("./Data/cvrp_distance_matrix.csv", index_col=0, sep=";")
    df_dist.columns = df_dist.columns.str.replace(r"^to_apl_", "", regex=True)
    df_dist.index = df_dist.index.str.replace(r"^to_apl_", "", regex=True)
    hub_id = df_dist.index[-1]

    df_apl = pd.read_csv("./Data/cflp_apl_deployment_summary.csv", sep=";")
    return df_apl, df_dist, hub_id


@app.cell
def _(
    df_apl,
    df_dist,
    expand_distance_matrix,
    process_apl_data_with_splitting,
):
    # Parameter

    vehicle_capacity = 250
    num_vehicles = 15
    depot_index = 0

    df_expanded, df_original = process_apl_data_with_splitting(df_apl=df_apl)
    apl_ids = df_expanded['APL_ID'].tolist()
    demands = [0] + df_expanded['Average_Demand'].astype(int).tolist()  # Hub + APLs
    daily_demand = [int(d/12/30) for d in demands]

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
    # DEBUGGING: Capacity analysis

    print("\n=== CAPACITY ANALYSIS ===")
    print(f"Demands: {daily_demand}")
    print(f"Number of nodes (incl. hub): {len(daily_demand)}")
    print(f"Total demand: {sum(daily_demand)}")
    print(f"Vehicle capacity: {vehicle_capacity}")
    print(f"Number of vehicles: {num_vehicles}")
    print(f"Total capacity: {vehicle_capacity * num_vehicles}")
    print(f"Sufficient capacity: {sum(daily_demand) <= vehicle_capacity * num_vehicles}")

    oversized_demands = [i for i, d in enumerate(daily_demand) if d > vehicle_capacity]
    if oversized_demands:
        print(f"WARNING: Demands greater than vehicle capacity at nodes: {oversized_demands}")
        for idx in oversized_demands:
            print(f"Node {idx}: demand {daily_demand[idx]} > capacity {vehicle_capacity}")
    return


@app.cell
def _(pd):
    # Split nodes with multiple APLs (necessary because of capacity constraints)

    def process_apl_data_with_splitting(df_apl):

            print(f"Original APLs: {len(df_apl)}")

            # Find APL locations with multiple setups
            multi_apls = df_apl[df_apl['Total_APLs_Opened'] > 1]
            if len(multi_apls) > 0:
                print(f"APLs with multiple openings: {len(multi_apls)}")
                for _, row in multi_apls.iterrows():
                    print(f"  {row['APL_ID']}: {row['Total_APLs_Opened']} Openings, Deamand: {row['Average_Demand']}")
            else:
                print("No APLs with multiple openings found")

            # Create expanded APL list
            expanded_apls = []

            for _, row in df_apl.iterrows():
                apl_id = row['APL_ID']
                total_opened = int(row['Total_APLs_Opened'])
                avg_demand = row['Average_Demand']

                if total_opened == 1:
                    # Standard APL, no changes
                    expanded_apls.append({
                        'APL_ID': apl_id,
                        'Original_APL_ID': apl_id,
                        'Part_Number': 1,
                        'Total_Parts': 1,
                        'Average_Demand': avg_demand,
                        'Original_Demand': avg_demand
                    })
                else:
                    # Split APL
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

            print(f"Expanded APLs: {len(df_expanded)}")

            return df_expanded, df_apl
    return (process_apl_data_with_splitting,)


@app.cell
def _(hub_id, pd):
    def expand_distance_matrix(df_expanded, df_dist):
        """
        Expands the distance matrix by the split APLs
        """    
        # Mapping from original ID to new APL IDs
        original_to_new = {}
        for _, row in df_expanded.iterrows():
            orig_id = row['Original_APL_ID']
            new_id = row['APL_ID']

            if orig_id not in original_to_new:
                original_to_new[orig_id] = []
            original_to_new[orig_id].append(new_id)

        # Create new distance matrix
        new_apl_ids = df_expanded['APL_ID'].tolist()
        all_ids = [hub_id] + new_apl_ids

        new_distance_matrix = pd.DataFrame(
            index=all_ids, 
            columns=all_ids, 
            dtype=float
        )

        # Expand distance matrix
        for i, id1 in enumerate(all_ids):
            for j, id2 in enumerate(all_ids):

                if i == j:
                    new_distance_matrix.loc[id1, id2] = 0
                    continue

                # Determine original IDs
                orig_id1 = id1 if id1 == hub_id else df_expanded[df_expanded['APL_ID'] == id1]['Original_APL_ID'].iloc[0]
                orig_id2 = id2 if id2 == hub_id else df_expanded[df_expanded['APL_ID'] == id2]['Original_APL_ID'].iloc[0]

                # Distance between APLs at the same location = 0
                if orig_id1 == orig_id2 and id1 != id2:
                    new_distance_matrix.loc[id1, id2] = 0
                else:
                    # Use original distance
                    if orig_id1 in df_dist.index and orig_id2 in df_dist.columns:
                        new_distance_matrix.loc[id1, id2] = df_dist.loc[orig_id1, orig_id2]
                    else:
                        print(f"Warning: Distance for {orig_id1} -> {orig_id2} not found")
                        new_distance_matrix.loc[id1, id2] = 999999  # Very big distance as fallback

        print(f"New distance matrix: {new_distance_matrix.shape}")
        print(f"APLs with distance 0 to each other (split APL): {(new_distance_matrix == 0).sum().sum() - len(all_ids)}")

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
    # Build Routing-Modell

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

    # Capacity constraints
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
    # Define search strategy
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.log_search = True

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)

    search_parameters.time_limit.seconds = 10  
    search_parameters.solution_limit = 10

    # Start search
    solution = routing.SolveWithParameters(search_parameters)
    return (solution,)


@app.cell
def _(apl_ids):
    # Read results
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
                if node_index == 0:
                    route.append("Hub") # node 0 is the hub
                    plan_output += " Hub ->"
                else:
                    route.append(apl_ids[node_index - 1])  # APL_ID for other nodes
                    plan_output += f' {apl_ids[node_index - 1]} (load: {route_load}) ->'
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            node_index = manager.IndexToNode(index)
            if node_index == 0:
                route.append("Hub")
            else:
                route.append(apl_ids[node_index - 1])
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
        print("No solution found.")

    # Save routes

    max_len = max(len(route) for route in routes)
    padded_routes = [route + [''] * (max_len - len(route)) for route in routes]
    df_routes = pd.DataFrame(padded_routes)
    df_routes = df_routes.T
    df_routes.to_csv('./Data/routes.csv', index=False, header=False, sep=";")
    return


@app.cell
def _(
    depot_index,
    num_vehicles,
    pd,
    pywrapcp,
    routing_enums_pb2,
    vehicle_capacity,
):
    # Read data (Heuristic)
    df_dist_h = pd.read_csv("./Data/cvrp_distance_matrix_heuristic.csv", index_col=0, sep=";")
    df_dist_h.columns = df_dist_h.columns.str.replace(r"^to_apl_", "", regex=True)
    df_dist_h.index = df_dist_h.index.str.replace(r"^to_apl_", "", regex=True)
    hub_id_h = df_dist_h.index[-1]
    df_apl_h = pd.read_csv("./Data/apl_customer_mapping_heuristic.csv", sep=";")


    apl_ids_h = df_apl_h['APL_ID'].unique().tolist()
    daily_demand_h = [0] + [100] * len(apl_ids_h)
    distance_matrix_h = df_dist_h.values.astype(int)

    # DEBUGGING: Capacity analysis
    print("\n=== KAPAZITÄTS-ANALYSE (Heuristik) ===")
    print(f"Demands (Heuristik): {daily_demand_h}")
    print(f"Anzahl Knoten (inkl. Hub) (Heuristik): {len(daily_demand_h)}")
    print(f"Gesamtbedarf (Heuristik): {sum(daily_demand_h)}")
    print(f"Fahrzeugkapazität: {vehicle_capacity}")
    print(f"Anzahl Fahrzeuge: {num_vehicles}")
    print(f"Gesamtkapazität: {vehicle_capacity * num_vehicles}")
    print(f"Kapazität ausreichend: {sum(daily_demand_h) <= vehicle_capacity * num_vehicles}")

    # Check demand vs. capacities (Heuristic) 
    oversized_demands_h = [i for i, d in enumerate(daily_demand_h) if d > vehicle_capacity]
    if oversized_demands_h:
        print(f"Warning: Demand exceeds vehicle capacity for node: {oversized_demands_h}")
        for idh in oversized_demands_h:
            print(f"  Node {idh}: Demand {daily_demand_h[idh]} > Capacity {vehicle_capacity}")

    # Build Routing Model (Heuristic)
    def create_data_model_h():
        return {
            'distance_matrix': distance_matrix_h.tolist(),
            'demands': daily_demand_h,
            'vehicle_capacities': [vehicle_capacity] * num_vehicles,
            'num_vehicles': num_vehicles,
            'depot': depot_index
        }

    data_h = create_data_model_h()

    manager_h = pywrapcp.RoutingIndexManager(len(data_h['distance_matrix']),
                                               data_h['num_vehicles'], data_h['depot'])

    routing_h = pywrapcp.RoutingModel(manager_h)

    def distance_callback_h(from_index, to_index):
        from_node = manager_h.IndexToNode(from_index)
        to_node = manager_h.IndexToNode(to_index)
        return data_h['distance_matrix'][from_node][to_node]

    transit_callback_index_h = routing_h.RegisterTransitCallback(distance_callback_h)
    routing_h.SetArcCostEvaluatorOfAllVehicles(transit_callback_index_h)

    # Capacity constraints (Heuristic)
    def demand_callback_h(from_index):
        from_node = manager_h.IndexToNode(from_index)
        return data_h['demands'][from_node]

    demand_callback_index_h = routing_h.RegisterUnaryTransitCallback(demand_callback_h)
    routing_h.AddDimensionWithVehicleCapacity(
        demand_callback_index_h,
        0,  # Zero capacity margin
        data_h['vehicle_capacities'],
        True,
        'Capacity'
    )

    # Define search strategy (Heuristic)
    search_parameters_h = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters_h.log_search = True

    search_parameters_h.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters_h.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)

    search_parameters_h.time_limit.seconds = 10
    search_parameters_h.solution_limit = 10

    # Start search (Heuristic)
    solution_h = routing_h.SolveWithParameters(search_parameters_h)

    # Read solutions (Heuristic)
    def get_routes_h(data, manager, routing, solution):
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
                node_index = manager_h.IndexToNode(index)
                route_load += data['demands'][node_index]
                if node_index == 0:
                    route.append("Hub")
                    plan_output += " Hub ->"
                else:
                    route.append(apl_ids_h[node_index - 1]) 
                    plan_output += f' {apl_ids_h[node_index - 1]} (load: {route_load}) ->'
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            node_index = manager_h.IndexToNode(index)
            if node_index == 0:
                route.append("Hub")
            else:
                route.append(apl_ids_h[node_index - 1])
            routes.append(route)
            plan_output += " Hub \n"
            plan_output += f'Distance: {route_distance}m, Load: {route_load}\n'
            print(plan_output)
            total_distance += route_distance
            total_load += route_load
        print(f'Total distance of all routes: {total_distance}m')
        print(f'Total load delivered: {total_load}')
        return routes

    if solution_h:
        routes_h = get_routes_h(data_h, manager_h, routing_h, solution_h)
    else:
        print("Keine Lösung gefunden (Heuristik).")

    # Save routes (Heuristic)
    if solution_h:
        max_len_h = max(len(route) for route in routes_h)
        padded_routes_h = [route + [''] * (max_len_h - len(route)) for route in routes_h]
        df_routes_h = pd.DataFrame(padded_routes_h)
        df_routes_h = df_routes_h.T
        df_routes_h.to_csv('./Data/routes_heuristic.csv', index=False, header=False, sep=";")
    return


if __name__ == "__main__":
    app.run()
