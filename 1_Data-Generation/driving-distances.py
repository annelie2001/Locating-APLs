import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import geopandas as gpd
    import pandas as pd
    import requests
    from tqdm import tqdm
    return gpd, pd, requests


@app.function
def normalize_coords(coords):
    """
    Normalizes coordinates in (lon, lat) tuple format.
    Supports lists, tuples and strings such as “(lon, lat)” or “lon,lat”.
    """
    normalized = []

    for coord in coords:
        try:
            # Liste oder Tupel
            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                lon, lat = float(coord[0]), float(coord[1])

            # String-Formate
            elif isinstance(coord, str):
                cleaned = coord.replace("(", "").replace(")", "").strip()
                parts = cleaned.split(",")
                if len(parts) == 2:
                    lon, lat = float(parts[0]), float(parts[1])
                else:
                    raise ValueError("String could not be devided in two parts")

            else:
                raise ValueError("Type not supported")

            normalized.append((lon, lat))

        except Exception as e:
            print(f"Warning: Coordinate in wrong format skipped: {coord} ({e})")

    return normalized


@app.cell
def _(gpd, requests):
    # Snap Hub-coordinates

    HUB_ID = "hub"
    HUB_COORDS = (9.999129, 49.772268)

    lon, lat = HUB_COORDS
    snap_url = f"http://localhost:5000/nearest/v1/car/{lon},{lat}?number=1"

    try:
        snap_response = requests.get(snap_url)
        snap_response.raise_for_status()
        hub_coord_snapped = snap_response.json()["waypoints"][0]["location"]
    except Exception as e:
        print(f"Snapping error: {e}")

    #-----------------------------
    # Load snapped APL coordinates
    #-----------------------------

    apl_gdf = gpd.read_file("./Data/apl_candidates_clusters_snapped.geojson")
    apl_ids = apl_gdf["Gitter_ID_100m"].tolist()
    apl_coords = apl_gdf["coord_car"].tolist()

    # Normalize coords and append Hub

    apl_coords = normalize_coords(apl_coords)
    hub_coord_snapped = normalize_coords([hub_coord_snapped])[0]
    print("Snapped hub-coordinate:", hub_coord_snapped)

    apl_coords.append(hub_coord_snapped)
    apl_ids.append(HUB_ID)

    # OSRM erwartet Koordinaten als LON,LAT → joinen mit ;
    coord_string = ";".join([f"{lon},{lat}" for lon, lat in apl_coords])


    #---------------------------------------
    # Load snapped heuristic APL coordinates
    #---------------------------------------

    heuristic_gdf = gpd.read_file("./Data/heuristic_result_with_customers_snapped.geojson")
    heuristic_ids = heuristic_gdf["Gitter_ID_100m"].tolist()
    heuristic_coords = heuristic_gdf["coord_car"].tolist()

    # Normalize coords and append Hub

    heuristic_coords = normalize_coords(heuristic_coords)

    heuristic_coords.append(hub_coord_snapped)
    heuristic_ids.append(HUB_ID)

    # OSRM erwartet Koordinaten als LON,LAT → joinen mit ;
    coord_string_heuristic = ";".join([f"{lon_},{lat_}" for lon_, lat_ in heuristic_coords])
    print(len(heuristic_ids))
    print(heuristic_ids)
    return apl_ids, coord_string, coord_string_heuristic, heuristic_ids


@app.cell
def _(apl_ids, coord_string, pd, requests):
    # Request an table-API senden

    url = f"http://localhost:5000/table/v1/driving/{coord_string}?annotations=distance"

    print("Send request to OSRM for distance matrix...")

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        distance_matrix = data["distances"]
    except Exception as e:
        print(f"Error retrieving the distance matrix: {e}")
        distance_matrix = []

    # Als DataFrame speichern
    df_dist = pd.DataFrame(distance_matrix, index=apl_ids, columns=[f"to_apl_{i}" for i in apl_ids])
    df_dist.index.name = "from_apl"

    df_dist.to_csv("./Data/cvrp_distance_matrix.csv", sep=";")
    print(f"Saved distance matrix")
    return


@app.cell
def _(coord_string_heuristic, heuristic_ids, pd, requests):
    #----------------------------
    # Heuristic
    #----------------------------
    # Request an table-API senden

    url_h = f"http://localhost:5000/table/v1/driving/{coord_string_heuristic}?annotations=distance"

    print("Send request to OSRM for distance matrix...")

    try:
        response_h = requests.get(url_h)
        response_h.raise_for_status()
        data_h = response_h.json()
        distance_matrix_heuristic = data_h["distances"]
    except Exception as e:
        print(f"Error retrieving the distance matrix: {e}")
        distance_matrix_heuristic = []

    # Als DataFrame speichern
    df_dist_h = pd.DataFrame(distance_matrix_heuristic, index=heuristic_ids, columns=[f"to_apl_{i}" for i in heuristic_ids])
    df_dist_h.index.name = "from_apl"

    df_dist_h.to_csv("./Data/cvrp_distance_matrix_heuristic.csv", sep=";")
    print(df_dist_h.shape)
    print(f"Saved distance matrix")
    return


if __name__ == "__main__":
    app.run()
