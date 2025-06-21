import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import geopandas as gpd
    import pandas as pd
    import requests
    from tqdm import tqdm
    import itertools

    # Paths
    CUST_PATH = "./Data/wuerzburg_bevoelkerung_300m_snapped.geojson"
    APL_PATH = "./Data/apl_candidates_clusters_snapped.geojson"
    OUTPUT_PATH = "./Data/walking_distance_matrix.csv"
    OSRM_URL = "http://localhost:5000"
    return APL_PATH, CUST_PATH, OSRM_URL, OUTPUT_PATH, gpd, pd, requests, tqdm


@app.cell
def _(APL_PATH, CUST_PATH, gpd):
    # Load GeoData
    apl_gdf = gpd.read_file(APL_PATH)
    cust_gdf = gpd.read_file(CUST_PATH)

    # Extract IDs and coordinates
    apl_ids = apl_gdf["Gitter_ID_100m"].tolist()
    apl_coords = apl_gdf["coord_foot"].tolist()

    cust_ids = cust_gdf["Gitter_ID_100m"].tolist()
    cust_coords = cust_gdf["coord_foot"].tolist()

    # Convert coordinates to tuple (lon, lat) if necessary
    apl_coords = [tuple(coord) if isinstance(coord, (list, tuple)) else tuple(eval(coord)) for coord in apl_coords]
    cust_coords = [tuple(coord) if isinstance(coord, (list, tuple)) else tuple(eval(coord)) for coord in cust_coords]
    return apl_coords, apl_ids, cust_coords, cust_ids


@app.cell
def _(apl_coords, cust_coords):
    print(apl_coords[0])
    print(cust_coords[0])
    return


@app.cell
def _(OSRM_URL, pd, requests, tqdm):
    def get_walking_distances_osrm(customers, customer_ids, apls, apl_ids):
        """
        Queries OSRM for the distance to all APLs for each customer coordinate (by footpath).
        Returns a distance matrix as a DataFrame.
        """
        dist_matrix = []

        for cust_id, cust_coord in tqdm(zip(customer_ids, customers), total=len(customers), desc="Calculate walking distances"):
            all_coords = [cust_coord] + apls
            coord_string = ";".join([f"{lon},{lat}" for lon, lat in all_coords])

            url = f"{OSRM_URL}/table/v1/foot/{coord_string}?sources=0&annotations=distance"

            try:
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                distances = data["distances"][0][1:]
                dist_matrix.append(distances)
            except Exception as e:
                print(f"Fehler bei {cust_id}: {e}")
                dist_matrix.append([float("inf")] * len(apl_ids))  # Fallback

        df = pd.DataFrame(dist_matrix, index=customer_ids, columns=[f"to_apl_{aid}" for aid in apl_ids])
        df.index.name = "from_customer"
        return df

    return (get_walking_distances_osrm,)


@app.cell
def _(
    OUTPUT_PATH,
    apl_coords,
    apl_ids,
    cust_coords,
    cust_ids,
    get_walking_distances_osrm,
):
    df_dist = get_walking_distances_osrm(cust_coords, cust_ids, apl_coords, apl_ids)

    # Save
    df_dist.to_csv(OUTPUT_PATH, sep=";")
    print(f"Walking distance matrix saved to: {OUTPUT_PATH}")

    return


if __name__ == "__main__":
    app.run()
