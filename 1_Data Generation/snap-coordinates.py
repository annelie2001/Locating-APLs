

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import geopandas as gpd
    import pandas as pd
    import requests
    from shapely.geometry import Point
    from tqdm import tqdm
    return gpd, requests, tqdm


@app.cell
def _(gpd):
    OSRM_URL = "http://localhost:5000"

    apl_gdf = gpd.read_file("./Data/apl_candidates_clusters.geojson").to_crs("EPSG:4326")
    customer_gdf = gpd.read_file("./Data/wuerzburg_bevoelkerung_300m.geojson").to_crs("EPSG:4326")
    return OSRM_URL, apl_gdf, customer_gdf


@app.cell
def _(OSRM_URL, requests, tqdm):
    def snap_points_osrm(gdf, mode="foot"):
        """
        Snappt Punkte eines GeoDataFrames an das nächstgelegene routbare Netzwerk mit OSRM.

        :param gdf: GeoDataFrame mit Punkt-Geometrien (WGS84)
        :param mode: "foot" oder "car"
        :return: Liste von (lon, lat) Tuples
        """
        snapped_coords = []

        for geom in tqdm(gdf.geometry, desc=f"Snapping ({mode})", total=len(gdf)):
            lon, lat = geom.centroid.x, geom.centroid.y
            url = f"{OSRM_URL}/nearest/v1/{mode}/{lon},{lat}?number=1"

            try:
                response = requests.get(url)
                response.raise_for_status()
                snapped = response.json()
                snapped_coord = tuple(snapped["waypoints"][0]["location"])
                snapped_coords.append(snapped_coord)
            except Exception as e:
                print(f"Fehler beim Snappen von Punkt {lon},{lat}: {e}")
                snapped_coords.append((lon, lat))  # Fallback: Originalposition

        return snapped_coords
    return (snap_points_osrm,)


@app.cell
def _(apl_gdf, customer_gdf, snap_points_osrm):
    # APL-Koordinaten
    apl_gdf["coord_foot"] = snap_points_osrm(apl_gdf, mode="foot")
    apl_gdf["coord_car"] = snap_points_osrm(apl_gdf, mode="car")

    # Kunden-Koordinaten
    customer_gdf["coord_foot"] = snap_points_osrm(customer_gdf, mode="foot")
    return


@app.cell
def _(apl_gdf, customer_gdf):
    apl_gdf.to_file("./Data/apl_candidates_clusters_snapped.geojson", driver="GeoJSON")
    customer_gdf.to_file("./Data/wuerzburg_bevoelkerung_300m_snapped.geojson", driver="GeoJSON")
    return


if __name__ == "__main__":
    app.run()
