import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium")


@app.cell
def _():
    import geopandas as gpd
    import numpy as np
    return gpd, np


@app.cell
def _(gpd):
    wuerzburg_gdf = gpd.read_file("./Data/wuerzburg_bevoelkerung_100m.geojson")
    return (wuerzburg_gdf,)


@app.cell
def _(np, wuerzburg_gdf):
    # Überprüfe das aktuelle CRS
    print("Aktuelles CRS:", wuerzburg_gdf.crs)

    # Definiere ein geeignetes projiziertes CRS für Würzburg
    # EPSG:25832 ist ein gängiges projiziertes CRS für Deutschland (ETRS89 / UTM zone 32N)
    projected_crs = "EPSG:25832"

    # Wandle das GeoDataFrame in das projizierte CRS um
    wuerzburg_gdf_projected = wuerzburg_gdf.to_crs(projected_crs)

    # Überprüfe das neue CRS
    print("Neues CRS:", wuerzburg_gdf_projected.crs)

    # Berechne die Zentroide jetzt mit dem projizierten GeoDataFrame
    centroids = wuerzburg_gdf_projected.geometry.centroid

    # Konvertiere die Zentroide in ein NumPy-Array von Koordinaten
    coords = np.array([(p.x, p.y) for p in centroids])

    # Berechne alle paarweisen Distanzen mit NumPy (vektorisiert)
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    np.save("./Data/distances.npy", distances)

    print("Distanzen berechnet und als .npy-Datei gespeichert.")
    return centroids, coords, distances, projected_crs, wuerzburg_gdf_projected


if __name__ == "__main__":
    app.run()
