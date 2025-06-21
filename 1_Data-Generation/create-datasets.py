import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Polygon, box, Point
    from sklearn.cluster import KMeans
    import numpy as np
    return KMeans, Point, Polygon, gpd, mo, np, pd


@app.cell
def _(Point, gpd):
    wuerzburg_center_wgs84 = gpd.GeoSeries([Point(9.9534, 49.7880)], crs='EPSG:4326')
    wuerzburg_center_3035 = wuerzburg_center_wgs84.to_crs('EPSG:3035')
    print(wuerzburg_center_3035)
    return


@app.cell
def _(mo):
    mo.md(r"""##100m-grid population data""")
    return


@app.cell
def _(Point, Polygon, gpd, pd):
    df = pd.read_csv("./Data/Zensus_Bevoelkerung_100m-Gitter.csv", sep=';')

    # Convert data type
    df['x_mp_100m'] = pd.to_numeric(df['x_mp_100m'])
    df['y_mp_100m'] = pd.to_numeric(df['y_mp_100m'])
    df['Einwohner'] = pd.to_numeric(df['Einwohner'])

    # Remove invalid values
    df = df[df['Einwohner'] >= 0]

    # Create list of point objects
    geometry = gpd.points_from_xy(df['x_mp_100m'], df['y_mp_100m'])
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:3035')

    # Bounding box around Würzburg center
    wuerzburg_center = Point(4317644.055, 2964301.536)
    buffer = 4500  # Radius in Metern
    wuerzburg_bbox = wuerzburg_center.buffer(buffer).envelope
    wuerzburg_gdf = gpd.GeoDataFrame(geometry=[wuerzburg_bbox], crs='EPSG:3035')

    # Add points to gdf
    wuerzburg_data = gpd.sjoin(gdf_points, wuerzburg_gdf, predicate='within')
    print(f"Gefunden: {len(wuerzburg_data)} Gitterzellen für Würzburg")
    wuerzburg_data = wuerzburg_data.drop(columns=['index_right'])

    # Convert points to 100x100m boxes
    def create_polygon(row):
        x = row['x_mp_100m'] - 50
        y = row['y_mp_100m'] - 50
        return Polygon([
            (x, y),
            (x + 100, y),
            (x + 100, y + 100),
            (x, y + 100)
        ])

    # Create new GeoDataFrame with polygons instead of points
    polygons = [create_polygon(row) for _, row in wuerzburg_data.iterrows()]
    wuerzburg_poly_gdf = gpd.GeoDataFrame(
        wuerzburg_data.drop(columns=['geometry']), 
        geometry=polygons,
        crs='EPSG:3035'
    )

    # Convert to WGS84 and save
    wuerzburg_poly_gdf = wuerzburg_poly_gdf.to_crs('EPSG:4326')

    wuerzburg_poly_gdf.to_file('./Data/wuerzburg_bevoelkerung_100m.geojson', driver='GeoJSON')

    print("Data successfully pre-processed and saved as GeoJSON.")
    return wuerzburg_data, wuerzburg_poly_gdf


@app.cell
def _(mo):
    mo.md(r"""##300m-grid population data""")
    return


@app.cell
def _(Polygon, get_top_left_id, gpd, wuerzburg_data):
    wuerzburg_data['x_300m'] = (wuerzburg_data['x_mp_100m'] // 300) * 300
    wuerzburg_data['y_300m'] = (wuerzburg_data['y_mp_100m'] // 300) * 300

    # Group and aggregate
    grouped_300m = wuerzburg_data.groupby(['x_300m', 'y_300m']).agg({
        'Einwohner': 'sum'
    }).reset_index()

    # Add ID
    grouped_300m['Gitter_ID_100m'] = wuerzburg_data.groupby(['x_300m', 'y_300m']).apply(get_top_left_id).values

    # Create 300m-grid geometries
    def create_300m_polygon(row):
        x = row['x_300m']
        y = row['y_300m']
        return Polygon([
            (x, y),
            (x + 300, y),
            (x + 300, y + 300),
            (x, y + 300)
        ])

    grouped_300m['geometry'] = grouped_300m.apply(create_300m_polygon, axis=1)

    gdf_300m = gpd.GeoDataFrame(grouped_300m, geometry='geometry', crs='EPSG:3035')
    gdf_300m = gdf_300m.to_crs('EPSG:4326')

    gdf_300m.to_file('./Data/wuerzburg_bevoelkerung_300m.geojson', driver='GeoJSON')
    print("Data successfully pre-processed and saved as GeoJSON.")
    return


@app.cell
def _(mo):
    mo.md("""##Population clusters as APL locations""")
    return


@app.cell
def _(KMeans, gpd, np, wuerzburg_poly_gdf):
    # Extract grid cell centers
    coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in wuerzburg_poly_gdf.geometry])

    # Calculate city center as mean of all centers
    center = np.mean(coords, axis=0)

    # Distances to city center
    distances = np.linalg.norm(coords - center, axis=1)

    # Weights based on distance to city center (close --> higher weight)
    distance_weights = 1 / (distances + 1)  # +1 to avoid dividing by o

    # Combine population-weights and distance-weights --> prefers central and densly-populated areas
    combined_weights = wuerzburg_poly_gdf["Einwohner"] * distance_weights

    # KMeans with combined weights
    n_clusters = 60
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords, sample_weight=combined_weights)
    wuerzburg_poly_gdf["cluster"] = kmeans.labels_

    cluster_centers = kmeans.cluster_centers_
    apl_geoms = []
    apl_records = []

    for cluster_id in range(n_clusters):
        cluster_cells = wuerzburg_poly_gdf[wuerzburg_poly_gdf["cluster"] == cluster_id]
        cell_coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in cluster_cells.geometry])

        nearest_idx = np.argmin(np.linalg.norm(cell_coords - cluster_centers[cluster_id], axis=1))
        nearest_cell = cluster_cells.iloc[nearest_idx]

        original_idx = cluster_cells.iloc[nearest_idx].name  # Original DataFrame Index
        array_idx = wuerzburg_poly_gdf.index.get_loc(original_idx)  # Position im Array

        apl_geoms.append(nearest_cell.geometry)
        apl_records.append({
            "Gitter_ID_100m": nearest_cell["Gitter_ID_100m"],
            "x_mp_100m": nearest_cell["x_mp_100m"],
            "y_mp_100m": nearest_cell["y_mp_100m"],
            "Einwohner": nearest_cell["Einwohner"],
            "cluster": cluster_id
        })

    # GeoDataFrame with cluster centers
    apl_candidates_gdf = gpd.GeoDataFrame(apl_records, geometry=apl_geoms, crs=wuerzburg_poly_gdf.crs)
    apl_candidates_gdf.to_file("./Data/apl_candidates_clusters.geojson", driver="GeoJSON")

    print("Data successfully pre-processed and saved as GeoJSON.")
    return


if __name__ == "__main__":
    app.run()
