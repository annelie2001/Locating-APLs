

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Polygon, box, Point
    from sklearn.cluster import KMeans
    return KMeans, Point, Polygon, gpd, pd


@app.cell
def _(Point, gpd):
    wuerzburg_center_wgs84 = gpd.GeoSeries([Point(9.9534, 49.7913)], crs='EPSG:4326')
    wuerzburg_center_3035 = wuerzburg_center_wgs84.to_crs('EPSG:3035')
    print(wuerzburg_center_3035)
    return


@app.cell
def _(Point, Polygon, gpd, pd):
    df = pd.read_csv("./Data/Zensus_Bevoelkerung_100m-Gitter.csv", sep=';')

    # Datentypen konvertieren
    df['x_mp_100m'] = pd.to_numeric(df['x_mp_100m'])
    df['y_mp_100m'] = pd.to_numeric(df['y_mp_100m'])
    df['Einwohner'] = pd.to_numeric(df['Einwohner'])

    # Filtern von ungültigen Werten
    df = df[df['Einwohner'] >= 0]

    # Punkte als Geometrien erstellen (explizit als Liste von Point-Objekten)
    geometry = gpd.points_from_xy(df['x_mp_100m'], df['y_mp_100m'])
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:3035')

    # Bounding Box automatisch um Würzburg-Mitte
    wuerzburg_center = Point(4317644.055, 2964301.536)
    buffer = 5000  # Radius in Metern
    wuerzburg_bbox = wuerzburg_center.buffer(buffer).envelope
    wuerzburg_gdf = gpd.GeoDataFrame(geometry=[wuerzburg_bbox], crs='EPSG:3035')

    # Räumlicher Filter
    wuerzburg_data = gpd.sjoin(gdf_points, wuerzburg_gdf, predicate='within')
    print(f"Gefunden: {len(wuerzburg_data)} Gitterzellen für Würzburg")

    # Nur die relevanten Spalten behalten und index_right entfernen, die durch den Join hinzugefügt wurde
    wuerzburg_data = wuerzburg_data.drop(columns=['index_right'])

    # Punkte in Polygone umwandeln
    def create_polygon(row):
        x = row['x_mp_100m'] - 50
        y = row['y_mp_100m'] - 50
        return Polygon([
            (x, y),
            (x + 100, y),
            (x + 100, y + 100),
            (x, y + 100)
        ])

    # Neue GeoDataFrame erstellen mit Polygonen statt Punkten
    polygons = [create_polygon(row) for _, row in wuerzburg_data.iterrows()]
    wuerzburg_poly_gdf = gpd.GeoDataFrame(
        wuerzburg_data.drop(columns=['geometry']),  # Entferne die alte geometry-Spalte
        geometry=polygons,  # Füge neue Polygon-Geometrien hinzu
        crs='EPSG:3035'
    )

    # Zu WGS84 konvertieren
    wuerzburg_poly_gdf = wuerzburg_poly_gdf.to_crs('EPSG:4326')

    # Als GeoJSON speichern
    wuerzburg_poly_gdf.to_file('./Data/wuerzburg_bevoelkerung_100m.geojson', driver='GeoJSON')

    print("Daten erfolgreich vorverarbeitet und als GeoJSON gespeichert.")
    return wuerzburg_data, wuerzburg_poly_gdf


@app.cell
def _(Polygon, gpd, wuerzburg_data):
    # 200m-Koordinaten bestimmen
    wuerzburg_data['x_200m'] = (wuerzburg_data['x_mp_100m'] // 200) * 200
    wuerzburg_data['y_200m'] = (wuerzburg_data['y_mp_100m'] // 200) * 200

    # Funktionen für Aggregation
    def get_top_left_id(group):
        top_left = group.sort_values(['y_mp_100m', 'x_mp_100m'], ascending=[False, True]).iloc[0]
        return top_left['Gitter_ID_100m']

    def get_all_ids(group):
        return list(group['Gitter_ID_100m'])

    # Gruppieren und aggregieren
    grouped_200m = wuerzburg_data.groupby(['x_200m', 'y_200m']).agg({
        'Einwohner': 'sum'
    }).reset_index()

    # Zusätzliche Infos hinzufügen
    grouped_200m['Gitter_ID_100m'] = wuerzburg_data.groupby(['x_200m', 'y_200m']).apply(get_top_left_id).values

    # Geometrie für 200m-Zellen erstellen
    def create_200m_polygon(row):
        x = row['x_200m']
        y = row['y_200m']
        return Polygon([
            (x, y),
            (x + 200, y),
            (x + 200, y + 200),
            (x, y + 200)
        ])

    grouped_200m['geometry'] = grouped_200m.apply(create_200m_polygon, axis=1)

    # GeoDataFrame bauen
    gdf_200m = gpd.GeoDataFrame(grouped_200m, geometry='geometry', crs='EPSG:3035')
    gdf_200m = gdf_200m.to_crs('EPSG:4326')

    # Speichern
    gdf_200m.to_file('./Data/wuerzburg_bevoelkerung_200m.geojson', driver='GeoJSON')
    print("Daten erfolgreich vorverarbeitet und als GeoJSON gespeichert.")
    return (get_top_left_id,)


@app.cell
def _(Polygon, get_top_left_id, gpd, wuerzburg_data):
    # 300m-Koordinaten bestimmen
    wuerzburg_data['x_300m'] = (wuerzburg_data['x_mp_100m'] // 300) * 300
    wuerzburg_data['y_300m'] = (wuerzburg_data['y_mp_100m'] // 300) * 300

    # Gruppieren und aggregieren
    grouped_300m = wuerzburg_data.groupby(['x_300m', 'y_300m']).agg({
        'Einwohner': 'sum'
    }).reset_index()

    # Zusätzliche Infos hinzufügen
    grouped_300m['Gitter_ID_100m'] = wuerzburg_data.groupby(['x_300m', 'y_300m']).apply(get_top_left_id).values

    # Geometrie für 300m-Zellen erstellen
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

    # GeoDataFrame bauen
    gdf_300m = gpd.GeoDataFrame(grouped_300m, geometry='geometry', crs='EPSG:3035')
    gdf_300m = gdf_300m.to_crs('EPSG:4326')

    # Speichern
    gdf_300m.to_file('./Data/wuerzburg_bevoelkerung_300m.geojson', driver='GeoJSON')
    print("Daten erfolgreich vorverarbeitet und als GeoJSON gespeichert.")
    return


@app.cell
def _(wuerzburg_poly_gdf):
    wuerzburg_poly_gdf.head()
    return


@app.cell
def _(KMeans, gdf, wuerzburg_poly_gdf):
    # Extrahiere relevante Infos
    X = wuerzburg_poly_gdf[["geometry"]].values

    # KMeans Clustering
    kmeans = KMeans(n_clusters=50, random_state=0)
    gdf["cluster"] = kmeans.fit_predict(X)

    # Cluster-Zentren abrufen
    cluster_centers = kmeans.cluster_centers_
    print(cluster_centers)
    # cluster_gdf = gpd.GeoDataFrame(geometry=[shapely.geometry.Point(xy) for xy in cluster_centers], crs="EPSG:4326")
    # cluster_gdf.to_file("./Data/potentielle_apl.geojson", driver="GeoJSON")
    return


if __name__ == "__main__":
    app.run()
