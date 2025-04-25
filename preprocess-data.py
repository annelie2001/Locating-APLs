import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Polygon, box, Point
    return Point, Polygon, box, gpd, pd


@app.cell
def _(Point, gpd):
    wuerzburg_center_wgs84 = gpd.GeoSeries([Point(9.9534, 49.7913)], crs='EPSG:4326')
    wuerzburg_center_3035 = wuerzburg_center_wgs84.to_crs('EPSG:3035')
    print(wuerzburg_center_3035)
    return wuerzburg_center_3035, wuerzburg_center_wgs84


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
    return (
        buffer,
        create_polygon,
        df,
        gdf_points,
        geometry,
        polygons,
        wuerzburg_bbox,
        wuerzburg_center,
        wuerzburg_data,
        wuerzburg_gdf,
        wuerzburg_poly_gdf,
    )


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
    grouped = wuerzburg_data.groupby(['x_200m', 'y_200m']).agg({
        'Einwohner': 'sum'
    }).reset_index()

    # Zusätzliche Infos hinzufügen
    grouped['Gitter_ID_100m'] = wuerzburg_data.groupby(['x_200m', 'y_200m']).apply(get_top_left_id).values
    #grouped['Gitter_IDs_alle_4'] = wuerzburg_data.groupby(['x_200m', 'y_200m']).apply(get_all_ids).values

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

    grouped['geometry'] = grouped.apply(create_200m_polygon, axis=1)

    # GeoDataFrame bauen
    gdf_200m = gpd.GeoDataFrame(grouped, geometry='geometry', crs='EPSG:3035')
    gdf_200m = gdf_200m.to_crs('EPSG:4326')

    # Speichern
    gdf_200m.to_file('./Data/wuerzburg_bevoelkerung_200m.geojson', driver='GeoJSON')
    print("Daten erfolgreich vorverarbeitet und als GeoJSON gespeichert.")
    return create_200m_polygon, gdf_200m, get_all_ids, get_top_left_id, grouped


if __name__ == "__main__":
    app.run()
