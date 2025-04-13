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
    wuerzburg_center = Point(4315908, 2964650)
    buffer = 6000  # Radius in Metern
    wuerzburg_bbox = wuerzburg_center.buffer(buffer).envelope
    wuerzburg_gdf = gpd.GeoDataFrame(geometry=[wuerzburg_bbox], crs='EPSG:3035')

    # Räumlicher Filter
    wuerzburg_data = gpd.sjoin(gdf_points, wuerzburg_gdf, predicate='within')
    print(f"Gefunden: {len(wuerzburg_data)} Gitterzellen für Würzburg")

    # Nur die relevanten Spalten behalten und index_right entfernen, die durch den Join hinzugefügt wurde
    wuerzburg_data = wuerzburg_data.drop(columns=['index_right'])

    # Jetzt die Punkte in Polygone umwandeln
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


if __name__ == "__main__":
    app.run()
