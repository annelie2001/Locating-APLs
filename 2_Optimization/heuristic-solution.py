

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import geopandas as gpd
    import pandas as pd
    from scipy.spatial import cKDTree
    return cKDTree, gpd, pd


@app.cell
def _(gpd):
    wuerzburg_gdf = gpd.read_file("./Data/wuerzburg_bevoelkerung_100m.geojson")
    wuerzburg_gdf_300m = gpd.read_file("./Data/wuerzburg_bevoelkerung_300m.geojson")
    num_periods = 10
    mindestabstand_zellen = 5
    anzahl = 40
    max_distanz_zellen = 1500  # 1.5km air distance
    return (
        anzahl,
        max_distanz_zellen,
        mindestabstand_zellen,
        wuerzburg_gdf,
        wuerzburg_gdf_300m,
    )


@app.cell
def _(anzahl, mindestabstand_zellen, pd, wuerzburg_gdf):
    gdf_sorted = wuerzburg_gdf.sort_values(by="Einwohner", ascending=False).copy()

    # Raster-Koordinaten berechnen (z. B. Gitterpositionen in "Zellen")
    gdf_sorted['row'] = ((gdf_sorted['y_mp_100m'] - gdf_sorted['y_mp_100m'].min()) / 100).astype(int)
    gdf_sorted['col'] = ((gdf_sorted['x_mp_100m'] - gdf_sorted['x_mp_100m'].min()) / 100).astype(int)

    belegte_zellen = []
    ausgewaehlte = []

    for _, row in gdf_sorted.iterrows():
        r, c = row['row'], row['col']

        # Prüfen, ob zu nah an bestehenden
        too_close = any(
            abs(r - br) <= mindestabstand_zellen and abs(c - bc) <= mindestabstand_zellen
            for br, bc in belegte_zellen
        )

        if not too_close:
            belegte_zellen.append((r, c))
            ausgewaehlte.append(row)
            if len(ausgewaehlte) >= anzahl:
                break

    apl_locations_heuristic = pd.DataFrame(ausgewaehlte)
    print(len(ausgewaehlte))
    apl_locations_heuristic.to_csv("./Data/heuristic_apl_setup.csv", sep=";")
    return (apl_locations_heuristic,)


@app.cell
def _(
    apl_locations_heuristic,
    cKDTree,
    gpd,
    max_distanz_zellen,
    pd,
    wuerzburg_gdf,
    wuerzburg_gdf_300m,
):
    # Vorbereitung: KDTree für die APL-Standorte erstellen
    apl_coords = apl_locations_heuristic[['x_mp_100m', 'y_mp_100m']].values
    customer_coords = wuerzburg_gdf_300m[['x_300m', 'y_300m']].values

    kdtree = cKDTree(apl_coords)

    # Zuordnung: Jeder Kunde wird der nächstgelegenen APL zugeordnet
    distances, indices = kdtree.query(customer_coords)

    # Initialisiere eine Liste für nicht bediente Kunden
    unserviced_customer_cells = []

    # Erstelle eine neue Spalte 'APL_ID_Heuristic' in wuerzburg_gdf_300m
    apl_ids = []
    for i, distance in enumerate(distances):
        if distance <= max_distanz_zellen:
            apl_index = indices[i]
            apl_id = apl_locations_heuristic.iloc[apl_index]['Gitter_ID_100m']
            apl_ids.append(apl_id)
        else:
            apl_ids.append(None)  # Oder einen anderen geeigneten Wert, z.B. -1
            unserviced_customer_cells.append(wuerzburg_gdf_300m.iloc[i]['Gitter_ID_100m'])

    wuerzburg_gdf_300m['APL_ID_Heuristic'] = apl_ids

    print(
        "Anzahl nicht bedienter Kunden:",
        len(unserviced_customer_cells),
    )

    # 1. Erstelle eine Liste der APL-IDs und zugehörigen Geometrien aus apl_locations_heuristic
    apl_geometries = apl_locations_heuristic[["Gitter_ID_100m", "geometry"]].copy()
    apl_geometries = apl_geometries.set_index("Gitter_ID_100m")

    # 2. Initialisiere ein Dictionary, um die Zuordnung von APL-ID zu Kundenzellen zu speichern
    apl_to_customer_cells = {apl_id: [] for apl_id in apl_geometries.index}

    # Iteriere über die Zeilen von wuerzburg_gdf_300m, um die Zuordnung zu erstellen
    for _, row_ in wuerzburg_gdf_300m.iterrows():
        apl_id = row_["APL_ID_Heuristic"]
        customer_id = row_["Gitter_ID_100m"]
        if apl_id is not None:
            apl_to_customer_cells[apl_id].append(customer_id)

    # 3. Füge die Liste der zugeordneten Kundenzellen als neue Spalte zu apl_geometries hinzu
    apl_geometries["zugeordnete_kundenzellen"] = apl_geometries.index.map(
        lambda apl_id: apl_to_customer_cells.get(apl_id, [])
    )

    # 4. Erstelle ein GeoDataFrame aus diesen Daten.
    apl_geometries_gdf = gpd.GeoDataFrame(apl_geometries, geometry="geometry", crs=wuerzburg_gdf.crs)
    apl_geometries_gdf = apl_geometries_gdf.reset_index()

    # 5. Speichere das GeoDataFrame als GeoJSON.
    apl_geometries_gdf.to_file("./Data/heuristic_result_with_customers.geojson", driver="GeoJSON")

    print("GeoJSON Datei erfolgreich gespeichert: ./Data/apl_locations_with_customers.geojson")

    # Erstelle eine Liste von Tupeln (APL_ID, Customer_ID) für die CSV-Datei
    apl_customer_list = []
    for apl_id, customer_ids in apl_to_customer_cells.items():
        for customer_id in customer_ids:
            apl_customer_list.append((apl_id, customer_id))

    # Erstelle ein DataFrame aus der Liste
    apl_customer_df = pd.DataFrame(apl_customer_list, columns=["APL_ID", "Customer_ID"])

    # Speichere das DataFrame als CSV-Datei
    apl_customer_df.to_csv("./Data/apl_customer_mapping_heuristic.csv", index=False, sep=";")

    print("CSV Datei erfolgreich gespeichert: ./Data/apl_customer_mapping_heuristic.csv")
    return


if __name__ == "__main__":
    app.run()
