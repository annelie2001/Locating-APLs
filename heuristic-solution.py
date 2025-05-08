

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
    anzahl = 36
    return (
        anzahl,
        mindestabstand_zellen,
        num_periods,
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
    return (apl_locations_heuristic,)


@app.cell
def _(apl_locations_heuristic, cKDTree, num_periods, pd, wuerzburg_gdf_300m):
    # Vorbereitung: KDTree für die APL-Standorte erstellen
    apl_coords = apl_locations_heuristic[['x_mp_100m', 'y_mp_100m']].values
    customer_coords = wuerzburg_gdf_300m[['x_300m', 'y_300m']].values

    kdtree = cKDTree(apl_coords)

    # Zuordnung: Jeder Kunde wird der nächstgelegenen APL zugeordnet
    distances, indices = kdtree.query(customer_coords)
    wuerzburg_gdf_300m['apl_index_heuristic'] = indices

    # Erstelle eine Mapping von Gitter_ID_100m zu APL_ID
    apl_locations_heuristic['apl_id_heuristic'] = range(len(apl_locations_heuristic))
    apl_id_mapping = apl_locations_heuristic.set_index('Gitter_ID_100m')['apl_id_heuristic'].to_dict()

    # Erstelle eine neue Spalte 'APL_ID_Heuristic' in wuerzburg_gdf_300m
    def get_apl_id(index):
        try:
            apl_id = apl_locations_heuristic.iloc[index]['Gitter_ID_100m']
            return apl_id
        except IndexError:
            return None  # Oder einen anderen geeigneten Wert, z.B. -1

    wuerzburg_gdf_300m['APL_ID_Heuristic'] = wuerzburg_gdf_300m['apl_index_heuristic'].apply(get_apl_id)
    #print("wuerzburg_gdf_300m head after assigning APL_ID_Heuristic:\n", wuerzburg_gdf_300m.head())

    # Erstelle results_df_heuristic
    results_list = []
    for period in range(1, num_periods + 1): 
        for _, row_ in wuerzburg_gdf_300m.iterrows():
            customer_id = row_['Gitter_ID_100m']
            apl_id = row_['APL_ID_Heuristic']
            results_list.append({'Period': period, 'APL_ID': apl_id, 'Customer_ID': customer_id})

    results_df_heuristic = pd.DataFrame(results_list)
    results_df_heuristic.to_csv("./Data/heuristic_results.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    app.run()
