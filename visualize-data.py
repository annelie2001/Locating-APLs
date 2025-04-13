import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import requests
    import folium
    import geopandas as gpd
    import pandas as pd
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon
    import json
    from folium.plugins import Fullscreen
    import branca.colormap as cm
    import numpy as np
    return (
        Fullscreen,
        Polygon,
        cm,
        folium,
        gpd,
        json,
        mo,
        np,
        pd,
        plt,
        requests,
    )


@app.cell
def _(gpd):
    wuerzburg_poly_gdf = gpd.read_file("./Data/wuerzburg_bevoelkerung_100m.geojson")
    return (wuerzburg_poly_gdf,)


@app.cell
def _(json, requests):
    # DHL-Daten Würzburg
    latitude = 49.7913
    longitude = 9.9534

    dhl_api_key='lsl7jeOhVdcGayp7IWypamqQ4a6DMYcO'

    url = 'https://api.dhl.com/location-finder/v1/find-by-geo'
    headers = {
        'DHL-API-Key': dhl_api_key,
        'Accept': 'application/json'
    }

    # Anfrageparameter
    params_dhl = {
        'latitude': latitude,
        'longitude': longitude,
        'radius': 5000,  
        'limit': 50,
        'locationType': 'locker'
    }

    response_dhl = requests.get(url, headers=headers, params=params_dhl)

    packstations = []
    if response_dhl.status_code == 200:
        data_dhl = response_dhl.json()
        for location in data_dhl.get('locations', []):
            geo = location.get('place', {}).get('geo', {})
            address = location.get('place', {}).get('address', {})
            lat = geo.get('latitude')
            lon = geo.get('longitude')

            if lat and lon:
                packstations.append({
                    'name': location.get('name', 'Packstation'),
                    'lat': lat,
                    'lon': lon,
                    'address': f"{address.get('streetAddress', '')}, {address.get('postalCode', '')} {address.get('addressLocality', '')}"
                })
        with open('./Data/dhl-parcel-lockers_wuerzburg.json', 'w', encoding='utf-8') as f:
            json.dump(packstations, f, ensure_ascii=False, indent=2)

        print("Packstationen gespeichert.")
    
    else:
        print(f"Fehler beim Abrufen der Packstationen: {response_dhl.status_code} - {response_dhl.text}")
    return (
        address,
        data_dhl,
        dhl_api_key,
        f,
        geo,
        headers,
        lat,
        latitude,
        location,
        lon,
        longitude,
        packstations,
        params_dhl,
        response_dhl,
        url,
    )


@app.cell
def _(gpd):
    def heuristic_APLs(gdf, anzahl=1, mindestabstand_zellen=3):
        gdf_sorted = gdf.sort_values(by="Einwohner", ascending=False).copy()

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

        return gpd.GeoDataFrame(ausgewaehlte, crs=gdf.crs)
    return (heuristic_APLs,)


@app.cell
def _(mo):
    anzahl_slider = mo.ui.slider(start=0, stop=100, value=0, step=1, label="Number of APLs")
    return (anzahl_slider,)


@app.cell
def _(
    Fullscreen,
    anzahl_slider,
    cm,
    folium,
    heuristic_APLs,
    mo,
    packstations,
    wuerzburg_poly_gdf,
):
    # Heuristik-Anwendung
    anzahl = anzahl_slider.value  # Vom Slider holen
    gewaehlte_packzellen = heuristic_APLs(wuerzburg_poly_gdf, anzahl=anzahl)

    m = folium.Map(location=[49.7925, 9.9380], zoom_start=13, tiles='cartodbpositron')
    Fullscreen().add_to(m)

    # Population-GeoJSON
    colormap = cm.linear.YlOrRd_09.scale(0, 200)
    colormap.caption = 'Population per 100m-grid-cell'

    folium.GeoJson(
        './Data/wuerzburg_bevoelkerung_100m.geojson',
        name='Population',
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties']['Einwohner']),
            'color': 'black',
            'weight': 0.3,
            'fillOpacity': 0.7,
        },
        tooltip=folium.GeoJsonTooltip(fields=['Einwohner'], aliases=['Einwohner:']),
    ).add_to(m)

    colormap.add_to(m)

    # DHL Layer
    dhl_layer = folium.FeatureGroup(name='DHL Parcel Lockers', show=False)

    for i, ps in enumerate(packstations):
        folium.Marker(
        location=[ps['lat'], ps['lon']],
        popup=f"{ps['name']}<br>{ps['address']}",
        icon=folium.Icon(color='blue', icon='envelope', prefix='glyphicon')
        ).add_to(dhl_layer)

    dhl_layer.add_to(m)

    # Dynamic APL Layer (heuristisch)
    heuristic_layer = folium.FeatureGroup(name='Heuristic APL Locations', show=True)

    for _, row in gewaehlte_packzellen.iterrows():
        centroid = row['geometry'].centroid
        folium.Marker(
            location=[centroid.y, centroid.x],
            popup=f"Einwohner: {row['Einwohner']}",
            icon=folium.Icon(color='green', icon='envelope')
        ).add_to(heuristic_layer)

    heuristic_layer.add_to(m)

    folium.LayerControl(position='bottomleft', collapsed=False).add_to(m)

    # Display Map
    mo.vstack([
        mo.md("#Population map Würzburg"),
        m,
        anzahl_slider,
        mo.md(
            f"""
            - Move the slider to adjust the number of APLs 
            - Select the card layers you want to display
            - For reference: There are {(len(packstations))} DHL parcel lockers in Würzburg

            **Heuristic:**
        
            - Set up APLs, starting with the most densely populated 100m grid cell and ending with the least densely populated grid cell 
            - APLs must be a minimum of three empty grid cells apart.
            """
        )
    ], gap=2)
    return (
        anzahl,
        centroid,
        colormap,
        dhl_layer,
        gewaehlte_packzellen,
        heuristic_layer,
        i,
        m,
        ps,
        row,
    )


if __name__ == "__main__":
    app.run()
