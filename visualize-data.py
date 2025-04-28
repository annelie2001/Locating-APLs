

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import requests
    import json
    import folium
    from folium.plugins import Fullscreen
    import pandas as pd
    import geopandas as gpd
    import altair as alt
    from shapely.geometry import Polygon
    import branca.colormap as cm
    import pysd
    import numpy as np
    return alt, cm, folium, gpd, json, mo, np, pd, pysd, requests


@app.cell
def _(gpd):
    wuerzburg_gdf = gpd.read_file("./Data/wuerzburg_bevoelkerung_100m.geojson")
    wuerzburg_gdf_200m = gpd.read_file("./Data/wuerzburg_bevoelkerung_200m.geojson")
    wuerzburg_gdf_300m = gpd.read_file("./Data/wuerzburg_bevoelkerung_300m.geojson")
    return wuerzburg_gdf, wuerzburg_gdf_200m, wuerzburg_gdf_300m


@app.cell
def _(wuerzburg_gdf, wuerzburg_gdf_200m, wuerzburg_gdf_300m):
    print(wuerzburg_gdf.shape)
    print(wuerzburg_gdf_200m.shape)
    print(wuerzburg_gdf_300m.shape)
    return


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
    return (packstations,)


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
    anzahl_slider,
    cm,
    folium,
    heuristic_APLs,
    mo,
    packstations,
    wuerzburg_gdf,
):
    # Heuristik-Anwendung
    anzahl = anzahl_slider.value  # Vom Slider holen
    gewaehlte_packzellen = heuristic_APLs(wuerzburg_gdf, anzahl=anzahl)

    m = folium.Map(location=[49.7925, 9.9380], zoom_start=13, tiles='cartodbpositron')
    #Fullscreen().add_to(m)

    # Population-GeoJSON
    colormap = cm.linear.YlOrRd_09.scale(0, 200)
    colormap.caption = 'Population per grid-cell'

    layer_100m = folium.FeatureGroup(name='Population 100m-Grid', overlay=True, control=True)
    folium.GeoJson(
        './Data/wuerzburg_bevoelkerung_100m.geojson',
        name='Population 100m-grid',
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties']['Einwohner']),
            'color': 'black',
            'weight': 0.3,
            'fillOpacity': 0.7,
        },
        tooltip=folium.GeoJsonTooltip(fields=['Einwohner'], aliases=['Einwohner:']),
    ).add_to(layer_100m)
    layer_100m.add_to(m)

    layer_200m = folium.FeatureGroup(name='Population 200m-Grid', overlay=True, control=True, show=False)
    folium.GeoJson(
        './Data/wuerzburg_bevoelkerung_200m.geojson',
        name='Population 200m-Grid',
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties']['Einwohner']),
            'color': 'black',
            'weight': 0.3,
            'fillOpacity': 0.7,
        },
        tooltip=folium.GeoJsonTooltip(fields=['Einwohner'], aliases=['Einwohner:']),
    ).add_to(layer_200m)
    layer_200m.add_to(m)

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

    mo.vstack([
        mo.md(
            r"""
            #Würzburg Status Quo: Population and APL Landscape

            ##Population Map Würzburg
            In order to plan the optimal placement of parcel lockers (APLs) in Würzburg, an understanding of the population distribution is crucial. The interactive map below visualizes population density at the level of 100m x 100m grid cells according to the Zensus 2022 data.
            """),
        m,
        anzahl_slider,
        mo.md(
            f"""
            To evaluate the effectiveness of different location strategies, two additional layers are displayed:

            * **DHL Packstations:** The locations of existing DHL Packstations serve as a real-world benchmark and enable a comparison with alternative approaches.
            * **Heuristic APL locations:** As a second benchmark, I applied a heurisitc method based on population density to identify potentially optimal APL locations. This method places APLs in the most densely populated areas while maintaining a minimum distance between locations.  

            (For reference: There are {(len(packstations))} DHL parcel lockers in Würzburg)
            """
        )
    ], gap=2)
    return


@app.cell
def _():
    constant_variables = ["Population", "Population growth rate", '"E-shopper share"', ''"E-shoppers growth rate"'', "Online purchase growth rate", "APL market growth rate", '"Avg. parcels per APL per month"']

    dynamic_variables = ["Market Size", '"Potential e-customers"', "APL users", "Purchases per month", "Number of deliveries", "Number of APLs"]
    return (dynamic_variables,)


@app.cell
def _(pysd):
    sd_model = pysd.read_vensim("./Vensim-Model/APL-SFD-Würzburg-V1.mdl")
    model_overview = sd_model.doc
    #print(model_overview)
    simulation_results = sd_model.run().reset_index()
    #print(simulation_results.head())
    return (simulation_results,)


@app.cell
def _(dynamic_variables, np, simulation_results):
    filtered_simulation_results = simulation_results[dynamic_variables].copy()

    for col in dynamic_variables:
        filtered_simulation_results[col] = filtered_simulation_results[col].apply(np.floor)
    return (filtered_simulation_results,)


@app.cell
def _(alt, mo, simulation_results):
    deliveries_chart = alt.Chart(simulation_results).mark_line().encode(
        x=alt.X("time:Q"),
        y=alt.Y("Number of deliveries:Q")
    )

    deliveries_chart = mo.ui.altair_chart(deliveries_chart, chart_selection="point")

    apl_users_chart = alt.Chart(simulation_results).mark_line().encode(
        x=alt.X("time:Q"),
        y=alt.Y("APL users")
    )

    apl_users_chart = mo.ui.altair_chart(apl_users_chart, chart_selection="point")

    market_size_chart = alt.Chart(simulation_results).mark_line().encode(
        x=alt.X("time:Q"),
        y=alt.Y("Market Size:Q")
    )

    market_size_chart = mo.ui.altair_chart(market_size_chart, chart_selection="point")
    return apl_users_chart, deliveries_chart, market_size_chart


@app.cell
def _(
    apl_users_chart,
    deliveries_chart,
    filtered_simulation_results,
    market_size_chart,
    mo,
):
    mo.vstack([
        mo.md(
            r"""
            #System Dynamics Model  
            In the next step, I adapted the Stocks and Flows Diagram by Rabe et al. to Würzburg to simulate the demand for APLs over the next 10 years.
            """),
        mo.image(src="./Vensim-Model/APL-SFD-Würzburg-V1.png",
                width=600, style={"margin-right": "auto", "margin-left": "auto"}),
        mo.md("##Simulation results"),
        mo.hstack([
            market_size_chart,
            apl_users_chart,
            deliveries_chart,
        ]),
        mo.callout(filtered_simulation_results),
    ], align="center")
    return


@app.cell
def _(alt, pd):
    scenario_long_df = pd.read_csv("./Data/generated_scenarios.csv", sep=";")
    scenario_long_df["Month"] = scenario_long_df.index
    scenario_long_df = scenario_long_df.melt(id_vars="Month", var_name="Scenario", value_name="Demand")

    color_scale = alt.Scale(scheme="redyellowgreen")
    return color_scale, scenario_long_df


@app.cell
def _(scenario_long_df):
    scenario_options = scenario_long_df["Scenario"].unique()

    default_scenarios = ["Scenario1", "Scenario10", "Scenario20"]
    return default_scenarios, scenario_options


@app.cell
def _(default_scenarios, mo, scenario_options):
    scenario_selection = mo.ui.multiselect(
        scenario_options, label="Select the scenarios you want to display", value=default_scenarios
    )
    return (scenario_selection,)


@app.cell
def _(alt, color_scale, scenario_long_df, scenario_selection):
    filtered_df = scenario_long_df[scenario_long_df["Scenario"].isin(scenario_selection.value)]

    interactive_chart = alt.Chart(filtered_df).mark_line().encode(
        x=alt.X("Month:Q", title="Month"),
        y=alt.Y("Demand:Q", title="Demand"),
        color=alt.Color("Scenario:N", scale=color_scale, legend=alt.Legend(title="Scenario")),
    ).properties(
        width=700,
        height=400,
        title="Generated demand scenarios"
    )
    return (interactive_chart,)


@app.cell
def _(interactive_chart, mo, scenario_selection):
    mo.vstack([
        mo.md(
            r"""
            ##Demand Scenarios

            After getting an idea of the demand development over the next 10 years, I created 20 demand scenarios following a normal distribution, with increasing uncertainty in future periods. Demand refers to the number of deliveries to APLs per month. The scenarios are ordered from pessimistic to optimistic. 
            The scenario and period dependent demand is modeled according to the following equation:

            $$
            D_{ts} = \frac{\delta}{\log(t + 1)} \cdot t \cdot \mu_{ts}
            $$

            with $\delta = 0.003$  
            and

            $$
            \mu_{ts} = D_t \cdot f_s
            $$ 

            with $D_t$ being the demand in month t according to the simulation and $f_s$ being a scenario dependent factor between 0.9 and 1.1

            """
        ),
        interactive_chart,
        scenario_selection
    ], gap=2, align="center")
    return


if __name__ == "__main__":
    app.run()
