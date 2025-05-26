

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
    return Fullscreen, alt, cm, folium, gpd, json, mo, np, pd, pysd, requests


@app.cell
def _(gpd, pd):
    wuerzburg_gdf = gpd.read_file("./Data/wuerzburg_bevoelkerung_100m.geojson")
    wuerzburg_gdf_200m = gpd.read_file("./Data/wuerzburg_bevoelkerung_200m.geojson")
    potential_locations_gdf = gpd.read_file("./Data/apl_candidates_clusters.geojson")

    #optimization results
    results_df = pd.read_csv("./Data/combined_results_with_setup_costs.csv", sep=";")
    apl_ids = results_df['APL_ID'].drop_duplicates()
    cust_ids = results_df['Customer_ID'].drop_duplicates()
    apl_gdf = wuerzburg_gdf[wuerzburg_gdf['Gitter_ID_100m'].isin(apl_ids)].copy()
    cust_gdf = wuerzburg_gdf[wuerzburg_gdf['Gitter_ID_100m'].isin(cust_ids)].copy()
    apl_gdf = apl_gdf.rename(columns={'Gitter_ID_100m': 'APL_ID', 'geometry': 'APL_geometry'})
    cust_gdf = cust_gdf.rename(columns={'Gitter_ID_100m': 'Customer_ID', 'geometry': 'Customer_geometry'})
    results_df = results_df.merge(apl_gdf[['APL_ID', 'APL_geometry']], on='APL_ID', how='left')
    results_df = results_df.merge(cust_gdf[['Customer_ID', 'Customer_geometry']], on='Customer_ID', how='left')

    summary_df = pd.read_csv("./Data/cflp_apl_deployment_summary.csv", sep=";")
    return potential_locations_gdf, results_df, summary_df, wuerzburg_gdf


@app.cell
def _(wuerzburg_gdf):
    print(len(wuerzburg_gdf["Gitter_ID_100m"]))
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
    anzahl_slider = mo.ui.slider(start=0, stop=50, value=0, step=1, label="Number of APLs")
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
    wuerzburg_gdf,
):
    # Heuristik-Anwendung
    anzahl = anzahl_slider.value  # Vom Slider holen
    gewaehlte_packzellen = heuristic_APLs(wuerzburg_gdf, anzahl=anzahl)

    m = folium.Map(location=[49.7925, 9.9380], zoom_start=13, tiles='cartodbpositron')
    Fullscreen().add_to(m)

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

    # colormap.add_to(m)

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

    folium.LayerControl(collapsed=True, position='bottomleft').add_to(m)

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
    return colormap, dhl_layer, layer_100m


@app.cell
def _():
    constant_variables = ["Population", "Population growth rate", '"E-shopper share"', ''"E-shoppers growth rate"'', "Online purchase growth rate", "APL market growth rate", '"Avg. parcels per APL per month"']

    dynamic_variables = ["Market Size", '"Potential e-customers"', "APL users", "Online purchases per month", "Number of deliveries", "Number of APLs"]
    return (dynamic_variables,)


@app.cell
def _(pysd):
    sd_model = pysd.read_vensim("./Vensim-Model/APL-SFD-Würzburg-V2.mdl")
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
        mo.image(src="./Vensim-Model/APL-SFD-Würzburg-V2.png",
                width=450, style={"margin-right": "auto", "margin-left": "auto"}),
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
    scenario_long_df['Scenario_Number'] = scenario_long_df['Scenario'].str.replace('Scenario', '').astype(int)

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
        color=alt.Color("Scenario_Number:N", scale=color_scale, legend=alt.Legend(title="Scenario")),
        tooltip=['Scenario', 'Month', 'Demand'] 
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


@app.cell
def _(
    Fullscreen,
    colormap,
    dhl_layer,
    folium,
    layer_100m,
    mo,
    potential_locations_gdf,
    results_df,
    summary_df,
):
    m1 = folium.Map(location=[49.7925, 9.9380], zoom_start=13, tiles='cartodbpositron')
    layer_100m.add_to(m1)
    dhl_layer.add_to(m1)
    Fullscreen().add_to(m1)

    # 300m layer
    layer_300m = folium.FeatureGroup(name='Population 300m-Grid', overlay=True, control=True, show=False)
    folium.GeoJson(
        './Data/wuerzburg_bevoelkerung_300m.geojson',
        name='Population 300m-Grid',
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties']['Einwohner']),
            'color': 'black',
            'weight': 0.3,
            'fillOpacity': 0.7,
        },
        tooltip=folium.GeoJsonTooltip(fields=['Einwohner'], aliases=['Einwohner:']),
    ).add_to(layer_300m)
    layer_300m.add_to(m1)

    # Potential Locations Layer
    cluster_layer = folium.FeatureGroup(name='Potential APL Locations (Population Clusters)', show=True)
    apl_centroids = potential_locations_gdf.copy()
    apl_centroids["geometry"] = apl_centroids["geometry"].centroid

    for idx, pt in apl_centroids.iterrows():
        lat1, lon1 = pt['geometry'].y, pt['geometry'].x
        folium.CircleMarker(
            location=(lat1, lon1),
            popup=f"APL-ID:{pt['Gitter_ID_100m']}",
            radius=4,
            color="blue",
            fill=True,
            fill_opacity=0.8
        ).add_to(cluster_layer)

    cluster_layer.add_to(m1)

    # Underperformer
    underutilized_apls = summary_df[summary_df["Underutilized_Most_Periods"] == True]["APL_ID"].tolist()
    highlighted_apls = apl_centroids[apl_centroids["Gitter_ID_100m"].isin(underutilized_apls)]

    low_utilization_layer = folium.FeatureGroup(name="Low capacity utilization APLs", show=False)

    for _, apl in highlighted_apls.iterrows():
        folium.Marker(
            location=[apl['geometry'].y, apl['geometry'].x],
            popup="Underutilized in >50% of periods",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(low_utilization_layer)

    low_utilization_layer.add_to(m1)

    # Für jede Periode ein Layer
    for t in sorted(results_df['Period'].unique()):
        group = folium.FeatureGroup(name=f"Period {t}", show=False)

        df_t = results_df[results_df['Period'] == t]

        for apl_id, sub_df in df_t.groupby("APL_ID"):
            apl_point = sub_df['APL_geometry'].iloc[0].centroid
            popup_text = f"APL {apl_id}<br>Kunden: {sub_df['Customer_ID'].nunique()}"
            folium.CircleMarker(
                location=[apl_point.y, apl_point.x],
                radius=4,
                color="#3182bd",
                fill=True,
                fill_opacity=0.8,
                popup=popup_text
            ).add_to(group)

            # Verbindungen zu Kunden
            for _, row_ in sub_df.iterrows():
                apl_centroid = row_['APL_geometry'].centroid
                cust_centroid = row_['Customer_geometry'].centroid
                folium.PolyLine(
                    locations=[[apl_centroid.y, apl_centroid.x],
                               [cust_centroid.y, cust_centroid.x]],
                    color="#3182bd",
                    weight=1
                ).add_to(group)

        group.add_to(m1)

    folium.LayerControl(collapsed=True, position="bottomleft").add_to(m1)

    mo.vstack([
        mo.md(
            f"""
            # Optimization

            In the next step, I modeled the problem as a Capacitated **Facility Location Problem (CFLP)**. The initial dataset consists of 100-meter grid cells across the city of Würzburg, each representing a potential facility site or demand point. However, including all grid cells as possible facility locations would result in an intractably large number of decision variables.  
            To address this issue, I used clustering techniques to reduce the candidate locations to **60 population-based clusters**, which serve as aggregated potential APL sites. For the demand sites, I started of with a **300-meter grid**. The resulting optimization model is implemented in Pyomo and solved using the CPLEX solver.  
            I solve the problem over ten periods (first month of ten consecutive years).
            In total, **{summary_df['Total_APLs_Opened'].sum()} APLs** are deployed.
            """
        ),
        m1
    ], align="center", gap=2)

    return


if __name__ == "__main__":
    app.run()
